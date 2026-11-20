"""LeWM (LeWorldModel, arXiv:2603.19312) — the pixel-world-model organ.

Inference-only, checkpoint-faithful reimplementation of the pieces needed by
the published PushT checkpoint (`.work/ckpt/pusht/`). This module is a leaf:
it imports only ``torch`` and the standard library, holds no global state, and
performs no network I/O. It knows nothing about the event ledger, so P1/P3 may
call it without coupling either direction.

Provenance of every architectural decision (see `docs/phase1-lewm-voe.md` for
the full audit trail):

* encoder  — HuggingFace ``ViTModel`` as constructed by
  ``stable_pretraining.backbone.utils.vit_hf(size="tiny", patch_size=14,
  image_size=224, pretrained=False, use_mask_token=False)``. Parameter names,
  pre-LN block order, explicit qkv matmul and ``layer_norm_eps=1e-12`` mirror
  `transformers`; the pooling layer is absent, so ``last_hidden_state``
  includes the CLS token under the final LayerNorm.
* predictor / action encoder / projectors — byte-for-byte the classes in
  `.work/le-wm/module.py` (ARPredictor, ConditionalBlock, Attention,
  FeedForward, Embedder, MLP, SIGReg) with ``einops.rearrange`` spelled out
  as view/permute so the library needs no third-party dependency.
* `Block` (the non-conditional block in `module.py`) is omitted because no
  LeWM module instantiates it; consequently ``Transformer`` here is
  conditional-only, which is exactly the code path LeWM exercises.

Frozen-weights contract: the checkpoint is the measurement basis. ``encode``
and ``predict`` are pure ``@torch.no_grad()`` reads of the parameters.
``LewM.train()`` is a no-op that keeps every submodule in eval mode, so
``BatchNorm1d`` running statistics can never be mutated by an accidental
``.train()``.

Units: `encode` expects *preprocessed* pixels — float, ImageNet-normalised,
``[B,T,3,224,224]`` or ``[B,3,224,224]``. Raw ``uint8`` frames must go through
``imagenet_normalize`` first (``encode`` refuses them rather than silently
feeding 0..255 values to a model trained on normalised input). `predict`
expects *normalised* actions: the training pipeline z-scores them with the
per-dataset mean/std of the action column (see `normalize_actions`). The
PushT statistics measured from the training set are in `docs/phase1-lewm-voe.md`.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "EMBED_DIM",
    "HISTORY_SIZE",
    "DEFAULT_CHECKPOINT_DIR",
    "EncoderOutput",
    "SIGReg",
    "FeedForward",
    "Attention",
    "ConditionalBlock",
    "Transformer",
    "Embedder",
    "MLP",
    "ARPredictor",
    "ViTPatchEmbeddings",
    "ViTEmbeddings",
    "ViTSelfAttention",
    "ViTAttention",
    "ViTIntermediate",
    "ViTOutput",
    "ViTLayer",
    "ViTEncoder",
    "ViTTiny",
    "LewM",
    "imagenet_normalize",
    "normalize_actions",
    "surprise",
    "surprise_per_step",
    "checkpoint_report",
    "load_lewm",
]

# ---------------------------------------------------------------------------
# Constants pinned by the published checkpoint
# ---------------------------------------------------------------------------

#: ImageNet statistics used by `stable_pretraining`'s dataset_stats.ImageNet,
#: i.e. the normalisation the encoder was trained under.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Latent width of the published PushT model.
EMBED_DIM = 192
#: Number of context frames the predictor sees at once (``num_frames``).
HISTORY_SIZE = 3

#: Dev-time default location of the published checkpoint inside this repo.
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "LEWM_CHECKPOINT_DIR",
        Path(__file__).resolve().parent.parent / ".work" / "ckpt" / "pusht",
    )
)
DEFAULT_WEIGHTS_NAME = "weights.pt"
DEFAULT_CONFIG_NAME = "config.json"

#: ``vit_hf`` size table from stable_pretraining.backbone.utils.
_VIT_SIZES = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
}


class EncoderOutput(NamedTuple):
    """Minimal stand-in for ``BaseModelOutputWithPooling``."""

    last_hidden_state: torch.Tensor


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------


def _channel_axis_view(t: torch.Tensor) -> torch.Tensor:
    """Return a broadcastable view of (mean, std) against a ``[..., 3, H, W]`` tensor."""
    extra = t.dim() - 4  # number of leading batch dims
    return (1,) * (extra + 1) + (3, 1, 1)


def imagenet_normalize(frames: torch.Tensor) -> torch.Tensor:
    """Convert raw frames to the encoder's expected input.

    ``frames`` is ``uint8`` in ``[0, 255]`` (any leading batch dims, channel
    axis at ``-3``, 3 channels). The result is ``float32`` in the ImageNet
    normalised space used by the training pipeline:

    ``(frames / 255 - mean) / std`` with the per-channel statistics above.

    Accepts float input only if it is already in ``[0, 255]`` scale; callers
    holding ``[0, 1]`` floats should not call this twice.
    """
    if frames.dim() < 4:
        raise ValueError(
            f"imagenet_normalize expects at least [B, 3, H, W]; got shape {tuple(frames.shape)}"
        )
    if frames.size(-3) != 3:
        raise ValueError(f"expected 3 channels at axis -3; got {tuple(frames.shape)}")
    x = frames.to(torch.float32)
    if frames.dtype != torch.uint8 and frames.is_floating_point():
        mx = float(x.max()) if x.numel() else 0.0
        if mx <= 1.0:
            raise ValueError(
                "imagenet_normalize expects 0..255-scale frames; this tensor looks "
                "like it is already in 0..1. Normalise from the uint8 source instead."
            )
    x = x / 255.0
    view = _channel_axis_view(x)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(view)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(view)
    return (x - mean) / std


def normalize_actions(
    actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Z-score actions exactly as the training pipeline does.

    ``mean``/``std`` are the per-dimension statistics of the action column
    computed over the training set (``ZScoreNormalizer`` in the reference
    `.work/le-wm/utils.py`). This helper exists so callers do not re-derive
    the convention; it is pure arithmetic.
    """
    return ((actions - mean) / std).to(torch.float32)


# ---------------------------------------------------------------------------
# LeWM predictor stack (mirrors .work/le-wm/module.py)
# ---------------------------------------------------------------------------


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).

    Copied from the reference implementation. Note that ``forward`` draws
    fresh random projections on every call, so its value is stochastic even
    in eval mode; the organ does not use it at inference time (it is exposed
    for tests of the latent statistics).
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """``proj``: ``(T, B, D)``; returns a scalar statistic."""
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation used by the conditional blocks."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking.

    ``einops.rearrange`` in the reference is spelled here as
    ``view(...).transpose(...)``; the arithmetic is identical.
    """

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """``x``: ``(B, T, D)``."""
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        b, t, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        head_dim = q.size(-1) // self.heads
        q = q.view(b, t, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        # Reference initialisation (a functional no-op once the checkpoint is
        # loaded strictly; kept so a from-scratch build starts from the same
        # zero-gated block as training did).
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    """Transformer with AdaLN-zero conditioning (LeWM's inference path).

    The reference additionally supports non-conditional blocks; LeWM never
    instantiates one, so that branch is not reproduced (see module docstring).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    """Action chunk encoder: per-timestep Conv1d smoothing + 2-layer MLP."""

    def __init__(
        self,
        input_dim: int = 10,
        smoothed_dim: int = 10,
        emb_dim: int = 10,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, T, D)`` -> ``(B, T, emb_dim)``."""
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Linear -> norm -> activation -> linear (the latent projector)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn: type[nn.Module] | None = nn.LayerNorm,
        act_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction.

    Slot ``i`` of the output is the prediction for frame ``i + 1`` given
    frames ``0..i`` and actions ``0..i`` — this is the supervision the model
    was trained under (``pred_emb`` against ``emb[:, num_preds:]``), and it is
    why a rollout only consumes the last slot.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, T, d)``; ``c``: ``(B, T, act_dim)``."""
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, c)


# ---------------------------------------------------------------------------
# Encoder: HuggingFace ViTModel (add_pooling_layer=False, use_mask_token=False)
# ---------------------------------------------------------------------------


class ViTPatchEmbeddings(nn.Module):
    """Conv2d patch projection, exactly as HF ``ViTPatchEmbeddings``."""

    def __init__(self, hidden_size: int, patch_size: int, num_channels: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    """CLS token + patch embeddings + (optionally interpolated) position."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        image_size: int,
        num_channels: int,
        hidden_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_embeddings = ViTPatchEmbeddings(hidden_size, patch_size, num_channels)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Bicubic position-embedding interpolation, as HF does it."""
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        dim = embeddings.shape[-1]
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        h0 = w0 = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channels, got {num_channels}"
            )
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)


class ViTSelfAttention(nn.Module):
    """Multi-head self-attention with explicit matmul (HF arithmetic)."""

    def __init__(
        self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        ) / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        return context_layer.view(*context_layer.size()[:-2], self.all_head_size)


class ViTSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class ViTAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output = ViTSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden_states))


class ViTIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class ViTOutput(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class ViTLayer(nn.Module):
    """Pre-LN ViT block: LN -> attn -> residual -> LN -> MLP -> residual."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.attention = ViTAttention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        self.intermediate = ViTIntermediate(hidden_size, intermediate_size)
        self.output = ViTOutput(hidden_size, intermediate_size, hidden_dropout_prob)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attention(self.layernorm_before(hidden_states)) + hidden_states
        hidden_states = self.output(self.intermediate(self.layernorm_after(hidden_states))) + hidden_states
        return hidden_states


class ViTEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                ViTLayer(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    layer_norm_eps,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class ViTTiny(nn.Module):
    """The encoder half of LeWM, matching `vit_hf(..., add_pooling_layer=False)`.

    ``forward`` returns an :class:`EncoderOutput`; ``last_hidden_state`` is the
    post-``layernorm`` sequence (CLS at index 0), which is what
    ``JEPA.encode`` slices.
    """

    def __init__(
        self,
        *,
        size: str = "tiny",
        patch_size: int = 14,
        image_size: int = 224,
        num_channels: int = 3,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        intermediate_size: int | None = None,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-12,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()
        if size not in _VIT_SIZES:
            raise ValueError(f"unknown ViT size {size!r}; choose from {sorted(_VIT_SIZES)}")
        cfg = dict(_VIT_SIZES[size])
        if hidden_size is not None:
            cfg["hidden_size"] = hidden_size
        if num_hidden_layers is not None:
            cfg["num_hidden_layers"] = num_hidden_layers
        if num_attention_heads is not None:
            cfg["num_attention_heads"] = num_attention_heads

        hidden = cfg["hidden_size"]
        intermediate = intermediate_size or hidden * 4
        if use_mask_token:
            raise ValueError(
                "use_mask_token=True is not supported: the published checkpoint "
                "was built with use_mask_token=False and has no mask_token entry"
            )
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_size = hidden
        self.num_hidden_layers = cfg["num_hidden_layers"]
        self.num_attention_heads = cfg["num_attention_heads"]
        self.layer_norm_eps = layer_norm_eps

        self.embeddings = ViTEmbeddings(
            hidden, patch_size, image_size, num_channels, hidden_dropout_prob
        )
        self.encoder = ViTEncoder(
            cfg["num_hidden_layers"],
            hidden,
            cfg["num_attention_heads"],
            intermediate,
            layer_norm_eps,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
        )
        self.layernorm = nn.LayerNorm(hidden, eps=layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> EncoderOutput:
        embedding_output = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        sequence_output = self.encoder(embedding_output)
        sequence_output = self.layernorm(sequence_output)
        return EncoderOutput(last_hidden_state=sequence_output)


# ---------------------------------------------------------------------------
# The organ
# ---------------------------------------------------------------------------


def _strip_target(d: Mapping[str, Any], allowed: set[str], what: str) -> dict[str, Any]:
    """Return ``d`` restricted to ``allowed`` keys, ignoring ``_target_``."""
    out = {k: v for k, v in d.items() if k in allowed}
    unknown = set(d) - allowed - {"_target_", "_partial_"}
    if unknown:
        raise ValueError(f"unexpected {what} config keys: {sorted(unknown)}")
    return out


def _resolve_norm_fn(spec: Any) -> type[nn.Module] | None:
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        spec = spec.get("_target_")
    if spec is None:
        return None
    table = {
        "torch.nn.BatchNorm1d": nn.BatchNorm1d,
        "torch.nn.LayerNorm": nn.LayerNorm,
        "torch.nn.Identity": nn.Identity,
    }
    if spec not in table:
        raise ValueError(f"unsupported norm_fn target {spec!r}")
    return table[spec]


class LewM(nn.Module):
    """Inference-only LeWM organ built from the published checkpoint config.

    Submodule attribute names (``encoder``, ``predictor``, ``action_encoder``,
    ``projector``, ``pred_proj``) are chosen so that the state-dict keys match
    the published ``weights.pt`` exactly; loading is strict in both directions.

    All public methods are ``@torch.no_grad()`` and the module is permanently in
    eval mode; the parameters are frozen at construction.
    """

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_config = json.loads(json.dumps(dict(model_config)))

        enc = self.model_config["encoder"]
        # `pretrained` is a dev-time flag of the reference builder; the organ is
        # always built from the config and then strict-loaded from disk.
        if enc.get("pretrained", False):
            raise ValueError(
                "encoder.pretrained=True is not supported: the organ never downloads"
            )
        self.encoder = ViTTiny(
            size=enc.get("size", "tiny"),
            patch_size=enc.get("patch_size", 14),
            image_size=enc.get("image_size", 224),
            use_mask_token=enc.get("use_mask_token", False),
        )

        pred_cfg = _strip_target(
            self.model_config["predictor"],
            {
                "num_frames",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "depth",
                "heads",
                "mlp_dim",
                "dim_head",
                "dropout",
                "emb_dropout",
            },
            "predictor",
        )
        self.predictor = ARPredictor(**pred_cfg)

        act_cfg = _strip_target(
            self.model_config["action_encoder"], {"input_dim", "emb_dim"}, "action_encoder"
        )
        self.action_encoder = Embedder(**act_cfg)

        self.projector = self._build_mlp("projector")
        self.pred_proj = self._build_mlp("pred_proj")

        #: Latent width implied by the built projector (192 for PushT).
        self.latent_dim = self.projector.net[0].in_features
        #: Flattened action-chunk width implied by the action encoder (10 for PushT).
        self.action_dim = self.action_encoder.patch_embed.in_channels

        self.eval()
        self.requires_grad_(False)

    def _build_mlp(self, key: str) -> MLP:
        cfg = self.model_config[key]
        if cfg.get("_target_", "").rsplit(".", 1)[-1] != "MLP":
            raise ValueError(f"{key} must be an MLP in module.py; got {cfg.get('_target_')}")
        kwargs = _strip_target(
            cfg, {"input_dim", "hidden_dim", "output_dim", "norm_fn", "act_fn"}, key
        )
        norm = _resolve_norm_fn(kwargs.pop("norm_fn", None))
        act = kwargs.pop("act_fn", None)
        if act is not None and act not in ("torch.nn.GELU",):
            raise ValueError(f"unsupported act_fn target {act!r}")
        return MLP(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs.get("output_dim"),
            norm_fn=norm,
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | str | Path
    ) -> "LewM":
        """Build from the published ``config.json`` (mapping or path)."""
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        return cls(config)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path | None = None,
        *,
        weights_name: str = DEFAULT_WEIGHTS_NAME,
        config_name: str = DEFAULT_CONFIG_NAME,
        map_location: str | torch.device = "cpu",
    ) -> "LewM":
        """Build from ``config.json`` and strict-load ``weights.pt``.

        Raises ``RuntimeError`` on any missing or unexpected key, and
        ``ValueError`` if the config and the state dict disagree on shape.
        """
        root = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
        config_path = root / config_name
        weights_path = root / weights_name
        if not config_path.is_file():
            raise FileNotFoundError(f"missing LeWM config: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing LeWM weights: {weights_path}")

        model = cls.from_config(config_path)
        state_dict = torch.load(weights_path, map_location=map_location, weights_only=False)
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"{weights_path} did not contain a state dict; got {type(state_dict).__name__}"
            )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.requires_grad_(False)
        return model

    # -- frozen / eval invariants ------------------------------------------

    def train(self, mode: bool = True) -> "LewM":  # noqa: D401 - deliberate override
        """No-op override: this organ is permanently in eval mode.

        A stray ``.train()`` would make ``BatchNorm1d`` consume batch
        statistics and mutate its running buffers, silently changing the
        measurement basis. Inference-only is a property of the organ, not a
        caller convention, so the request is rejected by keeping eval mode.
        """
        super().train(False)
        return self

    # -- public pure functions ---------------------------------------------

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latents, mirroring ``JEPA.encode``.

        Parameters
        ----------
        frames:
            ``[B, T, 3, H, W]`` (or ``[B, 3, H, W]`` for a single frame),
            ``float32`` already ImageNet-normalised (see
            :func:`imagenet_normalize`). ``uint8`` input is rejected rather
            than silently mis-scaled.

        Returns
        -------
        ``[B, T, EMBED_DIM]``. For a 4-D input the time axis is kept with
        ``T = 1`` (``[B, 1, EMBED_DIM]``) so the shape contract never changes.

        The pipeline is exactly: ``pixels.float()`` -> flatten ``B*T`` -> ViT
        with ``interpolate_pos_encoding=True`` -> CLS token ->
        ``projector`` (Linear-2048-BatchNorm1d-GELU-Linear).
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError(f"frames must be a Tensor, got {type(frames).__name__}")
        squeeze = frames.dim() == 4
        if squeeze:
            frames = frames.unsqueeze(1)
        if frames.dim() != 5:
            raise ValueError(
                "frames must be [B, T, 3, H, W] or [B, 3, H, W]; "
                f"got shape {tuple(frames.shape)}"
            )
        if frames.dtype == torch.uint8:
            raise ValueError(
                "encode() expects ImageNet-normalised float pixels; run raw uint8 "
                "frames through amateur.lewm.imagenet_normalize() first"
            )
        b, t, c, h, w = frames.shape
        if t == 0:
            raise ValueError("frames must contain at least one timestep")
        if c != 3:
            raise ValueError(f"expected 3 channels, got {c}")
        if h != w:
            raise ValueError(f"expected square frames, got {h}x{w}")

        pixels = frames.float()
        flat = pixels.reshape(b * t, c, h, w)
        out = self.encoder(flat, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        z = self.projector(cls)
        return z.reshape(b, t, -1)

    @torch.no_grad()
    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next-step latents, mirroring ``JEPA.predict``.

        Parameters
        ----------
        z:
            ``[B, T, EMBED_DIM]`` context latents (from :meth:`encode`).
        actions:
            ``[B, T, A]`` context actions, already z-scored with the training
            statistics (see :func:`normalize_actions`). ``A`` is the flattened
            action chunk (``frameskip * action_dim`` = 10 for PushT).

        Returns
        -------
        ``[B, T, EMBED_DIM]``; slot ``i`` is the prediction for frame ``i+1``.
        ``T`` may not exceed the predictor's ``num_frames`` (3 for the
        published checkpoint) because the positional table has exactly that
        many entries — a rollout must truncate to its last ``HISTORY_SIZE``
        steps, as the reference ``JEPA.rollout`` does.
        """
        if z.dim() != 3:
            raise ValueError(f"z must be [B, T, D]; got shape {tuple(z.shape)}")
        if actions.dim() != 3:
            raise ValueError(
                f"actions must be [B, T, A]; got shape {tuple(actions.shape)}"
            )
        if z.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"z {tuple(z.shape[:2])} and actions {tuple(actions.shape[:2])} "
                "must agree on batch and time"
            )
        if z.size(-1) != self.latent_dim:
            raise ValueError(
                f"z latent width must be {self.latent_dim}, got {z.size(-1)}"
            )
        if actions.size(-1) != self.action_dim:
            raise ValueError(
                f"action width must be {self.action_dim} "
                f"(frameskip * action_dim), got {actions.size(-1)}"
            )
        t = z.size(1)
        if t == 0:
            raise ValueError("predict needs at least one context step")
        if t > self.predictor.num_frames:
            raise ValueError(
                f"predict accepts at most num_frames={self.predictor.num_frames} "
                f"context steps; got T={t}. Truncate to the last "
                f"{HISTORY_SIZE} steps (as JEPA.rollout does)."
            )

        b = z.size(0)
        act_emb = self.action_encoder(actions)
        preds = self.predictor(z, act_emb)
        preds = self.pred_proj(preds.reshape(b * t, -1))
        return preds.reshape(b, t, -1)

    @torch.no_grad()
    def surprise(
        self, z_hat: torch.Tensor, z_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Squared latent prediction error (VoE "surprise").

        Returns ``(per_sample [B], mean scalar)``. Reduction: the mean of the
        squared error over the time and latent axes for each sample, then the
        mean over the batch. No square root — this is the same squared-error
        quantity as the paper's prediction loss
        ``(pred_emb - tgt_emb).pow(2).mean()``, reported per sample.
        """
        return surprise(z_hat, z_next)

    @torch.no_grad()
    def surprise_per_step(
        self, z_hat: torch.Tensor, z_next: torch.Tensor
    ) -> torch.Tensor:
        """Per-timestep surprise ``[B, T]`` (mean squared error over latents)."""
        return surprise_per_step(z_hat, z_next)


# ---------------------------------------------------------------------------
# Model-free pure functions
# ---------------------------------------------------------------------------


def surprise(
    z_hat: torch.Tensor, z_next: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared latent prediction error, per sample and as a scalar mean.

    ``z_hat`` and ``z_next`` must have identical shape, either ``[B, D]`` or
    ``[B, T, D]``. Reduction: mean over all non-batch axes per sample, then
    mean over the batch. Returns ``(per_sample [B], mean scalar)``.
    """
    if z_hat.shape != z_next.shape:
        raise ValueError(
            f"z_hat and z_next must have the same shape; got {tuple(z_hat.shape)} "
            f"and {tuple(z_next.shape)}"
        )
    if z_hat.dim() not in (2, 3):
        raise ValueError(f"expected [B, D] or [B, T, D]; got {tuple(z_hat.shape)}")
    sq = (z_hat.to(torch.float32) - z_next.to(torch.float32)).pow(2)
    per_sample = sq.flatten(1).mean(dim=1)
    return per_sample, per_sample.mean()


def surprise_per_step(z_hat: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
    """Per-timestep surprise ``[B, T]``: mean squared error over latents."""
    if z_hat.shape != z_next.shape or z_hat.dim() != 3:
        raise ValueError(
            "surprise_per_step expects two same-shaped [B, T, D] tensors; got "
            f"{tuple(z_hat.shape)} and {tuple(z_next.shape)}"
        )
    return (z_hat.to(torch.float32) - z_next.to(torch.float32)).pow(2).mean(dim=-1)


# ---------------------------------------------------------------------------
# Dev helper: inspect a published checkpoint without building a model
# ---------------------------------------------------------------------------


def checkpoint_report(
    checkpoint_dir: str | Path | None = None,
    *,
    weights_name: str = DEFAULT_WEIGHTS_NAME,
) -> dict[str, Any]:
    """Summarise a published checkpoint directory (keys, shapes, param count)."""
    root = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    state_dict = torch.load(root / weights_name, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint did not contain a state dict")
    num_params = sum(
        int(v.numel()) for v in state_dict.values() if isinstance(v, torch.Tensor)
    )
    dtypes = {str(v.dtype) for v in state_dict.values() if isinstance(v, torch.Tensor)}
    return {
        "path": str(root / weights_name),
        "num_keys": len(state_dict),
        "num_params": num_params,
        "dtypes": sorted(dtypes),
        "keys": list(state_dict.keys()),
    }


def load_lewm(
    checkpoint_dir: str | Path | None = None, **kwargs: Any
) -> LewM:
    """Convenience alias for :meth:`LewM.load_checkpoint`."""
    return LewM.load_checkpoint(checkpoint_dir, **kwargs)
