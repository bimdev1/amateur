"""A small LeWM-style JEPA for CPU study of the flippable 2-D world.

Architecture
------------
* **Encoder** -- a 3-layer strided CNN (``3 -> 16 -> 32 -> 32``, stride 2)
  followed by a linear map to ``latent_dim``.  A 96x96 frame becomes a 12x12
  feature map; a 64x64 frame becomes 8x8.  Frames may be ``uint8`` (values
  scaled by 1/255) or float in ``[0, 1]``.
* **Predictor** -- an action embedding, a 2-layer GRU over
  ``[z_t, embed(a_t)]``, and a residual MLP head that predicts
  ``z_{t+1} = z_t + delta``.  The head's last layer is zero-initialised, so
  the model starts as the identity predictor and learns the dynamics.
* **SIGReg** -- the Epps-Pulley isotropic-Gaussian regulariser from the LeWM
  reference implementation: random unit projections of the latents, the
  empirical characteristic function compared against the standard-normal one
  at ``knots`` frequencies, weighted by the reference window.  Applied to
  encoder latents ``(T, B, D)`` it penalises collapse (a constant latent makes
  the characteristic function a point mass and the statistic non-zero).

Training loss (LeWM two-loss form)::

    loss = MSE(z_hat_{t+1}, z_{t+1}) + sigreg_weight * SIGReg(z)

The **binding ruling for the epoch**: the encoder is frozen after fitting and
phase-5 overlays adapt the predictor only.  The model therefore exposes
``encoder_parameters()`` and ``predictor_parameters()`` (disjoint lists),
``freeze_encoder()`` (``requires_grad=False`` + ``eval()``, and ``train()`` is
overridden so the encoder stays in eval mode), ``predictor_optimizer()`` for
predictor-only adaptation, and checkpoints that record the frozen flag.

Size
----
Default configuration at 96x96: ~0.52M parameters total (encoder ~0.31M,
predictor ~0.21M, SIGReg 0 trainable parameters) -- comfortably below the
~2M budget.  ``param_counts()`` reports exact numbers at runtime.

Checkpoints
-----------
``save(path)`` writes a ``torch.save`` dict with the config, full
``state_dict``, the encoder-frozen flag, parameter counts, and optional JSON-
safe metadata.  ``load(path)`` uses ``weights_only=False`` because the
payload contains a config dict (no pickled callables are stored).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import RolloutBatch, sample_window_batch

__all__ = [
    "MicroLeWM",
    "MicroLeWMConfig",
    "SIGReg",
]

CHECKPOINT_FORMAT = "micro-lewm/1"


@dataclass(frozen=True)
class MicroLeWMConfig:
    """Configuration for :class:`MicroLeWM` (all fields are JSON-safe)."""

    frame_size: int = 96
    in_channels: int = 3
    latent_dim: int = 64
    encoder_widths: tuple[int, ...] = (16, 32, 32)
    predictor_hidden: int = 128
    predictor_layers: int = 2
    action_dim: int = 5
    act_emb_dim: int = 8
    sigreg_knots: int = 17
    sigreg_projections: int = 256
    sigreg_weight: float = 0.09
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_widths", tuple(int(v) for v in self.encoder_widths))
        if self.frame_size < 32 or self.frame_size % 8 != 0:
            raise ValueError("frame_size must be >= 32 and divisible by 8")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not self.encoder_widths or any(w <= 0 for w in self.encoder_widths):
            raise ValueError("encoder_widths must be positive")
        if self.predictor_hidden <= 0 or self.predictor_layers <= 0:
            raise ValueError("predictor dims must be positive")
        if self.action_dim <= 0 or self.act_emb_dim <= 0:
            raise ValueError("action dims must be positive")
        if self.sigreg_knots < 3:
            raise ValueError("sigreg_knots must be >= 3")
        if self.sigreg_projections <= 0:
            raise ValueError("sigreg_projections must be positive")
        if self.sigreg_weight < 0.0:
            raise ValueError("sigreg_weight must be non-negative")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MicroLeWMConfig":
        return cls(**dict(data))


class SIGReg(nn.Module):
    """Epps-Pulley isotropic-Gaussian regulariser (reference LeWM form).

    ``forward`` expects latents shaped ``(T, B, D)``.  It draws
    ``num_proj`` random unit-norm directions, evaluates the empirical
    characteristic function on ``knots`` frequencies in ``[0, 3]``, compares
    the real part against ``exp(-t^2 / 2)`` and the imaginary part against
    zero (the standard-normal characteristic function), weights by the
    reference triangular/window weights, and averages.  The statistic is
    multiplied by ``B`` (samples per time slice) as in the reference.
    """

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.num_proj = int(num_proj)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("SIGReg expects latents shaped (T, B, D)")
        a = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        a = a.div_(a.norm(p=2, dim=0, keepdim=True))
        x_t = (z @ a).unsqueeze(-1) * self.t  # (T, B, P, K)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * z.size(-2)
        return statistic.mean()


class _Encoder(nn.Module):
    def __init__(self, in_channels: int, widths: tuple[int, ...], latent_dim: int, frame_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for index, width in enumerate(widths):
            kernel, stride, padding = (5, 2, 2) if index == 0 else (3, 2, 1)
            layers.append(nn.Conv2d(channels, width, kernel, stride=stride, padding=padding))
            layers.append(nn.GELU())
            channels = width
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, in_channels, frame_size, frame_size))
        self.head = nn.Linear(int(probe.numel()), latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1))


class _Predictor(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        act_emb_dim: int,
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.act_embed = nn.Embedding(action_dim, act_emb_dim)
        self.gru = nn.GRU(
            input_size=latent_dim + act_emb_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + act_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict the next latent for every position of ``z``.

        ``z`` is ``(B, T, D)``, ``actions`` is ``(B, T)``; the result is
        ``(B, T, D)`` and simply ignores the last action.
        """
        emb = self.act_embed(actions)
        out, _ = self.gru(torch.cat([z, emb], dim=-1))
        delta = self.head(torch.cat([self.norm(out), z, emb], dim=-1))
        return z + delta

    def step(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.act_embed(action)
        out, hidden = self.gru(torch.cat([z_t, emb], dim=-1).unsqueeze(1), hidden)
        out = out[:, 0]
        delta = self.head(torch.cat([self.norm(out), z_t, emb], dim=-1))
        return z_t + delta, hidden


class MicroLeWM(nn.Module):
    """Micro JEPA world model: CNN encoder + GRU predictor + SIGReg."""

    def __init__(self, config: MicroLeWMConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        if config is not None and overrides:
            raise TypeError("pass either config or keyword overrides, not both")
        self.config = config if config is not None else MicroLeWMConfig(**overrides)
        torch.manual_seed(int(self.config.seed))

        self.encoder = _Encoder(
            self.config.in_channels,
            self.config.encoder_widths,
            self.config.latent_dim,
            self.config.frame_size,
        )
        self.predictor = _Predictor(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            act_emb_dim=self.config.act_emb_dim,
            hidden_dim=self.config.predictor_hidden,
            layers=self.config.predictor_layers,
        )
        self.sigreg = SIGReg(self.config.sigreg_knots, self.config.sigreg_projections)

        self._optimizer: torch.optim.Optimizer | None = None
        self._encoder_frozen = False

    # ------------------------------------------------------------- properties
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def encoder_frozen(self) -> bool:
        return self._encoder_frozen

    def param_counts(self) -> dict[str, int]:
        encoder = sum(p.numel() for p in self.encoder.parameters())
        predictor = sum(p.numel() for p in self.predictor.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": int(encoder), "predictor": int(predictor), "total": int(total)}

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.encoder.parameters()]

    def predictor_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.predictor.parameters()]

    # ------------------------------------------------------------ frame input
    def _prepare_frames(self, frames: Any) -> tuple[torch.Tensor, bool]:
        if isinstance(frames, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(frames))
        elif torch.is_tensor(frames):
            tensor = frames
        else:
            raise TypeError(f"frames must be numpy or torch, got {type(frames)!r}")

        single = tensor.ndim == 4
        if tensor.ndim == 5:
            if tensor.shape[2] == self.config.in_channels:
                pass
            elif tensor.shape[-1] == self.config.in_channels:
                tensor = tensor.permute(0, 1, 4, 2, 3)
            else:
                raise ValueError("cannot infer channel axis for 5-D frames")
        elif single:
            if tensor.shape[1] == self.config.in_channels:
                pass
            elif tensor.shape[-1] == self.config.in_channels:
                tensor = tensor.permute(0, 3, 1, 2)
            else:
                raise ValueError("cannot infer channel axis for 4-D frames")
        else:
            raise ValueError("frames must be 4-D (B,C,H,W) or 5-D (B,T,C,H,W)")

        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif tensor.is_floating_point():
            if tensor.numel() > 0 and float(tensor.max()) > 1.5:
                tensor = tensor / 255.0
        else:
            raise TypeError(f"unsupported frame dtype {tensor.dtype}")
        tensor = tensor.to(self.device)
        if tensor.shape[-1] != self.config.frame_size or tensor.shape[-2] != self.config.frame_size:
            raise ValueError(
                f"frames must be {self.config.frame_size}x{self.config.frame_size}, "
                f"got {tuple(tensor.shape[-2:])}"
            )
        return tensor, single

    def _prepare_actions(self, actions: Any, batch: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.ndim != 2:
            raise ValueError("actions must be (B, T) or (T,)")
        if torch.is_floating_point(actions) or actions.dtype == torch.bool:
            raise TypeError("actions must be integer-valued")
        actions = actions.long()
        actions = actions.to(device=device, dtype=torch.long)
        if actions.shape[0] not in (1, batch):
            raise ValueError("action batch does not match latent batch")
        if actions.shape[0] == 1 and batch > 1:
            actions = actions.expand(batch, -1)
        if actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= self.config.action_dim):
            raise ValueError(f"actions must lie in 0..{self.config.action_dim - 1}")
        return actions

    # ---------------------------------------------------------------- forward
    def encode(self, frames: Any) -> torch.Tensor:
        """Encode frames.

        ``(B, C, H, W)`` or ``(B, H, W, C)`` -> ``(B, D)``;
        ``(B, T, C, H, W)`` or ``(B, T, H, W, C)`` -> ``(B, T, D)``.
        ``uint8`` frames are scaled by 1/255; float frames with values above
        1.5 are treated as 0..255 and scaled as well.
        """
        tensor, single = self._prepare_frames(frames)
        if single:
            return self.encoder(tensor)
        b, t, c, h, w = tensor.shape
        z = self.encoder(tensor.reshape(b * t, c, h, w))
        return z.reshape(b, t, self.config.latent_dim)

    def predict(self, z: torch.Tensor, actions: Any) -> torch.Tensor:
        """Predict ``z_{t+1}`` for every step of a latent sequence.

        ``z`` is ``(B, T, D)``.  ``actions`` is ``(B, T-1)`` (action per
        transition) or ``(B, T)`` (an extra trailing action is ignored).
        Returns ``(B, T-1, D)`` aligned with ``z[:, 1:]``.
        """
        if not torch.is_tensor(z):
            z = torch.as_tensor(z)
        if z.ndim != 3:
            raise ValueError("z must be (B, T, D)")
        if z.shape[1] < 2:
            raise ValueError("predict needs at least two latent steps; use predict_step")
        actions = self._prepare_actions(actions, z.shape[0], z.device)
        needed = z.shape[1] - 1
        if actions.shape[1] < needed:
            raise ValueError("need at least T-1 actions to predict T-1 transitions")
        return self.predictor(z[:, :-1], actions[:, :needed])

    def predict_step(
        self,
        z_t: torch.Tensor,
        action: Any,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One online predictor step; returns ``(z_hat_next, new_hidden)``."""
        if not torch.is_tensor(z_t):
            z_t = torch.as_tensor(z_t)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if z_t.ndim != 2:
            raise ValueError("z_t must be (B, D)")
        actions = self._prepare_actions(
            torch.as_tensor(action).reshape(1) if np.isscalar(action) else action,
            z_t.shape[0],
            z_t.device,
        )[:, 0]
        return self.predictor.step(z_t, actions, hidden)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        hidden = self.config.predictor_hidden
        layers = self.config.predictor_layers
        return torch.zeros(layers, batch_size, hidden, device=self.device)

    def surprise(self, frames: Any, actions: Any, *, reduce: bool = False) -> torch.Tensor:
        """Squared latent prediction error per transition.

        Returns ``(B, T-1)`` (or the scalar mean with ``reduce=True``):
        ``surprise[b, i]`` is the error when predicting the latent of
        ``frames[b, i+1]`` from ``frames[b, :i+1]`` and ``actions[b, :i+1]``.
        Runs under ``torch.no_grad``.
        """
        with torch.no_grad():
            z = self.encode(frames)
            if z.ndim != 3:
                raise ValueError("surprise needs a frame sequence")
            pred = self.predict(z, actions)
            errors = (pred - z[:, 1:]).pow(2).mean(dim=-1)
        return errors.mean() if reduce else errors

    # ---------------------------------------------------------------- training
    def loss_terms(self, frames: Any, actions: Any) -> dict[str, torch.Tensor]:
        """Two-loss terms: ``pred MSE + sigreg_weight * SIGReg``."""
        z = self.encode(frames)
        if z.ndim != 3:
            raise ValueError("loss_terms needs a frame sequence")
        pred = self.predict(z, actions)
        target = z[:, 1:]
        pred_loss = F.mse_loss(pred, target)
        sigreg = self.sigreg(z.transpose(0, 1))
        loss = pred_loss + self.config.sigreg_weight * sigreg
        return {"loss": loss, "pred_mse": pred_loss, "sigreg": sigreg}

    def configure_optimizer(
        self,
        lr: float | None = None,
        weight_decay: float | None = None,
        *,
        predictor_only: bool = False,
    ) -> torch.optim.Optimizer:
        params: Iterable[nn.Parameter]
        params = self.predictor_parameters() if predictor_only else [p for p in self.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            list(params),
            lr=self.config.lr if lr is None else lr,
            weight_decay=self.config.weight_decay if weight_decay is None else weight_decay,
        )
        return self._optimizer

    def predictor_optimizer(self, lr: float | None = None) -> torch.optim.Optimizer:
        """An optimizer over predictor parameters only (phase-5 overlay hook)."""
        return self.configure_optimizer(lr=lr, predictor_only=True)

    def train_step(self, frames: Any, actions: Any) -> dict[str, float]:
        """One optimizer step; returns detached scalar losses."""
        if self._optimizer is None:
            raise RuntimeError("call configure_optimizer() (or fit()) before train_step()")
        self._optimizer.zero_grad(set_to_none=True)
        terms = self.loss_terms(frames, actions)
        terms["loss"].backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.parameters() if p.requires_grad],
                self.config.grad_clip,
            )
        self._optimizer.step()
        return {name: float(value.detach()) for name, value in terms.items()}

    def fit(
        self,
        train_rollouts: RolloutBatch,
        *,
        steps: int,
        seq_len: int,
        batch_size: int,
        eval_rollouts: RolloutBatch | None = None,
        eval_every: int = 100,
        freeze_after: bool = True,
        log: bool = True,
    ) -> dict[str, Any]:
        """Fit on rollout windows and optionally freeze the encoder at the end.

        Reproducibility: seeds ``torch`` from ``config.seed`` and draws window
        positions from ``numpy.random.default_rng(config.seed + 12345)``.
        Returns a history dict with loss curves, eval points, wall time, and
        the final frozen flag.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not 1 <= seq_len <= train_rollouts.horizon:
            raise ValueError("seq_len out of range for train rollouts")

        torch.manual_seed(int(self.config.seed) + 1)
        rng = np.random.default_rng(int(self.config.seed) + 12345)

        if self._optimizer is None:
            self.configure_optimizer()

        eval_batch = None
        if eval_rollouts is not None:
            eval_size = min(batch_size, max(1, len(eval_rollouts) // 2))
            eval_frames, eval_actions = sample_window_batch(
                eval_rollouts,
                seq_len,
                eval_size,
                np.random.default_rng(int(self.config.seed) + 999),
            )
            eval_batch = (eval_frames, eval_actions)

        history: dict[str, Any] = {
            "loss": [],
            "pred_mse": [],
            "sigreg": [],
            "eval_steps": [],
            "eval_pred_mse": [],
            "config": self.config.to_dict(),
            "param_counts": self.param_counts(),
            "steps": int(steps),
            "seq_len": int(seq_len),
            "batch_size": int(batch_size),
        }
        self.train()
        start = time.perf_counter()
        for step in range(1, steps + 1):
            frames, actions = sample_window_batch(train_rollouts, seq_len, batch_size, rng)
            metrics = self.train_step(frames, actions)
            history["loss"].append(metrics["loss"])
            history["pred_mse"].append(metrics["pred_mse"])
            history["sigreg"].append(metrics["sigreg"])
            if eval_batch is not None and (step % eval_every == 0 or step == 1):
                self.eval()
                with torch.no_grad():
                    z = self.encode(eval_batch[0])
                    pred = self.predict(z, eval_batch[1])
                    eval_loss = float(F.mse_loss(pred, z[:, 1:]))
                history["eval_steps"].append(step)
                history["eval_pred_mse"].append(eval_loss)
                self.train()
            if log and (step % max(1, steps // 10) == 0 or step == 1):
                print(
                    f"[fit] step {step:5d}/{steps}  loss {metrics['loss']:.4f}  "
                    f"pred {metrics['pred_mse']:.4f}  sigreg {metrics['sigreg']:.4f}",
                    flush=True,
                )
        history["wall_seconds"] = float(time.perf_counter() - start)
        history["ms_per_step"] = 1000.0 * history["wall_seconds"] / steps
        if freeze_after:
            self.freeze_encoder()
        history["encoder_frozen"] = self.encoder_frozen
        return history

    # ------------------------------------------------------------ freeze/state
    def freeze_encoder(self) -> None:
        """Freeze and eval-mode the encoder; the predictor stays trainable."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        """Explicitly reverse :meth:`freeze_encoder` (for experiments only)."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self._encoder_frozen = False

    def train(self, mode: bool = True) -> "MicroLeWM":
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
        """Save config + state dict + frozen flag to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "encoder_frozen": bool(self.encoder_frozen),
            "param_counts": self.param_counts(),
            "meta": dict(extra or {}),
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "MicroLeWM":
        """Load a checkpoint written by :meth:`save`.

        ``weights_only=False`` is used because the payload carries a plain
        config dict; nothing executable is stored.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unexpected checkpoint format: {payload.get('format')!r}")
        config = MicroLeWMConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        if payload.get("encoder_frozen", False):
            model.freeze_encoder()
        model.eval()
        model.checkpoint_meta = payload.get("meta", {})
        return model
