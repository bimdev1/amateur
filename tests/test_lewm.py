"""Tests for the LeWM organ (`amateur/lewm.py`).

The published PushT checkpoint is the measurement basis: these tests pin the
load contract (exact key set, exact parameter count, strict load in both
directions), the shape/determinism contract of ``encode``/``predict``, and the
VoE signal itself — zero surprise for identical latents, strictly rising under
latent perturbation, and a large surprise spike when an object teleports inside
an otherwise smooth frame sequence.

Nothing here needs the 13 GB PushT dataset except `TestDatasetVoE`, which skips
with an explicit reason when the file (or `h5py`) is unavailable.

Runtime: ~2-4 s on 8 CPU threads; the slowest single item is the checkpoint
load (~0.3 s) plus a handful of 224x224 ViT-tiny forward passes.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from amateur import lewm

_REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = lewm.DEFAULT_CHECKPOINT_DIR
WEIGHTS = CHECKPOINT_DIR / lewm.DEFAULT_WEIGHTS_NAME
CONFIG = CHECKPOINT_DIR / lewm.DEFAULT_CONFIG_NAME
DATASET = _REPO_ROOT / ".work" / "data" / "pusht_expert_train.h5"

CHECKPOINT_AVAILABLE = WEIGHTS.is_file() and CONFIG.is_file()
CHECKPOINT_REASON = (
    f"published LeWM checkpoint absent ({WEIGHTS} / {CONFIG}); "
    "the organ is only meaningful against those weights"
)

#: Number of entries in the published state dict (303).
EXPECTED_NUM_KEYS = 303
#: Sum of `numel()` over *every* tensor in the published state dict, i.e.
#: parameters plus BatchNorm running statistics plus `num_batches_tracked`.
EXPECTED_STATE_DICT_ELEMENTS = 18_042_672
#: Sum over `model.parameters()` only (buffers excluded).
EXPECTED_PARAMETERS = 18_034_478

#: Loaded once for the whole module.
MODEL: lewm.LewM | None = None
STATE_DICT: dict[str, torch.Tensor] | None = None


def setUpModule() -> None:
    """Load the checkpoint once; the model is read-only for every test."""
    global MODEL, STATE_DICT
    # The task pins 8 CPU threads; the baseline suite has no torch users, so
    # this cannot slow anything else down.
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if not CHECKPOINT_AVAILABLE:
        return
    MODEL = lewm.LewM.load_checkpoint(CHECKPOINT_DIR)
    STATE_DICT = torch.load(WEIGHTS, map_location="cpu", weights_only=False)


requires_checkpoint = unittest.skipUnless(CHECKPOINT_AVAILABLE, CHECKPOINT_REASON)


# ---------------------------------------------------------------------------
# Deterministic synthetic PushT-like frames (no dataset, no network)
# ---------------------------------------------------------------------------

_IMAGE_SIZE = 224
_CELL = 14
_BACKGROUND = (0.95, 0.95, 0.95)
_BLOCK_RGB = (0.93, 0.36, 0.72)
_AGENT_RGB = (0.25, 0.45, 0.85)
#: T-shape footprint in 14 px cells, matching the PushT block's coarse layout.
_T_CELLS = ((0, 0), (1, 0), (2, 0), (1, 1), (1, 2))


def _frame(block_xy: tuple[int, int], agent_xy: tuple[int, int]) -> torch.Tensor:
    """A crude but in-distribution-ish PushT frame in float RGB [0, 1]."""
    img = torch.empty(3, _IMAGE_SIZE, _IMAGE_SIZE)
    for channel, value in enumerate(_BACKGROUND):
        img[channel] = value
    bx, by = block_xy
    for dx, dy in _T_CELLS:
        x0, y0 = bx + dx * _CELL, by + dy * _CELL
        for channel, value in enumerate(_BLOCK_RGB):
            img[channel, y0 : y0 + _CELL, x0 : x0 + _CELL] = value
    ax, ay = agent_xy
    gy, gx = torch.meshgrid(
        torch.arange(_IMAGE_SIZE), torch.arange(_IMAGE_SIZE), indexing="ij"
    )
    disk = (gx - ax) ** 2 + (gy - ay) ** 2 <= 12**2
    for channel, value in enumerate(_AGENT_RGB):
        img[channel][disk] = value
    return img


def _block_mask(frame: torch.Tensor) -> torch.Tensor:
    return (frame[0] > 0.85) & (frame[1] < 0.5) & (frame[2] > 0.6)


def _sequence(
    block_positions: list[tuple[int, int]], agent_positions: list[tuple[int, int]]
) -> torch.Tensor:
    return torch.stack(
        [_frame(b, a) for b, a in zip(block_positions, agent_positions)]
    )


def _to_pixels(rgb_sequences: torch.Tensor) -> torch.Tensor:
    """float [N,T,3,H,W] in [0,1] -> ImageNet-normalised float32."""
    raw = (rgb_sequences.clamp(0.0, 1.0) * 255).round().to(torch.uint8)
    return lewm.imagenet_normalize(raw)


def _synthetic_voe_batch() -> torch.Tensor:
    """Four 4-frame conditions: unperturbed, teleported, recoloured, unperturbed.

    Smooth linear motion for the first two frames in every condition; from
    frame 2 on, the physical condition teleports the block to a far corner
    while the visual condition only changes its colour. Every condition keeps
    its own self-consistent frames after the perturbation, so the *only* thing
    that differs at the measured step is the discontinuity.
    """
    block = [(40 + 3 * i, 60 + 3 * i) for i in range(4)]
    agent = [(150 - 2 * i, 150) for i in range(4)]
    base = _sequence(block, agent)
    teleported = _sequence(block[:2] + [(120, 120), (123, 123)], agent)
    recoloured = base.clone()
    for t in (2, 3):
        mask = _block_mask(recoloured[t])
        recoloured[t][0][mask] = 0.20
        recoloured[t][1][mask] = 0.65
        recoloured[t][2][mask] = 0.30
    return _to_pixels(torch.stack([base, teleported, recoloured, base]))


# ---------------------------------------------------------------------------
# Load contract
# ---------------------------------------------------------------------------


@requires_checkpoint
class TestCheckpointLoad(unittest.TestCase):
    """The checkpoint must load strictly and the organ must reproduce it exactly."""

    def test_state_dict_shape(self) -> None:
        self.assertEqual(len(STATE_DICT), EXPECTED_NUM_KEYS)
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in STATE_DICT.values()))

    def test_state_dict_element_count(self) -> None:
        total = sum(int(v.numel()) for v in STATE_DICT.values())
        self.assertEqual(total, EXPECTED_STATE_DICT_ELEMENTS)
        params = sum(p.numel() for p in MODEL.parameters())
        self.assertEqual(params, EXPECTED_PARAMETERS)
        buffers = sum(b.numel() for b in MODEL.buffers())
        self.assertEqual(params + buffers, EXPECTED_STATE_DICT_ELEMENTS)

    def test_keys_and_shapes_round_trip(self) -> None:
        model_keys = set(MODEL.state_dict())
        self.assertEqual(model_keys, set(STATE_DICT))
        self.assertEqual(len(model_keys), EXPECTED_NUM_KEYS)
        for key, tensor in STATE_DICT.items():
            self.assertEqual(
                tuple(MODEL.state_dict()[key].shape),
                tuple(tensor.shape),
                f"shape disagreement on {key}",
            )

    def test_strict_load_from_config(self) -> None:
        """A fresh build from `config.json` must accept the weights exactly."""
        rebuilt = lewm.LewM.from_config(CONFIG)
        result = rebuilt.load_state_dict(STATE_DICT, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_strict_load_rejects_missing_key(self) -> None:
        model = lewm.LewM.from_config(CONFIG)
        broken = dict(STATE_DICT)
        broken.pop("predictor.pos_embedding")
        with self.assertRaises(RuntimeError):
            model.load_state_dict(broken, strict=True)

    def test_strict_load_rejects_unexpected_key(self) -> None:
        model = lewm.LewM.from_config(CONFIG)
        broken = dict(STATE_DICT)
        broken["encoder.embeddings.graffiti"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(broken, strict=True)

    def test_config_agrees_with_state_dict(self) -> None:
        """Every architectural claim in `config.json` shows up in the weights."""
        config = json.loads(CONFIG.read_text())
        self.assertEqual(MODEL.model_config, config)

        sd = STATE_DICT
        # ViT-tiny / patch 14 / image 224 / 12 layers / 3 heads / mlp 768.
        self.assertEqual(
            tuple(sd["encoder.embeddings.patch_embeddings.projection.weight"].shape),
            (192, 3, 14, 14),
        )
        self.assertEqual(
            tuple(sd["encoder.embeddings.position_embeddings"].shape), (1, 257, 192)
        )
        self.assertEqual(tuple(sd["encoder.embeddings.cls_token"].shape), (1, 1, 192))
        self.assertEqual(
            tuple(sd["encoder.encoder.layer.0.intermediate.dense.weight"].shape),
            (768, 192),
        )
        layer_ids = {
            int(key.split(".")[3])
            for key in sd
            if key.startswith("encoder.encoder.layer.")
        }
        self.assertEqual(layer_ids, set(range(12)))
        # 3 attention heads: qkv packs 3 x 3 heads x 64 dim_head = 576 -> 3*576.
        self.assertEqual(
            tuple(sd["encoder.encoder.layer.0.attention.attention.query.weight"].shape),
            (192, 192),
        )

        # Predictor: 3 context frames, 6 conditional blocks, AdaLN 6*192.
        self.assertEqual(tuple(sd["predictor.pos_embedding"].shape), (1, 3, 192))
        block_ids = {
            int(key.split(".")[3])
            for key in sd
            if key.startswith("predictor.transformer.layers.")
        }
        self.assertEqual(block_ids, set(range(6)))
        self.assertEqual(
            tuple(sd["predictor.transformer.layers.0.adaLN_modulation.1.weight"].shape),
            (1152, 192),
        )
        # 16 heads x 64 dim_head = 1024 inner width.
        self.assertEqual(
            tuple(sd["predictor.transformer.layers.0.attn.to_qkv.weight"].shape),
            (3072, 192),
        )
        self.assertEqual(
            tuple(sd["predictor.transformer.layers.0.attn.to_out.0.weight"].shape),
            (192, 1024),
        )
        self.assertEqual(MODEL.predictor.num_frames, config["predictor"]["num_frames"])

        # Action encoder: frameskip 5 x 2 action dims = 10; MLP 10 -> 768 -> 192.
        self.assertEqual(
            tuple(sd["action_encoder.patch_embed.weight"].shape), (10, 10, 1)
        )
        self.assertEqual(tuple(sd["action_encoder.embed.0.weight"].shape), (768, 10))
        self.assertEqual(tuple(sd["action_encoder.embed.2.weight"].shape), (192, 768))

        # Projectors: Linear(192, 2048) -> BatchNorm1d(2048) -> GELU -> Linear(2048, 192).
        for name in ("projector", "pred_proj"):
            self.assertIsInstance(
                MODEL.get_submodule(name).net[1], torch.nn.BatchNorm1d
            )
            self.assertEqual(
                tuple(sd[f"{name}.net.0.weight"].shape), (2048, 192)
            )
            self.assertEqual(tuple(sd[f"{name}.net.3.weight"].shape), (192, 2048))
            self.assertIn(f"{name}.net.1.running_mean", sd)
            self.assertIn(f"{name}.net.1.running_var", sd)

    def test_rejects_unsupported_config(self) -> None:
        config = json.loads(CONFIG.read_text())
        bad = json.loads(json.dumps(config))
        bad["encoder"]["pretrained"] = True
        with self.assertRaises(ValueError):
            lewm.LewM(bad)
        bad = json.loads(json.dumps(config))
        bad["encoder"]["use_mask_token"] = True
        with self.assertRaises(ValueError):
            lewm.LewM(bad)


# ---------------------------------------------------------------------------
# encode / predict shapes and determinism
# ---------------------------------------------------------------------------


@requires_checkpoint
class TestEncode(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.frames = torch.randn(2, 2, 3, _IMAGE_SIZE, _IMAGE_SIZE) * 0.5

    def test_shape_and_determinism(self) -> None:
        z = MODEL.encode(self.frames)
        self.assertEqual(tuple(z.shape), (2, 2, lewm.EMBED_DIM))
        self.assertEqual(z.dtype, torch.float32)
        z_again = MODEL.encode(self.frames)
        self.assertTrue(torch.equal(z, z_again), "encode must be deterministic")
        self.assertFalse(z.requires_grad)

    def test_single_frame_input_keeps_time_axis(self) -> None:
        z = MODEL.encode(self.frames[:, 0])
        self.assertEqual(tuple(z.shape), (2, 1, lewm.EMBED_DIM))
        # Broadcasting a single frame across T must give identical latents.
        z_full = MODEL.encode(self.frames[:, :1])
        self.assertTrue(torch.equal(z, z_full))

    def test_batch_grouping_invariance(self) -> None:
        """Encoding is per-frame: batching must not change any latent."""
        z = MODEL.encode(self.frames)
        z_split = MODEL.encode(self.frames.reshape(4, 1, 3, _IMAGE_SIZE, _IMAGE_SIZE))
        self.assertTrue(
            torch.equal(z.reshape(4, lewm.EMBED_DIM), z_split.reshape(4, lewm.EMBED_DIM))
        )

    def test_rejects_uint8_pixels(self) -> None:
        raw = torch.randint(0, 255, (1, 1, 3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            MODEL.encode(raw)

    def test_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            MODEL.encode(torch.randn(3, _IMAGE_SIZE, _IMAGE_SIZE))
        with self.assertRaises(ValueError):
            MODEL.encode(torch.randn(1, 1, 4, _IMAGE_SIZE, _IMAGE_SIZE))
        with self.assertRaises(ValueError):
            MODEL.encode(torch.randn(1, 1, 3, _IMAGE_SIZE, 96))
        with self.assertRaises(ValueError):
            MODEL.encode(torch.randn(1, 0, 3, _IMAGE_SIZE, _IMAGE_SIZE))

    def test_imagenet_normalize(self) -> None:
        raw = torch.full((2, 3, 3, 4, 4), 128, dtype=torch.uint8)
        out = lewm.imagenet_normalize(raw)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (2, 3, 3, 4, 4))
        expected = (128 / 255 - torch.tensor(lewm.IMAGENET_MEAN)) / torch.tensor(
            lewm.IMAGENET_STD
        )
        self.assertTrue(torch.allclose(out[0, 0, :, 0, 0], expected, atol=1e-6))
        with self.assertRaises(ValueError):
            lewm.imagenet_normalize(torch.rand(1, 3, 4, 4))  # already 0..1
        with self.assertRaises(ValueError):
            lewm.imagenet_normalize(torch.zeros(2, 4, 4))


@requires_checkpoint
class TestPredict(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.z = torch.randn(2, 3, lewm.EMBED_DIM)
        self.actions = torch.randn(2, 3, 10)

    def test_shape_and_determinism(self) -> None:
        z_hat = MODEL.predict(self.z, self.actions)
        self.assertEqual(tuple(z_hat.shape), (2, 3, lewm.EMBED_DIM))
        self.assertTrue(torch.equal(z_hat, MODEL.predict(self.z, self.actions)))
        self.assertFalse(z_hat.requires_grad)

    def test_rejects_more_than_history(self) -> None:
        too_long = torch.randn(2, 4, lewm.EMBED_DIM)
        with self.assertRaises(ValueError):
            MODEL.predict(too_long, torch.randn(2, 4, 10))

    def test_rejects_mismatched_actions(self) -> None:
        with self.assertRaises(ValueError):
            MODEL.predict(self.z, torch.randn(2, 3, 9))
        with self.assertRaises(ValueError):
            MODEL.predict(self.z, torch.randn(3, 3, 10))
        with self.assertRaises(ValueError):
            MODEL.predict(self.z, torch.randn(2, 2, 10))
        with self.assertRaises(ValueError):
            MODEL.predict(torch.randn(2, 3, 128), self.actions)


# ---------------------------------------------------------------------------
# Surprise
# ---------------------------------------------------------------------------


@requires_checkpoint
class TestSurprise(unittest.TestCase):
    def test_zero_for_identical_latents(self) -> None:
        torch.manual_seed(2)
        z = torch.randn(4, 3, lewm.EMBED_DIM)
        per_sample, mean = lewm.surprise(z, z)
        self.assertTrue(torch.equal(per_sample, torch.zeros(4)))
        self.assertEqual(float(mean), 0.0)
        self.assertTrue(torch.equal(MODEL.surprise(z, z)[0], torch.zeros(4)))

    def test_reduction_matches_definition(self) -> None:
        torch.manual_seed(3)
        a = torch.randn(2, 3, lewm.EMBED_DIM)
        b = torch.randn(2, 3, lewm.EMBED_DIM)
        per_sample, mean = lewm.surprise(a, b)
        expected = (a - b).pow(2).mean(dim=(1, 2))
        self.assertTrue(torch.allclose(per_sample, expected))
        self.assertAlmostEqual(float(mean), float(expected.mean()), places=6)
        per_step = lewm.surprise_per_step(a, b)
        self.assertTrue(torch.allclose(per_step, (a - b).pow(2).mean(dim=-1)))
        self.assertTrue(
            torch.allclose(per_step.mean(dim=1), per_sample, atol=1e-6)
        )

    def test_strictly_positive_and_rising_under_latent_perturbation(self) -> None:
        torch.manual_seed(4)
        z_next = torch.randn(3, 3, lewm.EMBED_DIM)
        z_hat = z_next + 0.01 * torch.randn(3, 3, lewm.EMBED_DIM)
        _, base = lewm.surprise(z_hat, z_next)
        self.assertGreater(float(base), 0.0)
        previous = 0.0
        for eps in (0.01, 0.1, 0.5, 1.0):
            _, value = lewm.surprise(z_next + eps, z_next)
            self.assertGreater(float(value), previous)
            previous = float(value)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            lewm.surprise(torch.zeros(2, 3, lewm.EMBED_DIM), torch.zeros(2, 4, lewm.EMBED_DIM))
        with self.assertRaises(ValueError):
            lewm.surprise(torch.zeros(2), torch.zeros(2))

    def test_frame_level_voe_teleport_and_colour_change(self) -> None:
        """The VoE signal end-to-end: perturbation at t=2 must raise surprise.

        Four conditions share frames 0-1 (smooth motion): unperturbed,
        block teleported from t=2 on, block recoloured from t=2 on, and a
        second unperturbed copy. Surprise is measured at the step whose target
        is the first perturbed frame, using a 3-frame causal context.
        """
        frames = _synthetic_voe_batch()
        z = MODEL.encode(frames)
        z_hat = MODEL.predict(z[:, :3], torch.zeros(4, 3, 10))
        per_step = MODEL.surprise_per_step(z_hat, z[:, 1:])

        base = float(per_step[0, 0])  # slot 0 is unaffected by construction
        self.assertTrue(
            torch.allclose(per_step[0, 1], per_step[3, 1], atol=1e-6),
            "the two unperturbed copies must agree",
        )
        step = 1  # slot 1 predicts frame 2, the first perturbed frame
        unperturbed = float(per_step[0, step])
        teleported = float(per_step[1, step])
        recoloured = float(per_step[2, step])

        self.assertGreater(teleported, 10.0 * unperturbed)
        self.assertGreater(recoloured, unperturbed)
        self.assertGreater(teleported, recoloured)
        # The colour change leaves no trace one step later; the teleport does.
        self.assertLess(float(per_step[2, 2]), 2.0 * float(per_step[0, 2]))
        self.assertGreater(float(per_step[1, 2]), 2.0 * float(per_step[0, 2]))
        self.assertAlmostEqual(float(per_step[0, 0]), base)


# ---------------------------------------------------------------------------
# Frozen-weights contract
# ---------------------------------------------------------------------------


@requires_checkpoint
class TestFrozenWeights(unittest.TestCase):
    def test_no_parameters_track_gradients(self) -> None:
        self.assertFalse(any(p.requires_grad for p in MODEL.parameters()))
        self.assertTrue(all(not m.training for m in MODEL.modules()))

    def test_train_is_a_no_op(self) -> None:
        before = MODEL.projector.net[1].running_mean.clone()
        self.assertIs(MODEL.train(), MODEL)
        self.assertTrue(all(not m.training for m in MODEL.modules()))
        MODEL.encode(torch.zeros(1, 1, 3, _IMAGE_SIZE, _IMAGE_SIZE))
        self.assertTrue(
            torch.equal(before, MODEL.projector.net[1].running_mean),
            "BatchNorm running statistics must not move",
        )

    def test_forward_does_not_mutate_parameters(self) -> None:
        torch.manual_seed(5)
        snapshot = {k: v.detach().clone() for k, v in MODEL.state_dict().items()}
        MODEL.encode(torch.randn(1, 2, 3, _IMAGE_SIZE, _IMAGE_SIZE))
        MODEL.predict(torch.randn(1, 3, lewm.EMBED_DIM), torch.randn(1, 3, 10))
        for key, value in MODEL.state_dict().items():
            self.assertTrue(torch.equal(value, snapshot[key]), f"{key} mutated")


# ---------------------------------------------------------------------------
# Dataset-dependent VoE protocol (real PushT frames)
# ---------------------------------------------------------------------------
#
# The published HDF5 stores `pixels` under HDF5 filter id 32001 — HDF5's Blosc1
# meta-compressor with the LZ4 codec (16-byte header + per-block LZ4), *not* raw
# zstd. Two ways to read it: (a) import `hdf5plugin`, which registers the filter,
# and then plain h5py indexing works; (b) read the filtered chunks with
# `Dataset.id.read_direct_chunk` and decode Blosc1 against system `liblz4`.
# Path (a) is preferred; (b) exists because `hdf5plugin` is a scratch
# (`PYTHONPATH`) dependency and its wheels lag new CPython versions.
#
# Schema (verified against `state`; see docs/phase1-lewm-voe.md):
#   pixels     (N, 224, 224, 3) uint8   chunked 100 rows, filter 32001
#   action     (N, 2) float32           per-env-step action; training chunks 5
#   state      (N, 7) float32           [agent_xy, block_xy, block_angle, ...]
#   episode_idx / step_idx / ep_offset / ep_len
# Episode rows are contiguous: row = ep_offset[ep] + step.

FRAMESKIP = 5
BLOCK_RGB = (144, 162, 184)
NEW_BLOCK_RGB = (200, 45, 45)
IMAGE_EXTENT = 512.0  # PushT world coordinates span [0, 512]; frames are 224 px


def load_liblz4():
    """Return the liblz4 handle, or None when it is unavailable."""
    import ctypes

    for name in ("liblz4.so.1", "liblz4.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.LZ4_decompress_safe.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.LZ4_decompress_safe.restype = ctypes.c_int
        return lib
    return None


def decode_blosc1(raw: bytes, lz4) -> bytes:
    """Decode one Blosc1 buffer (format 2, LZ4 codec, typesize 1)."""
    import struct

    version, _versionlz, flags, typesize = raw[0], raw[1], raw[2], raw[3]
    if version != 2:
        raise ValueError(f"unsupported Blosc format version {version}")
    if typesize != 1:
        raise ValueError(f"byte-shuffle with typesize {typesize} is not implemented")
    nbytes, blocksize, cbytes = struct.unpack("<III", raw[4:16])
    if flags & 0x02:  # memcpyed: whole buffer stored verbatim
        return raw[16 : 16 + nbytes]
    if cbytes != len(raw):
        raise ValueError(f"cbytes {cbytes} != buffer length {len(raw)}")
    nblocks = -(-nbytes // blocksize)
    offsets = struct.unpack(f"<{nblocks}I", raw[16 : 16 + 4 * nblocks])
    out = bytearray(nbytes)
    scratch = None
    for j in range(nblocks):
        start = offsets[j]
        end = offsets[j + 1] if j + 1 < nblocks else cbytes
        block_bytes = min(blocksize, nbytes - j * blocksize)
        (compressed,) = struct.unpack("<I", raw[start : start + 4])
        if compressed == 0:
            out[j * blocksize : j * blocksize + block_bytes] = raw[
                start + 4 : start + 4 + block_bytes
            ]
            continue
        if scratch is None:
            import ctypes

            scratch = ctypes.create_string_buffer(blocksize)
        produced = lz4.LZ4_decompress_safe(
            raw[start + 4 : end], scratch, end - start - 4, block_bytes
        )
        if produced != block_bytes:
            raise ValueError(
                f"block {j}: LZ4 produced {produced} bytes, expected {block_bytes}"
            )
        out[j * blocksize : (j + 1) * blocksize] = scratch.raw[:block_bytes]
    return bytes(out)


class PushTRows:
    """Row-addressable reader for the published PushT HDF5.

    Raises ``unittest.SkipTest`` (with the concrete reason) when the pixel
    filter cannot be applied and there is no decoder to fall back on.
    """

    def __init__(self, path: Path) -> None:
        import h5py
        import numpy as np

        self.needs_filter = True
        try:  # registers HDF5 filter 32001 for plain h5py reads
            import hdf5plugin  # noqa: F401

            self.needs_filter = False
        except ImportError:
            pass
        if self.needs_filter:
            self.lz4 = load_liblz4()
            if self.lz4 is None:
                raise unittest.SkipTest(
                    "pixels use HDF5 filter 32001 (Blosc1/LZ4) and neither "
                    "hdf5plugin nor liblz4 is available"
                )
        self.handle = h5py.File(path, "r")
        self.pixels = self.handle["pixels"]
        self._np = np
        self.chunk_rows = int(self.pixels.chunks[0])
        self._chunks: dict[int, "np.ndarray"] = {}
        self.ep_offset = np.asarray(self.handle["ep_offset"])
        self.ep_len = np.asarray(self.handle["ep_len"])
        self.step_idx = np.asarray(self.handle["step_idx"])
        self.actions = self.handle["action"]
        self.state = np.asarray(self.handle["state"])

    def close(self) -> None:
        self.handle.close()

    def _chunk(self, index: int):
        import numpy as np

        if index not in self._chunks:
            mask, raw = self.pixels.id.read_direct_chunk(
                (index * self.chunk_rows, 0, 0, 0)
            )
            if mask != 0:
                raise ValueError(f"chunk {index} has unapplied filters: {mask}")
            data = decode_blosc1(raw, self.lz4)
            rows = min(
                self.chunk_rows, self.pixels.shape[0] - index * self.chunk_rows
            )
            self._chunks[index] = np.frombuffer(data, dtype=np.uint8).reshape(
                -1, *self.pixels.shape[1:]
            )[:rows]
            if len(self._chunks) > 4:  # ~60 MB per decoded chunk
                self._chunks.pop(next(iter(self._chunks)))
        return self._chunks[index]

    def frames(self, rows):
        """Decode the requested rows, keeping one cached chunk per 100 rows."""
        import numpy as np

        rows = [int(r) for r in rows]
        if not self.needs_filter:
            return np.asarray(self.pixels[sorted(rows)])[
                np.argsort(np.argsort(rows))
            ]
        return np.stack(
            [self._chunk(r // self.chunk_rows)[r % self.chunk_rows] for r in rows]
        )

    def action_chunk(self, row: int, episode: int) -> "np.ndarray":
        """frameskip-5 action chunk flattened to a 1-D vector."""
        import numpy as np

        if self.actions.shape[1:] == (2,):
            base = int(self.ep_offset[episode]) + int(self.step_idx[row])
            chunk = np.asarray(self.actions[base : base + FRAMESKIP])
            if chunk.shape[0] != FRAMESKIP:
                raise ValueError(f"short action chunk at row {row}")
            return chunk.reshape(-1)
        return np.asarray(self.actions[row]).reshape(-1)

    def eligible_episodes(self, min_rows: int):
        """Episode ids long enough for ``min_rows`` raw frames."""
        import numpy as np

        return np.nonzero(self.ep_len >= min_rows)[0]


def to_pixels(raw) -> torch.Tensor:
    """uint8 NHWC frames -> ImageNet-normalised float ``[1, T, 3, 224, 224]``."""
    import numpy as np

    t = torch.from_numpy(np.ascontiguousarray(raw))
    if t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    if tuple(t.shape[1:]) != (3, _IMAGE_SIZE, _IMAGE_SIZE):
        raise ValueError(f"unexpected frame shape {tuple(t.shape)}")
    return lewm.imagenet_normalize(t.unsqueeze(0))


def block_mask(frame: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the PushT block in an ImageNet-normalised ``[3,H,W]``."""
    mean = torch.tensor(lewm.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(lewm.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    rgb = (frame * std + mean) * 255.0
    ref = torch.tensor(BLOCK_RGB, dtype=torch.float32).view(3, 1, 1)
    return ((rgb - ref).pow(2).sum(dim=0)).sqrt() < 22.0


def recolour_block(pixels: torch.Tensor, from_step: int) -> tuple[torch.Tensor, int]:
    """Solidly recolour the block from ``from_step`` on.

    The core slate pixels are dilated by 2 px so the anti-aliased fringe is
    replaced too; otherwise a grey outline would survive the recolour and blunt
    the visual perturbation. Returns the new pixels and the smallest mask size.
    """
    out = pixels.clone()
    counts = []
    for t in range(from_step, out.shape[1]):
        mask = block_mask(out[0, t]).to(torch.float32)
        mask = torch.nn.functional.max_pool2d(
            mask.view(1, 1, _IMAGE_SIZE, _IMAGE_SIZE), 5, 1, 2
        )
        mask = mask.view(_IMAGE_SIZE, _IMAGE_SIZE) > 0
        counts.append(int(mask.sum()))
        if counts[-1] < 400:
            raise AssertionError(f"block mask vanished at t={t}: {counts[-1]} px")
        for channel, value in enumerate(NEW_BLOCK_RGB):
            out[0, t, channel][mask] = (
                value / 255.0 - lewm.IMAGENET_MEAN[channel]
            ) / lewm.IMAGENET_STD[channel]
    return out, min(counts)


def dataset_action_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Mean/std of the *chunked* action column over the whole dataset.

    Reproduces `get_column_normalizer` from the reference training script:
    statistics over every valid frameskip-5 chunk (chunks that would run past an
    episode end are dropped, exactly like the loader's NaN rows).
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        actions = np.asarray(handle["action"], dtype=np.float64)
        offset = np.asarray(handle["ep_offset"])
        length = np.asarray(handle["ep_len"])
    starts = np.concatenate(
        [
            offset[ep] + np.arange(length[ep] - (FRAMESKIP - 1))
            for ep in range(len(offset))
        ]
    )
    chunks = actions[starts[:, None] + np.arange(FRAMESKIP)]  # (N, 5, 2)
    flat = chunks.reshape(len(starts), -1)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return (
        torch.from_numpy(mean.astype("float32")),
        torch.from_numpy(std.astype("float32")),
        int(len(starts)),
    )


def build_conditions(
    rows: PushTRows,
    episode: int,
    *,
    window: int,
    perturb_step: int,
    jump: int,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build unperturbed / teleport / colour conditions for one episode.

    * unperturbed — frames ``5*i`` of the episode, its own action chunks.
    * teleport    — frames ``5*i`` for ``i < k`` then frames ``5*(i+jump)`` for
      ``i >= k`` (a discontinuous state jump into the same episode's later
      trajectory, so the goal pose is unchanged by construction); actions are
      spliced at the same boundary so the post-jump dynamics stay consistent.
    * colour      — the unperturbed stream with the block recoloured from ``k``.
    """
    import numpy as np

    base_row = int(rows.ep_offset[episode])
    clean_steps = [FRAMESKIP * i for i in range(window)]
    jump_steps = [FRAMESKIP * (i + jump) for i in range(window)]

    clean_raw = rows.frames([base_row + s for s in clean_steps])
    jump_raw = rows.frames([base_row + s for s in jump_steps])
    clean_actions = np.stack(
        [rows.action_chunk(base_row + s, episode) for s in clean_steps]
    )
    jump_actions = np.stack(
        [rows.action_chunk(base_row + s, episode) for s in jump_steps]
    )

    teleport_raw = np.concatenate([clean_raw[:perturb_step], jump_raw[perturb_step:]])
    teleport_actions = np.concatenate(
        [clean_actions[:perturb_step], jump_actions[perturb_step:]]
    )

    def norm(actions) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(actions).astype("float32"))
        return lewm.normalize_actions(t, act_mean, act_std).unsqueeze(0)

    clean_pixels = to_pixels(clean_raw)
    colour_pixels, _ = recolour_block(clean_pixels, perturb_step)
    return {
        "unperturbed": (clean_pixels, norm(clean_actions)),
        "teleport": (to_pixels(teleport_raw), norm(teleport_actions)),
        "colour": (colour_pixels, norm(clean_actions)),
    }


def surprise_curve(
    model: lewm.LewM, pixels: torch.Tensor, actions: torch.Tensor, window: int
) -> list[float]:
    """Per-target surprise for targets ``3 .. window-1`` (3-frame causal window)."""
    z = model.encode(pixels)
    preds = [
        model.predict(z[:, t - 3 : t], actions[:, t - 3 : t])[:, -1:]
        for t in range(3, window)
    ]
    z_hat = torch.cat(preds, dim=1)
    return model.surprise_per_step(z_hat, z[:, 3:])[0].tolist()


def teleport_displacement(
    rows: PushTRows, episode: int, perturb_step: int, jump: int
) -> tuple[float, float]:
    """Block/agent displacement (px) between the pre-jump and post-jump frames.

    ``state`` is ``[agent_x, agent_y, block_x, block_y, block_angle, ...]`` in
    PushT's 512-unit world; frames are 224 px, and the frame axes were verified
    to map as ``pixel = coordinate * 224 / 512`` (no vertical flip).
    """
    import numpy as np

    base_row = int(rows.ep_offset[episode])
    before = rows.state[base_row + FRAMESKIP * perturb_step]
    after = rows.state[base_row + FRAMESKIP * (perturb_step + jump)]
    scale = float(_IMAGE_SIZE) / IMAGE_EXTENT
    block = float(np.hypot(after[2] - before[2], after[3] - before[3]) * scale)
    agent = float(np.hypot(after[0] - before[0], after[1] - before[1]) * scale)
    return block, agent


@unittest.skipUnless(DATASET.is_file(), f"PushT HDF5 dataset absent ({DATASET})")
class TestDatasetVoE(unittest.TestCase):
    """Replay-based VoE on real expert frames: unperturbed vs teleport vs colour."""

    WINDOW = 12       # condition frames
    PERTURB_STEP = 6  # first perturbed frame index
    JUMP = 12         # teleport distance in frameskip steps
    EPISODES = 3

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(
                "h5py not importable; run with PYTHONPATH=$PWD/.work/pydeps "
                f"({exc})"
            ) from exc
        cls.rows = PushTRows(DATASET)
        cls.need_rows = FRAMESKIP * (cls.WINDOW + cls.JUMP) + 1
        cls.sample = cls.rows.eligible_episodes(cls.need_rows)
        if len(cls.sample) < cls.EPISODES:
            cls.rows.close()
            raise unittest.SkipTest(
                f"only {len(cls.sample)} episodes reach {cls.need_rows} rows"
            )
        cls.episodes = [int(e) for e in cls.sample[: cls.EPISODES]]
        cls.act_mean, cls.act_std, cls.act_chunks = dataset_action_stats(DATASET)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.rows.close()

    def setUp(self) -> None:
        torch.set_num_threads(min(8, os.cpu_count() or 1))

    def test_block_mask_finds_the_block(self) -> None:
        import numpy as np

        episode = self.episodes[0]
        base_row = int(self.rows.ep_offset[episode])
        frame = to_pixels(self.rows.frames([base_row]))[0, 0]
        mask = block_mask(frame)
        self.assertGreater(int(mask.sum()), 400)
        ys, xs = np.nonzero(mask.numpy())
        state = self.rows.state[base_row]
        scale = float(_IMAGE_SIZE) / IMAGE_EXTENT
        self.assertLess(
            abs(float(xs.mean()) - float(state[2]) * scale),
            20.0,
            "block mask centroid should track state[2] (block x)",
        )
        self.assertLess(abs(float(ys.mean()) - float(state[3]) * scale), 30.0)

    def test_curves(self) -> None:
        import numpy as np

        curves: dict[str, list[list[float]]] = {}
        displacements = []
        for episode in self.episodes:
            conditions = build_conditions(
                self.rows,
                episode,
                window=self.WINDOW,
                perturb_step=self.PERTURB_STEP,
                jump=self.JUMP,
                act_mean=self.act_mean,
                act_std=self.act_std,
            )
            displacements.append(
                teleport_displacement(self.rows, episode, self.PERTURB_STEP, self.JUMP)
            )
            for name, (pixels, actions) in conditions.items():
                curves.setdefault(name, []).append(
                    surprise_curve(MODEL, pixels, actions, self.WINDOW)
                )

        self.curves = {k: np.asarray(v) for k, v in curves.items()}
        self.displacements = displacements
        targets = list(range(3, self.WINDOW))
        step = targets.index(self.PERTURB_STEP)
        unperturbed = self.curves["unperturbed"][:, step].mean()
        teleported = self.curves["teleport"][:, step].mean()
        recoloured = self.curves["colour"][:, step].mean()

        # The teleport is the largest violation at the jump ...
        self.assertGreater(teleported, 3.0 * unperturbed)
        self.assertGreater(teleported, recoloured)
        # ... it decays once the new state has been observed ...
        self.assertLess(
            self.curves["teleport"][:, step + 1].mean(),
            self.curves["teleport"][:, step].mean(),
        )
        # ... while the unperturbed replay never reaches the spike.
        self.assertLess(self.curves["unperturbed"].max(), teleported)


if __name__ == "__main__":
    unittest.main()
