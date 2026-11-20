"""Rollout generation and window sampling for the flippable 2-D world.

The dataset is generated *from* :class:`amateur.physics.env.PhysicsEnv`; no
external data is read.  Episodes are stored as ``uint8`` frames with the
action that produced each transition, plus the gravity sign observed at each
frame.  Two evaluation regimes are supported and kept distinct:

* **in-distribution** -- gravity sign constant for the whole episode (by
  default the sign the model was fit on, +1 = downward);
* **flipped** -- either the whole episode starts with sign -1, or gravity is
  flipped mid-episode at ``flip_at``.  The mid-episode flip is the phase-5
  acceptance event: the frame distribution is unchanged, only the dynamics
  switch.

Frames are stored channels-last because that is what the renderer returns:
``(n_episodes, horizon + 1, frame_size, frame_size, 3)``, ``uint8``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .env import PhysicsEnv

__all__ = [
    "RolloutBatch",
    "concat_rollouts",
    "frames_tensor",
    "generate_rollouts",
    "sample_window_batch",
]

_SEED_STREAM = 0x5EED_0001
_ACTION_STREAM = 0xAC71_0002


@dataclass(frozen=True)
class RolloutBatch:
    """A batch of env rollouts.

    Attributes
    ----------
    frames : np.ndarray
        ``(N, horizon + 1, H, W, 3)`` ``uint8``; ``frames[:, t]`` is the frame
        *before* action ``t`` (and ``frames[:, 0]`` the reset frame).
    actions : np.ndarray
        ``(N, horizon)`` ``int64`` in ``0..4``; ``actions[:, t]`` produced
        ``frames[:, t + 1]``.
    gravity_sign : np.ndarray
        ``(N, horizon + 1)`` ``int8`` sign observed at each frame (+1 down,
        -1 up).
    seeds : np.ndarray
        ``(N,)`` ``int64`` reset seeds.
    flip_at : np.ndarray
        ``(N,)`` ``int64`` episode step at which gravity was flipped, or -1.
    horizon : int
    frame_size : int
    """

    frames: np.ndarray
    actions: np.ndarray
    gravity_sign: np.ndarray
    seeds: np.ndarray
    flip_at: np.ndarray
    horizon: int
    frame_size: int

    def __post_init__(self) -> None:
        n, steps, *rest = self.frames.shape
        if len(rest) != 3 or rest[2] != 3:
            raise ValueError("frames must be (N, T+1, H, W, 3)")
        if steps != self.horizon + 1:
            raise ValueError("frames length must be horizon + 1")
        if rest[0] != self.frame_size or rest[1] != self.frame_size:
            raise ValueError("frames do not match frame_size")
        if self.frames.dtype != np.uint8:
            raise ValueError("frames must be uint8")
        if self.actions.shape != (n, self.horizon):
            raise ValueError("actions must be (N, horizon)")
        if not np.issubdtype(self.actions.dtype, np.integer):
            raise ValueError("actions must be integer")
        if int(self.actions.min(initial=0)) < 0 or int(self.actions.max(initial=0)) > 4:
            raise ValueError("actions must lie in 0..4")
        if self.gravity_sign.shape != (n, self.horizon + 1):
            raise ValueError("gravity_sign must be (N, horizon + 1)")
        if self.seeds.shape != (n,) or self.flip_at.shape != (n,):
            raise ValueError("seeds/flip_at must be (N,)")

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_transitions(self) -> int:
        return int(self.frames.shape[0]) * self.horizon


def _episode_seed(seed: int, index: int) -> int:
    ss = np.random.SeedSequence([int(seed) & 0xFFFF_FFFF, int(index), _SEED_STREAM])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _action_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed) & 0xFFFF_FFFF, int(index), _ACTION_STREAM])
    )


def generate_rollouts(
    env: PhysicsEnv,
    n_episodes: int,
    horizon: int,
    seed: int,
    *,
    gravity_sign: int = 1,
    flip_at: int | None = None,
    action_policy: str = "random",
) -> RolloutBatch:
    """Roll out ``n_episodes`` seeded episodes of ``horizon`` steps.

    ``gravity_sign`` sets the sign at reset; ``flip_at`` (if not ``None``)
    flips it before the action taken at that step, so frames after
    ``flip_at`` are post-flip.  Actions are drawn from a per-episode
    ``numpy`` generator derived from ``seed`` (independent of ``PhysicsEnv``'s
    own reset RNG), so the action stream is reproducible and decoupled from
    the initial-state stream.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if flip_at is not None and not 0 <= flip_at < horizon:
        raise ValueError("flip_at must lie in [0, horizon)")
    if gravity_sign not in (-1, 1):
        raise ValueError("gravity_sign must be +1 or -1")
    if action_policy not in ("random", "noop"):
        raise ValueError("action_policy must be 'random' or 'noop'")

    size = env.config.frame_size
    frames = np.empty((n_episodes, horizon + 1, size, size, 3), dtype=np.uint8)
    actions = np.empty((n_episodes, horizon), dtype=np.int64)
    signs = np.empty((n_episodes, horizon + 1), dtype=np.int8)
    seeds = np.empty(n_episodes, dtype=np.int64)
    flips = np.full(n_episodes, -1 if flip_at is None else int(flip_at), dtype=np.int64)

    for episode in range(n_episodes):
        ep_seed = _episode_seed(seed, episode)
        seeds[episode] = ep_seed
        rng = _action_rng(seed, episode)
        frames[episode, 0] = env.reset(seed=ep_seed, gravity_sign=gravity_sign)
        signs[episode, 0] = env.gravity_sign
        for t in range(horizon):
            if flip_at is not None and t == flip_at:
                env.flip_gravity()
            if action_policy == "random":
                action = int(rng.integers(0, env.n_actions))
            else:
                action = 0
            actions[episode, t] = action
            frames[episode, t + 1] = env.step(action)
            signs[episode, t + 1] = env.gravity_sign

    return RolloutBatch(
        frames=frames,
        actions=actions,
        gravity_sign=signs,
        seeds=seeds,
        flip_at=flips,
        horizon=int(horizon),
        frame_size=int(size),
    )


def frames_tensor(batch: RolloutBatch) -> torch.Tensor:
    """All frames of a batch as a ``uint8 (N, T, C, H, W)`` torch tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(batch.frames))
    return tensor.permute(0, 1, 4, 2, 3)


def sample_window_batch(
    batch: RolloutBatch,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample contiguous windows of ``seq_len`` transitions.

    Returns ``(frames, actions)`` with shapes ``(B, seq_len + 1, C, H, W)``
    (``uint8`` torch tensor) and ``(B, seq_len)`` (``int64``).  Windows are
    sampled uniformly over episode start positions; the caller controls
    reproducibility by passing a seeded ``numpy`` generator.
    """
    if not 1 <= seq_len <= batch.horizon:
        raise ValueError("seq_len must lie in [1, horizon]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = len(batch)
    n_starts = batch.horizon - seq_len + 1
    episodes = rng.integers(0, n, size=batch_size)
    starts = rng.integers(0, n_starts, size=batch_size)

    frames = np.stack(
        [batch.frames[int(ep), int(s) : int(s) + seq_len + 1] for ep, s in zip(episodes, starts)]
    )
    actions = np.stack(
        [batch.actions[int(ep), int(s) : int(s) + seq_len] for ep, s in zip(episodes, starts)]
    )
    frames_t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 1, 4, 2, 3)
    actions_t = torch.from_numpy(np.ascontiguousarray(actions)).long()
    return frames_t, actions_t


def concat_rollouts(batches: list[RolloutBatch]) -> RolloutBatch:
    """Concatenate batches sharing ``horizon`` and ``frame_size``."""
    if not batches:
        raise ValueError("need at least one batch")
    horizon = batches[0].horizon
    size = batches[0].frame_size
    for batch in batches:
        if batch.horizon != horizon or batch.frame_size != size:
            raise ValueError("all batches must share horizon/frame_size")
    return RolloutBatch(
        frames=np.concatenate([b.frames for b in batches], axis=0),
        actions=np.concatenate([b.actions for b in batches], axis=0),
        gravity_sign=np.concatenate([b.gravity_sign for b in batches], axis=0),
        seeds=np.concatenate([b.seeds for b in batches], axis=0),
        flip_at=np.concatenate([b.flip_at for b in batches], axis=0),
        horizon=horizon,
        frame_size=size,
    )
