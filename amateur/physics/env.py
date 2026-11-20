"""A deterministic, flippable 2-D Newtonian world.

This is the *environment boundary* for the physics epoch: the world is ours
(no Two-Room, no external dataset), it is small enough to render on CPU, and
its only "law" that can be changed is the sign of gravity.  A gravity flip is
a change of the dynamics -- it is never drawn into the frame.  Two worlds in
the same physical state render byte-identical frames whether gravity points
down or up; the trajectories diverge only after the next ``step``.

World model
-----------
Coordinates are normalised to ``[0, 1] x [0, 1]`` with ``x`` right and ``y``
up.  The renderer maps ``y`` to pixel rows as ``row = (1 - y) * (H - 1)``.

* Box: closed walls on all four sides.  A body that crosses a wall is
  reflected and its wall-normal velocity is multiplied by ``restitution``.
* Bodies: one *agent* (thrust-controlled disc) and one *object* (passive
  disc).  Both obey the same Newtonian law::

      a_body = a_thrust + (0, gravity_sign * gravity)
      v <- clip(v + a * dt, max_speed)
      p <- p + v * dt

  There is no body-body collision (the two discs pass through each other);
  this keeps the dynamics separable and analytically checkable and is
  documented rather than hidden.
* Actions: five discrete thrust commands applied to the agent only::

      ACTIONS = ("noop", "left", "right", "down", "up")

  The four directional actions add ``thrust * dt`` to the agent velocity in
  that direction; the object is never actuated.
* Gravity: ``gravity`` is the magnitude in world-units / s^2 and
  ``gravity_sign`` is +1 (downward, y decreasing) or -1 (upward).
  :meth:`PhysicsEnv.flip_gravity` toggles the sign.  The renderer never
  receives the sign, so the flip cannot leak into the observation except
  through the physical state.

Frames
------
``reset`` and ``step`` return ``numpy.ndarray`` of shape
``(frame_size, frame_size, 3)`` and dtype ``uint8``, values in ``[0, 255]``.
The frame shows a static dark background, a static 8-pixel grid and border
(for position reference), the cyan agent and the orange object.  Nothing in
the frame encodes ``gravity_sign``, ``tick``, or ``last_action``.

Determinism
-----------
All physics is ``float64`` numpy.  ``reset(seed)`` draws initial positions
and velocities from ``numpy.random.default_rng(seed)``; ``step`` contains no
randomness.  The same seed plus the same action sequence reproduces the same
states and frames exactly on the same machine/numpy build.

Bounds
------
Positions stay inside ``[radius, 1 - radius]`` per axis, speeds are clipped
to ``max_speed``, and rendered values are clipped to ``[0, 255]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ACTIONS",
    "ACTION_DELTAS",
    "PhysicsEnv",
    "WorldConfig",
]

#: Discrete action names; index == action value.
ACTIONS: tuple[str, ...] = ("noop", "left", "right", "down", "up")

#: Unit thrust directions in world coordinates (x right, y up).
ACTION_DELTAS: np.ndarray = np.array(
    [
        [0.0, 0.0],   # noop
        [-1.0, 0.0],  # left
        [1.0, 0.0],   # right
        [0.0, -1.0],  # down
        [0.0, 1.0],   # up
    ],
    dtype=np.float64,
)

_BG = np.array([0.055, 0.063, 0.094], dtype=np.float32)
_GRID = np.array([0.125, 0.137, 0.180], dtype=np.float32)
_WALL = np.array([0.28, 0.30, 0.38], dtype=np.float32)
_AGENT_RGB = np.array([0.35, 0.78, 1.00], dtype=np.float32)   # cyan
_OBJECT_RGB = np.array([1.00, 0.55, 0.20], dtype=np.float32)  # orange
_GRID_PX = 8
_AA_PX = 1.35


@dataclass(frozen=True)
class WorldConfig:
    """Static configuration of :class:`PhysicsEnv`.

    Values are deliberately small and bounded; see the module docstring.
    """

    frame_size: int = 96
    dt: float = 0.1
    gravity: float = 4.0
    gravity_sign: int = 1
    restitution: float = 0.90
    thrust: float = 6.0
    max_speed: float = 2.0
    agent_radius: float = 0.075
    object_radius: float = 0.060
    spawn_margin: float = 0.18
    initial_speed: float = 0.50
    min_separation: float = 0.18

    def __post_init__(self) -> None:
        if self.frame_size < 32:
            raise ValueError("frame_size must be >= 32")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.gravity < 0.0:
            raise ValueError("gravity magnitude must be non-negative")
        if self.gravity_sign not in (-1, 1):
            raise ValueError("gravity_sign must be +1 or -1")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must lie in [0, 1]")
        if self.thrust < 0.0 or self.max_speed <= 0.0:
            raise ValueError("thrust/max_speed out of range")
        for name in ("agent_radius", "object_radius"):
            value = float(getattr(self, name))
            if not 0.0 < value < 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5)")
        if not 0.0 <= self.spawn_margin < 0.5:
            raise ValueError("spawn_margin out of range")
        if self.initial_speed < 0.0:
            raise ValueError("initial_speed must be non-negative")


class PhysicsEnv:
    """A seedable 2-D box with an actuated agent, a passive object, and a
    signed gravity constant that can be flipped at any tick.

    ``reset`` returns the first frame; ``step`` returns the next frame.  The
    observation is always ``uint8 (H, W, 3)``.  ``state()`` exposes the
    physical state for tests and for the phase-4/5 consumers.
    """

    def __init__(self, config: WorldConfig | None = None, **overrides: Any) -> None:
        if config is not None and overrides:
            raise TypeError("pass either config or keyword overrides, not both")
        self.config = config if config is not None else WorldConfig(**overrides)

        # Static image scaffolding (identical for every gravity sign).
        rows = np.arange(self.config.frame_size, dtype=np.float32)[:, None]
        cols = np.arange(self.config.frame_size, dtype=np.float32)[None, :]
        self._row = np.broadcast_to(rows, (self.config.frame_size, self.config.frame_size))
        self._col = np.broadcast_to(cols, (self.config.frame_size, self.config.frame_size))

        # Physical state -- populated by reset().
        self._agent_pos = np.zeros(2, dtype=np.float64)
        self._agent_vel = np.zeros(2, dtype=np.float64)
        self._object_pos = np.zeros(2, dtype=np.float64)
        self._object_vel = np.zeros(2, dtype=np.float64)
        self._gravity_sign = int(self.config.gravity_sign)
        self._tick = 0
        self._last_action = 0
        self._flips = 0
        self._episode_seed: int | None = None

    # ------------------------------------------------------------------ API
    @property
    def gravity_sign(self) -> int:
        return self._gravity_sign

    @property
    def action_space(self) -> tuple[str, ...]:
        return ACTIONS

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        size = self.config.frame_size
        return (size, size, 3)

    def reset(self, seed: int | None = None, *, gravity_sign: int | None = None) -> np.ndarray:
        """Reset to a seeded initial state and return the first frame.

        ``seed=None`` uses fresh entropy (non-reproducible by design).  The
        optional ``gravity_sign`` keyword sets the initial sign for this
        episode (used to build whole-episode flipped-gravity evaluation
        data); otherwise the config sign is restored.
        """
        sign = self.config.gravity_sign if gravity_sign is None else int(gravity_sign)
        if sign not in (-1, 1):
            raise ValueError("gravity_sign must be +1 or -1")
        rng = np.random.default_rng(seed)
        self._episode_seed = None if seed is None else int(seed)
        self._gravity_sign = sign
        self._tick = 0
        self._last_action = 0
        self._flips = 0

        a_pos = self._random_positions(rng, self.config.agent_radius)
        for _ in range(64):
            o_pos = self._random_positions(rng, self.config.object_radius)
            if float(np.linalg.norm(a_pos - o_pos)) >= self.config.min_separation:
                break
        else:  # pragma: no cover - rejection is overwhelmingly likely to pass
            o_pos = np.array([1.0 - self.config.spawn_margin] * 2, dtype=np.float64)

        speed = self.config.initial_speed
        self._agent_pos = a_pos
        self._object_pos = o_pos
        self._agent_vel = rng.uniform(-speed, speed, size=2).astype(np.float64)
        self._object_vel = rng.uniform(-speed, speed, size=2).astype(np.float64)
        return self.render()

    def step(self, action: int) -> np.ndarray:
        """Advance the world one tick and return the resulting frame."""
        action = self._validate_action(action)
        self._last_action = action

        # Agent thrust first, then the common gravitational acceleration.
        self._agent_vel = self._agent_vel + ACTION_DELTAS[action] * (self.config.thrust * self.config.dt)
        # gravity_sign=+1 points down (-y); sign=-1 points up (+y).
        self._agent_vel[1] -= self._gravity_sign * self.config.gravity * self.config.dt
        self._object_vel[1] -= self._gravity_sign * self.config.gravity * self.config.dt

        self._integrate(self._agent_pos, self._agent_vel, self.config.agent_radius)
        self._integrate(self._object_pos, self._object_vel, self.config.object_radius)

        self._tick += 1
        return self.render()

    def flip_gravity(self) -> int:
        """Reverse the sign of gravity; returns the new sign.

        This changes the acceleration law for both bodies at the next
        integration step.  It does not touch positions, velocities, tick,
        or the renderer.
        """
        self._gravity_sign = -self._gravity_sign
        self._flips += 1
        return self._gravity_sign

    def state(self) -> dict[str, Any]:
        """Return a defensive copy of the full physical state."""
        return {
            "tick": int(self._tick),
            "gravity": float(self.config.gravity),
            "gravity_sign": int(self._gravity_sign),
            "signed_gravity": float(-self._gravity_sign * self.config.gravity),
            "gravity_flips": int(self._flips),
            "agent_pos": self._agent_pos.copy(),
            "agent_vel": self._agent_vel.copy(),
            "object_pos": self._object_pos.copy(),
            "object_vel": self._object_vel.copy(),
            "agent_radius": float(self.config.agent_radius),
            "object_radius": float(self.config.object_radius),
            "frame_size": int(self.config.frame_size),
            "last_action": int(self._last_action),
            "episode_seed": self._episode_seed,
        }

    def render(self) -> np.ndarray:
        """Render the current state as a ``uint8 (H, W, 3)`` frame.

        The renderer is a pure function of the physical positions: the
        gravity sign, tick, action history, and seed are not consulted.
        """
        h = w = self.config.frame_size
        img = np.empty((h, w, 3), dtype=np.float32)
        img[:] = _BG
        img[::_GRID_PX, :, :] = _GRID
        img[:, ::_GRID_PX, :] = _GRID
        img[0:1, :, :] = _WALL
        img[-1:, :, :] = _WALL
        img[:, 0:1, :] = _WALL
        img[:, -1:, :] = _WALL

        img = self._draw_disc(img, self._agent_pos, self.config.agent_radius, _AGENT_RGB)
        img = self._draw_disc(img, self._object_pos, self.config.object_radius, _OBJECT_RGB)
        return np.clip(img * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------- internals
    def _random_positions(self, rng: np.random.Generator, radius: float) -> np.ndarray:
        margin = self.config.spawn_margin
        lo, hi = margin, 1.0 - margin
        # Keep the disc fully inside the box regardless of spawn_margin.
        lo = max(lo, radius)
        hi = min(hi, 1.0 - radius)
        return rng.uniform(lo, hi, size=2).astype(np.float64)

    def _integrate(self, pos: np.ndarray, vel: np.ndarray, radius: float) -> None:
        """Semi-implicit Euler with reflective walls and a speed cap."""
        dt = self.config.dt
        speed = float(np.linalg.norm(vel))
        if speed > self.config.max_speed:
            vel *= self.config.max_speed / speed
        pos += vel * dt

        lo, hi = radius, 1.0 - radius
        for axis in (0, 1):
            if pos[axis] < lo:
                pos[axis] = 2.0 * lo - pos[axis]
                vel[axis] = -vel[axis] * self.config.restitution
            elif pos[axis] > hi:
                pos[axis] = 2.0 * hi - pos[axis]
                vel[axis] = -vel[axis] * self.config.restitution
            pos[axis] = min(max(pos[axis], lo), hi)

    @staticmethod
    def _validate_action(action: int) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be an int in 0..{len(ACTIONS) - 1}")
        value = int(action)
        if not 0 <= value < len(ACTIONS):
            raise ValueError(f"action must be in 0..{len(ACTIONS) - 1}, got {value}")
        return value

    def _draw_disc(
        self,
        img: np.ndarray,
        position: np.ndarray,
        radius: float,
        rgb: np.ndarray,
    ) -> np.ndarray:
        h = w = self.config.frame_size
        cx = float(position[0]) * (w - 1)
        cy = (1.0 - float(position[1])) * (h - 1)
        r_px = float(radius) * (h - 1)
        dist = np.sqrt((self._col - cx) ** 2 + (self._row - cy) ** 2)
        alpha = np.clip((r_px - dist) / _AA_PX + 0.5, 0.0, 1.0).astype(np.float32)
        return img + (rgb - img) * alpha[..., None]
