"""Physics boundary slice for the v0.10 physics epoch.

Contents (all CPU-only, no external data):

* :mod:`amateur.physics.env` -- a deterministic, seedable 2-D Newtonian world
  with a signed gravity constant that can be flipped.  Frames are
  ``uint8 (H, W, 3)``; the flip is a dynamics change, never a pixel change.
* :mod:`amateur.physics.data` -- seeded rollout generation and contiguous
  window sampling.
* :mod:`amateur.physics.micro_lewm` -- a micro LeWM-style JEPA (CNN encoder +
  GRU predictor + SIGReg) with separately addressable encoder/predictor
  parameters, an enforced encoder freeze, and checkpoint save/load.
* :mod:`amateur.physics.train` -- the training/evaluation CLI that writes
  ``.work/forge/checkpoint.pt`` and ``.work/forge/stats.json``.
* :mod:`amateur.physics.organ` -- the P5 digest + P6 predictor-overlay
  organ (phase 4-5a), standalone and testable.
* :mod:`amateur.physics.run` -- the phase-4-5b executor wiring
  (``PhysicsSource`` content-only hook + ``PhysicsRun`` single-stream
  acceptance).

The organ layer is the boundary that phases 4-5 consume (digest clustering,
ratchet overlay, gravity-flip acceptance).  The run layer is the thin
adapter into the ledger executor; it does not change the executor's ops
table, schedule, or channel semantics.
"""
from __future__ import annotations

from .env import ACTIONS, PhysicsEnv, WorldConfig
from .micro_lewm import MicroLeWM, MicroLeWMConfig, SIGReg
from .organ import (
    DigestThresholds,
    PhysicsOrgan,
    PredictorOverlay,
)
from .run import PhysicsRun, PhysicsSource

__all__ = [
    "ACTIONS",
    "DigestThresholds",
    "MicroLeWM",
    "MicroLeWMConfig",
    "PhysicsEnv",
    "PhysicsOrgan",
    "PhysicsRun",
    "PhysicsSource",
    "PredictorOverlay",
    "SIGReg",
    "WorldConfig",
]
