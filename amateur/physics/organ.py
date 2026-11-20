"""Phase 4-5a physics organ: P5 digest + P6 predictor overlay.

This module is the **standalone physics organ** for the flippable 2-D world.
It owns E1 (the environment) and the phase-3 micro-LeWM (frozen encoder,
trained predictor) and implements the phase-5/6 state machines as pure,
testable objects -- there is **no executor coupling** here.  The executor
wiring plan is in ``docs/phase4-5-design.md``.

Three layers, each independently testable
-----------------------------------------
1. :class:`Observation` -- one causal observation tick: the frame latent,
   the latent state delta (#12), the model surprise (#6/#12), the residual
   vector (the real debris candidate), a 3-frame causal window, and a short
   *resolved* latent rollout (#10).
2. :class:`ResidualDigester` (P5) -- a bounded online residual clusterer.
   Only residuals above the measured normal q99 enter; residuals are
   assigned to centroids within a merge radius, clusters are merged to make
   room, and the pool is hard-capped at ``debris_capacity``.  It emits
   :class:`DigestedDebris` (D3 bodies), :class:`StructuralMotif` (when a
   cluster is stable), and :class:`CollapseMandate` (when the ceiling forces
   a collapse).
3. :class:`OverlayTable` + :class:`PredictorOverlay` (P6) -- a cheap
   predictor-only patch keyed by ``overlay_id``, stored in a side table,
   with ``(epoch, overlay_id, status)`` records, birth / revalidate /
   stale / evict transitions and capacity 4.  **The encoder is never
   touched:** fitting happens on a deep copy of the predictor and the
   encoder parameters are asserted bit-identical (``torch.equal``) before
   and after every fit.

Threshold derivation (measured phase-3 statistics, not synthetic noise)
-----------------------------------------------------------------------
All numeric thresholds come from ``.work/forge/stats.json`` (phase 3) via
:meth:`Phase3Stats.from_stats` / :func:`derive_thresholds`:

===============  =============================================  =========
threshold        derivation                                      value
===============  =============================================  =========
debris_threshold ``normal_q99`` (1 % normal false-positive)      0.413042
spike_threshold  ``normal_q99`` (same measured tail)             0.413042
merge_radius     ``sqrt(latent_dim * post_flip_q95)``            6.7436
normal_baseline  ``normal_mean``                                 0.049386
===============  =============================================  =========

The ``merge_radius`` is the Euclidean norm of a post-flip residual at its
measured q95 (``||e||**2 == latent_dim * surprise``); the measured RMS
distance of flipped residual vectors to their centroid is ``3.919`` and the
measured q95 spread is ``6.790``, so one cluster with radius ``6.744``
covers the flipped regime.  Structural constants (pool capacity, overlay
capacity, motif support, persistence gate) are **design choices**, listed
with their justification in :class:`DigestThresholds` and in the design
memo's honesty notes.

Measured limits (honest, see the memo's numbered notes)
-------------------------------------------------------
* A predictor-only patch fitted on a **single** stream's window does not
  improve forward surprise (measured ratios 0.99-1.18x); it only fits its
  window.  The overlay is therefore demonstrated on a **bank of streams**
  with a fit/held-out split, which is a shaped probe, not the executor's
  single-stream flow.
* A head-only patch is used by default: it leaves the GRU (and therefore
  the online hidden state) unchanged, so applying/clearing an overlay is
  exact with no hidden-state replay.  A full-predictor patch is available
  for measurement only.
"""
from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .data import _action_rng, _episode_seed
from .env import PhysicsEnv, WorldConfig
from .micro_lewm import MicroLeWM

__all__ = [
    "AcceptanceTick",
    "AcceptanceTrace",
    "CollapseMandate",
    "D4StandIn",
    "DigestResult",
    "DigestThresholds",
    "DigestedDebris",
    "FitRecord",
    "Observation",
    "OverlayRecord",
    "OverlayTable",
    "Phase3Stats",
    "PhysicsOrgan",
    "PredictorOverlay",
    "RatchetResult",
    "ResidualDigester",
    "StructuralMotif",
    "derive_thresholds",
]

#: Default artifact paths (relative to the repo root; ``.work`` is ignored).
DEFAULT_CHECKPOINT = Path(".work/forge/checkpoint.pt")
DEFAULT_STATS = Path(".work/forge/stats.json")

#: The phase-3 checkpoint's latent width (used to derive radii from stats).
DEFAULT_LATENT_DIM = 64

#: Design constants.  ``DEBRIS_CAPACITY`` and ``OVERLAY_CAPACITY`` mirror the
#: executor's D2 capacity (``amateur/payloads.py::MAT_CAPACITY == 4``).
#: ``MOTIF_MIN_COUNT`` is the smallest measured separating window (the
#: phase-3 doc's "first 4 post-flip" window, 10.03x normal); four residuals
#: at the measured 1 % normal false-positive rate are statistically decisive.
#: ``MOTIF_MIN_PERSISTENCE`` is *probe-derived*: forward-generalising patches
#: appear at >= 10 post-flip ticks of banked support (measured in the design
#: memo), so adoption waits for that depth.
DEBRIS_CAPACITY = 8
OVERLAY_CAPACITY = 4
MOTIF_MIN_COUNT = 4
MOTIF_MIN_PERSISTENCE = 10
ROLLOUT_HORIZON = 4
WINDOW_FRAMES = 3
STALE_PATIENCE = 2
STALE_AFTER = 3
COOLDOWN_TICKS = 4
#: A stable motif may miss an isolated support tick; two consecutive misses reset.
MOTIF_STREAK_GAP = 2
DEFAULT_PATCH = "head"
DEFAULT_FIT_STEPS = 40
DEFAULT_FIT_LR = 1e-3
DEFAULT_CACHE_LEN = 64


# --------------------------------------------------------------------------
# Threshold derivation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Phase3Stats:
    """The measured phase-3 surprise statistics this organ consumes."""

    normal_mean: float
    normal_median: float
    normal_q95: float
    normal_q99: float
    post_flip_mean: float
    post_flip_q95: float
    immediate4_mean: float
    latent_dim: int = DEFAULT_LATENT_DIM

    @classmethod
    def from_stats(cls, path: str | Path = DEFAULT_STATS) -> "Phase3Stats":
        """Read the phase-3 ``stats.json`` artifact."""
        data = json.loads(Path(path).read_text())
        normal = data["evaluation"]["in_distribution"]["surprise"]
        post = data["evaluation"]["flipped_mid_episode"]["post_flip"]
        immediate = data["evaluation"]["flipped_mid_episode"]["immediate4"]
        return cls(
            normal_mean=float(normal["mean"]),
            normal_median=float(normal["median"]),
            normal_q95=float(normal["q95"]),
            normal_q99=float(normal["q99"]),
            post_flip_mean=float(post["mean"]),
            post_flip_q95=float(post["q95"]),
            immediate4_mean=float(immediate["mean"]),
            latent_dim=int(data["model_config"]["latent_dim"]),
        )


@dataclass(frozen=True)
class DigestThresholds:
    """The digest/ratchet operating point, derived from :class:`Phase3Stats`.

    ``debris_threshold`` / ``spike_threshold`` are the measured normal q99.
    ``merge_radius`` is the norm of a post-flip residual at its measured q95
    (``sqrt(D * q95)``).  ``normal_*``/``post_flip_*`` carry the raw measured
    reference values so tests and the memo can cite them without re-reading
    the artifact.  The remaining fields are structural design constants; the
    dataclass makes them explicit rather than scattered literals.
    """

    debris_threshold: float
    spike_threshold: float
    merge_radius: float
    normal_baseline: float
    normal_median: float
    normal_q95: float
    post_flip_mean: float
    post_flip_q95: float
    immediate4_mean: float
    debris_capacity: int = DEBRIS_CAPACITY
    overlay_capacity: int = OVERLAY_CAPACITY
    motif_min_count: int = MOTIF_MIN_COUNT
    motif_min_persistence: int = MOTIF_MIN_PERSISTENCE
    rollout_horizon: int = ROLLOUT_HORIZON
    window_frames: int = WINDOW_FRAMES
    stale_patience: int = STALE_PATIENCE
    stale_after: int = STALE_AFTER
    cooldown_ticks: int = COOLDOWN_TICKS
    motif_streak_gap: int = MOTIF_STREAK_GAP

    def __post_init__(self) -> None:
        if self.debris_capacity < 2:
            raise ValueError("debris_capacity must be >= 2")
        if self.overlay_capacity < 1:
            raise ValueError("overlay_capacity must be >= 1")
        if self.motif_min_count < 1 or self.motif_min_persistence < 1:
            raise ValueError("motif gates must be positive")
        if self.stale_patience < 1 or self.stale_after < 0:
            raise ValueError("stale gates out of range")


def derive_thresholds(stats: Phase3Stats) -> DigestThresholds:
    """Derive the digest operating point from measured phase-3 statistics."""
    merge_radius = float(np.sqrt(stats.latent_dim * stats.post_flip_q95))
    return DigestThresholds(
        debris_threshold=float(stats.normal_q99),
        spike_threshold=float(stats.normal_q99),
        merge_radius=merge_radius,
        normal_baseline=float(stats.normal_mean),
        normal_median=float(stats.normal_median),
        normal_q95=float(stats.normal_q95),
        post_flip_mean=float(stats.post_flip_mean),
        post_flip_q95=float(stats.post_flip_q95),
        immediate4_mean=float(stats.immediate4_mean),
    )


def _default_thresholds() -> DigestThresholds:
    """Phase-3 artifact if present, otherwise the recorded constants.

    The fallback values are the *recorded* phase-3 measurements (the same
    numbers the artifact carries); they are not a synthetic substitute.
    """
    try:
        return derive_thresholds(Phase3Stats.from_stats(DEFAULT_STATS))
    except (OSError, KeyError, ValueError):
        recorded = Phase3Stats(
            normal_mean=0.04938631785656374,
            normal_median=0.02428896352648735,
            normal_q95=0.17219337970018384,
            normal_q99=0.4130416682362567,
            post_flip_mean=0.28597303529750207,
            post_flip_q95=0.7105834186077118,
            immediate4_mean=0.34670642018318176,
            latent_dim=DEFAULT_LATENT_DIM,
        )
        return derive_thresholds(recorded)


# --------------------------------------------------------------------------
# Payload-shaped bodies (immutable snapshots)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Observation:
    """One causal observation tick (the #6/#10/#12 source bodies).

    ``tick`` is the frame index within the stream (0 is the reset frame).
    ``surprise`` is the measured mean squared latent prediction error and
    ``residual`` the signed error vector ``z_hat - z`` (surprise ==
    ``mean(residual**2)``).  At the reset tick there is nothing to predict:
    ``surprise`` is ``nan`` and ``residual``/``latent_delta`` are zeros.
    """

    tick: int
    stream: int
    action: int | None
    latent: np.ndarray
    latent_delta: np.ndarray
    surprise: float
    residual: np.ndarray
    rollout: np.ndarray
    window: np.ndarray
    overlay_id: str | None
    base_surprise: float = float("nan")
    ingest: bool = True
    frame: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.latent.ndim != 1:
            raise ValueError("latent must be (D,)")
        if self.residual.shape != self.latent.shape:
            raise ValueError("residual shape must match latent")
        if self.latent_delta.shape != self.latent.shape:
            raise ValueError("latent_delta shape must match latent")
        if self.rollout.ndim != 2 or self.rollout.shape[1] != self.latent.size:
            raise ValueError("rollout must be (horizon, D)")
        if self.window.ndim != 2 or self.window.shape[1] != self.latent.size:
            raise ValueError("window must be (w, D)")


@dataclass(frozen=True)
class DigestedDebris:
    """A D3 debris body: a real residual cluster (never synthetic noise)."""

    debris_id: str
    centroid: np.ndarray
    count: int
    spread: float
    first_tick: int
    last_tick: int
    peak_surprise: float

    def __post_init__(self) -> None:
        if self.centroid.ndim != 1:
            raise ValueError("centroid must be (D,)")
        if self.count < 1:
            raise ValueError("count must be positive")
        if self.spread < 0.0:
            raise ValueError("spread must be non-negative")


@dataclass(frozen=True)
class StructuralMotif:
    """An extracted rule (#21): a stable D3 cluster with persistence."""

    motif_id: str
    debris_id: str
    centroid: np.ndarray
    count: int
    spread: float
    persistence: int
    drift: float
    first_tick: int
    last_tick: int


@dataclass(frozen=True)
class CollapseMandate:
    """Ceiling pressure (#17): the bounded pool had to collapse."""

    reason: str
    pressure: int
    capacity: int
    active_bodies: int
    freed_body: str | None
    tick: int


@dataclass(frozen=True)
class DigestResult:
    """P5 output for one observation: debris, motif, mandate, pool state."""

    observation: Observation
    tick: int
    ingested: bool
    assigned_debris_id: str | None
    created: DigestedDebris | None
    merged: tuple[DigestedDebris, ...]
    evicted: tuple[DigestedDebris, ...]
    motif: StructuralMotif | None
    mandate: CollapseMandate | None
    pool: tuple[DigestedDebris, ...]


# --------------------------------------------------------------------------
# P5 -- bounded online residual clusterer
# --------------------------------------------------------------------------
@dataclass
class _Body:
    debris_id: str
    centroid: np.ndarray
    count: int
    ssd: float
    first_tick: int
    last_tick: int
    peak_surprise: float
    first_centroid: np.ndarray
    support_ticks: set[int] = field(default_factory=set)
    streak: int = 1
    streak_first_tick: int = 0
    streak_first_centroid: np.ndarray | None = None
    last_support_tick: int = 0

    def snapshot(self) -> DigestedDebris:
        spread = float(np.sqrt(self.ssd / self.count)) if self.count > 0 else 0.0
        return DigestedDebris(
            debris_id=self.debris_id,
            centroid=self.centroid.copy(),
            count=int(self.count),
            spread=spread,
            first_tick=int(self.first_tick),
            last_tick=int(self.last_tick),
            peak_surprise=float(self.peak_surprise),
        )

    def absorb(self, residual: np.ndarray, surprise: float, tick: int, gap: int = MOTIF_STREAK_GAP) -> None:
        if tick == self.last_support_tick:
            pass  # several streams may support the same tick; one tick, one step
        elif tick - self.last_support_tick <= gap:
            self.streak += 1
        else:
            # A gap breaks the motif streak; re-origin the stability window.
            self.streak = 1
            self.streak_first_tick = tick
            self.streak_first_centroid = self.centroid.copy()
        self.last_support_tick = max(self.last_support_tick, tick)
        self.count += 1
        delta = residual - self.centroid
        self.centroid = self.centroid + delta / self.count
        self.ssd += float(delta @ delta) * (self.count - 1) / self.count
        self.last_tick = max(self.last_tick, tick)
        self.peak_surprise = max(self.peak_surprise, float(surprise))
        self.support_ticks.add(int(tick))


class ResidualDigester:
    """P5: bounded online clustering of real prediction residuals.

    Capacity is a hard ceiling.  On an unmatched residual with a full pool
    the digester either merges the closest body pair (when they are close
    enough to be the same phenomenon) or evicts the least-supported body,
    then admits the new body.  A :class:`CollapseMandate` is emitted whenever
    a ceiling collapse happened, naming the freed body.
    """

    def __init__(self, thresholds: DigestThresholds | None = None) -> None:
        self.thresholds = thresholds or _default_thresholds()
        self._bodies: dict[str, _Body] = {}
        self._serial = 0
        self.observations = 0
        self.ingested = 0
        self.merges = 0
        self.evictions = 0
        self.mandates = 0

    # ------------------------------------------------------------- helpers
    def _new_id(self) -> str:
        self._serial += 1
        return f"debris-{self._serial:03d}"

    def _nearest(self, residual: np.ndarray) -> tuple[_Body | None, float]:
        best: _Body | None = None
        best_distance = float("inf")
        for body in self._bodies.values():
            distance = float(np.linalg.norm(residual - body.centroid))
            if distance < best_distance:
                best = body
                best_distance = distance
        return best, best_distance

    def _merge_pair(self, a: _Body, b: _Body) -> _Body:
        # Keep the better-supported body's identity; merge the other into it.
        if (b.count, b.debris_id) > (a.count, a.debris_id):
            a, b = b, a
        total = a.count + b.count
        centroid = (a.centroid * a.count + b.centroid * b.count) / total
        gap = float(np.linalg.norm(a.centroid - b.centroid))
        # Merge the within-cluster scatter plus the between-centroid gap.
        ssd = a.ssd + b.ssd + gap * gap * a.count * b.count / total
        if abs(a.last_support_tick - b.last_support_tick) <= MOTIF_STREAK_GAP:
            source = a if a.streak >= b.streak else b
            streak = max(a.streak, b.streak)
        else:
            source = a if a.last_support_tick >= b.last_support_tick else b
            streak = source.streak
        merged = _Body(
            debris_id=a.debris_id,
            centroid=centroid,
            count=total,
            ssd=ssd,
            first_tick=min(a.first_tick, b.first_tick),
            last_tick=max(a.last_tick, b.last_tick),
            peak_surprise=max(a.peak_surprise, b.peak_surprise),
            first_centroid=(a.first_centroid * a.count + b.first_centroid * b.count) / total,
            support_ticks=set(a.support_ticks) | set(b.support_ticks),
            streak=streak,
            streak_first_tick=source.streak_first_tick,
            streak_first_centroid=source.streak_first_centroid,
            last_support_tick=max(a.last_support_tick, b.last_support_tick),
        )
        del self._bodies[b.debris_id]
        self._bodies[a.debris_id] = merged
        self.merges += 1
        return merged

    def _compact(self, min_distance: float) -> list[DigestedDebris]:
        """Merge centroid pairs closer than ``min_distance`` (deterministic)."""
        merged: list[DigestedDebris] = []
        changed = True
        while changed and len(self._bodies) > 1:
            changed = False
            ids = list(self._bodies)
            for i, left in enumerate(ids):
                for right in ids[i + 1 :]:
                    a, b = self._bodies.get(left), self._bodies.get(right)
                    if a is None or b is None:
                        continue
                    if float(np.linalg.norm(a.centroid - b.centroid)) < min_distance:
                        survivor = self._merge_pair(a, b)
                        merged.append(survivor.snapshot())
                        changed = True
                        break
                if changed:
                    break
        return merged

    def _evict_least_supported(self) -> _Body | None:
        if not self._bodies:
            return None
        victim = min(
            self._bodies.values(),
            key=lambda body: (body.count, -body.last_tick, body.debris_id),
        )
        del self._bodies[victim.debris_id]
        self.evictions += 1
        return victim

    def _make_room(self) -> tuple[list[DigestedDebris], list[DigestedDebris], _Body | None]:
        """Free one slot; returns (merged, evicted, freed-body)."""
        merged: list[DigestedDebris] = []
        evicted: list[DigestedDebris] = []
        ids = list(self._bodies)
        best_pair: tuple[_Body, _Body] | None = None
        best_distance = float("inf")
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                a, b = self._bodies[left], self._bodies[right]
                distance = float(np.linalg.norm(a.centroid - b.centroid))
                if distance < best_distance:
                    best_pair = (a, b)
                    best_distance = distance
        # Close enough to be one phenomenon: collapse the pair.  Otherwise
        # evict the least-supported body (count tie-break -- "evict by count").
        if best_pair is not None and best_distance <= 2.0 * self.thresholds.merge_radius:
            survivor = self._merge_pair(*best_pair)
            merged.append(survivor.snapshot())
            return merged, evicted, None
        victim = self._evict_least_supported()
        if victim is not None:
            evicted.append(victim.snapshot())
        return merged, evicted, victim

    # ----------------------------------------------------------------- API
    def observe(self, observation: Observation) -> DigestResult:
        """Ingest one observation; returns the P5 bodies and pool state."""
        thresholds = self.thresholds
        self.observations += 1
        tick = observation.tick
        surprise = observation.surprise
        residual = observation.residual
        finite = np.isfinite(surprise) and bool(np.all(np.isfinite(residual)))
        if not observation.ingest or not finite or surprise <= thresholds.debris_threshold:
            return DigestResult(
                observation=observation,
                tick=tick,
                ingested=False,
                assigned_debris_id=None,
                created=None,
                merged=(),
                evicted=(),
                motif=None,
                mandate=None,
                pool=self.snapshot(),
            )

        self.ingested += 1
        created: DigestedDebris | None = None
        merged: list[DigestedDebris] = []
        evicted: list[DigestedDebris] = []
        mandate: CollapseMandate | None = None
        body, distance = self._nearest(residual)
        if body is not None and distance <= thresholds.merge_radius:
            body.absorb(residual, surprise, tick, thresholds.motif_streak_gap)
            assigned = body.debris_id
        else:
            if len(self._bodies) >= thresholds.debris_capacity:
                was_full = len(self._bodies)
                room_merged, room_evicted, freed = self._make_room()
                merged.extend(room_merged)
                evicted.extend(room_evicted)
                self.mandates += 1
                mandate = CollapseMandate(
                    reason="ceiling-pressure",
                    pressure=was_full,
                    capacity=thresholds.debris_capacity,
                    active_bodies=len(self._bodies),
                    freed_body=None if freed is None else freed.debris_id,
                    tick=tick,
                )
            new_id = self._new_id()
            body = _Body(
                debris_id=new_id,
                centroid=residual.copy(),
                count=1,
                ssd=0.0,
                first_tick=tick,
                last_tick=tick,
                peak_surprise=float(surprise),
                first_centroid=residual.copy(),
                support_ticks={tick},
                streak=1,
                streak_first_tick=tick,
                streak_first_centroid=residual.copy(),
                last_support_tick=tick,
            )
            self._bodies[new_id] = body
            assigned = new_id
            created = body.snapshot()

        merged.extend(self._compact(thresholds.merge_radius / 2.0))
        motif = self._motif_for(assigned)
        if mandate is not None:
            mandate = replace(mandate, active_bodies=len(self._bodies))
        return DigestResult(
            observation=observation,
            tick=tick,
            ingested=True,
            assigned_debris_id=assigned,
            created=created,
            merged=tuple(merged),
            evicted=tuple(evicted),
            motif=motif,
            mandate=mandate,
            pool=self.snapshot(),
        )

    def _motif_for(self, debris_id: str | None) -> StructuralMotif | None:
        if debris_id is None:
            return None
        body = self._bodies.get(debris_id)
        if body is None:
            return None
        thresholds = self.thresholds
        persistence = int(body.streak)
        if body.count < thresholds.motif_min_count:
            return None
        if persistence < thresholds.motif_min_persistence:
            return None
        if body.last_support_tick != body.last_tick:
            return None
        spread = float(np.sqrt(body.ssd / body.count)) if body.count else 0.0
        if spread > thresholds.merge_radius:
            return None
        origin = body.streak_first_centroid
        if origin is None:  # pragma: no cover - always set on creation/absorption
            origin = body.first_centroid
        drift = float(np.linalg.norm(body.centroid - origin))
        if drift > thresholds.merge_radius:
            return None
        return StructuralMotif(
            motif_id=f"motif-{body.debris_id}",
            debris_id=body.debris_id,
            centroid=body.centroid.copy(),
            count=int(body.count),
            spread=spread,
            persistence=persistence,
            drift=drift,
            first_tick=int(body.streak_first_tick),
            last_tick=int(body.last_tick),
        )

    def snapshot(self) -> tuple[DigestedDebris, ...]:
        """The current D3 pool, oldest body first, as immutable snapshots."""
        return tuple(body.snapshot() for body in self._bodies.values())

    def __len__(self) -> int:
        return len(self._bodies)


# --------------------------------------------------------------------------
# P6 -- predictor-only overlay + lifecycle table
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictorOverlay:
    """A cheap predictor-only patch, stored by ``overlay_id`` in a side table.

    ``deltas`` maps predictor parameter names to additive tensors; it never
    contains encoder parameters.  ``changes_gru`` is true when the patch
    touches the GRU (a full-predictor fit), which requires replaying the
    causal cache to reseed each stream's hidden state when the overlay is
    applied or cleared.  A head-only patch leaves the GRU untouched and
    needs no replay.
    """

    overlay_id: str
    deltas: dict[str, torch.Tensor]
    param_count: int
    fit_transitions: int
    fit_streams: int
    loss_before: float
    loss_after: float
    patch: str

    @property
    def changes_gru(self) -> bool:
        return any(name.startswith("gru.") for name in self.deltas)

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.deltas.values())

    def apply_to(self, model: MicroLeWM) -> None:
        """Add the patch to ``model.predictor`` in place (no grad)."""
        parameters = dict(model.predictor.named_parameters())
        with torch.no_grad():
            for name, delta in self.deltas.items():
                parameters[name].add_(delta)


@dataclass(frozen=True)
class FitRecord:
    """Lightweight diagnostics of one overlay fit (no tensors).

    The patch tensors belong to the D2 store (``MatBoard``) while the
    overlay is resident; this record keeps only scalars so the organ holds
    no tensor after an eviction (Q1-A "no organ-side retention").
    """

    overlay_id: str
    param_count: int
    nbytes: int
    fit_transitions: int
    fit_streams: int
    loss_before: float
    loss_after: float
    patch: str


@dataclass(frozen=True)
class OverlayRecord:
    """One D2-shaped overlay record: ``(epoch, overlay_id, status)`` plus
    the lifecycle bookkeeping the executor needs for gating."""

    epoch: int
    overlay_id: str
    status: str
    motif_id: str
    birth_tick: int
    last_validated_tick: int
    stale_tick: int | None = None

    def token(self) -> tuple[int, str, str]:
        """The exact payload triple the D2 mat carries (#23/#24/#25)."""
        return (self.epoch, self.overlay_id, self.status)


class OverlayTable:
    """Bounded overlay lifecycle keyed by ``overlay_id`` (capacity 4).

    The adapter lives in a **side table** (``self._adapters``) keyed by id;
    the records carry only ``(epoch, overlay_id, status)`` and bookkeeping,
    so the executor's D2 payloads stay small and provenance-clean.  The
    table enforces: birth at epoch 1, revalidation advances the epoch,
    stale records stop resolving to an adapter, and eviction removes both
    the record and the adapter.
    """

    def __init__(self, thresholds: DigestThresholds | None = None) -> None:
        self.thresholds = thresholds or _default_thresholds()
        self._records: dict[str, OverlayRecord] = {}
        self._adapters: dict[str, PredictorOverlay] = {}
        self.births: list[OverlayRecord] = []
        self.revalidations: list[OverlayRecord] = []
        self.stales: list[OverlayRecord] = []
        self.evictions: list[OverlayRecord] = []

    # ------------------------------------------------------------- helpers
    def _sync(self, record: OverlayRecord) -> OverlayRecord:
        self._records[record.overlay_id] = record
        return record

    def _evict_oldest(self, *, stale_only: bool) -> OverlayRecord | None:
        candidates = [
            record
            for record in self._records.values()
            if not stale_only or record.status == "stale"
        ]
        if not candidates:
            return None
        victim = min(
            candidates,
            key=lambda record: (
                record.last_validated_tick if record.status == "stale" else record.birth_tick,
                record.overlay_id,
            ),
        )
        return self.evict(victim.overlay_id)

    # ----------------------------------------------------------------- API
    def birth(
        self,
        overlay: PredictorOverlay,
        *,
        motif_id: str,
        tick: int,
    ) -> OverlayRecord:
        if overlay.overlay_id in self._records:
            raise ValueError(f"overlay {overlay.overlay_id!r} already exists")
        while len(self._records) >= self.thresholds.overlay_capacity:
            victim = self._evict_oldest(stale_only=True) or self._evict_oldest(stale_only=False)
            if victim is None:  # pragma: no cover - capacity >= 1 guarantees one
                break
        record = OverlayRecord(
            epoch=1,
            overlay_id=overlay.overlay_id,
            status="active",
            motif_id=motif_id,
            birth_tick=tick,
            last_validated_tick=tick,
        )
        self._records[record.overlay_id] = record
        self._adapters[record.overlay_id] = overlay
        self.births.append(record)
        return record

    def revalidate(self, overlay_id: str, tick: int) -> OverlayRecord:
        record = self._records.get(overlay_id)
        if record is None:
            raise KeyError(f"no overlay {overlay_id!r}")
        record = replace(
            record,
            epoch=record.epoch + 1,
            status="active",
            last_validated_tick=max(record.last_validated_tick, tick),
            stale_tick=None,
        )
        self.revalidations.append(record)
        return self._sync(record)

    def mark_stale(self, overlay_id: str, tick: int) -> OverlayRecord:
        record = self._records.get(overlay_id)
        if record is None:
            raise KeyError(f"no overlay {overlay_id!r}")
        record = replace(record, status="stale", stale_tick=tick)
        self.stales.append(record)
        return self._sync(record)

    def evict(self, overlay_id: str) -> OverlayRecord | None:
        record = self._records.pop(overlay_id, None)
        self._adapters.pop(overlay_id, None)
        if record is not None:
            self.evictions.append(record)
        return record

    def evict_due(self, tick: int) -> tuple[OverlayRecord, ...]:
        due = [
            record
            for record in self._records.values()
            if record.status == "stale"
            and record.stale_tick is not None
            and tick - record.stale_tick >= self.thresholds.stale_after
        ]
        return tuple(record for record in (self.evict(r.overlay_id) for r in due) if record)

    # ---------------------------------------------------------------- reads
    def record(self, overlay_id: str) -> OverlayRecord | None:
        return self._records.get(overlay_id)

    def record_for_motif(self, motif_id: str) -> OverlayRecord | None:
        for record in self._records.values():
            if record.motif_id == motif_id:
                return record
        return None

    def adapter(self, overlay_id: str) -> PredictorOverlay | None:
        """The stored patch for ``overlay_id`` regardless of status (Q1-A).

        The D2 store holds the patch; this accessor is how the one-way
        post-commit hook hands it to the store.  ``resolve`` remains the
        prediction path (active only).
        """
        return self._adapters.get(overlay_id)

    def all_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def reset(self) -> None:
        """Clear records, adapters and lifecycle event lists (fresh P6)."""
        self._records.clear()
        self._adapters.clear()
        self.births.clear()
        self.revalidations.clear()
        self.stales.clear()
        self.evictions.clear()

    def prune_to(self, resident_ids: "set[str] | frozenset[str]") -> tuple[OverlayRecord, ...]:
        """Evict records/patches no longer resident in D2 (Q1-A).

        Used by the executor's one-way post-commit prune so the organ keeps
        no overlay-side retention after a store eviction.
        """
        removed = [
            record
            for record in self._records.values()
            if record.overlay_id not in resident_ids
        ]
        for record in removed:
            self.evict(record.overlay_id)
        return tuple(removed)

    def resolve(self, overlay_id: str) -> PredictorOverlay | None:
        """The adapter, but only while the record is **active**."""
        record = self._records.get(overlay_id)
        if record is None or record.status != "active":
            return None
        return self._adapters.get(overlay_id)

    def active(self) -> tuple[OverlayRecord, ...]:
        return tuple(record for record in self._records.values() if record.status == "active")

    def records(self) -> tuple[tuple[int, str, str], ...]:
        """The D2 payload triples, insertion order."""
        return tuple(record.token() for record in self._records.values())

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True)
class RatchetResult:
    """P6 output for one finalised tick."""

    tick: int
    births: tuple[OverlayRecord, ...]
    revalidations: tuple[OverlayRecord, ...]
    stales: tuple[OverlayRecord, ...]
    evictions: tuple[OverlayRecord, ...]
    applied_overlay_id: str | None
    cooldown_remaining: int
    d4_writes: int
    records: tuple[tuple[int, str, str], ...]

    @property
    def empty(self) -> bool:
        return not (
            self.births or self.revalidations or self.stales or self.evictions
        )


# --------------------------------------------------------------------------
# D4 stand-in (Base Core): the acceptance gate is that it gets zero writes
# --------------------------------------------------------------------------
class D4StandIn:
    """Stand-in for D4 Base Core; counts every attempted write.

    The physics organ never calls :meth:`write` -- the acceptance scenario
    asserts ``writes == 0``.  The ledger's #22 sediment edge is out of scope
    for this slice; overlays live on D2 / in the side table.
    """

    def __init__(self) -> None:
        self.writes = 0
        self.payloads: list[Any] = []

    def write(self, payload: Any) -> None:  # pragma: no cover - must stay unused
        self.writes += 1
        self.payloads.append(payload)


# --------------------------------------------------------------------------
# Per-stream state
# --------------------------------------------------------------------------
class _StreamState:
    __slots__ = (
        "env",
        "frame_size",
        "tick",
        "hidden",
        "prev_latent",
        "window",
        "latents",
        "actions",
        "cache_start",
        "seed",
    )

    def __init__(self, frame_size: int, seed: int | None) -> None:
        self.env = PhysicsEnv(WorldConfig(frame_size=frame_size))
        self.frame_size = frame_size
        self.tick = 0
        self.hidden: torch.Tensor | None = None
        self.prev_latent: torch.Tensor | None = None
        self.window: deque[np.ndarray] = deque(maxlen=WINDOW_FRAMES)
        self.latents: deque[np.ndarray] = deque()
        self.actions: deque[int] = deque()
        self.cache_start = 0
        self.seed = seed

    def trim_cache(self, cache_len: int) -> None:
        while len(self.actions) > cache_len:
            self.actions.popleft()
            self.latents.popleft()
            self.cache_start += 1


# --------------------------------------------------------------------------
# The organ
# --------------------------------------------------------------------------
class PhysicsOrgan:
    """E1 + frozen-encoder micro-LeWM + P5 digest + P6 ratchet.

    Public API (the executor wiring surface)::

        organ = PhysicsOrgan(checkpoint_path)
        obs = organ.reset(seed=7)                 # E1 reset, first frame
        obs = organ.step(action)                  # E1 step + observation
        digest = organ.digest(obs)                # P5
        ratchet = organ.ratchet(digest)           # P6
        # or, in one call: obs, digest, ratchet = organ.tick(action)

    The same methods take ``stream=<int>`` for a bank of independent
    streams; digest and ratchet state are shared (one organ = one world).
    """

    def __init__(
        self,
        model: MicroLeWM | str | Path | None = None,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        stats_path: str | Path = DEFAULT_STATS,
        thresholds: DigestThresholds | None = None,
        digester: ResidualDigester | None = None,
        overlay_table: OverlayTable | None = None,
        d4: D4StandIn | None = None,
        fit_streams: Iterable[int] | None = None,
        cache_len: int = DEFAULT_CACHE_LEN,
        patch: str = DEFAULT_PATCH,
        fit_steps: int = DEFAULT_FIT_STEPS,
        fit_lr: float = DEFAULT_FIT_LR,
        fit_seed: int = 0,
        store_frames: bool = False,
    ) -> None:
        if model is None:
            model = MicroLeWM.load(checkpoint_path)
        elif isinstance(model, (str, Path)):
            model = MicroLeWM.load(model)
        if not isinstance(model, MicroLeWM):
            raise TypeError("model must be a MicroLeWM or a checkpoint path")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.base_model = model
        self.working_model = copy.deepcopy(model)
        self.working_model.eval()
        self._snapshot_encoder()

        if thresholds is None:
            try:
                thresholds = derive_thresholds(Phase3Stats.from_stats(stats_path))
            except (OSError, KeyError, ValueError):
                thresholds = _default_thresholds()
        self.thresholds = thresholds
        if patch not in ("head", "full"):
            raise ValueError("patch must be 'head' or 'full'")
        self.patch = patch
        self.fit_steps = int(fit_steps)
        self.fit_lr = float(fit_lr)
        self.fit_seed = int(fit_seed)
        self.cache_len = int(cache_len)
        self.store_frames = bool(store_frames)

        self.digester = digester or ResidualDigester(thresholds)
        self.overlay_table = overlay_table or OverlayTable(thresholds)
        self.d4 = d4 or D4StandIn()
        self.fit_streams: set[int] | None = (
            None if fit_streams is None else {int(s) for s in fit_streams}
        )

        self._streams: dict[int, _StreamState] = {}
        self._active_overlay_id: str | None = None
        self._active_adapter: PredictorOverlay | None = None
        self._overlay_serial = 0
        self._cooldown_until = -1
        self._stale_run = 0
        self._decided_tick = -1
        self._pending: dict[int, list[DigestResult]] = {}
        #: Published #22 gate verdict (adapted_mean, base_mean), or None.
        self._gate_evidence: tuple[float, float] | None = None
        self._gate_adapted_sum = 0.0
        self._gate_base_sum = 0.0
        self._gate_evidence_ticks = 0
        self.history: list[RatchetResult] = []
        self.observations: list[Observation] = []
        self.fit_history: list[FitRecord] = []

    # ------------------------------------------------------------ internals
    def _snapshot_encoder(self) -> None:
        self._encoder_snapshot = [
            p.detach().clone() for p in self.base_model.encoder_parameters()
        ]

    def encoder_bit_identical(self) -> bool:
        """True iff every base-encoder parameter equals its init snapshot."""
        current = [p.detach() for p in self.base_model.encoder_parameters()]
        if len(current) != len(self._encoder_snapshot):
            return False
        return all(torch.equal(now, then) for now, then in zip(current, self._encoder_snapshot))

    def _state(self, stream: int) -> _StreamState:
        state = self._streams.get(int(stream))
        if state is None:
            state = _StreamState(self.base_model.config.frame_size, seed=None)
            self._streams[int(stream)] = state
        return state

    def _encode(self, frame: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.base_model.encode(np.asarray(frame)[None]).reshape(-1)

    def _rollout(self, state: _StreamState, latent: torch.Tensor, action: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        current = latent
        hidden = state.hidden
        with torch.no_grad():
            for _ in range(self.thresholds.rollout_horizon):
                current, hidden = self.working_model.predict_step(current, action, hidden)
                rows.append(current.detach().cpu().numpy().reshape(-1).astype(np.float64))
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------ E1 surface
    def begin_stream(
        self,
        frame: np.ndarray,
        *,
        stream: int = 0,
        seed: int | None = None,
        ingest: bool = True,
    ) -> Observation:
        """Start (or restart) a stream from a reset frame; no prediction yet."""
        state = self._state(stream)
        state.tick = 0
        state.hidden = None
        state.prev_latent = None
        state.window.clear()
        state.latents.clear()
        state.actions.clear()
        state.cache_start = 0
        state.seed = seed
        return self._observe(frame, None, stream=stream, ingest=ingest)

    def reset(
        self,
        seed: int | None = None,
        *,
        gravity_sign: int | None = None,
        stream: int = 0,
        ingest: bool = True,
    ) -> Observation:
        """Reset the owned environment for ``stream`` and observe frame 0."""
        state = self._state(stream)
        frame = state.env.reset(seed=seed, gravity_sign=gravity_sign)
        return self.begin_stream(frame, stream=stream, seed=seed, ingest=ingest)

    def step(self, action: int, *, stream: int = 0, ingest: bool = True) -> Observation:
        """Advance the owned environment one tick and observe the new frame."""
        state = self._state(stream)
        frame = state.env.step(action)
        return self._observe(frame, int(action), stream=stream, ingest=ingest)

    def flip_gravity(self, *, stream: int | None = 0) -> int | dict[int, int]:
        """Flip gravity on one owned environment (default stream 0).

        ``stream=None`` flips every owned environment and returns the new
        signs keyed by stream; a single stream returns the new sign.
        """
        if stream is None:
            return {key: state.env.flip_gravity() for key, state in self._streams.items()}
        return self._state(stream).env.flip_gravity()

    # ---------------------------------------------------------- observation
    def observe(
        self,
        frame: np.ndarray,
        action: int | None,
        *,
        stream: int = 0,
        ingest: bool = True,
    ) -> Observation:
        """Observe the next frame of a stream (public, executor entry point)."""
        return self._observe(frame, action, stream=stream, ingest=ingest)

    def _observe(
        self,
        frame: np.ndarray,
        action: int | None,
        *,
        stream: int,
        ingest: bool,
    ) -> Observation:
        state = self._state(stream)
        frame = np.asarray(frame)
        if frame.shape != (state.frame_size, state.frame_size, 3):
            raise ValueError(
                f"frame must be ({state.frame_size},{state.frame_size},3), got {frame.shape}"
            )
        latent = self._encode(frame)
        surprise = float("nan")
        base_surprise = float("nan")
        residual = np.zeros(latent.numel(), dtype=np.float64)
        overlay_id = self._active_overlay_id
        if state.prev_latent is not None:
            if action is None:
                raise ValueError("action is required after the first frame")
            with torch.no_grad():
                prediction, hidden = self.working_model.predict_step(
                    state.prev_latent, int(action), state.hidden
                )
                if self._active_adapter is None:
                    base_prediction = prediction
                else:
                    # Head-only patches leave the GRU untouched, so the base
                    # predictor's own step uses the same hidden state.
                    base_prediction, _ = self.base_model.predict_step(
                        state.prev_latent, int(action), state.hidden
                    )
            state.hidden = hidden
            error = prediction - latent
            residual = error.detach().cpu().numpy().reshape(-1).astype(np.float64)
            surprise = float(np.mean(residual * residual))
            base_error = base_prediction - latent
            base_surprise = float(
                torch.mean(base_error * base_error).detach()
            )
        elif self._active_adapter is None:
            base_surprise = surprise
        if state.prev_latent is not None:
            delta = latent - state.prev_latent
        else:
            delta = torch.zeros_like(latent)
        state.prev_latent = latent
        latent_np = latent.detach().cpu().numpy().astype(np.float64)
        state.window.append(latent_np)
        state.latents.append(latent_np)
        if action is not None:
            state.actions.append(int(action))
        state.trim_cache(self.cache_len)

        rollout = self._rollout(state, latent, 0 if action is None else int(action))
        observation = Observation(
            tick=state.tick,
            stream=int(stream),
            action=None if action is None else int(action),
            latent=latent_np,
            latent_delta=delta.detach().cpu().numpy().astype(np.float64),
            surprise=surprise,
            residual=residual,
            rollout=rollout,
            window=np.stack(list(state.window), axis=0),
            overlay_id=overlay_id,
            base_surprise=base_surprise,
            ingest=bool(ingest),
            frame=frame.copy() if self.store_frames else None,
        )
        state.tick += 1
        self.observations.append(observation)
        return observation

    # ---------------------------------------------------------------- P5
    def digest(self, observation: Observation) -> DigestResult:
        """Run P5 on one observation."""
        return self.digester.observe(observation)

    # ---------------------------------------------------------------- P6
    def ratchet(self, digest: DigestResult) -> RatchetResult:
        """Run P6 on one digest result.

        Per-stream digests are accumulated by tick; a tick's lifecycle
        decision is finalised when the next tick's first digest arrives
        (or on :meth:`flush`).  The returned result is the finalised tick,
        or an empty result when no tick was finalised.
        """
        tick = digest.tick
        self._pending.setdefault(tick, []).append(digest)
        births: list[OverlayRecord] = []
        revalidations: list[OverlayRecord] = []
        stales: list[OverlayRecord] = []
        evictions: list[OverlayRecord] = []
        while self._decided_tick + 1 < tick:
            target = self._decided_tick + 1
            decisions = self._pending.pop(target, [])
            b, r, s, e = self._decide_tick(target, decisions)
            births.extend(b)
            revalidations.extend(r)
            stales.extend(s)
            evictions.extend(e)
            self._decided_tick = target
        if not (births or revalidations or stales or evictions):
            return RatchetResult(
                tick=tick,
                births=(),
                revalidations=(),
                stales=(),
                evictions=(),
                applied_overlay_id=self._active_overlay_id,
                cooldown_remaining=max(0, self._cooldown_until - tick),
                d4_writes=self.d4.writes,
                records=self.overlay_table.records(),
            )
        result = RatchetResult(
            tick=self._decided_tick,
            births=tuple(births),
            revalidations=tuple(revalidations),
            stales=tuple(stales),
            evictions=tuple(evictions),
            applied_overlay_id=self._active_overlay_id,
            cooldown_remaining=max(0, self._cooldown_until - self._decided_tick),
            d4_writes=self.d4.writes,
            records=self.overlay_table.records(),
        )
        self.history.append(result)
        return result

    def flush(self) -> tuple[RatchetResult, ...]:
        """Finalise every pending tick (call at the end of a run)."""
        results: list[RatchetResult] = []
        for target in sorted(self._pending):
            if target <= self._decided_tick:
                continue
            decisions = self._pending.pop(target, [])
            b, r, s, e = self._decide_tick(target, decisions)
            if b or r or s or e:
                result = RatchetResult(
                    tick=target,
                    births=tuple(b),
                    revalidations=tuple(r),
                    stales=tuple(s),
                    evictions=tuple(e),
                    applied_overlay_id=self._active_overlay_id,
                    cooldown_remaining=max(0, self._cooldown_until - target),
                    d4_writes=self.d4.writes,
                    records=self.overlay_table.records(),
                )
                self.history.append(result)
                results.append(result)
            self._decided_tick = target
        return tuple(results)

    def _decide_tick(
        self, tick: int, digests: Sequence[DigestResult]
    ) -> tuple[
        list[OverlayRecord], list[OverlayRecord], list[OverlayRecord], list[OverlayRecord]
    ]:
        births: list[OverlayRecord] = []
        revalidations: list[OverlayRecord] = []
        stales: list[OverlayRecord] = []
        if not digests:
            return births, revalidations, stales, []
        surprises = [
            result.observation.surprise
            for result in digests
            if np.isfinite(result.observation.surprise)
        ]
        mean_surprise = float(np.mean(surprises)) if surprises else float("nan")
        active_id = self._active_overlay_id
        if active_id is not None:
            base_surprises = [
                result.observation.base_surprise
                for result in digests
                if np.isfinite(result.observation.base_surprise)
            ]
            mean_base = float(np.mean(base_surprises)) if base_surprises else float("nan")
            # Gate evidence counts only ticks in which the patch was actually
            # applied to the prediction (``observation.overlay_id``); the
            # adoption tick's observation predates the patch and carries no
            # usefulness verdict.
            applied = [
                result.observation
                for result in digests
                if result.observation.overlay_id == active_id
                and np.isfinite(result.observation.surprise)
                and np.isfinite(result.observation.base_surprise)
            ]
            if applied:
                self._accrue_gate_evidence(
                    float(np.mean([observation.surprise for observation in applied])),
                    float(np.mean([observation.base_surprise for observation in applied])),
                )
            # Stale when the patch stops improving on what it has to correct:
            # the base-predictor residual is smaller than the patched one.
            hurts = (
                np.isfinite(mean_surprise)
                and np.isfinite(mean_base)
                and mean_surprise > mean_base
            )
            if hurts:
                self._stale_run += 1
            else:
                self._stale_run = 0
            if self._stale_run >= self.thresholds.stale_patience:
                stales.append(self.overlay_table.mark_stale(active_id, tick))
                self._set_active_overlay(None)
                self._cooldown_until = tick + self.thresholds.cooldown_ticks
                self._stale_run = 0
            elif (
                np.isfinite(mean_surprise)
                and mean_surprise <= self.thresholds.debris_threshold
            ):
                revalidations.append(self.overlay_table.revalidate(active_id, tick))
        else:
            motif = self._best_motif(digests)
            if motif is not None and tick >= self._cooldown_until:
                record = self._adopt_or_revalidate(motif, tick)
                if record is not None:
                    if record.birth_tick == tick and record.epoch == 1:
                        births.append(record)
                    else:
                        revalidations.append(record)
        evictions = list(self.overlay_table.evict_due(tick))
        return births, revalidations, stales, evictions

    @staticmethod
    def _best_motif(digests: Sequence[DigestResult]) -> StructuralMotif | None:
        motifs = [result.motif for result in digests if result.motif is not None]
        if not motifs:
            return None
        return max(motifs, key=lambda motif: (motif.persistence, motif.count))

    def _adopt_or_revalidate(self, motif: StructuralMotif, tick: int) -> OverlayRecord | None:
        existing = self.overlay_table.record_for_motif(motif.motif_id)
        if existing is not None:
            record = self.overlay_table.revalidate(existing.overlay_id, tick)
            self._set_active_overlay(existing.overlay_id)
            return record
        overlay = self._fit_overlay(motif, tick)
        if overlay is None:
            return None
        record = self.overlay_table.birth(overlay, motif_id=motif.motif_id, tick=tick)
        self._set_active_overlay(record.overlay_id)
        return record

    # ------------------------------------------------------ executor surface
    def live_overlay_ids(self) -> frozenset[str]:
        """The organ-side overlay ids (used by the executor prune)."""
        return self.overlay_table.all_ids()

    def prune_overlays(self, resident_ids: "set[str] | frozenset[str]") -> tuple[OverlayRecord, ...]:
        """Drop organ-side records/patches that D2 no longer holds (Q1-A)."""
        removed = self.overlay_table.prune_to(resident_ids)
        if self._active_overlay_id is not None and self._active_overlay_id not in resident_ids:
            self._set_active_overlay(None)
        return removed

    def ratchet_gate_passed(self) -> tuple[bool, float]:
        """The real #22 sediment gate, derived from stored P6 evidence (Q4).

        The organ records adapted-vs-base surprise evidence on every
        finalized tick in which the active overlay was actually applied to
        the prediction.  The production recorder
        (:meth:`_accrue_gate_evidence`) publishes cumulative means only
        after ``stale_patience`` applied ticks -- the same patience the
        staleness decision uses -- so one transient tick cannot sediment an
        invariant.

        The gate passes only when a verdict exists, an overlay is active,
        and the cumulative adapted mean is strictly below the base mean; the
        confidence is ``clamp(1 - adapted/base, 0, 1)``.  No verdict, no
        active overlay, non-finite values, ``base <= 0``, or ``adapted >=
        base`` all return ``(False, 0.0)``, and the source emits the
        zero-confidence ``ratchet-gate-blocked`` body.
        """
        if self._gate_evidence is None or not self.overlay_table.active():
            return False, 0.0
        adapted, base = self._gate_evidence
        if not (np.isfinite(adapted) and np.isfinite(base)) or base <= 0.0:
            return False, 0.0
        if adapted >= base:
            return False, 0.0
        return True, float(min(1.0, max(0.0, 1.0 - adapted / base)))

    def _accrue_gate_evidence(self, adapted: float, base: float) -> None:
        """Accumulate one applied-tick usefulness measurement (P6).

        Publishes cumulative ``(adapted_mean, base_mean)`` once at least
        ``stale_patience`` applied ticks have been recorded.  Non-finite or
        degenerate measurements (``base <= 0``) carry no verdict and are
        ignored rather than poisoning the evidence.
        """
        if not (np.isfinite(adapted) and np.isfinite(base)) or base <= 0.0:
            return
        self._gate_adapted_sum += float(adapted)
        self._gate_base_sum += float(base)
        self._gate_evidence_ticks += 1
        if self._gate_evidence_ticks >= self.thresholds.stale_patience:
            ticks = float(self._gate_evidence_ticks)
            self._gate_evidence = (
                self._gate_adapted_sum / ticks,
                self._gate_base_sum / ticks,
            )

    def _record_gate_evidence(self, adapted: float, base: float) -> None:
        """Publish a completed adapted-vs-base verdict directly.

        Evidence-injection hook: the production path accrues per-tick
        evidence and publishes after ``stale_patience`` applied ticks; this
        method publishes a supplied verdict so the gate's passing branch can
        be exercised without a successful adaptation.  This slice has none
        -- the registered acceptance's cumulative evidence is harmful -- so
        the passing branch is only reachable through this hook.
        """
        adapted = float(adapted)
        base = float(base)
        if not (np.isfinite(adapted) and np.isfinite(base)) or base <= 0.0:
            raise ValueError("gate evidence must be finite with base > 0")
        self._gate_adapted_sum = adapted
        self._gate_base_sum = base
        self._gate_evidence_ticks = max(1, int(self.thresholds.stale_patience))
        self._gate_evidence = (adapted, base)

    def reset_p6_state(self) -> None:
        """Clear all P6 decision state, lifecycle table and fit artifacts.

        Called by ``PhysicsRun.__init__`` and ``PhysicsRun.reset`` so a
        second run on the same organ is a fresh experiment: no pending
        decisions, cooldown, stale run, gate evidence, overlay
        records/patches or fit history survive.  E1 (environment/streams)
        and the frozen base model are untouched; the working model is
        restored to the base copy.
        """
        self._active_overlay_id = None
        self._active_adapter = None
        self._overlay_serial = 0
        self._cooldown_until = -1
        self._stale_run = 0
        self._decided_tick = -1
        self._pending = {}
        self._gate_evidence = None
        self._gate_adapted_sum = 0.0
        self._gate_base_sum = 0.0
        self._gate_evidence_ticks = 0
        self.overlay_table.reset()
        self.history.clear()
        self.observations.clear()
        self.fit_history.clear()
        self.working_model = copy.deepcopy(self.base_model)
        self.working_model.eval()

    def adopt_overlay_from_buffer(
        self,
        tick: int,
        *,
        fit_window: int = 16,
        motif_id: str = "single-stream",
    ) -> OverlayRecord | None:
        """Single-stream adaptation: fit the predictor patch on the last
        ``fit_window`` real transitions of the same stream and install it.

        This is the pre-registered bounded-replay policy for the executor
        acceptance: same-stream data only, no cross-stream bank data, no
        post-hoc fit-variant selection.  It reuses the ratchet's fitting
        path (``_fit_overlay``) with a synthetic motif whose window is the
        replay buffer.  Returns the birth record, or ``None`` when the fit
        cannot improve its own window (the caller must report that
        honestly, not retry with another variant).
        """
        if fit_window < 2:
            raise ValueError("fit_window must be >= 2")
        first_tick = max(1, int(tick) - int(fit_window) + 1)
        if first_tick >= tick:
            return None
        motif = StructuralMotif(
            motif_id=motif_id,
            debris_id="debris-replay",
            centroid=np.zeros(self.base_model.config.latent_dim, dtype=np.float64),
            count=int(fit_window),
            spread=0.0,
            persistence=int(fit_window),
            drift=0.0,
            first_tick=int(first_tick),
            last_tick=int(tick) - 1,
        )
        overlay = self._fit_overlay(motif, int(tick))
        if overlay is None:
            return None
        record = self.overlay_table.birth(overlay, motif_id=motif.motif_id, tick=int(tick))
        self._set_active_overlay(record.overlay_id)
        self._stale_run = 0
        return record

    # --------------------------------------------------------- overlay apply
    def _set_active_overlay(self, overlay_id: str | None) -> None:
        adapter = None if overlay_id is None else self.overlay_table.resolve(overlay_id)
        if overlay_id is not None and adapter is None:
            overlay_id = None
        if overlay_id != self._active_overlay_id:
            # A different overlay (or none) means the previous evidence
            # verdict belongs to another patch and must not be reused.
            self._gate_evidence = None
            self._gate_adapted_sum = 0.0
            self._gate_base_sum = 0.0
            self._gate_evidence_ticks = 0
        previous = self._active_adapter
        if overlay_id == self._active_overlay_id and adapter is previous:
            return
        self._active_overlay_id = overlay_id
        self._active_adapter = adapter
        self.working_model = copy.deepcopy(self.base_model)
        self.working_model.eval()
        if adapter is not None:
            adapter.apply_to(self.working_model)
        needs_replay = (adapter is not None and adapter.changes_gru) or (
            previous is not None and previous.changes_gru
        )
        if needs_replay:
            self._reseed_hidden()

    def _reseed_hidden(self) -> None:
        """Replay each stream's causal cache through the working predictor."""
        for state in self._streams.values():
            if not state.actions:
                state.hidden = None
                continue
            latents = torch.as_tensor(np.stack(list(state.latents), axis=0), dtype=torch.float32)
            actions = torch.as_tensor(list(state.actions), dtype=torch.long)
            with torch.no_grad():
                embedding = self.working_model.predictor.act_embed(actions)
                joined = torch.cat([latents[:-1], embedding], dim=-1).unsqueeze(0)
                _, hidden = self.working_model.predictor.gru(joined)
            state.hidden = hidden

    # ----------------------------------------------------------- overlay fit
    def fit_overlay(self, motif: StructuralMotif, tick: int) -> PredictorOverlay | None:
        """Public fit entry point (the ratchet calls :meth:`_fit_overlay`).

        Exposed so an acceptance probe can fit against a synthetic motif
        built from the organ's real cached observations.
        """
        return self._fit_overlay(motif, tick)

    def _fit_overlay(self, motif: StructuralMotif, tick: int) -> PredictorOverlay | None:
        """Fit a predictor-only patch on the real cached post-motif residuals.

        Fits on a deep copy of the base model; the base encoder and predictor
        are never mutated.  Returns ``None`` when there is not enough data or
        the loss is not finite / did not improve.
        """
        lo = max(0, motif.first_tick - 1)
        sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
        for stream in self._fit_stream_ids():
            state = self._streams.get(stream)
            if state is None or not state.actions:
                continue
            if state.cache_start > lo:
                # The causal cache scrolled past the fit window; skip rather
                # than silently fit a truncated prefix.
                continue
            z = np.stack(list(state.latents), axis=0)
            a = np.asarray(list(state.actions), dtype=np.int64)
            end = min(len(a), tick)
            if end <= lo:
                continue
            sequences.append(
                (
                    torch.as_tensor(z[: end + 1], dtype=torch.float32),
                    torch.as_tensor(a[:end], dtype=torch.long),
                )
            )
        if not sequences:
            return None
        common = min(int(item[1].shape[0]) for item in sequences)
        if common <= lo:
            return None
        z_batch = torch.stack([item[0][: common + 1] for item in sequences], dim=0)
        a_batch = torch.stack([item[1][:common] for item in sequences], dim=0)
        foreground = slice(lo, common)

        model = copy.deepcopy(self.base_model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if self.patch == "head":
            parameters = list(model.predictor.head.parameters()) + list(
                model.predictor.norm.parameters()
            )
        else:
            parameters = list(model.predictor.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(parameters, lr=self.fit_lr, weight_decay=1e-4)
        torch.manual_seed(self.fit_seed)
        with torch.no_grad():
            before = model.predict(z_batch, a_batch)
            loss_before = float(
                (before[:, foreground] - z_batch[:, 1:][:, foreground]).pow(2).mean()
            )
        for _ in range(self.fit_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.predict(z_batch, a_batch)
            loss = (prediction[:, foreground] - z_batch[:, 1:][:, foreground]).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        with torch.no_grad():
            after = model.predict(z_batch, a_batch)
            loss_after = float(
                (after[:, foreground] - z_batch[:, 1:][:, foreground]).pow(2).mean()
            )
        if not np.isfinite(loss_before) or not np.isfinite(loss_after) or loss_after >= loss_before:
            return None
        trainable_names = {
            name for name, parameter in model.predictor.named_parameters() if parameter.requires_grad
        }
        base_parameters = dict(self.base_model.predictor.named_parameters())
        deltas = {
            name: (parameter.detach().clone() - base_parameters[name].detach())
            for name, parameter in model.predictor.named_parameters()
            if name in trainable_names
        }
        if not deltas:
            return None
        self._overlay_serial += 1
        overlay = PredictorOverlay(
            overlay_id=f"ovr-{self._overlay_serial:02d}",
            deltas=deltas,
            param_count=int(sum(delta.numel() for delta in deltas.values())),
            fit_transitions=int((common - lo) * len(sequences)),
            fit_streams=len(sequences),
            loss_before=loss_before,
            loss_after=loss_after,
            patch=self.patch,
        )
        self.fit_history.append(
            FitRecord(
                overlay_id=overlay.overlay_id,
                param_count=overlay.param_count,
                nbytes=overlay.nbytes,
                fit_transitions=overlay.fit_transitions,
                fit_streams=overlay.fit_streams,
                loss_before=overlay.loss_before,
                loss_after=overlay.loss_after,
                patch=overlay.patch,
            )
        )
        return overlay

    def _fit_stream_ids(self) -> list[int]:
        if self.fit_streams is not None:
            return sorted(stream for stream in self.fit_streams if stream in self._streams)
        return sorted(self._streams)

    # -------------------------------------------------------------- accept
    def acceptance_scenario(
        self,
        *,
        seed: int = 31,
        n_fit_streams: int = 24,
        n_heldout_streams: int = 8,
        horizon: int = 36,
        flip_at: int = 8,
        flip_back_at: int = 28,
        episode_offset: int = 0,
    ) -> "AcceptanceTrace":
        """Run the phase-5 acceptance probe and return its numbered trace.

        A **bank** of paired streams (the phase-3 evaluation seeds/actions)
        runs normal -> gravity flip -> ... -> gravity flip back.  The first
        ``n_fit_streams`` streams feed the digest and the overlay fit; the
        last ``n_heldout_streams`` are evaluation-only (their residuals are
        never ingested).  This is a shaped probe: the executor sees one
        stream, and the measured single-stream overlay does *not* generalise
        forward (see the module docstring).
        """
        total = int(n_fit_streams) + int(n_heldout_streams)
        env = PhysicsEnv(WorldConfig(frame_size=self.base_model.config.frame_size))
        frames, actions = _three_phase_rollouts(
            env,
            total,
            int(horizon),
            seed=int(seed),
            flip_at=int(flip_at),
            flip_back_at=int(flip_back_at),
            episode_offset=int(episode_offset),
        )
        normal_frames, normal_actions = _normal_rollouts(
            env,
            total,
            int(horizon),
            seed=int(seed),
            episode_offset=int(episode_offset),
        )
        fit_ids = set(range(int(n_fit_streams)))
        heldout_ids = list(range(int(n_fit_streams), total))
        self.fit_streams = fit_ids
        self._streams.clear()
        self._pending.clear()
        self._decided_tick = -1
        self._active_overlay_id = None
        self._active_adapter = None
        self._cooldown_until = -1
        self._stale_run = 0
        self.history.clear()
        self.observations.clear()
        self.working_model = copy.deepcopy(self.base_model)

        events: list[RatchetResult] = []
        ticks: list[AcceptanceTick] = []
        for stream in range(total):
            self.begin_stream(frames[stream, 0], stream=stream, seed=int(seed) + stream)
        for index in range(int(horizon)):
            tick = index + 1
            for stream in range(total):
                observation = self.observe(
                    frames[stream, tick],
                    int(actions[stream, index]),
                    stream=stream,
                    ingest=stream in fit_ids,
                )
                digest = self.digest(observation)
                result = self.ratchet(digest)
                if not result.empty:
                    events.append(result)
            window = self.observations[-total:]
            ticks.append(
                AcceptanceTick(
                    tick=tick,
                    phase=_phase(tick, int(flip_at), int(flip_back_at)),
                    mean_surprise=float(np.mean([o.surprise for o in window])),
                    fit_surprise=float(
                        np.mean([o.surprise for o in window[: int(n_fit_streams)]])
                    ),
                    heldout_surprise=float(
                        np.mean([o.surprise for o in window[int(n_fit_streams) :]])
                    ),
                    active_overlay_id=self._active_overlay_id,
                )
            )
        events.extend(self.flush())

        # Counterfactual base-predictor surprise on the same held-out frames.
        with torch.no_grad():
            z_held = self.base_model.encode(frames_tensor_from(frames[heldout_ids]))
            a_held = torch.from_numpy(actions[heldout_ids]).long()
            base_held = (
                (self.base_model.predict(z_held, a_held) - z_held[:, 1:])
                .pow(2)
                .mean(dim=-1)
                .cpu()
                .numpy()
            )
            z_normal = self.base_model.encode(frames_tensor_from(normal_frames[heldout_ids]))
            a_normal = torch.from_numpy(normal_actions[heldout_ids]).long()
            normal_held = (
                (self.base_model.predict(z_normal, a_normal) - z_normal[:, 1:])
                .pow(2)
                .mean(dim=-1)
                .cpu()
                .numpy()
            )
        return AcceptanceTrace(
            tick_rows=tuple(ticks),
            events=tuple(events),
            base_heldout=base_held,
            normal_heldout=normal_held,
            flip_at=int(flip_at),
            flip_back_at=int(flip_back_at),
            n_fit_streams=int(n_fit_streams),
            n_heldout_streams=int(n_heldout_streams),
            d4_writes=int(self.d4.writes),
            encoder_bit_identical=self.encoder_bit_identical(),
            thresholds=self.thresholds,
        )


# --------------------------------------------------------------------------
# Rollout helpers (mirror amateur.physics.data's seed/action streams)
# --------------------------------------------------------------------------
def _phase(tick: int, flip_at: int, flip_back_at: int | None) -> str:
    """Scene phase of a 1-based frame tick."""
    if tick <= flip_at:
        return "normal"
    if flip_back_at is None or tick <= flip_back_at:
        return "flipped"
    return "recovered"


def frames_tensor_from(frames: np.ndarray) -> torch.Tensor:
    """``(N, T, H, W, 3)`` uint8 numpy -> ``(N, T, 3, H, W)`` torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 1, 4, 2, 3)


def _normal_rollouts(
    env: PhysicsEnv,
    n_episodes: int,
    horizon: int,
    *,
    seed: int,
    episode_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The paired no-flip twin: identical seeds and action streams."""
    return _rollouts(env, n_episodes, horizon, seed, episode_offset=episode_offset)


def _three_phase_rollouts(
    env: PhysicsEnv,
    n_episodes: int,
    horizon: int,
    *,
    seed: int,
    flip_at: int,
    flip_back_at: int,
    episode_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out ``normal -> flipped -> normal`` episodes (paired streams)."""
    if not 0 <= flip_at < flip_back_at < horizon:
        raise ValueError("need 0 <= flip_at < flip_back_at < horizon")
    size = env.config.frame_size
    frames = np.empty((n_episodes, horizon + 1, size, size, 3), dtype=np.uint8)
    actions = np.empty((n_episodes, horizon), dtype=np.int64)
    for episode in range(n_episodes):
        episode_seed = _episode_seed(seed, episode + episode_offset)
        rng = _action_rng(seed, episode + episode_offset)
        frames[episode, 0] = env.reset(seed=episode_seed, gravity_sign=1)
        for t in range(horizon):
            if t == flip_at:
                env.flip_gravity()
            if t == flip_back_at:
                env.flip_gravity()
            action = int(rng.integers(0, env.n_actions))
            actions[episode, t] = action
            frames[episode, t + 1] = env.step(action)
    return frames, actions


def _rollouts(
    env: PhysicsEnv,
    n_episodes: int,
    horizon: int,
    seed: int,
    *,
    episode_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    size = env.config.frame_size
    frames = np.empty((n_episodes, horizon + 1, size, size, 3), dtype=np.uint8)
    actions = np.empty((n_episodes, horizon), dtype=np.int64)
    for episode in range(n_episodes):
        episode_seed = _episode_seed(seed, episode + episode_offset)
        rng = _action_rng(seed, episode + episode_offset)
        frames[episode, 0] = env.reset(seed=episode_seed, gravity_sign=1)
        for t in range(horizon):
            action = int(rng.integers(0, env.n_actions))
            actions[episode, t] = action
            frames[episode, t + 1] = env.step(action)
    return frames, actions


# --------------------------------------------------------------------------
# Acceptance trace
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AcceptanceTick:
    """One scene tick of the acceptance probe (per-stream means)."""

    tick: int
    phase: str
    mean_surprise: float
    fit_surprise: float
    heldout_surprise: float
    active_overlay_id: str | None


@dataclass(frozen=True)
class AcceptanceTrace:
    """The numbered acceptance record; every number is measured."""

    tick_rows: tuple[AcceptanceTick, ...]
    events: tuple[RatchetResult, ...]
    base_heldout: np.ndarray
    normal_heldout: np.ndarray
    flip_at: int
    flip_back_at: int
    n_fit_streams: int
    n_heldout_streams: int
    d4_writes: int
    encoder_bit_identical: bool
    thresholds: DigestThresholds

    # --------------------------------------------------------------- helpers
    def event(self, kind: str) -> list[RatchetResult]:
        attr = {
            "birth": "births",
            "revalidate": "revalidations",
            "stale": "stales",
            "evict": "evictions",
        }[kind]
        return [item for item in self.events if getattr(item, attr)]

    @property
    def adoption_tick(self) -> int | None:
        births = [item for item in self.events if item.births]
        return births[0].tick if births else None

    @property
    def stale_tick(self) -> int | None:
        stales = [item for item in self.events if item.stales]
        return stales[0].tick if stales else None

    @property
    def eviction_tick(self) -> int | None:
        evictions = [item for item in self.events if item.evictions]
        return evictions[0].tick if evictions else None

    def transition_slice(self, start: int, stop: int) -> slice:
        """Transition-index slice; ``transition i`` predicts frame ``i + 1``."""
        return slice(start, stop)

    def heldout_surprise_array(self) -> np.ndarray:
        """Observed held-out per-transition surprise, shape ``(H, T)``."""
        values = np.full((self.n_heldout_streams, len(self.tick_rows)), np.nan)
        for row in self.tick_rows:
            values[:, row.tick - 1] = row.heldout_surprise
        return values

    def summary(self) -> dict[str, Any]:
        """Measured summary used by the report and the design memo."""
        held = self.heldout_surprise_array()
        post = slice(self.flip_at, self.flip_back_at)
        normal_post = float(self.normal_heldout[:, post].mean())
        base_post = float(self.base_heldout[:, post].mean())
        peak = float(self.base_heldout[:, post].mean(axis=0).max())
        adoption = self.adoption_tick
        if adoption is not None:
            after = slice(max(self.flip_at, adoption), self.flip_back_at)
            adapted_after = float(np.nanmean(held[:, after]))
            base_after = float(self.base_heldout[:, after].mean())
            normal_after = float(self.normal_heldout[:, after].mean())
        else:  # pragma: no cover - no adoption means no recovery claim
            adapted_after = float("nan")
            base_after = float("nan")
            normal_after = float("nan")
        return {
            "normal_baseline_mean": float(self.thresholds.normal_baseline),
            "normal_phase_heldout_mean": float(
                self.base_heldout[:, : self.flip_at + 1].mean()
            ),
            "post_flip_base_mean": base_post,
            "post_flip_peak_tick_mean": peak,
            "post_flip_over_normal": base_post / normal_post if normal_post else float("nan"),
            "adoption_tick": adoption,
            "stale_tick": self.stale_tick,
            "eviction_tick": self.eviction_tick,
            "adapted_post_mean": adapted_after,
            "base_post_adoption_mean": base_after,
            "adapted_over_base": (
                adapted_after / base_after if base_after else float("nan")
            ),
            "adapted_over_normal": (
                adapted_after / normal_after if normal_after else float("nan")
            ),
            "normal_post_mean": normal_post,
            "d4_writes": int(self.d4_writes),
            "encoder_bit_identical": bool(self.encoder_bit_identical),
        }

    def __str__(self) -> str:  # pragma: no cover - human display
        lines = ["acceptance trace"]
        for row in self.tick_rows:
            lines.append(
                f"  t={row.tick:2d} {row.phase:9s} mean={row.mean_surprise:.4f} "
                f"held={row.heldout_surprise:.4f} overlay={row.active_overlay_id}"
            )
        for event in self.events:
            if event.births:
                lines.append(f"  birth @ {event.tick}: {event.births[-1].token()}")
            if event.stales:
                lines.append(f"  stale @ {event.tick}: {event.stales[-1].token()}")
            if event.evictions:
                lines.append(f"  evict @ {event.tick}: {[r.token() for r in event.evictions]}")
        lines.append(f"  summary: {self.summary()}")
        return "\n".join(lines)
