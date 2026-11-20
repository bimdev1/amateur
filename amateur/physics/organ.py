"""Phase 4-5 physics organ: tagged-cohort P5 digest + P6 candidate nursery.

This module is the **standalone physics organ** for the flippable 2-D world.
It owns E1 (the environment) and the phase-3 micro-LeWM (frozen encoder,
trained predictor).  There is still **no executor coupling** here: the
executor wiring lives in :mod:`amateur.physics.run`, and the payload types
live in :mod:`amateur.payloads`.

Round 11 architecture (rulings C, E, F) -- the candidate is the evidence
carrier, not the live overlay
-------------------------------------------------------------------------
1. :class:`Observation` -- one causal observation tick: the frame latent,
   the latent state delta (#12), the model surprise (#6/#12), the residual
   vector (the real debris candidate), a 3-frame causal window, and a short
   *resolved* latent rollout (#10).
2. :class:`ResidualDigester` (P5) -- a bounded online residual clusterer.
   Only residuals above the measured normal q99 enter; residuals are
   assigned to centroids within a merge radius, clusters are merged to make
   room, and the pool is hard-capped at ``debris_capacity``.  Round 11:
   every digest item is **tagged** into one of the four cohort classes
   (``motif`` / ``residual`` / ``scaffold`` / ``trail``) and exposed as
   ``DigestResult.cohorts``; that tuple is what the source packs into the
   single tagged ``#19`` composite body.  Merge sources and capacity
   evictions are surfaced as *scaffold* cohorts (the support machinery P4
   recycles), and each assignment leaves a *trail* cohort (the routing
   trace).
3. :class:`CandidateTable` (the organ-side D2 **nursery**, #18) -- a
   candidate is a capacity-bounded low-rank predictor patch that is
   **substance and never steering**.  :meth:`PhysicsOrgan.birth_candidate`
   fits it from the last real transitions of a stream (the same
   capacity-bounded fit and refusal records as the ratchet), stores it in
   the nursery, and never applies it: :meth:`PhysicsOrgan.shadow_scores`
   evaluates every resident candidate's patch on a *shadow copy of the
   predictor* against the base predictor on each tick's real transition and
   asserts the encoder and the live predictor are bit-identical afterwards.
   Accumulated residence evidence (``adapted_mean / base_mean`` over at
   least ``RESIDENCE_N`` shadow-scored ticks) is what ``#23`` surfaces: the
   survivor is the **lowest-ratio** resident with enough residence, and
   ``qualified`` (``ratio < 1.0``) is carried as a flag, **not** applied as a
   selection filter -- a harmful survivor must stay observable so that
   ``#22``'s refusal is testable and ADMISSION is not vacuous.  Only a
   *qualified* survivor may be lineage-consumed by ``#22``.
4. :class:`OverlayTable` + :class:`PredictorOverlay` (P6) -- the **live**
   ratchet.  Live patches arrive only via ``#24``, driven by a ``#21`` motif
   cohort read out of D3: :meth:`PhysicsOrgan.adopt_motif` fits (capacity
   bounded) or revalidates the overlay behind that motif and activates it.
   Birth / revalidate / stale / cooldown / evict semantics are unchanged
   from round 9a.

Ruling (C): the live-overlay D4 gate is **gone**.  Round 9a's
``ratchet_gate_passed`` / ``_accrue_gate_evidence`` / ``_record_gate_evidence``
decided ``#22`` from live-overlay evidence; round 11 rules that nothing
reaches D4 that did not reside in D2 and get surfaced by ``#23``.  The
decision path is replaced by candidate residence evidence
(:meth:`PhysicsOrgan.residence_evidence`).  Staleness -- which genuinely
needs the live overlay's own adapted-vs-base comparison -- is kept: a live
patch that stops improving on what it has to correct goes stale and is then
evicted.

Capacity bound (Round 9a, unchanged)
------------------------------------
The stored patch is a **capacity-bounded low-rank delta of the last head
layer only** (``head.2`` weight + bias, factors ``u @ v.T``), fitted under
``K <= gamma * N * D`` with ``gamma = 5.0`` in
:mod:`amateur.physics.capacity`.  ``N`` is the number of real evidence
transitions actually used, counted from the data.  A dense head+norm or
full-predictor patch is inadmissible (a different patch class, not a larger
budget).  Every refusal is recorded in ``fit_history`` with a reason
(``capacity`` / ``no-evidence`` / ``no-improvement``) and the budget it was
judged against -- never silent.

**The encoder is never touched:** every fit happens on a deep copy of the
base model, and the encoder parameters are asserted bit-identical
(``torch.equal``) before and after every fit, every shadow score and every
overlay activation.

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
measured q95 (``||e||**2 == latent_dim * surprise``).  Structural constants
(pool capacity, overlay capacity, nursery capacity, residence N, motif
support, persistence gate) are **design choices**, listed with their
justification in :class:`DigestThresholds` and in the design memo's honesty
notes; ``RESIDENCE_N = 4`` and ``gamma = 5.0`` are pre-registered design
constants, not measurements.

Measured limits (honest, see the memo's numbered notes)
-------------------------------------------------------
* A predictor-only patch fitted on a **single** stream's window is not
  guaranteed to improve forward surprise.  Round 9a's single-stream
  result was mixed (it helped on the flipped window and harmed on the full
  active window of the registered trace; see
  ``docs/phase4-5-design.md`` §15.4/§15.10, historical, old 26-flow
  topology).  Round 11 therefore does not re-assert any round 7-9a number:
  the evidence carrier changed, and the numbers are re-measured in the
  round's registered runs.
* Only the last head layer is adapted, so applying/clearing an overlay is
  exact with no replay, and ``changes_gru`` is false everywhere.  This is
  also what makes shadow scoring sound: a head-only patch leaves the GRU
  hidden state untouched, so the shadow predictor can be stepped with the
  live stream's own ``hidden_before``.
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

from .capacity import EvidenceBudget

try:  # the round-11 schema is owned by ``amateur/payloads.py`` (wave 1)
    from ..payloads import NURSERY_CAPACITY, RESIDENCE_N
except ImportError:  # pragma: no cover - in-flight wave-1 schema window
    # The contract pins both values (§1.8: NURSERY_CAPACITY = 4,
    # RESIDENCE_N = 4).  The fallback keeps the organ importable while
    # ``payloads.py`` is being rewritten; ``tests/test_physics_organ.py``
    # asserts they equal the payload constants once those land.
    NURSERY_CAPACITY = 4
    RESIDENCE_N = 4

from .data import _action_rng, _episode_seed
from .env import PhysicsEnv, WorldConfig
from .micro_lewm import MicroLeWM

__all__ = [
    "COHORT_KINDS",
    "CandidateRecord",
    "CandidateTable",
    "CollapseMandate",
    "CohortRecord",
    "D4StandIn",
    "DigestResult",
    "DigestThresholds",
    "DigestedDebris",
    "EvidenceBudget",
    "FitRecord",
    "NurseryTick",
    "NurseryTrace",
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
#:
#: **Measured reachability (Round 11 wave 3, registered single-stream
#: scenario, 36 ticks).**  Seed 7: 6 ingest ticks (12, 19, 25, 28, 29, 30),
#: longest support streak 1, largest body count 2.  Seed 0: 5 ingest ticks
#: (9, 10, 11, 12, 14), longest support streak 2, largest body count 3.
#: Both motif gates are therefore unreachable on one stream (persistence 10,
#: count 4), and the run produces **no motif cohort at all**: the ``#21`` ->
#: ``#24`` live-overlay path does not fire.  The gate was derived on a
#: multi-stream bank, so this is a *reported limitation* of the
#: single-stream regime, not a tuned constant: both values are unchanged and
#: no threshold is relaxed to make the path reachable.  The end-to-end
#: lifecycle is demonstrated by the 16-stream
#: :meth:`PhysicsOrgan.nursery_scenario` probe (POST-HOC PROBE / NOT A GATE).
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
#: Round 11: the default number of trailing real transitions a candidate fit
#: sees (the same window the round-9a proposal used).
DEFAULT_FIT_WINDOW = 16
#: The four D3 cohort classes (#19).  Same names as
#: ``amateur.payloads.COHORT_KINDS`` (one string vocabulary, two modules: the
#: organ produces the tags, the payload layer transports them).
COHORT_KINDS = ("motif", "scaffold", "trail", "residual")


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
    nursery_capacity: int = NURSERY_CAPACITY
    residence_n: int = RESIDENCE_N

    def __post_init__(self) -> None:
        if self.debris_capacity < 2:
            raise ValueError("debris_capacity must be >= 2")
        if self.overlay_capacity < 1:
            raise ValueError("overlay_capacity must be >= 1")
        if self.motif_min_count < 1 or self.motif_min_persistence < 1:
            raise ValueError("motif gates must be positive")
        if self.stale_patience < 1 or self.stale_after < 0:
            raise ValueError("stale gates out of range")
        if self.nursery_capacity < 1:
            raise ValueError("nursery_capacity must be >= 1")
        if self.residence_n < 1:
            raise ValueError("residence_n must be >= 1")


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
class CohortRecord:
    """One tagged digest item (a D3 cohort, #19).

    Round 11 tags every digest item into exactly one of :data:`COHORT_KINDS`:

    ``motif``
        the assigned body is **stable** this tick (persistence, count, spread
        and drift inside the existing :meth:`ResidualDigester._motif_for`
        gates) -- the rule the ratchet mines (#21).
    ``residual``
        the assigned body is not yet stable -- raw debris (#27's residual
        drain, plus expired cohorts).
    ``scaffold``
        a structure **consumed** this tick: a merge source from
        :meth:`ResidualDigester._merge_pair` or a capacity victim from
        :meth:`ResidualDigester._evict_least_supported` -- the support
        machinery P4 recycles (#20).
    ``trail``
        the per-tick assignment trace: the mean assigned residual and the
        number of assignments this tick -- the routing trail (#26).

    A cohort is an **issuance**, not a store record: depositing the same body
    on two ticks mints two cohorts.  ``cohort_id`` is deterministic and
    unique per ``(kind, body, tick)`` (the emitting stream is part of the id
    so a bank probe cannot collide).
    """

    cohort_id: str
    kind: str
    centroid: np.ndarray
    count: int
    spread: float
    first_tick: int
    last_tick: int
    peak_surprise: float
    #: The D3 body this issuance came from ('' for the trail cohort).
    body_id: str = ""

    def __post_init__(self) -> None:
        if self.kind not in COHORT_KINDS:
            raise ValueError(f"unknown cohort kind {self.kind!r}")
        if not self.cohort_id:
            raise ValueError("cohort_id must be non-empty")
        if self.centroid.ndim != 1:
            raise ValueError("cohort centroid must be (D,)")
        if self.count < 1:
            raise ValueError("cohort count must be positive")
        if self.spread < 0.0:
            raise ValueError("cohort spread must be non-negative")


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
    #: Round 11 (#19): the tagged cohort items this tick digested.  Empty on
    #: a non-ingest tick (the source then emits the ``noop_debris`` sentinel).
    cohorts: tuple[CohortRecord, ...] = ()


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

    Round 11: every ingest tick also produces **tagged cohorts**
    (:class:`CohortRecord`) in ``DigestResult.cohorts`` -- the assignment as
    a ``motif`` or ``residual`` cohort, every structure consumed this tick as
    a ``scaffold`` cohort, and the routing trail as a ``trail`` cohort.  A
    non-ingest tick produces no cohorts (the source emits the #19 no-op).
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
        #: Structures consumed by the *current* observe call (merge sources
        #: and capacity evictions): the scaffold cohort source.
        self._consumed: list[DigestedDebris] = []

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
        # Round 11: both merge sources are structures consumed this tick;
        # they become the scaffold cohort (snapshot before the survivor is
        # mutated again by a later merge in the same observe pass).
        self._consumed.append(a.snapshot())
        self._consumed.append(b.snapshot())
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
        self._consumed.append(victim.snapshot())
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
        self._consumed = []
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

        # The assignment-time snapshot: if compaction folds this body into
        # another, the tick's assignment must still be issued as a cohort
        # (a digest item is never silently lost).
        assigned_body = None if assigned is None else self._bodies.get(assigned)
        assigned_snapshot = (
            None if assigned_body is None else assigned_body.snapshot()
        )
        merged.extend(self._compact(thresholds.merge_radius / 2.0))
        motif = self._motif_for(assigned)
        if mandate is not None:
            mandate = replace(mandate, active_bodies=len(self._bodies))
        cohorts = self._cohorts(
            tick=tick,
            stream=int(observation.stream),
            assigned=assigned,
            assigned_snapshot=assigned_snapshot,
            motif=motif,
            consumed=self._consumed,
            residual=residual,
            surprise=float(surprise),
        )
        self._consumed = []
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
            cohorts=cohorts,
        )

    def _cohorts(
        self,
        *,
        tick: int,
        stream: int,
        assigned: str | None,
        assigned_snapshot: DigestedDebris | None,
        motif: StructuralMotif | None,
        consumed: Sequence[DigestedDebris],
        residual: np.ndarray,
        surprise: float,
    ) -> tuple[CohortRecord, ...]:
        """Tag this observation's digest items into the four cohort classes.

        Rules (Round 11, contract §3.3):

        * ``motif`` when the assigned body is stable this tick (the same
          :meth:`_motif_for` gates the ratchet mines), else ``residual``;
        * one ``scaffold`` per structure consumed this tick (merge sources
          and capacity victims), in consumption order;
        * one ``trail`` for the routing trace: the assigned residual and the
          number of assignments recorded by this observation (exactly one,
          because a digest is one observation; the single-stream run has one
          observation per tick, so this is the tick's assignment trace).

        Ids are deterministic and unique per ``(kind, body, tick)``.  The
        ``-s<stream>`` and ``-<i>`` suffixes make them unique when several
        streams support the same tick and when one body is consumed twice in
        the same observe pass (a merge survivor that is merged again).
        """
        out: list[CohortRecord] = []
        body = None if assigned is None else self._bodies.get(assigned)
        snapshot = assigned_snapshot if body is None else body.snapshot()
        if snapshot is not None:
            kind = "motif" if motif is not None else "residual"
            out.append(
                CohortRecord(
                    cohort_id=f"{kind}-{snapshot.debris_id}-s{stream}-{tick:04d}",
                    kind=kind,
                    centroid=snapshot.centroid,
                    count=snapshot.count,
                    spread=snapshot.spread,
                    first_tick=snapshot.first_tick,
                    last_tick=snapshot.last_tick,
                    peak_surprise=snapshot.peak_surprise,
                    body_id=snapshot.debris_id,
                )
            )
        else:  # pragma: no cover - an ingest tick always has an assignment
            raise AssertionError("an ingested tick must have an assigned body")
        for index, victim in enumerate(consumed):
            out.append(
                CohortRecord(
                    cohort_id=(
                        f"scaffold-{victim.debris_id}-s{stream}-{tick:04d}-{index:02d}"
                    ),
                    kind="scaffold",
                    centroid=victim.centroid,
                    count=victim.count,
                    spread=victim.spread,
                    first_tick=victim.first_tick,
                    last_tick=victim.last_tick,
                    peak_surprise=victim.peak_surprise,
                    body_id=victim.debris_id,
                )
            )
        out.append(
            CohortRecord(
                cohort_id=f"trail-s{stream}-{tick:04d}",
                kind="trail",
                centroid=np.asarray(residual, dtype=np.float64),
                count=1,
                spread=0.0,
                first_tick=tick,
                last_tick=tick,
                peak_surprise=float(surprise),
            )
        )
        return tuple(out)

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
    """A capacity-bounded, predictor-only **low-rank** patch.

    The stored substance is the last head layer's deltas in factored form:

    * ``u`` -- ``(latent_dim, rank)``: left factor, singular values absorbed;
    * ``v`` -- ``(head_in, rank)``: right factor;
    * ``bias`` -- ``(latent_dim,)``: the bias delta.

    ``apply_to`` materialises ``u @ v.T`` at apply time; the dense weight
    delta is **never stored** (Round 9a's evidence bound).  ``param_count``
    and ``nbytes`` are derived from the stored tensors, so a record cannot
    claim a size its factors do not have.

    ``changes_gru`` is always false: the patch touches only ``head.2``, so
    the GRU (and therefore the online hidden state) is unchanged and
    applying/clearing an overlay needs no replay.
    """

    overlay_id: str
    u: torch.Tensor
    v: torch.Tensor
    bias: torch.Tensor
    fit_transitions: int
    fit_streams: int
    loss_before: float
    loss_after: float
    patch: str
    #: ``K(max_rank)`` of the :class:`~amateur.physics.capacity.EvidenceBudget`
    #: that admitted this patch (0 for an externally built probe patch).
    capacity: int = 0
    #: ``N``, the real evidence transitions the budget was derived from.
    budget_n: int = 0

    @property
    def rank(self) -> int:
        return int(self.u.shape[1])

    @property
    def param_count(self) -> int:
        return (
            int(self.u.numel()) + int(self.v.numel()) + int(self.bias.numel())
        )

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.u, self.v, self.bias)
        )

    @property
    def changes_gru(self) -> bool:
        return False

    @property
    def delta_shapes(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int,]]:
        """The materialised shape triple (audit: no dense storage)."""
        return (
            (int(self.u.shape[0]), int(self.v.shape[0])),
            (int(self.v.shape[0]), int(self.v.shape[1])),
            (int(self.bias.shape[0]),),
        )

    def materialise(self) -> torch.Tensor:
        """The dense ``(latent_dim, head_in)`` weight delta (apply time only)."""
        return torch.matmul(self.u, self.v.t())

    def apply_to_predictor(self, predictor: Any) -> None:
        """Add the materialised patch to ``predictor.head.2`` (no grad).

        Only the last head layer is touched: the norm, ``head.0``, the GRU
        are left bit-identical.  Round 11 shadow scoring applies a patch to a
        standalone copy of the **predictor** only, so the encoder is not even
        copied (and provably cannot be touched).
        """
        head = predictor.head[-1]
        with torch.no_grad():
            head.weight.add_(self.materialise().to(head.weight.dtype))
            head.bias.add_(self.bias.to(head.bias.dtype))

    def apply_to(self, model: MicroLeWM) -> None:
        """Add the materialised patch to ``model.predictor.head.2`` (no grad)."""
        self.apply_to_predictor(model.predictor)


@dataclass(frozen=True)
class FitRecord:
    """Lightweight diagnostics of one overlay fit attempt (no tensors).

    The patch factors belong to the D2 store (``MatBoard``) while the
    overlay is resident; this record keeps only scalars so the organ holds
    no tensor after an eviction (Q1-A "no organ-side retention").

    ``reason`` is empty for an admitted fit and one of ``capacity`` /
    ``no-evidence`` / ``no-improvement`` for a refusal.  A refusal is
    **recorded**, never silent, and carries the evidence budget it was
    judged against (Round 9a).  (Round 11 dropped the ``cooldown`` reason:
    the cooldown now gates :meth:`PhysicsOrgan.adopt_motif`, which returns
    ``None`` without attempting a fit -- there is no fit to refuse.)
    """

    overlay_id: str
    param_count: int
    nbytes: int
    fit_transitions: int
    fit_streams: int
    loss_before: float
    loss_after: float
    patch: str
    rank: int = 0
    budget_n: int = 0
    budget_capacity: float = 0.0
    budget_max_params: int = 0
    requested_params: int = 0
    reason: str = ""


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
        """The ``(epoch, overlay_id, status)`` record the D2 overlay index holds.

        Round 11 per-flow pins: ``#24`` writes this record (``active``), ``#8``
        reads the active head, ``#25`` reads the head labelled ``stale``.
        ``#23`` does **not** carry this type -- it carries the nursery
        survivor (``SurvivingCrystalBody``), which is a read-time selection
        over candidate records, not a D2 overlay record.
        """
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
# Round 11 (E): the D2 nursery -- candidates carry the patch, never steer it
# --------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    """One resident D2 nursery record: the shadow patch and its evidence.

    ``overlay`` is the candidate's **substance**: a capacity-bounded
    low-rank predictor patch.  It is never applied to the live model; it is
    applied to a shadow copy of the predictor by
    :meth:`PhysicsOrgan.shadow_scores`, which accumulates ``evidence`` as
    ``(tick, adapted, base)`` -- one paired measurement per resident tick.

    ``ratio = adapted_mean / base_mean`` is the residence evidence
    ``#23``/``#22`` read (candidate ``qualified`` iff ``ticks >=
    RESIDENCE_N`` and ``ratio < 1.0``).
    """

    candidate_id: str
    overlay: PredictorOverlay
    born_tick: int
    evidence: list[tuple[int, float, float]] = field(default_factory=list)
    promoted: bool = False
    closed_tick: int | None = None
    cause: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.overlay.overlay_id != self.candidate_id:
            raise ValueError(
                "the candidate's patch must be keyed by the candidate id "
                f"({self.overlay.overlay_id!r} != {self.candidate_id!r})"
            )

    # ------------------------------------------------------------- evidence
    @property
    def residence_ticks(self) -> int:
        return len(self.evidence)

    @property
    def adapted_mean(self) -> float:
        if not self.evidence:
            return float("nan")
        return float(np.mean([sample[1] for sample in self.evidence]))

    @property
    def base_mean(self) -> float:
        if not self.evidence:
            return float("nan")
        return float(np.mean([sample[2] for sample in self.evidence]))

    @property
    def ratio(self) -> float:
        """``adapted_mean / base_mean``; ``inf`` when there is no valid base."""
        base = self.base_mean
        if not np.isfinite(base) or base <= 0.0:
            return float("inf")
        return float(self.adapted_mean / base)

    def qualified(self, residence_n: int = RESIDENCE_N) -> bool:
        """The #23 selection predicate: enough residence time and a real win."""
        ratio = self.ratio
        return (
            self.residence_ticks >= int(residence_n)
            and np.isfinite(ratio)
            and ratio < 1.0
        )

    def summary(self) -> dict[str, Any]:
        """The residence-evidence row (the pre-registered §15.11 curve)."""
        return {
            "candidate_id": self.candidate_id,
            "born_tick": int(self.born_tick),
            "residence_ticks": int(self.residence_ticks),
            "adapted_mean": float(self.adapted_mean),
            "base_mean": float(self.base_mean),
            "ratio": float(self.ratio),
            "qualified": bool(self.qualified()),
        }


@dataclass(frozen=True)
class PromotionRecord:
    """One candidate promotion (#22 lineage consumption), with its evidence.

    ``ratio`` is the candidate's **final** residence ratio at the moment of
    promotion -- the value the Round-11 ADMISSION gate reads (a promoted
    candidate with ``ratio >= 1.0`` is a wiring defect, not a tuning
    opportunity).
    """

    candidate_id: str
    born_tick: int
    promoted_tick: int
    residence_ticks: int
    adapted_mean: float
    base_mean: float
    ratio: float
    qualified: bool


class CandidateTable:
    """Bounded D2 nursery: ``candidate_id -> CandidateRecord`` (capacity 4).

    The organ owns the operative candidate patch; the store (``MatBoard``,
    ``amateur/payloads.py``) owns the accounting mirror.  The table mirrors
    the store's candidate index policy exactly: FIFO capacity eviction,
    ``nursery_observe`` bookkeeping only for **resident** ids (an unknown id
    is a loud failure), promote/retire remove the record and its patch
    together, and ``prune_to`` aligns the organ with the store's residents.

    Every mutation raises on illegal input; there are no silent no-ops.
    """

    def __init__(self, capacity: int = NURSERY_CAPACITY) -> None:
        if int(capacity) < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._records: dict[str, CandidateRecord] = {}
        #: ``(candidate_id, born_tick, close_tick, cause)`` -- the eviction log.
        self.evictions: list[tuple[str, int, int, str]] = []
        #: The promotion log (the ADMISSION gate's evidence).
        self.promotions: list[PromotionRecord] = []

    # ------------------------------------------------------------- mutations
    def birth(self, record: CandidateRecord) -> CandidateRecord:
        if record.candidate_id in self._records:
            raise ValueError(f"candidate {record.candidate_id!r} already resident")
        while len(self._records) >= self.capacity:
            oldest = next(iter(self._records.values()))
            self._close(oldest.candidate_id, record.born_tick, "capacity")
        self._records[record.candidate_id] = record
        return record

    def observe(self, candidate_id: str, tick: int, adapted: float, base: float) -> None:
        """Append one shadow measurement to a **resident** candidate."""
        record = self.require(candidate_id)
        adapted = float(adapted)
        base = float(base)
        if not (np.isfinite(adapted) and np.isfinite(base)):
            raise ValueError("shadow scores must be finite")
        record.evidence.append((int(tick), adapted, base))

    def retire(self, candidate_id: str, tick: int) -> CandidateRecord:
        return self._close(candidate_id, tick, "retired")

    def promote(self, candidate_id: str, tick: int) -> PromotionRecord:
        """Promote a resident candidate: log the evidence, remove record+patch."""
        record = self.require(candidate_id)
        if record.promoted:  # pragma: no cover - promotion removes the record
            raise ValueError(f"candidate {candidate_id!r} was already promoted")
        record.promoted = True
        promotion = PromotionRecord(
            candidate_id=candidate_id,
            born_tick=int(record.born_tick),
            promoted_tick=int(tick),
            residence_ticks=int(record.residence_ticks),
            adapted_mean=float(record.adapted_mean),
            base_mean=float(record.base_mean),
            ratio=float(record.ratio),
            qualified=bool(record.qualified()),
        )
        self.promotions.append(promotion)
        self._close(candidate_id, tick, "promoted")
        return promotion

    def prune_to(self, resident_ids: "set[str] | frozenset[str]") -> tuple[CandidateRecord, ...]:
        """Drop records the store no longer holds (the Q1-A one-way prune)."""
        removed = [
            record
            for name, record in self._records.items()
            if name not in resident_ids
        ]
        for record in removed:
            self._close(record.candidate_id, record.born_tick, "pruned")
        return tuple(removed)

    def reset(self) -> None:
        """Clear records and both logs (a fresh nursery)."""
        self._records.clear()
        self.evictions.clear()
        self.promotions.clear()

    def _close(self, candidate_id: str, tick: int, cause: str) -> CandidateRecord:
        record = self._records.pop(candidate_id)
        record.closed_tick = int(tick)
        record.cause = str(cause)
        self.evictions.append((candidate_id, int(record.born_tick), int(tick), str(cause)))
        return record

    # ----------------------------------------------------------------- reads
    def require(self, candidate_id: str) -> CandidateRecord:
        record = self._records.get(str(candidate_id))
        if record is None:
            raise KeyError(f"no resident candidate {candidate_id!r}")
        return record

    def record(self, candidate_id: str) -> CandidateRecord | None:
        return self._records.get(str(candidate_id))

    def ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def resident(self) -> tuple[CandidateRecord, ...]:
        """Resident records, oldest birth first (insertion order)."""
        return tuple(self._records.values())

    def patch(self, candidate_id: str) -> PredictorOverlay | None:
        record = self._records.get(str(candidate_id))
        return None if record is None else record.overlay

    def evidence(self, candidate_id: str) -> tuple[tuple[int, float, float], ...]:
        return tuple(self.require(candidate_id).evidence)

    def survivor(self) -> CandidateRecord | None:
        """The #23 selection: lowest ratio, **no** ``qualified`` pre-filter.

        Mirrors ``MatBoard.nursery_survivor`` exactly (contract §1.9): among
        residents with ``residence_ticks >= RESIDENCE_N`` and a finite
        positive base mean, the lowest ratio wins with the oldest birth as
        tie-break; a harmful candidate (``ratio >= 1``) is returned too, with
        ``qualified() is False``.  ``None`` iff no resident has enough
        residence.  Surfacing the harmful case is what makes the ``#22``
        refusal and the ADMISSION metric non-vacuous.
        """
        best: CandidateRecord | None = None
        for record in self._records.values():
            if record.residence_ticks < RESIDENCE_N:
                continue
            base = record.base_mean
            if not np.isfinite(base) or base <= 0.0:
                continue
            if best is None or record.ratio < best.ratio:
                best = record
        return best

    def promotion(self, candidate_id: str) -> PromotionRecord | None:
        for promotion in reversed(self.promotions):
            if promotion.candidate_id == str(candidate_id):
                return promotion
        return None

    def __len__(self) -> int:
        return len(self._records)


# --------------------------------------------------------------------------
# D4 stand-in (Base Core): the acceptance gate is that it gets zero writes
# --------------------------------------------------------------------------
class D4StandIn:
    """Stand-in for D4 Base Core; counts every attempted write.

    The organ never calls :meth:`write`: in Round 11 the ``#22`` sediment is
    produced by the **executor** source from the ``#23`` survivor read
    (:meth:`PhysicsOrgan.residence_evidence` /
    :meth:`PhysicsOrgan.survivor`), not by the organ, so the organ-level
    probes assert ``writes == 0``.  A live overlay can never write D4.
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
        "transitions",
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
        #: ``(frame_tick, prev_latent, action, hidden_before, latent)`` for the
        #: last ``cache_len`` transitions: the shadow-scoring probe needs the
        #: exact GRU input state each prediction used (Round 11, E).
        self.transitions: deque[
            tuple[int, torch.Tensor, int, torch.Tensor | None, torch.Tensor]
        ] = deque()

    def trim_cache(self, cache_len: int) -> None:
        while len(self.actions) > cache_len:
            self.actions.popleft()
            self.latents.popleft()
            self.cache_start += 1
        while len(self.transitions) > cache_len:
            self.transitions.popleft()


# --------------------------------------------------------------------------
# The organ
# --------------------------------------------------------------------------
class PhysicsOrgan:
    """E1 + frozen-encoder micro-LeWM + P5 tagged digest + P6 nursery/ratchet.

    Public API (the executor wiring surface, Round 11)::

        organ = PhysicsOrgan(checkpoint_path)
        obs = organ.reset(seed=7)                 # E1 reset, first frame
        obs = organ.step(action)                  # E1 step + observation
        digest = organ.digest(obs)                # P5 (tagged cohorts)
        candidate = organ.birth_candidate(tick)   # #18 (fit, never apply)
        shadow = organ.shadow_scores(tick)        # residence evidence
        record = organ.adopt_motif(motif, tick)   # #21 -> #24 (live patch)
        ratchet = organ.ratchet(digest)           # P6 staleness/eviction

    The same methods take ``stream=<int>`` for a bank of independent
    streams; digest, nursery and ratchet state are shared (one organ = one
    world).
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
        self.history: list[RatchetResult] = []
        self.observations: list[Observation] = []
        self.fit_history: list[FitRecord] = []
        self.refusals: list[dict[str, Any]] = []
        #: The budget of the most recent fit attempt (diagnostic).
        self.last_budget: EvidenceBudget | None = None
        # ---- Round 11 (E): the D2 nursery, organ-side ---------------------
        #: Candidate records + evidence (the store's ``MatBoard`` nursery
        #: index is the accounting mirror).
        self.nursery = CandidateTable(self.thresholds.nursery_capacity)
        self._candidate_serial = 0
        #: One shadow predictor copy per resident candidate, built at birth
        #: with the candidate's patch applied.  It is a copy of the **base
        #: predictor** and is never the live model.
        self._shadow_predictors: dict[str, Any] = {}

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
        # The shadow-scoring probe belongs to the stream too: a restart must
        # not let a later score replay a transition from the previous episode.
        state.transitions.clear()
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
            hidden_before = state.hidden
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
            # Round 11 (E): keep the exact probe a shadow score needs -- the
            # previous latent, the action, the GRU hidden state *before* this
            # step and the observed latent.  Shadow scoring replays this
            # transition on a copy of the predictor; the live model is never
            # touched.  Head-only patches leave the GRU untouched, so this
            # hidden state is valid for the base predictor too.
            state.transitions.append(
                (
                    int(state.tick),
                    state.prev_latent,
                    int(action),
                    hidden_before,
                    latent,
                )
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
        """Finalise one tick of the **live-overlay** lifecycle (P6).

        Round 11 (ruling C) narrowed this pass to what staleness genuinely
        needs.  Adoption is no longer decided here: the live patch arrives
        only through ``#24``, from a ``#21`` motif cohort, via
        :meth:`adopt_motif`.  The live overlay no longer produces ``#22``
        evidence either -- D4 is reached only by a candidate surfaced by
        ``#23`` (see :meth:`residence_evidence`).  What remains is the
        staleness rule: a live patch that stops improving on the residual it
        has to correct (the base-predictor error is smaller for
        ``stale_patience`` consecutive ticks) is marked stale, deactivated
        and cooled down; a patch that is still improving on a quiet stream
        is revalidated (epoch advances).
        """
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
        evictions = list(self.overlay_table.evict_due(tick))
        return births, revalidations, stales, evictions

    # ------------------------------------------------- motif path (#21 -> #24)
    @staticmethod
    def _as_structural_motif(motif: Any) -> StructuralMotif:
        """Accept a :class:`StructuralMotif`, a motif :class:`CohortRecord`,
        or a payload ``StructuralMotifBody`` (duck-typed) and normalise it.

        ``#21`` is an executor-derived D3 store read, so the source hands the
        organ whatever body it read; the fit only needs the window
        (``first_tick``/``last_tick``) and the lineage id.
        """
        if isinstance(motif, StructuralMotif):
            return motif
        if isinstance(motif, CohortRecord):
            if motif.kind != "motif":
                raise ValueError(
                    f"adopt_motif needs a motif cohort, got {motif.kind!r}"
                )
            return StructuralMotif(
                motif_id=motif.cohort_id,
                debris_id=motif.body_id,
                centroid=np.asarray(motif.centroid, dtype=np.float64),
                count=int(motif.count),
                spread=float(motif.spread),
                persistence=max(1, int(motif.last_tick) - int(motif.first_tick) + 1),
                drift=0.0,
                first_tick=int(motif.first_tick),
                last_tick=int(motif.last_tick),
            )
        motif_id = getattr(motif, "motif_id", None)
        if not motif_id:
            raise TypeError("adopt_motif needs a motif body with a motif_id")
        centroid = getattr(motif, "centroid", None)
        centroid = (
            np.zeros(DEFAULT_LATENT_DIM, dtype=np.float64)
            if centroid is None
            else np.asarray(centroid, dtype=np.float64)
        )
        return StructuralMotif(
            motif_id=str(motif_id),
            debris_id=str(getattr(motif, "debris_id", "")),
            centroid=centroid,
            count=int(getattr(motif, "count", 1)),
            spread=float(getattr(motif, "spread", 0.0)),
            persistence=int(getattr(motif, "persistence", 1)),
            drift=float(getattr(motif, "drift", 0.0)),
            first_tick=int(getattr(motif, "first_tick", 0)),
            last_tick=int(getattr(motif, "last_tick", 0)),
        )

    def adopt_motif(self, motif: Any, tick: int) -> OverlayRecord | None:
        """Fit or revalidate the **live** patch from a ``#21`` motif cohort.

        This is the only way a live patch arrives (Round 11 ruling C/E): the
        executor reads the motif cohort out of D3 (``#21``) and the source
        calls this method; a birth emits ``#24`` with
        ``motif_lineage = motif_id``.  Semantics are the round-9a ratchet's,
        unchanged:

        * a motif that already maps to a record is **revalidated** (epoch
          advances, status back to active) and re-activated;
        * otherwise the overlay is fitted under the unchanged capacity bound
          and **born** (epoch 1, active);
        * nothing happens during the cooldown window (returns ``None``; the
          source emits the ``#24`` no-op);
        * the ratchet keeps **one active overlay**: while a *different*
          overlay is active, a new motif is refused (returns ``None``) rather
          than silently replacing it.  Staleness/eviction free the slot.

        Returns the birth/revalidation record, or ``None``.
        """
        tick = int(tick)
        structural = self._as_structural_motif(motif)
        if tick < self._cooldown_until:
            return None
        existing = self.overlay_table.record_for_motif(structural.motif_id)
        if existing is not None:
            record = self.overlay_table.revalidate(existing.overlay_id, tick)
            self._set_active_overlay(existing.overlay_id)
            self._stale_run = 0
            return record
        if self._active_overlay_id is not None:
            return None
        overlay = self._fit_overlay(structural, tick)
        if overlay is None:
            return None
        record = self.overlay_table.birth(
            overlay, motif_id=structural.motif_id, tick=tick
        )
        self._set_active_overlay(record.overlay_id)
        self._stale_run = 0
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

    # ------------------------------------------------ candidate nursery (E)
    def birth_candidate(
        self,
        tick: int,
        *,
        fit_window: int = DEFAULT_FIT_WINDOW,
        stream: int = 0,
    ) -> CandidateRecord | None:
        """Fit a **shadow** patch and birth it into the nursery (``#18``).

        The fit is the existing capacity-bounded one (``_fit_overlay``:
        ``K <= gamma*N*D`` with ``gamma = 5.0``, last head layer only, SVD
        truncation, 40 AdamW steps on a deep copy of the base model) on the
        **last ``fit_window`` real transitions ending at ``tick``** -- the
        same evidence window the round-9a proposal used, and the same
        ``FitRecord``/refusal-reason machinery (``capacity``, ``no-evidence``,
        ``no-improvement``).

        Round 11 (E): the fitted patch is **not applied** -- not activated, not
        attached to the live model, not written to the overlay table.  It is
        stored in :class:`CandidateTable` as the candidate's substance, and
        a shadow predictor copy is built for it.  ``None`` means the fit was
        refused (the refusal is recorded, never silent).
        """
        tick = int(tick)
        if int(fit_window) < 2:
            raise ValueError("fit_window must be >= 2")
        first_tick = max(1, tick - int(fit_window) + 1)
        if first_tick >= tick:
            self._record_refusal("no-evidence", 0, EvidenceBudget(0), tick=tick)
            return None
        motif = StructuralMotif(
            motif_id=f"candidate-window-s{int(stream)}-{tick:04d}",
            debris_id="debris-candidate",
            centroid=np.zeros(self.base_model.config.latent_dim, dtype=np.float64),
            count=int(fit_window),
            spread=0.0,
            persistence=int(fit_window),
            drift=0.0,
            first_tick=int(first_tick),
            last_tick=int(tick),
        )
        overlay = self._fit_overlay(motif, tick)
        if overlay is None:
            return None
        # No peeking: every fitted transition ends at or before the birth tick
        # (the fit path can only see the causal cache, this is the falsifier).
        self._self_guard_fit_boundary(motif, tick)
        self._candidate_serial += 1
        candidate_id = f"cand-{self._candidate_serial:02d}"
        overlay = replace(overlay, overlay_id=candidate_id)
        record = CandidateRecord(
            candidate_id=candidate_id,
            overlay=overlay,
            born_tick=tick,
        )
        self.nursery.birth(record)
        # A standalone copy of the base *predictor* with the candidate's patch
        # applied once.  It is never the live model; bit-identity of the live
        # model is asserted on every ``shadow_scores`` call.
        self._shadow_predictors[candidate_id] = self._build_shadow_predictor(overlay)
        return record

    def _build_shadow_predictor(self, overlay: PredictorOverlay) -> Any:
        shadow = copy.deepcopy(self.base_model.predictor)
        shadow.eval()
        for parameter in shadow.parameters():
            parameter.requires_grad_(False)
        overlay.apply_to_predictor(shadow)
        return shadow

    def _shadow_predictor(self, record: CandidateRecord) -> Any:
        shadow = self._shadow_predictors.get(record.candidate_id)
        if shadow is None:  # pragma: no cover - built at birth; store-side prune
            shadow = self._build_shadow_predictor(record.overlay)
            self._shadow_predictors[record.candidate_id] = shadow
        return shadow

    def _transitions_at(
        self, tick: int, *, stream: int | None = None
    ) -> list[tuple[int, torch.Tensor, int, torch.Tensor | None, torch.Tensor]]:
        """The real transitions whose frame tick is ``tick`` (one per stream)."""
        found: list[tuple[int, torch.Tensor, int, torch.Tensor | None, torch.Tensor]] = []
        for key in sorted(self._streams):
            if stream is not None and int(key) != int(stream):
                continue
            for item in self._streams[key].transitions:
                if int(item[0]) == int(tick):
                    found.append(item)
        return found

    def shadow_scores(
        self,
        tick: int,
        *,
        stream: int | None = None,
        exclude_born_at: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Shadow-score every resident candidate on this tick's transition.

        For each resident candidate the candidate's patch is evaluated on the
        **shadow copy of the predictor** built at birth, and the base
        predictor is evaluated on the same transition; both use the stream's
        own ``hidden_before``, which is valid because a head-only patch leaves
        the GRU untouched.  Returns ``candidate_id -> (adapted, base)`` and
        appends exactly one ``(tick, adapted, base)`` sample per resident
        candidate (re-scoring the same tick is idempotent: the existing
        sample is returned, not duplicated).

        **The candidate never steers.**  After scoring, the encoder, the base
        predictor and the live working model are asserted bit-identical to
        their values before the call; a difference raises ``AssertionError``
        (a wiring defect, never a silent drift).

        ``adapted``/``base`` are the mean squared latent prediction errors
        over every stream's transition at ``tick`` (one stream in the
        executor run).

        ``exclude_born_at`` skips every candidate born at that tick.  A
        candidate is fitted on the last ``fit_window`` transitions ending at
        its birth tick, so its birth-tick transition is **in-sample**: the
        patch was optimised on it by construction.  Counting that sample
        would bias the residence evidence favourably, and the executor's
        store-side annotation runs before the ``#18`` birth commits, so the
        store can never carry it.  The registered run therefore scores
        candidates from ``born_tick + 1`` (out-of-sample only), keeping the
        organ's evidence set identical to the store's.
        """
        tick = int(tick)
        transitions = self._transitions_at(tick, stream=stream)
        if not transitions:
            return {}
        encoder_before = [
            parameter.detach().clone()
            for parameter in self.base_model.encoder_parameters()
        ]
        base_before = [
            parameter.detach().clone()
            for parameter in self.base_model.predictor.parameters()
        ]
        working_before = [
            parameter.detach().clone()
            for parameter in self.working_model.parameters()
        ]
        scores: dict[str, tuple[float, float]] = {}
        for record in self.nursery.resident():
            if (
                exclude_born_at is not None
                and int(record.born_tick) == int(exclude_born_at)
            ):
                continue
            existing = record.evidence[-1] if record.evidence else None
            if existing is not None and int(existing[0]) == tick:
                scores[record.candidate_id] = (float(existing[1]), float(existing[2]))
                continue
            shadow = self._shadow_predictor(record)
            adapted_errors: list[float] = []
            base_errors: list[float] = []
            with torch.no_grad():
                for _, prev_latent, action, hidden_before, latent in transitions:
                    z_t = (
                        prev_latent.unsqueeze(0)
                        if prev_latent.ndim == 1
                        else prev_latent
                    )
                    actions = torch.as_tensor([int(action)], dtype=torch.long)
                    adapted_prediction, _ = shadow.step(z_t, actions, hidden_before)
                    base_prediction, _ = self.base_model.predictor.step(
                        z_t, actions, hidden_before
                    )
                    adapted_error = adapted_prediction.reshape(-1) - latent.reshape(-1)
                    base_error = base_prediction.reshape(-1) - latent.reshape(-1)
                    adapted_errors.append(
                        float(torch.mean(adapted_error * adapted_error))
                    )
                    base_errors.append(float(torch.mean(base_error * base_error)))
            adapted = float(np.mean(adapted_errors))
            base = float(np.mean(base_errors))
            if not (np.isfinite(adapted) and np.isfinite(base)):
                continue
            self.nursery.observe(record.candidate_id, tick, adapted, base)
            scores[record.candidate_id] = (adapted, base)
        self._assert_never_steered(encoder_before, base_before, working_before)
        return scores

    def _assert_never_steered(
        self,
        encoder_before: Sequence[torch.Tensor],
        base_before: Sequence[torch.Tensor],
        working_before: Sequence[torch.Tensor],
    ) -> None:
        """Assert the shadow probe left the encoder and live predictors alone."""
        current_encoder = [
            parameter.detach()
            for parameter in self.base_model.encoder_parameters()
        ]
        current_base = [
            parameter.detach()
            for parameter in self.base_model.predictor.parameters()
        ]
        current_working = [
            parameter.detach() for parameter in self.working_model.parameters()
        ]
        same = (
            len(current_encoder) == len(encoder_before)
            and len(current_base) == len(base_before)
            and len(current_working) == len(working_before)
            and all(
                torch.equal(now, then)
                for now, then in zip(current_encoder, encoder_before)
            )
            and all(
                torch.equal(now, then) for now, then in zip(current_base, base_before)
            )
            and all(
                torch.equal(now, then)
                for now, then in zip(current_working, working_before)
            )
        )
        if not same:
            raise AssertionError(
                "a candidate patch touched the live model: shadow scoring must "
                "never steer (encoder and live predictor must stay bit-identical)"
            )

    def retire_candidate(self, candidate_id: str, tick: int) -> CandidateRecord:
        """Retire a resident candidate and drop its shadow predictor."""
        record = self.nursery.retire(candidate_id, tick)
        self._shadow_predictors.pop(record.candidate_id, None)
        return record

    def promote_candidate(self, candidate_id: str, tick: int) -> PromotionRecord:
        """Promote a candidate (``#22`` lineage consumption) and log evidence."""
        promotion = self.nursery.promote(candidate_id, tick)
        self._shadow_predictors.pop(promotion.candidate_id, None)
        return promotion

    def prune_candidates(
        self, resident_ids: "set[str] | frozenset[str]"
    ) -> tuple[CandidateRecord, ...]:
        """Drop organ-side candidates the store no longer holds (Q1-A)."""
        removed = self.nursery.prune_to(resident_ids)
        for record in removed:
            self._shadow_predictors.pop(record.candidate_id, None)
        return removed

    # ------------------------------------------------------- nursery reads
    @property
    def promotions(self) -> tuple[PromotionRecord, ...]:
        """The promotion log, newest last (the ADMISSION gate's evidence)."""
        return tuple(self.nursery.promotions)

    def candidate_ids(self) -> tuple[str, ...]:
        return self.nursery.ids()

    def candidate_record(self, candidate_id: str) -> CandidateRecord | None:
        return self.nursery.record(candidate_id)

    def candidate_patch(self, candidate_id: str) -> PredictorOverlay | None:
        """The candidate's stored patch (the store's ``#18`` attach hook)."""
        return self.nursery.patch(candidate_id)

    def residence_evidence(
        self, candidate_id: str
    ) -> tuple[int, float, float, float]:
        """``(ticks, adapted_mean, base_mean, ratio)`` for one candidate.

        This is the Round-11 replacement for the deleted live-overlay gate:
        the ``#23`` survivor body and the ``#22`` decision are both derived
        from this evidence.
        """
        record = self.nursery.require(candidate_id)
        return (
            int(record.residence_ticks),
            float(record.adapted_mean),
            float(record.base_mean),
            float(record.ratio),
        )

    def qualified(self, candidate_id: str) -> bool:
        """True iff the candidate passes the ``#23`` residence selection."""
        record = self.nursery.require(candidate_id)
        return bool(record.qualified(self.thresholds.residence_n))

    def survivor(self) -> CandidateRecord | None:
        """The ``#23`` selection, organ-side: lowest ratio, oldest tie-break.

        Delegates to :meth:`CandidateTable.survivor`: the selection does
        **not** pre-filter on ``qualified`` (contract §1.9), so a harmful
        survivor surfaces with ``qualified(...) is False`` and the ``#22``
        gate is the thing that refuses it.  The store's
        ``MatBoard.nursery_survivor`` is the same selection on the same
        evidence; this mirror exists for the organ-level probe.
        """
        return self.nursery.survivor()

    def reset_p6_state(self) -> None:
        """Clear all P6 decision state, lifecycle table and fit artifacts.

        Called by ``PhysicsRun.__init__`` and ``PhysicsRun.reset`` so a
        second run on the same organ is a fresh experiment: no pending
        decisions, cooldown, stale run, overlay records/patches, nursery
        candidates/evidence/promotions or fit history survive.  E1 (environment/streams)
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
        self.overlay_table.reset()
        self.history.clear()
        self.observations.clear()
        self.fit_history.clear()
        self.refusals.clear()
        self.last_budget = None
        # Round 11 (E): the nursery is part of P6 state.  Records, evidence,
        # promotions, eviction log and every shadow predictor are dropped;
        # a second run on the same organ is a fresh experiment.
        self.nursery.reset()
        self._candidate_serial = 0
        self._shadow_predictors.clear()
        self.working_model = copy.deepcopy(self.base_model)
        self.working_model.eval()

    def _self_guard_fit_boundary(self, motif: StructuralMotif, tick: int) -> None:
        """No peeking: every fitted transition must end at or before ``tick``."""
        if int(motif.last_tick) > int(tick):
            raise AssertionError(
                f"candidate evidence extends past the birth tick: motif "
                f"last_tick={motif.last_tick} > tick={tick}"
            )

    # --------------------------------------------------------- overlay apply
    def _set_active_overlay(self, overlay_id: str | None) -> None:
        adapter = None if overlay_id is None else self.overlay_table.resolve(overlay_id)
        if overlay_id is not None and adapter is None:
            overlay_id = None
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
        """Fit a capacity-bounded low-rank ``head.2`` patch on real evidence.

        Round 9a (user decision 1): the stored patch scales with the
        evidence.  A dense delta of the last head layer (weight + bias) is
        fitted on the real transitions in the motif window using the
        pre-registered replay path (40 AdamW steps, fixed seed), then
        SVD-truncated to the rank the evidence budget admits (Eckart-Young)
        and stored as factors ``u (D x r)``, ``v (H x r)`` and the bias
        delta.  The dense delta is never stored and the norm / ``head.0`` /
        GRU are untouched.

        Fits on a deep copy of the base model; the base encoder and
        predictor are never mutated.  Every refusal is recorded in
        ``fit_history`` with a reason and the budget it was judged against:

        * ``capacity`` -- ``patch="full"`` (the dense full-predictor patch)
          or a budget whose admissible rank is < 1;
        * ``no-evidence`` -- no real transitions in the window;
        * ``no-improvement`` -- the stored low-rank patch does not improve
          its own window.
        """
        tick = int(tick)
        head_layer = self.base_model.predictor.head[-1]
        latent_dim = int(head_layer.weight.shape[0])
        head_in = int(head_layer.weight.shape[1])
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
            self.last_budget = EvidenceBudget(
                0, latent_dim=latent_dim, head_in=head_in
            )
            self._record_refusal("no-evidence", 0, self.last_budget)
            return None
        common = min(int(item[1].shape[0]) for item in sequences)
        if common <= lo:
            self.last_budget = EvidenceBudget(
                0, latent_dim=latent_dim, head_in=head_in
            )
            self._record_refusal("no-evidence", 0, self.last_budget)
            return None
        z_batch = torch.stack([item[0][: common + 1] for item in sequences], dim=0)
        a_batch = torch.stack([item[1][: common] for item in sequences], dim=0)
        foreground = slice(lo, common)
        # N is the number of REAL evidence transitions the loss is computed
        # on -- counted from the data, never the requested window or motif.
        fit_transitions = int((common - lo) * len(sequences))
        budget = EvidenceBudget(
            fit_transitions, latent_dim=latent_dim, head_in=head_in
        )
        self.last_budget = budget

        # Capacity gate BEFORE any fit or overlay: the admissible patch class
        # is the capacity-bounded low-rank last-layer delta.
        if self.patch == "full":
            requested = int(
                sum(parameter.numel() for parameter in self.base_model.predictor.parameters())
            )
            self._record_refusal(
                "capacity",
                fit_transitions,
                budget,
                patch="full",
                requested_params=requested,
                fit_streams=len(sequences),
            )
            return None
        if budget.max_rank < 1:
            self._record_refusal(
                "capacity",
                fit_transitions,
                budget,
                fit_streams=len(sequences),
            )
            return None

        model = copy.deepcopy(self.base_model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        head = model.predictor.head[-1]
        trainable = list(head.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(trainable, lr=self.fit_lr, weight_decay=1e-4)
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
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        base_head = self.base_model.predictor.head[-1]
        delta_w = (head.weight.detach() - base_head.weight.detach()).float()
        delta_b = (head.bias.detach() - base_head.bias.detach()).float()
        rank = budget.max_rank
        try:
            u_full, s_full, vh_full = torch.linalg.svd(delta_w, full_matrices=False)
        except RuntimeError:  # pragma: no cover - LAPACK failure is not expected
            self._record_refusal(
                "capacity", fit_transitions, budget, fit_streams=len(sequences)
            )
            return None
        u = (u_full[:, :rank] * s_full[:rank]).contiguous()
        v = vh_full[:rank, :].transpose(0, 1).contiguous()
        self._overlay_serial += 1
        candidate = PredictorOverlay(
            overlay_id=f"ovr-{self._overlay_serial:02d}",
            u=u,
            v=v,
            bias=delta_b.clone(),
            fit_transitions=fit_transitions,
            fit_streams=len(sequences),
            loss_before=loss_before,
            loss_after=float("nan"),
            patch=self.patch,
            capacity=budget.max_params,
            budget_n=fit_transitions,
        )
        probe = copy.deepcopy(self.base_model)
        probe.eval()
        candidate.apply_to(probe)
        with torch.no_grad():
            after = probe.predict(z_batch, a_batch)
            loss_after = float(
                (after[:, foreground] - z_batch[:, 1:][:, foreground]).pow(2).mean()
            )
        if not np.isfinite(loss_before) or not np.isfinite(loss_after) or loss_after >= loss_before:
            self._overlay_serial -= 1
            self._record_refusal(
                "no-improvement",
                fit_transitions,
                budget,
                rank=rank,
                fit_streams=len(sequences),
            )
            return None
        overlay = replace(candidate, loss_after=loss_after)
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
                rank=overlay.rank,
                budget_n=fit_transitions,
                budget_capacity=budget.capacity_params,
                budget_max_params=budget.max_params,
            )
        )
        return overlay

    def _record_refusal(
        self,
        reason: str,
        n_transitions: int,
        budget: EvidenceBudget | None,
        *,
        tick: int | None = None,
        patch: str | None = None,
        requested_params: int = 0,
        rank: int = 0,
        fit_streams: int = 0,
    ) -> None:
        """Record a refused fit attempt (never silent, Round 9a).

        Refusals carry the evidence budget they were judged against, so the
        cap is auditable from ``fit_history`` alone.
        """
        self.refusals.append(
            {
                "reason": str(reason),
                "n_transitions": int(n_transitions),
                "tick": None if tick is None else int(tick),
                "patch": self.patch if patch is None else str(patch),
                "requested_params": int(requested_params),
                "max_rank": 0 if budget is None else int(budget.max_rank),
                "max_params": 0 if budget is None else int(budget.max_params),
                "capacity_params": (
                    0.0 if budget is None else float(budget.capacity_params)
                ),
            }
        )
        self.fit_history.append(
            FitRecord(
                overlay_id="",
                param_count=0,
                nbytes=0,
                fit_transitions=int(n_transitions),
                fit_streams=int(fit_streams),
                loss_before=float("nan"),
                loss_after=float("nan"),
                patch=self.patch if patch is None else str(patch),
                rank=int(rank),
                budget_n=int(n_transitions),
                budget_capacity=0.0 if budget is None else float(budget.capacity_params),
                budget_max_params=0 if budget is None else int(budget.max_params),
                requested_params=int(requested_params),
                reason=str(reason),
            )
        )

    def _fit_stream_ids(self) -> list[int]:
        if self.fit_streams is not None:
            return sorted(stream for stream in self.fit_streams if stream in self._streams)
        return sorted(self._streams)

    # ------------------------------------------------- nursery probe (E/F)
    def nursery_scenario(
        self,
        *,
        seed: int = 0,
        n_streams: int = 4,
        horizon: int = 36,
        flip_at: int = 8,
        flip_back_at: int = 28,
        fit_window: int = DEFAULT_FIT_WINDOW,
        episode_offset: int = 0,
    ) -> "NurseryTrace":
        """Run the organ-level nursery probe on a real flip scenario.

        A bank of ``n_streams`` paired streams runs
        ``normal -> gravity flip -> ... -> flip back`` (the shape the round-11
        registered run uses, pre-registered in
        ``docs/phase4-5-design.md`` §15.11).  Per tick, after every stream's
        observation and digest:

        1. the **first** tick whose mean surprise exceeds the measured
           ``spike_threshold`` births one candidate (``#18``): a shadow patch
           fitted on the last ``fit_window`` transitions, never applied;
        2. every resident candidate is shadow-scored against the base
           predictor (:meth:`shadow_scores`); the candidate never steers;
        3. a candidate whose residence evidence qualifies is **promoted**
           (the organ-side mirror of ``#22`` consuming the ``#23`` survivor);
        4. the tick's ``motif`` cohort (from ``DigestResult.cohorts``) is
           offered to :meth:`adopt_motif` -- the organ-side mirror of the
           ``#21`` store read driving ``#24``;
        5. the P6 staleness/eviction pass runs through ``ratchet``.

        This is a **probe, not the executor run**: the executor's registered
        gates are computed from ``PhysicsRun`` rows in wave 3.  Every number
        it returns is measured on this run; no round 7-9a number is
        re-asserted.  ``self.d4`` is never written (the organ has no D4 path).
        """
        total = int(n_streams)
        if total < 1:
            raise ValueError("n_streams must be >= 1")
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
        self.reset_p6_state()
        self.fit_streams = set(range(total))
        self._streams.clear()

        events: list[RatchetResult] = []
        rows: list[NurseryTick] = []
        born: list[dict[str, Any]] = []
        cohort_counts: dict[str, int] = {kind: 0 for kind in COHORT_KINDS}
        promoted_ids: list[str] = []
        active_candidate: CandidateRecord | None = None
        #: Per-tick ``(tick, adapted, base)`` means over the observations the
        #: **live** overlay actually steered -- the counterfactual base side
        #: comes from ``Observation.base_surprise`` (the same transition
        #: re-predicted by the unpatched base model).
        live_rows: list[tuple[int, float, float]] = []
        for stream in range(total):
            self.begin_stream(frames[stream, 0], stream=stream, seed=int(seed) + stream)
        for index in range(int(horizon)):
            tick = index + 1
            tick_cohorts: list[CohortRecord] = []
            for stream in range(total):
                observation = self.observe(
                    frames[stream, tick],
                    int(actions[stream, index]),
                    stream=stream,
                    ingest=True,
                )
                digest = self.digest(observation)
                tick_cohorts.extend(digest.cohorts)
                for cohort in digest.cohorts:
                    cohort_counts[cohort.kind] += 1
                result = self.ratchet(digest)
                if not result.empty:
                    events.append(result)
            window = self.observations[-total:]
            applied = [
                item
                for item in window
                if item.overlay_id is not None
                and np.isfinite(item.surprise)
                and np.isfinite(item.base_surprise)
            ]
            if applied:
                live_rows.append(
                    (
                        tick,
                        float(np.mean([item.surprise for item in applied])),
                        float(np.mean([item.base_surprise for item in applied])),
                    )
                )
            mean_surprise = float(np.mean([item.surprise for item in window]))
            fit_surprise = mean_surprise
            all_surprises = [
                item.surprise for item in window if np.isfinite(item.surprise)
            ]
            if all_surprises:
                mean_surprise = float(np.mean(all_surprises))
            # 1. #18: immediate birth on the first spike (never delayed).
            candidate: CandidateRecord | None = None
            if (
                active_candidate is None
                and tick > 2
                and np.isfinite(mean_surprise)
                and mean_surprise > self.thresholds.spike_threshold
            ):
                candidate = self.birth_candidate(tick, fit_window=fit_window, stream=0)
                if candidate is not None:
                    active_candidate = candidate
                    born.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "born_tick": tick,
                            "rank": candidate.overlay.rank,
                            "param_count": candidate.overlay.param_count,
                            "fit_transitions": candidate.overlay.fit_transitions,
                            "loss_before": candidate.overlay.loss_before,
                            "loss_after": candidate.overlay.loss_after,
                        }
                    )
            # 2. shadow scoring: one paired sample per resident candidate,
            #    out-of-sample only (the birth tick is in the fit window).
            scores = self.shadow_scores(
                tick,
                exclude_born_at=(
                    None if candidate is None else candidate.born_tick
                ),
            )
            shadow_ratio = None
            if scores:
                adapted = float(np.mean([pair[0] for pair in scores.values()]))
                base = float(np.mean([pair[1] for pair in scores.values()]))
                shadow_ratio = adapted / base if base > 0.0 else float("inf")
            # 3. #22 mirror: promote the **qualified** survivor only.
            survivor = self.survivor()
            survivor_qualified = (
                survivor is not None
                and survivor.qualified(self.thresholds.residence_n)
            )
            promoted_id = None
            if (
                survivor is not None
                and survivor_qualified
                and survivor.candidate_id not in promoted_ids
            ):
                promotion = self.promote_candidate(survivor.candidate_id, tick)
                promoted_ids.append(promotion.candidate_id)
                promoted_id = promotion.candidate_id
                if active_candidate is not None and (
                    active_candidate.candidate_id == promotion.candidate_id
                ):
                    active_candidate = None
            # 4. #21 -> #24 mirror: adopt the tick's motif cohort.
            motif = _best_motif_cohort(tick_cohorts)
            motif_id = None
            adopted: OverlayRecord | None = None
            if motif is not None:
                motif_id = motif.cohort_id
                adopted = self.adopt_motif(motif, tick)
            rows.append(
                NurseryTick(
                    tick=tick,
                    phase=_phase(tick, int(flip_at), int(flip_back_at)),
                    mean_surprise=mean_surprise,
                    candidates=self.candidate_ids(),
                    shadow_ratio=shadow_ratio,
                    survivor_id=None if survivor is None else survivor.candidate_id,
                    promoted_id=promoted_id,
                    motif_id=motif_id,
                    adopted_overlay_id=(
                        None if adopted is None else adopted.overlay_id
                    ),
                    active_overlay_id=self._active_overlay_id,
                    cohort_kinds=tuple(sorted({c.kind for c in tick_cohorts})),
                    survivor_qualified=bool(survivor_qualified),
                )
            )
        events.extend(self.flush())
        return NurseryTrace(
            tick_rows=tuple(rows),
            events=tuple(events),
            cohort_counts=cohort_counts,
            born=tuple(born),
            promotions=tuple(self.nursery.promotions),
            live_rows=tuple(live_rows),
            overlay_births=tuple(self.overlay_table.births),
            overlay_revalidations=tuple(self.overlay_table.revalidations),
            overlay_stales=tuple(self.overlay_table.stales),
            overlay_evictions=tuple(self.overlay_table.evictions),
            d4_writes=int(self.d4.writes),
            encoder_bit_identical=self.encoder_bit_identical(),
            thresholds=self.thresholds,
            flip_at=int(flip_at),
            flip_back_at=int(flip_back_at),
            n_streams=total,
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


# --------------------------------------------------------------------------
# The nursery probe trace (Round 11)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NurseryTick:
    """One scene tick of the nursery probe (per-stream means)."""

    tick: int
    phase: str
    mean_surprise: float
    candidates: tuple[str, ...]
    shadow_ratio: float | None
    survivor_id: str | None
    promoted_id: str | None
    motif_id: str | None
    adopted_overlay_id: str | None
    active_overlay_id: str | None
    cohort_kinds: tuple[str, ...]
    #: ``#23``'s ``qualified`` flag for ``survivor_id`` (Round 11: the
    #: survivor is surfaced regardless, and this records whether ``#22``
    #: would pass or refuse it).
    survivor_qualified: bool = False


def _best_motif_cohort(cohorts: Sequence[CohortRecord]) -> CohortRecord | None:
    """The tick's motif cohort the ratchet would mine: most support first."""
    motifs = [cohort for cohort in cohorts if cohort.kind == "motif"]
    if not motifs:
        return None
    return max(motifs, key=lambda cohort: (cohort.count, cohort.last_tick))


@dataclass(frozen=True)
class NurseryTrace:
    """The numbered nursery record; every number is measured on this run."""

    tick_rows: tuple[NurseryTick, ...]
    events: tuple[RatchetResult, ...]
    cohort_counts: dict[str, int]
    born: tuple[dict[str, Any], ...]
    promotions: tuple[PromotionRecord, ...]
    #: Per-tick ``(tick, adapted, base)`` means over the observations the live
    #: overlay steered, with the paired base counterfactual (measured state).
    live_rows: tuple[tuple[int, float, float], ...]
    #: The live-overlay lifecycle logs (``adopt_motif`` births/revalidations
    #: no longer flow through ``_decide_tick``; the counters are the organ's
    #: own tables, read directly so no event is lost).
    overlay_births: tuple[OverlayRecord, ...]
    overlay_revalidations: tuple[OverlayRecord, ...]
    overlay_stales: tuple[OverlayRecord, ...]
    overlay_evictions: tuple[OverlayRecord, ...]
    d4_writes: int
    encoder_bit_identical: bool
    thresholds: DigestThresholds
    flip_at: int
    flip_back_at: int
    n_streams: int

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
        return self.overlay_births[0].birth_tick if self.overlay_births else None

    @property
    def stale_tick(self) -> int | None:
        return self.overlay_stales[0].stale_tick if self.overlay_stales else None

    @property
    def eviction_tick(self) -> int | None:
        """The tick the eviction was decided at.

        ``OverlayRecord`` carries the *staleness* tick, not the eviction tick
        (the record is the D2 payload triple); the eviction decision is
        logged on the :class:`RatchetResult` of the tick that ran
        ``evict_due``, so that event tick is authoritative here.
        """
        for event in self.events:
            if event.evictions:
                return event.tick
        return None

    @property
    def candidate_birth_tick(self) -> int | None:
        return int(self.born[0]["born_tick"]) if self.born else None

    def row(self, tick: int) -> NurseryTick:
        for item in self.tick_rows:
            if item.tick == int(tick):
                return item
        raise KeyError(f"no row for tick {tick}")

    def shadow_rows(self) -> list[tuple[int, str, float]]:
        """``(tick, phase, shadow_ratio)`` for every shadow-scored row."""
        return [
            (row.tick, row.phase, float(row.shadow_ratio))
            for row in self.tick_rows
            if row.shadow_ratio is not None
        ]

    def adaptation_ratio(self) -> float:
        """Mean shadow ratio over ``flipped`` rows (the organ-side ADAPTATION).

        This is the *shadow* half of the round-11 gate, computed over this
        probe's rows only.  The registered gate is computed from the
        executor's ``RunTick`` rows (wave 3); this value is reported as the
        organ-level measurement, never as the gate.
        """
        ratios = [
            float(row.shadow_ratio)
            for row in self.tick_rows
            if row.phase == "flipped" and row.shadow_ratio is not None
        ]
        if not ratios:
            return float("nan")
        return float(np.mean(ratios))

    def live_ratio(self, *, phase: str | None = None) -> float:
        """Mean live adapted / mean live base over the steered ticks.

        ``phase`` restricts the rows to one scene phase (``"flipped"`` /
        ``"recovered"`` / ``"normal"``).  Reported as measured state
        (contract §5(iii)), never as a gate.
        """
        rows = [
            row
            for row in self.live_rows
            if phase is None
            or _phase(row[0], self.flip_at, self.flip_back_at) == phase
        ]
        if not rows:
            return float("nan")
        base = float(np.mean([row[2] for row in rows]))
        if not np.isfinite(base) or base <= 0.0:
            return float("inf")
        return float(np.mean([row[1] for row in rows]) / base)

    def lifecycle(self) -> dict[str, Any]:
        """The lifecycle event ticks (measured, may be ``None``)."""
        return {
            "candidate_birth": self.candidate_birth_tick,
            "promotion": (
                self.promotions[0].promoted_tick if self.promotions else None
            ),
            "overlay_birth": self.adoption_tick,
            "overlay_stale": self.stale_tick,
            "overlay_evict": self.eviction_tick,
            "d4_writes": int(self.d4_writes),
            "encoder_bit_identical": bool(self.encoder_bit_identical),
            "cohort_counts": dict(self.cohort_counts),
            # ``#23`` surfacing is counted separately from promotion so a
            # harmful survivor that the ``#22`` mirror refused stays visible.
            "survivor_ticks": sum(
                1 for row in self.tick_rows if row.survivor_id is not None
            ),
            "refused_harmful_survivors": sum(
                1
                for row in self.tick_rows
                if row.survivor_id is not None and not row.survivor_qualified
            ),
        }

    def summary(self) -> dict[str, Any]:
        """Measured summary used by the report and the design memo."""
        return {
            "scenario": {
                "n_streams": int(self.n_streams),
                "flip_at": int(self.flip_at),
                "flip_back_at": int(self.flip_back_at),
                "ticks": len(self.tick_rows),
            },
            "cohort_counts": dict(self.cohort_counts),
            "candidates": tuple(self.born),
            "promotions": tuple(
                promotion.__dict__ for promotion in self.promotions
            ),
            "shadow_adaptation_ratio": self.adaptation_ratio(),
            "live_overlay_ratio": self.live_ratio(),
            "live_overlay_ratio_flipped": self.live_ratio(phase="flipped"),
            "live_ticks": len(self.live_rows),
            "lifecycle": self.lifecycle(),
        }

    def __str__(self) -> str:  # pragma: no cover - human display
        lines = ["nursery trace"]
        for row in self.tick_rows:
            lines.append(
                f"  t={row.tick:2d} {row.phase:9s} surprise={row.mean_surprise:.4f} "
                f"cands={row.candidates} shadow={row.shadow_ratio} "
                f"live={row.active_overlay_id}"
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
