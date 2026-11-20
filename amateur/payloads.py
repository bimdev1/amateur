"""Typed payloads for the geometry and operational stores, schema-enforced.

Round 11 (rulings A-G): the D2 write kinds are pinned **per flow**, never
per store.  #18 carries `EmergentCrystalBody` (the nursery candidate kind)
and #24 carries `MatOverlay` (the ratcheted overlay kind); the candidate
lives in its own index and is never a `MatOverlay` status.  D2 reads are
pinned too: #8/#25 read `MatOverlay`, #23 reads `SurvivingCrystalBody`.
D3 is a cohort-level store (`DebrisCohort`), with per-edge read filters and
a destructive #27 drain.  `None` is no longer permitted on a modeled store
channel: a "no event" is a real exact-type sentinel.  D1 `BoardStore`
(#13/#14) is frozen work and keeps its prior behaviour (see the D1 section).

INV-EMPTINESS: the D4 write channel accepts only `BiasPrimitive`; the D2
write channels are exact-type per flow.  INV-NO-OSSIFY applies to the
overlay index, whose `mat_*_policy` semantics are **unchanged**; the
candidate index has its own FIFO capacity (`NURSERY_CAPACITY`) and is not
a second overlay stream.

Physics-epoch bodies are *content*, never provenance: the ``Envelope``
carries ``(source_fid, tick_born)`` and the exact-type gate compares body
types only.  All bodies are frozen dataclasses whose fields are tuples and
scalars, so they are hashable exactly when the envelope needs them to be.
"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ledger import FLOW_LEDGER, STORES

if TYPE_CHECKING:  # runtime import is local to BoardStore.apply (cycle guard)
    from .provenance import Provenance

_D2 = next(s for s in STORES if s.startswith("D2"))
_D4 = next(s for s in STORES if s.startswith("D4"))

#: Latent width of the phase-3 boundary slice; bodies that carry a latent
#: vector declare their own length, but sentinels and validation use this.
PAYLOAD_LATENT_DIM = 64

#: Reserved ids for the fixed-cadence "no event" bodies.  They are real,
#: exact-type bodies transported on their flow; the executor/store treats
#: them as no-ops (the same pattern as the Q4 zero-confidence #22 sentinel).
NOOP_OVERLAY_ID = "ovr-noop"
NO_DEBRIS_ID = "debris-none"
NO_MOTIF_ID = "motif-none"
NO_MANDATE_REASON = "none"

#: D3 production bounds (Round 11, cohort-level).  The old record-level
#: capacity of 8 became 32 cohorts: ~2-4 cohorts are emitted per tick
#: (residual + trail, plus scaffold/motif), so a capacity of 8 would evict a
#: motif cohort before #21 can read it on the next tick and silently starve
#: the #21 -> #24 live-overlay path.  32 cohorts ~= the old 8 bodies x 4 kinds
#: of material, keeps the byte ceiling comfortable (~18 KiB at ~560 B per
#: cohort), and still leaves capacity pressure as a real mechanism (#17).
DEBRIS_CAPACITY = 32
DEBRIS_MAX_BYTES = 65_536

#: D1 production bound: 8 context slots (design constant, like D3's 8).
#: No byte ceiling: D1 retains metadata + content digest only, so there is
#: no represented payload to bound (Lead adjudication 2026-11-20, §9.7).
BOARD_CAPACITY = 8

#: The only board-operation verbs #13 may carry.
BOARD_OP_VERBS = ("write", "erase", "resize", "evict")

#: D2 patch bounds (Q1-A).  Per patch: a head patch (136,960 B) fits, a
#: full-predictor patch (843,680 B) does not.  Per table: four per-patch
#: bounds, i.e. the MAT_CAPACITY worst case.
MAX_PATCH_BYTES = 262_144
MAX_PATCH_TOTAL_BYTES = 1_048_576

#: Round 11 D2 candidate index and D3 cohort model (contract §1.8/§1.9).
NURSERY_CAPACITY = 4
RESIDENCE_N = 4
DEBRIS_TTL = 12
COHORT_KINDS = ("motif", "scaffold", "trail", "residual")

#: Reserved no-event identity for the candidate kind (#18/#23).
NO_CANDIDATE_ID = "cand-none"


# ---------------------------------------------------------------------------
# Geometry records (unchanged)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatOverlay:
    """One D2 ratcheted overlay: epoch-stamped so staleness is checkable.

    Round 11: the `candidate` status is **removed** (candidates are a
    separate kind in the candidate index).  ``motif_lineage`` is the #21
    motif id for a #24 write; it is required non-None by the #24 execution
    gate, while the seed/read records carry None.
    """

    epoch: int
    overlay_id: str
    status: str
    motif_lineage: str | None = None

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or isinstance(self.epoch, bool):
            raise TypeError(f"epoch must be an int, got {self.epoch!r}")
        if self.epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {self.epoch}")
        if type(self.overlay_id) is not str or not self.overlay_id:
            raise ValueError("overlay_id must be a non-empty string")
        if self.status not in ("active", "stale"):
            raise ValueError(
                f"status must be active|stale, got {self.status!r}"
            )
        if self.motif_lineage is not None and (
            type(self.motif_lineage) is not str or not self.motif_lineage
        ):
            raise ValueError("motif_lineage must be a non-empty string or None")


@dataclass(frozen=True)
class BiasPrimitive:
    """One D4 sedimented bias: confidence-weighted, trajectory-free.

    Round 11 adds the #23 candidate lineage; it is required non-None by the
    #22 execution gate (physics mode), while the Q4 zero-confidence sentinel
    carries None.
    """

    bias_primitive: str
    confidence: float
    candidate_lineage: str | None = None

    def __post_init__(self) -> None:
        if type(self.bias_primitive) is not str or not self.bias_primitive:
            raise ValueError("bias_primitive must be a non-empty string")
        _require_finite("confidence", self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.candidate_lineage is not None and (
            type(self.candidate_lineage) is not str
            or not self.candidate_lineage
        ):
            raise ValueError(
                "candidate_lineage must be a non-empty string or None"
            )


# ---------------------------------------------------------------------------
# Physics-epoch body records (frozen, hashable)
# ---------------------------------------------------------------------------
def _require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number!r}")
    return number


def _require_vector(name: str, value: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of floats")
    if len(value) != PAYLOAD_LATENT_DIM:
        raise ValueError(
            f"{name} must carry {PAYLOAD_LATENT_DIM} values, got {len(value)}"
        )
    return tuple(_require_finite(name, item) for item in value)


@dataclass(frozen=True)
class FrameBody:
    """#3 E1 -> P1: the perturbation frame that started the tick."""

    frame: bytes
    height: int
    width: int
    tick: int

    def __post_init__(self) -> None:
        if not isinstance(self.frame, bytes):
            raise TypeError("frame must be bytes")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("frame dimensions must be positive")
        if len(self.frame) != self.height * self.width * 3:
            raise ValueError(
                "frame byte length does not match height*width*3: "
                f"{len(self.frame)} != {self.height * self.width * 3}"
            )
        if self.tick < 0:
            raise ValueError("tick must be non-negative")


@dataclass(frozen=True)
class PredictionErrorBody:
    """#6 P1 -> P2: the measured surprise and its signed residual vector."""

    surprise: float
    residual: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_finite("surprise", self.surprise)
        _require_vector("residual", self.residual)


@dataclass(frozen=True)
class LatentRolloutBody:
    """#10 P3 -> P1: the short resolved latent trajectory."""

    latents: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.latents, tuple) or not self.latents:
            raise TypeError("latents must be a non-empty tuple of vectors")
        for row in self.latents:
            _require_vector("rollout row", row)


@dataclass(frozen=True)
class StateDeltaBody:
    """#12 P3 -> P4: the latent state delta and its surprise."""

    delta: tuple[float, ...]
    surprise: float

    def __post_init__(self) -> None:
        _require_vector("delta", self.delta)
        _require_finite("surprise", self.surprise)


@dataclass(frozen=True)
class BoardOpBody:
    """#13 P4 -> D1: one board operation (write/erase/resize/evict).

    The body is the *operation*, not the context content: ``body_id`` is a
    content-addressed digest of the ``#12`` delta that the write represents,
    and ``nbytes`` is ``8 * len(delta)`` (represented bytes, computed at the
    regulator, never a literal).  For ``resize``, ``nbytes`` is the new slot
    capacity; for ``erase``/``evict`` it is 0 and ``body_id`` is ``None``.
    """

    op: str
    slot_id: str
    body_id: str | None
    nbytes: int
    reason: str
    tick: int

    def __post_init__(self) -> None:
        if self.op not in BOARD_OP_VERBS:
            raise ValueError(
                f"op must be one of {BOARD_OP_VERBS}, got {self.op!r}"
            )
        if type(self.slot_id) is not str or not self.slot_id:
            raise ValueError("slot_id must be a non-empty string")
        if self.body_id is not None and (
            type(self.body_id) is not str or not self.body_id
        ):
            raise ValueError("body_id must be a non-empty string or None")
        if type(self.nbytes) is not int or isinstance(self.nbytes, bool):
            raise TypeError(f"nbytes must be an int, got {self.nbytes!r}")
        if self.nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if type(self.tick) is not int or isinstance(self.tick, bool):
            raise TypeError(f"tick must be an int, got {self.tick!r}")
        if self.tick < 0:
            raise ValueError("tick must be non-negative")
        if self.op == "write":
            if self.body_id is None or self.nbytes <= 0:
                raise ValueError("write requires body_id and nbytes > 0")
        elif self.op == "resize":
            if self.body_id is not None or self.nbytes <= 0:
                raise ValueError("resize requires body_id=None and nbytes > 0")
        else:  # erase / evict
            if self.body_id is not None or self.nbytes != 0:
                raise ValueError("erase/evict require body_id=None and nbytes=0")


@dataclass(frozen=True)
class SlotStateBody:
    """#14 D1 -> P4: the board's slot identity and occupancy snapshot.

    No content bytes are retained by this slice (Lead adjudication Q2); the
    body is a pure state snapshot (no tick field) so the executor read is
    derived from D1, not from the clock.
    """

    capacity: int
    occupied: int
    nbytes: int
    slot_ids: tuple[str, ...]
    oldest_slot_id: str | None

    def __post_init__(self) -> None:
        for name in ("capacity", "occupied", "nbytes"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not isinstance(self.slot_ids, tuple):
            raise TypeError("slot_ids must be a tuple")
        for slot_id in self.slot_ids:
            if type(slot_id) is not str or not slot_id:
                raise ValueError("slot_ids entries must be non-empty strings")
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("slot_ids must be unique")
        if self.occupied != len(self.slot_ids):
            raise ValueError("occupied must equal len(slot_ids)")
        if self.occupied > self.capacity:
            raise ValueError("occupied exceeds capacity")
        expected_oldest = self.slot_ids[0] if self.slot_ids else None
        if self.oldest_slot_id != expected_oldest:
            raise ValueError(
                "oldest_slot_id must be slot_ids[0] or None if empty"
            )


@dataclass(frozen=True)
class CostErrorBody:
    """#16 P4 -> P5: allocation cost against digest error.

    ``cost`` is the *represented* bytes D1 retains a handle to
    (``sum(slot.nbytes)``, i.e. ``8 * len(delta)`` per written record), not
    the store's own metadata footprint (Lead adjudication Q5).
    """

    cost: float
    error: float

    def __post_init__(self) -> None:
        _require_finite("cost", self.cost)
        _require_finite("error", self.error)


@dataclass(frozen=True)
class CollapseMandateBody:
    """#17 P5 -> P4: ceiling pressure forced a collapse."""

    reason: str
    pressure: int
    capacity: int
    active_bodies: int
    freed_body: str | None
    tick: int

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        for name in ("pressure", "capacity", "active_bodies", "tick"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.freed_body is not None and not isinstance(self.freed_body, str):
            raise TypeError("freed_body must be a string or None")


@dataclass(frozen=True)
class EmergentCrystalBody:
    """#18 P3 -> D2: the nursery candidate kind (Round 11 ruling A).

    This is the *only* body type #18 may carry and the only type stored in
    the candidate index.  ``born_tick`` must equal the #18 emission tick;
    there is no ``candidate`` status on ``MatOverlay`` (candidates are a
    separate kind, not a tagged overlay stream).
    """

    candidate_id: str
    residual: tuple[float, ...]
    surprise: float
    born_tick: int

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        _require_vector("residual", self.residual)
        _require_finite("surprise", self.surprise)
        if type(self.born_tick) is not int or isinstance(self.born_tick, bool):
            raise TypeError(f"born_tick must be an int, got {self.born_tick!r}")
        if self.born_tick < 0:
            raise ValueError("born_tick must be non-negative")


@dataclass(frozen=True)
class SurvivingCrystalBody:
    """#23 D2 -> P6: a candidate surfaced by its residence evidence.

    ``qualified`` is the read-time selection (`ratio < 1` and
    ``residence_ticks >= RESIDENCE_N``); there is no stored "surviving"
    kind and no third overlay status.
    """

    candidate_id: str
    born_tick: int
    residence_ticks: int
    adapted_mean: float
    base_mean: float
    ratio: float
    qualified: bool

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        for name in ("born_tick", "residence_ticks"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("adapted_mean", "base_mean", "ratio"):
            _require_finite(name, getattr(self, name))
        if self.base_mean <= 0.0:
            raise ValueError("base_mean must be positive")
        if type(self.qualified) is not bool:
            raise TypeError("qualified must be a bool")
        expected = self.ratio < 1.0 and self.residence_ticks >= RESIDENCE_N
        if self.qualified is not expected:
            raise ValueError(
                "qualified must be exactly (ratio < 1 and residence_ticks "
                ">= RESIDENCE_N)"
            )


@dataclass(frozen=True)
class DebrisCohort:
    """One D3 cohort: tagged material class with depositing provenance.

    ``provenance`` is validated structurally (it must expose
    ``source_fid``/``tick_born``) so ``payloads`` never imports
    ``provenance`` at module scope; the executor passes a real
    ``Provenance``.
    """

    cohort_id: str
    kind: str
    centroid: tuple[float, ...]
    count: int
    spread: float
    first_tick: int
    last_tick: int
    peak_surprise: float
    provenance: object

    def __post_init__(self) -> None:
        if type(self.cohort_id) is not str or not self.cohort_id:
            raise ValueError("cohort_id must be a non-empty string")
        if self.kind not in COHORT_KINDS:
            raise ValueError(
                f"kind must be one of {COHORT_KINDS}, got {self.kind!r}"
            )
        _require_vector("centroid", self.centroid)
        if type(self.count) is not int or isinstance(self.count, bool):
            raise TypeError(f"count must be an int, got {self.count!r}")
        if self.count < 0:
            raise ValueError("count must be non-negative")
        _require_finite("spread", self.spread)
        for name in ("first_tick", "last_tick"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        _require_finite("peak_surprise", self.peak_surprise)
        _require_provenance_shape(self.provenance)


def _require_provenance_shape(value: object) -> None:
    """Structural provenance check (avoids payloads <-> provenance cycle)."""
    source_fid = getattr(value, "source_fid", None)
    tick_born = getattr(value, "tick_born", None)
    if type(source_fid) is not int or isinstance(source_fid, bool):
        raise TypeError("provenance.source_fid must be an int")
    if source_fid not in {flow.fid for flow in FLOW_LEDGER}:
        raise ValueError("provenance.source_fid is not a ledger flow id")
    if type(tick_born) is not int or isinstance(tick_born, bool):
        raise TypeError("provenance.tick_born must be an int")
    if tick_born < 0:
        raise ValueError("provenance.tick_born must be non-negative")


@dataclass(frozen=True)
class DigestedDebrisBody:
    """#19 P5 -> D3: a tagged composite of cohort classes (Round 11 F)."""

    debris_id: str
    cohorts: tuple[DebrisCohort, ...]

    def __post_init__(self) -> None:
        if type(self.debris_id) is not str or not self.debris_id:
            raise ValueError("debris_id must be a non-empty string")
        if not isinstance(self.cohorts, tuple):
            raise TypeError("cohorts must be a tuple of DebrisCohort")
        for cohort in self.cohorts:
            if type(cohort) is not DebrisCohort:
                raise TypeError(
                    "cohorts entries must be exact DebrisCohort bodies"
                )
        ids = tuple(cohort.cohort_id for cohort in self.cohorts)
        if len(set(ids)) != len(ids):
            raise ValueError("cohort ids must be unique within one body")


@dataclass(frozen=True)
class ScaffoldsBody:
    """#20 D3 -> P4: scaffold cohorts only (per-edge read filter)."""

    cohorts: tuple[DebrisCohort, ...]
    no_event: bool = False

    def __post_init__(self) -> None:
        _validate_filtered_cohorts(self.cohorts, "scaffold", self.no_event)


@dataclass(frozen=True)
class StructuralMotifBody:
    """#21 D3 -> P6: the motif cohort read from the debris pool."""

    motif_id: str
    debris_id: str
    centroid: tuple[float, ...]
    count: int
    spread: float
    persistence: int
    first_tick: int
    last_tick: int

    def __post_init__(self) -> None:
        for name in ("motif_id", "debris_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        _require_vector("centroid", self.centroid)
        for name in ("count", "persistence"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        _require_finite("spread", self.spread)
        for name in ("first_tick", "last_tick"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class AvoidanceTrailsBody:
    """#26 D3 -> P2: trail cohorts only (per-edge read filter)."""

    cohorts: tuple[DebrisCohort, ...]
    no_event: bool = False

    def __post_init__(self) -> None:
        _validate_filtered_cohorts(self.cohorts, "trail", self.no_event)


@dataclass(frozen=True)
class DissipatedEntropyBody:
    """#27 D3 -> P1: residual + expired cohorts, drained destructively."""

    cohort_ids: tuple[str, ...]
    magnitudes: tuple[float, ...]
    tick: int

    def __post_init__(self) -> None:
        if not isinstance(self.cohort_ids, tuple):
            raise TypeError("cohort_ids must be a tuple")
        if not isinstance(self.magnitudes, tuple):
            raise TypeError("magnitudes must be a tuple")
        for cohort_id in self.cohort_ids:
            if type(cohort_id) is not str or not cohort_id:
                raise ValueError("cohort_ids entries must be non-empty strings")
        if len(set(self.cohort_ids)) != len(self.cohort_ids):
            raise ValueError("cohort_ids must be unique")
        if len(self.magnitudes) != len(self.cohort_ids):
            raise ValueError("magnitudes must match cohort_ids length")
        for magnitude in self.magnitudes:
            _require_finite("magnitude", magnitude)
        if type(self.tick) is not int or isinstance(self.tick, bool):
            raise TypeError(f"tick must be an int, got {self.tick!r}")
        if self.tick < 0:
            raise ValueError("tick must be non-negative")


def _validate_filtered_cohorts(
    cohorts: tuple[DebrisCohort, ...], kind: str, no_event: bool
) -> None:
    if not isinstance(cohorts, tuple):
        raise TypeError("cohorts must be a tuple of DebrisCohort")
    for cohort in cohorts:
        if type(cohort) is not DebrisCohort:
            raise TypeError("cohorts entries must be exact DebrisCohort bodies")
        if cohort.kind != kind:
            raise ValueError(
                f"this body carries {kind} cohorts only, got {cohort.kind!r}"
            )
    if type(no_event) is not bool:
        raise TypeError("no_event must be a bool")
    if no_event and cohorts:
        raise ValueError("a no-event body must carry no cohorts")


# ---------------------------------------------------------------------------
# Fixed-cadence no-op sentinels (exact-type, content-free)
# ---------------------------------------------------------------------------
def noop_crystal() -> EmergentCrystalBody:
    """The #18 no-event candidate body."""
    return EmergentCrystalBody(
        candidate_id=NO_CANDIDATE_ID,
        residual=(0.0,) * PAYLOAD_LATENT_DIM,
        surprise=0.0,
        born_tick=0,
    )


def noop_survivor() -> SurvivingCrystalBody:
    """The #23 no-event survivor body (not qualified)."""
    return SurvivingCrystalBody(
        candidate_id=NO_CANDIDATE_ID,
        born_tick=0,
        residence_ticks=0,
        adapted_mean=0.0,
        base_mean=1.0,
        ratio=0.0,
        qualified=False,
    )


def noop_debris() -> DigestedDebrisBody:
    """The #19 no-event body: a real body the D3 pool absorbs as a no-op."""
    return DigestedDebrisBody(debris_id=NO_DEBRIS_ID, cohorts=())


def noop_motif() -> StructuralMotifBody:
    """The #21 no-event body: a real body P6 consumes as a no-op."""
    return StructuralMotifBody(
        motif_id=NO_MOTIF_ID,
        debris_id=NO_DEBRIS_ID,
        centroid=(0.0,) * PAYLOAD_LATENT_DIM,
        count=0,
        spread=0.0,
        persistence=0,
        first_tick=0,
        last_tick=0,
    )


def noop_scaffolds() -> ScaffoldsBody:
    """The #20 no-event body: an empty scaffold cohort read."""
    return ScaffoldsBody(cohorts=(), no_event=True)


def noop_trails() -> AvoidanceTrailsBody:
    """The #26 no-event body: an empty trail cohort read."""
    return AvoidanceTrailsBody(cohorts=(), no_event=True)


def noop_entropy() -> DissipatedEntropyBody:
    """The #27 no-event body: nothing drainable."""
    return DissipatedEntropyBody(cohort_ids=(), magnitudes=(), tick=0)


def noop_mandate() -> CollapseMandateBody:
    """The #17 no-event body: a real body P4 consumes as a no-op."""
    return CollapseMandateBody(
        reason=NO_MANDATE_REASON,
        pressure=0,
        capacity=0,
        active_bodies=0,
        freed_body=None,
        tick=0,
    )


def noop_overlay() -> MatOverlay:
    """The #24 no-event token: a real MatOverlay the mat ignores."""
    return MatOverlay(0, NOOP_OVERLAY_ID, "stale")


def noop_bias() -> BiasPrimitive:
    """The #7/#22 no-event bias sentinel (Q4 zero confidence)."""
    return BiasPrimitive("ratchet-gate-blocked", 0.0, None)


# ---------------------------------------------------------------------------
# Flow bookkeeping and schema
# ---------------------------------------------------------------------------
def _store_flows(store: str) -> tuple[int, ...]:
    return tuple(
        f.fid
        for f in FLOW_LEDGER
        if f.target == store or f.source == store
    )


def _write_fids_of(store: str) -> tuple[int, ...]:
    return tuple(
        f.fid
        for f in FLOW_LEDGER
        if f.target == store and "(read)" not in f.flow
    )


D2_FLOWS: frozenset[int] = frozenset(_store_flows(_D2))
D4_FLOWS: frozenset[int] = frozenset(_store_flows(_D4))
D2_WRITES: frozenset[int] = frozenset(_write_fids_of(_D2))
D4_WRITES: frozenset[int] = frozenset(_write_fids_of(_D4))

#: Every flow whose source is a store (the executor derives these reads).
STORE_READ_FLOWS: frozenset[int] = frozenset(
    f.fid for f in FLOW_LEDGER if f.source in STORES
)

#: Every flow that is not a store read (the payload source supplies these).
NON_READ_FLOWS: frozenset[int] = frozenset(
    f.fid for f in FLOW_LEDGER if f.fid not in STORE_READ_FLOWS
)

_PLACEHOLDER = frozenset({type(None)})

#: Round 11 §1.1 (corrected): WRITE channels are exact-type with no None —
#: a promotion bug becomes untypeable, not merely rejected.  READ channels
#: allow the exact type OR None ("the store had nothing to hand over this
#: tick"); None keeps the A2 provenance rule intact (a body read must carry
#: the stored record's provenance with tick_born < emit_tick; None carries
#: self-origin (fid, tick)).  A sentinel body on a read would force a
#: sentinel-provenance exception and weaken the anti-mint gate.
#: D1 is frozen: #13/#14 keep their prior optional shapes and #11/#15 stay
#: None placeholders until #13 lands (contract §2.1).
WRITE_BODY: dict[int, type] = {
    18: EmergentCrystalBody,   # D2 candidate write
    19: DigestedDebrisBody,    # D3 deposit
    22: BiasPrimitive,         # D4 sediment
    24: MatOverlay,            # D2 ratcheted overlay write
}

READ_BODY: dict[int, type] = {
    7: BiasPrimitive,          # D4 read
    8: MatOverlay,             # D2 overlay read (active ratcheted only)
    16: CostErrorBody,         # derived this round
    20: ScaffoldsBody,         # D3 scaffold read
    21: StructuralMotifBody,   # D3 motif read
    23: SurvivingCrystalBody,  # D2 survivor read
    25: MatOverlay,            # D2 stale read
    26: AvoidanceTrailsBody,   # D3 trail read
    27: DissipatedEntropyBody, # D3 destructive drain read
}

#: Non-store real-flow bodies (None remains allowed on these channels so the
#: synthetic plumbing source and the not-yet-derived flows keep working).
REAL_BODIES: dict[int, type] = {
    3: FrameBody,
    6: PredictionErrorBody,
    10: LatentRolloutBody,
    12: StateDeltaBody,
    13: BoardOpBody,
    14: SlotStateBody,
    16: CostErrorBody,
    17: CollapseMandateBody,
    18: EmergentCrystalBody,
    19: DigestedDebrisBody,
    20: ScaffoldsBody,
    21: StructuralMotifBody,
    23: SurvivingCrystalBody,
    26: AvoidanceTrailsBody,
    27: DissipatedEntropyBody,
}


def _schema() -> dict[int, frozenset[type]]:
    schema: dict[int, frozenset[type]] = {}
    for flow in FLOW_LEDGER:
        fid = flow.fid
        if fid in WRITE_BODY:
            schema[fid] = frozenset({WRITE_BODY[fid]})
        elif fid in READ_BODY:
            schema[fid] = frozenset({READ_BODY[fid], type(None)})
        elif fid in REAL_BODIES:
            schema[fid] = frozenset({REAL_BODIES[fid], type(None)})
        else:
            schema[fid] = _PLACEHOLDER
    return schema


PAYLOAD_SCHEMA: dict[int, frozenset[type]] = _schema()

MAT_CAPACITY = 4
MAT_SEED = ("ovr-0", 0)


def _patch_attr(patch: object, name: str) -> int:
    value = getattr(patch, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(
            f"patch.{name} must be a non-negative int, got {value!r}"
        )
    return value


class MatBoard:
    """D2 lifecycle, mirrors the formal matTick policy exactly.

    Per tick: one birth (append, from surviving crystals), one
    revalidation (head popped and re-appended at the new epoch, from the
    stale read), then one eviction iff the board exceeds MAT_CAPACITY.
    Seeded with a single epoch-0 overlay so the #25 -> #24 pairing has a
    candidate from tick 1.

    Phase-4/5b extension (Q1-A): the board also holds the predictor patch
    keyed by ``overlay_id``.  ``attach_patch`` / ``patch`` /
    ``patches`` / ``patch_bytes`` account for substance; ``evict_if_needed``
    and ``enforce_patch_ceiling`` remove the record and its patch in one
    operation.  The **record** transition semantics above are unchanged
    (the Lean mirror bounds records only); the patch is extra Python-side
    state, asserted by ``assert_patch_bounds`` and marked
    *tested-not-proven* against the Lean bounds.

    Cadence honesty: the one-birth-per-tick minting schedule is dry-run
    plumbing; in the physics epoch #18/#24 are organ-owned and sentinel
    no-ops fill the fixed cadence. The Lean theorems that consume this
    policy quantify over every stream obeying the same update law
    (`mat_*_policy`), so a different schedule is a different law, never a
    silent assumption.
    """

    def __init__(
        self,
        *,
        seed: bool = True,
        max_patch_bytes: int = MAX_PATCH_BYTES,
        max_patch_total_bytes: int = MAX_PATCH_TOTAL_BYTES,
    ) -> None:
        if max_patch_bytes < 0 or max_patch_total_bytes < 0:
            raise ValueError("patch byte ceilings must be non-negative")
        if max_patch_bytes > max_patch_total_bytes:
            raise ValueError("per-patch ceiling cannot exceed the table ceiling")
        self._residents: deque[tuple[str, int]] = deque([MAT_SEED]) if seed else deque()
        self._patches: dict[str, object] = {}
        self.max_patch_bytes = int(max_patch_bytes)
        self.max_patch_total_bytes = int(max_patch_total_bytes)
        self.evictions: list[tuple[str, int, int]] = []
        #: Byte-ceiling evictions (a subset of ``evictions``), for tests.
        self.byte_evictions: list[tuple[str, int, int]] = []
        self.d4: list[BiasPrimitive] = []

        # Round 11 §1.9: the candidate index is a separate nursery, not a
        # tagged overlay stream.  The overlay index above is untouched.
        self._nursery: deque[tuple[str, int]] = deque()
        self._nursery_bodies: dict[str, EmergentCrystalBody] = {}
        self._nursery_seen: set[str] = set()
        self._candidate_patches: dict[str, object] = {}
        self._evidence: dict[str, list[tuple[int, float, float]]] = {}
        self._nursery_evictions: list[tuple[str, int, int, str]] = []
        self._nursery_promotions: list[tuple[str, int]] = []

    # ------------------------------------------------------------- D4 core
    def sediment(self, bias: BiasPrimitive) -> None:
        require_payload(22, bias)
        self.d4.append(bias)

    def latest_bias(self) -> BiasPrimitive | None:
        return self.d4[-1] if self.d4 else None

    # ---------------------------------------------------------- D2 records
    def head(self) -> tuple[str, int]:
        return self._residents[0]

    def birth(self, overlay: MatOverlay) -> None:
        require_payload(24, overlay)
        if overlay.overlay_id in self.resident_ids():
            raise ValueError(
                f"duplicate D2 birth for overlay_id {overlay.overlay_id!r}; "
                "a second record would share one patch entry"
            )
        self._residents.append((overlay.overlay_id, overlay.epoch))

    def revalidate(self, overlay: MatOverlay) -> None:
        require_payload(24, overlay)
        if not self._residents:
            raise RuntimeError(
                f"revalidation of {overlay.overlay_id!r} on an empty board"
            )
        head_id, _epoch = self._residents[0]
        if head_id != overlay.overlay_id:
            raise RuntimeError(
                f"revalidation pairing broken: head {head_id!r}, "
                f"write {overlay.overlay_id!r}"
            )
        self._residents.popleft()
        self._residents.append((overlay.overlay_id, overlay.epoch))

    def _remove(self, overlay_id: str, tick: int, *, cause: str) -> tuple[str, int] | None:
        for index, (resident_id, epoch) in enumerate(self._residents):
            if resident_id == overlay_id:
                del self._residents[index]
                self._patches.pop(overlay_id, None)
                self.evictions.append((overlay_id, epoch, tick))
                if cause == "patch-bytes":
                    self.byte_evictions.append((overlay_id, epoch, tick))
                return (overlay_id, epoch)
        return None

    def evict_if_needed(self, tick: int) -> None:
        while len(self._residents) > MAT_CAPACITY:
            overlay_id, _epoch = self._residents[0]
            self._remove(overlay_id, tick, cause="records")

    def evict_record(self, overlay_id: str, tick: int) -> tuple[str, int] | None:
        """Remove one resident record and its patch together (Q1-A).

        Used by the physics-mode one-way prune when the organ has evicted an
        overlay: the store must not retain a record whose patch owner is
        gone.  The normal ``evict_if_needed`` capacity policy is unchanged;
        this is the organ-owned lifecycle path, recorded in ``evictions``.
        """
        return self._remove(overlay_id, tick, cause="organ")

    def residents(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._residents)

    def resident_ids(self) -> frozenset[str]:
        return frozenset(overlay_id for overlay_id, _epoch in self._residents)

    # --------------------------------------------------- D2 candidate index
    def nursery_birth(self, body: EmergentCrystalBody, tick: int) -> None:
        """Append one #18 candidate; duplicate/used ids are refused."""
        require_payload(18, body)
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        if body.born_tick != tick:
            raise ValueError(
                f"candidate {body.candidate_id!r} born_tick {body.born_tick} "
                f"!= #18 emission tick {tick}"
            )
        if body.candidate_id in self._nursery_seen:
            raise ValueError(
                f"duplicate/resurrected candidate_id {body.candidate_id!r}"
            )
        self._nursery.append((body.candidate_id, body.born_tick))
        self._nursery_bodies[body.candidate_id] = body
        self._evidence[body.candidate_id] = []
        self._nursery_seen.add(body.candidate_id)
        while len(self._nursery) > NURSERY_CAPACITY:
            evicted_id, evicted_born = self._nursery.popleft()
            self._nursery_evictions.append(
                (evicted_id, evicted_born, tick, "capacity")
            )
            self._nursery_bodies.pop(evicted_id, None)
            self._evidence.pop(evicted_id, None)
            self._candidate_patches.pop(evicted_id, None)

    def nursery_observe(
        self, candidate_id: str, adapted: float, base: float, tick: int
    ) -> None:
        """Append one shadow measurement to a resident candidate."""
        evidence = self._evidence.get(candidate_id)
        if evidence is None or candidate_id not in self.nursery_ids():
            raise ValueError(f"unknown candidate_id {candidate_id!r}")
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        _require_finite("adapted", adapted)
        _require_finite("base", base)
        born_tick = self._nursery_bodies[candidate_id].born_tick
        if tick < born_tick:
            raise ValueError(
                f"shadow tick {tick} precedes candidate birth {born_tick}"
            )
        if evidence and tick <= evidence[-1][0]:
            raise ValueError(
                "shadow observations must have strictly increasing ticks"
            )
        evidence.append((tick, float(adapted), float(base)))

    def nursery_records(self) -> tuple[EmergentCrystalBody, ...]:
        """Resident candidate bodies, oldest-first."""
        return tuple(
            self._nursery_bodies[candidate_id]
            for candidate_id, _born in self._nursery
        )

    def nursery_evidence(
        self, candidate_id: str
    ) -> tuple[tuple[int, float, float], ...]:
        if candidate_id not in self._evidence:
            raise ValueError(f"unknown candidate_id {candidate_id!r}")
        return tuple(self._evidence[candidate_id])

    def nursery_survivor(self, now: int) -> SurvivingCrystalBody | None:
        """Surface the lowest-ratio resident with enough residence evidence.

        Correction (contract §1.9): this selection **must not pre-filter on
        ``qualified``**.  If #23 only ever surfaced ``ratio < 1`` candidates,
        #22's gate and the ADMISSION metric would be vacuous; the harmful
        survivor has to remain observable as surfaced-and-refused.  Among
        residents with ``residence_ticks >= RESIDENCE_N`` the lowest ratio
        wins (tie-break oldest birth, then FIFO order), regardless of
        ``ratio``; ``qualified = (ratio < 1.0)`` is carried as a flag.
        ``None`` is returned iff no resident has enough residence.
        ``residence_ticks`` is the number of shadow-scored observations
        (contract §1.3), not wall-clock age; ``now`` is validated and a
        candidate born after it is not considered.
        """
        if type(now) is not int or isinstance(now, bool) or now < 0:
            raise ValueError(f"now must be a non-negative int, got {now!r}")
        best: tuple[float, int, int] | None = None
        best_body: SurvivingCrystalBody | None = None
        for index, (candidate_id, born_tick) in enumerate(self._nursery):
            if born_tick > now:
                continue
            evidence = self._evidence[candidate_id]
            residence_ticks = len(evidence)
            if residence_ticks < RESIDENCE_N:
                continue
            adapted_mean = sum(item[1] for item in evidence) / residence_ticks
            base_mean = sum(item[2] for item in evidence) / residence_ticks
            if base_mean <= 0.0:
                continue
            ratio = adapted_mean / base_mean
            key = (ratio, born_tick, index)
            if best is None or key < best:
                best = key
                best_body = SurvivingCrystalBody(
                    candidate_id=candidate_id,
                    born_tick=born_tick,
                    residence_ticks=residence_ticks,
                    adapted_mean=adapted_mean,
                    base_mean=base_mean,
                    ratio=ratio,
                    qualified=ratio < 1.0,
                )
        return best_body

    def nursery_promote(self, candidate_id: str, tick: int) -> EmergentCrystalBody:
        """Remove a candidate and its patch atomically (INV-PROMOTE-ONCE)."""
        if candidate_id not in self.nursery_ids():
            raise ValueError(
                f"cannot promote {candidate_id!r}: unknown or already promoted"
            )
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        body = self._nursery_bodies.pop(candidate_id)
        for index, (resident_id, born_tick) in enumerate(self._nursery):
            if resident_id == candidate_id:
                del self._nursery[index]
                break
        self._candidate_patches.pop(candidate_id, None)
        self._nursery_promotions.append((candidate_id, tick))
        return body

    def nursery_evict_record(
        self, candidate_id: str, tick: int, cause: str
    ) -> EmergentCrystalBody:
        """Remove one candidate record and its patch (organ prune path)."""
        if candidate_id not in self.nursery_ids():
            raise ValueError(f"cannot evict unknown candidate {candidate_id!r}")
        if type(cause) is not str or not cause:
            raise ValueError("cause must be a non-empty string")
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        body = self._nursery_bodies.pop(candidate_id)
        born_tick = body.born_tick
        for index, (resident_id, _born) in enumerate(self._nursery):
            if resident_id == candidate_id:
                del self._nursery[index]
                break
        self._candidate_patches.pop(candidate_id, None)
        self._nursery_evictions.append((candidate_id, born_tick, tick, cause))
        return body

    def nursery_ids(self) -> tuple[str, ...]:
        return tuple(candidate_id for candidate_id, _born in self._nursery)

    def nursery_patch(self, candidate_id: str) -> object | None:
        return self._candidate_patches.get(candidate_id)

    def nursery_patch_bytes(self) -> int:
        return sum(
            _patch_attr(patch, "nbytes")
            for patch in self._candidate_patches.values()
        )

    def attach_candidate_patch(
        self, candidate_id: str, patch: object
    ) -> None:
        """Attach a candidate's substance patch (E); never live steering."""
        if candidate_id not in self.nursery_ids():
            raise AssertionError(
                f"cannot attach a patch to non-resident candidate "
                f"{candidate_id!r}"
            )
        patch_id = getattr(patch, "overlay_id", None)
        if patch_id is None:
            patch_id = getattr(patch, "candidate_id", None)
        if patch_id != candidate_id:
            raise ValueError(
                f"patch id {patch_id!r} does not match candidate "
                f"{candidate_id!r}"
            )
        nbytes = _patch_attr(patch, "nbytes")
        if nbytes > self.max_patch_bytes:
            raise ValueError(
                f"candidate patch {candidate_id!r} is {nbytes} bytes, over "
                f"the per-patch ceiling {self.max_patch_bytes}"
            )
        _patch_attr(patch, "param_count")
        self._candidate_patches[candidate_id] = patch

    # ----------------------------------------------------------- D2 patches
    def attach_patch(self, overlay_id: str, patch: object) -> None:
        """Attach the predictor patch for a resident record (Q1-A)."""
        if overlay_id not in self.resident_ids():
            raise AssertionError(
                f"cannot attach a patch to non-resident overlay {overlay_id!r}"
            )
        patch_id = getattr(patch, "overlay_id", None)
        if patch_id != overlay_id:
            raise ValueError(
                f"patch overlay_id {patch_id!r} does not match {overlay_id!r}"
            )
        nbytes = _patch_attr(patch, "nbytes")
        if nbytes > self.max_patch_bytes:
            raise ValueError(
                f"patch {overlay_id!r} is {nbytes} bytes, over the per-patch "
                f"ceiling {self.max_patch_bytes}"
            )
        _patch_attr(patch, "param_count")
        self._patches[overlay_id] = patch

    def patch(self, overlay_id: str) -> object | None:
        return self._patches.get(overlay_id)

    def patches(self) -> tuple[tuple[str, object], ...]:
        return tuple(self._patches.items())

    def patch_bytes(self) -> int:
        """Total D2 substance: overlay patches + candidate patches (E)."""
        return (
            sum(_patch_attr(patch, "nbytes") for patch in self._patches.values())
            + sum(
                _patch_attr(patch, "nbytes")
                for patch in self._candidate_patches.values()
            )
        )

    def patch_params(self) -> int:
        return (
            sum(
                _patch_attr(patch, "param_count")
                for patch in self._patches.values()
            )
            + sum(
                _patch_attr(patch, "param_count")
                for patch in self._candidate_patches.values()
            )
        )

    def enforce_patch_ceiling(self, tick: int) -> None:
        """Evict patched residents until the combined D2 byte bound holds.

        Overlay victims are removed first (their record+patch semantics are
        unchanged); candidate victims are used only when no overlay patch
        remains.  The total bound covers both indices (E).
        """
        while self.patch_bytes() > self.max_patch_total_bytes:
            victims = [
                overlay_id
                for overlay_id, _epoch in self._residents
                if overlay_id in self._patches
            ]
            if victims:
                self._remove(victims[0], tick, cause="patch-bytes")
                continue
            candidates = [
                candidate_id
                for candidate_id in self.nursery_ids()
                if candidate_id in self._candidate_patches
            ]
            if not candidates:
                break
            self.nursery_evict_record(candidates[0], tick, "patch-bytes")

    def assert_patch_bounds(self, *, require_complete: bool = False) -> None:
        """Python-asserted substance bound (tested-not-proven; Lean: records).

        Raises ``AssertionError`` when the patch table and the resident
        records are out of step, a patch exceeds the per-patch bound, the
        total exceeds the table bound, or (physics mode) a resident record
        has no patch.
        """
        residents = self.resident_ids()
        patch_ids = frozenset(self._patches)
        candidates = frozenset(self.nursery_ids())
        candidate_patch_ids = frozenset(self._candidate_patches)
        assert patch_ids <= residents, (
            f"orphan overlay patches {sorted(patch_ids - residents)} "
            "have no resident"
        )
        assert candidate_patch_ids <= candidates, (
            f"orphan candidate patches "
            f"{sorted(candidate_patch_ids - candidates)} have no resident"
        )
        assert len(residents) <= MAT_CAPACITY, "record capacity exceeded"
        assert len(candidates) <= NURSERY_CAPACITY, "nursery capacity exceeded"
        assert self.patch_bytes() <= self.max_patch_total_bytes, (
            f"patch table {self.patch_bytes()} > {self.max_patch_total_bytes} bytes"
        )
        for table in (self._patches, self._candidate_patches):
            for owner_id, patch in table.items():
                nbytes = _patch_attr(patch, "nbytes")
                assert nbytes <= self.max_patch_bytes, (
                    f"patch {owner_id!r} {nbytes} > "
                    f"{self.max_patch_bytes} bytes"
                )
        if require_complete:
            seed_exempt = {MAT_SEED[0]} if MAT_SEED[0] in residents else set()
            missing_overlays = sorted(residents - patch_ids - seed_exempt)
            missing_candidates = sorted(candidates - candidate_patch_ids)
            assert not missing_overlays, (
                f"resident overlays without a patch: {missing_overlays}"
            )
            assert not missing_candidates, (
                f"resident candidates without a patch: {missing_candidates}"
            )


# ---------------------------------------------------------------------------
# D3 debris pool (Round 11: cohort-level, tagged, destructive #27 drain)
# ---------------------------------------------------------------------------
class DebrisCeilingExceeded(RuntimeError):
    """The D3 byte ceiling would be exceeded without an allowed eviction."""


def _cohort_bytes(cohort: DebrisCohort) -> int:
    """Deterministic represented size of one cohort (no provenance bytes)."""
    return (
        len(cohort.centroid) * 8
        + 8 * 5  # count, spread, first_tick, last_tick, peak_surprise
        + len(cohort.cohort_id.encode("utf-8"))
        + len(cohort.kind.encode("utf-8"))
    )


class DebrisPool:
    """D3 Debris & Scaffolding Pool: cohort-level store (F).

    Storage is cohort-level, not body-level: ``deposit`` splits the #19
    composite into `DebrisCohort` entries, each retaining its depositing
    provenance.  Reads are **non-destructive** and per-kind; #27 is the one
    destructive drain (``drain``).  Capacity overflow evicts FIFO; byte
    overflow raises `DebrisCeilingExceeded` unless the caller allows the
    recorded eviction backstop.  Two accounts are kept, both named and never
    substituted for each other (prereg §7): :meth:`conservation_account`
    asserts ``deposits == resident + drained + evicted`` (contract §0/§1.8)
    with reads explicitly excluded, and :meth:`issuance_account` reports the
    ruling's intended cohort-issuance partition.  ``mass_balance()`` carries
    both plus the Round-11 flat keys ``run.py`` reads.
    """

    def __init__(
        self,
        capacity: int = DEBRIS_CAPACITY,
        max_bytes: int = DEBRIS_MAX_BYTES,
    ) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive int")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive int")
        self.capacity = int(capacity)
        self.max_bytes = int(max_bytes)
        self._entries: deque[DebrisCohort] = deque()
        self._deposits = 0
        self._evicted = 0
        self._ceiling_evicted = 0
        self._drained = 0
        self._promoted: set[str] = set()
        self._reads_by_kind: dict[str, int] = {
            kind: 0 for kind in COHORT_KINDS
        }
        self.evictions: list[DebrisCohort] = []
        self.drains: list[DebrisCohort] = []
        self.promotions: list[tuple[str, int]] = []

    # ------------------------------------------------------------- deposit
    def deposit(
        self,
        body: DigestedDebrisBody,
        provenance: object,
        tick: int,
        *,
        allow_ceiling_evict: bool = False,
    ) -> tuple[DebrisCohort, ...]:
        """Deposit a tagged composite; return the cohorts it displaced.

        Capacity overflow (FIFO) is always handled.  Byte-ceiling overflow
        raises `DebrisCeilingExceeded` unless `allow_ceiling_evict=True`, in
        which case oldest cohorts are evicted and counted as
        ``ceiling_evicted``.  The operation is rejected atomically: on a
        raised ceiling no deposit/eviction counter changes.
        """
        if type(body) is not DigestedDebrisBody:
            raise TypeError(
                f"DebrisPool accepts DigestedDebrisBody, got "
                f"{type(body).__name__}"
            )
        if body.debris_id == NO_DEBRIS_ID or not body.cohorts:
            raise ValueError("the fixed-cadence no-op is not deposited")
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        _require_provenance_shape(provenance)
        cohorts = list(body.cohorts)

        combined = list(self._entries) + cohorts
        capacity_evicted: list[DebrisCohort] = []
        while len(combined) > self.capacity:
            capacity_evicted.append(combined.pop(0))

        ceiling_evicted: list[DebrisCohort] = []
        if sum(_cohort_bytes(c) for c in combined) > self.max_bytes:
            if not allow_ceiling_evict:
                raise DebrisCeilingExceeded(
                    f"deposit would make D3 exceed {self.max_bytes} bytes "
                    "(drain or allow the recorded eviction backstop)"
                )
            while (
                combined
                and sum(_cohort_bytes(c) for c in combined) > self.max_bytes
            ):
                ceiling_evicted.append(combined.pop(0))

        self._entries = deque(combined)
        self._deposits += len(cohorts)
        for cohort in capacity_evicted:
            self._record_evicted(cohort, ceiling=False)
        for cohort in ceiling_evicted:
            self._record_evicted(cohort, ceiling=True)
        self._assert_balance()
        return tuple(capacity_evicted + ceiling_evicted)

    def _record_evicted(self, cohort: DebrisCohort, *, ceiling: bool) -> None:
        self.evictions.append(cohort)
        self._evicted += 1
        if ceiling:
            self._ceiling_evicted += 1

    def evict_to_ceiling(self, tick: int) -> tuple[DebrisCohort, ...]:
        """Recorded FIFO backstop when a drained pool is still over ceiling."""
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        evicted: list[DebrisCohort] = []
        while (
            self._entries
            and sum(_cohort_bytes(c) for c in self._entries) > self.max_bytes
        ):
            cohort = self._entries.popleft()
            self._record_evicted(cohort, ceiling=True)
            evicted.append(cohort)
        self._assert_balance()
        return tuple(evicted)

    # --------------------------------------------------------------- reads
    def read(self, kind: str) -> tuple[DebrisCohort, ...]:
        """Non-destructive per-kind read, oldest-first; counts the read."""
        if kind not in COHORT_KINDS:
            raise ValueError(
                f"kind must be one of {COHORT_KINDS}, got {kind!r}"
            )
        self._reads_by_kind[kind] += 1
        return tuple(cohort for cohort in self._entries if cohort.kind == kind)

    def best_motif(self) -> DebrisCohort | None:
        """Highest (count, persistence) motif; oldest-first tie-break.

        `DebrisCohort` has no separate persistence field in the contract
        schema, so persistence is the cohort's observed span,
        ``last_tick - first_tick`` (reported as a contract ambiguity).
        """
        best: DebrisCohort | None = None
        best_key: tuple[int, int] | None = None
        for cohort in self._entries:
            if cohort.kind != "motif":
                continue
            key = (cohort.count, cohort.last_tick - cohort.first_tick)
            if best_key is None or key > best_key:
                best, best_key = cohort, key
        return best

    def drainable(self, now: int) -> tuple[DebrisCohort, ...]:
        """Residual cohorts plus cohorts older than DEBRIS_TTL, oldest-first."""
        if type(now) is not int or isinstance(now, bool) or now < 0:
            raise ValueError(f"now must be a non-negative int, got {now!r}")
        return tuple(
            cohort
            for cohort in self._entries
            if cohort.kind == "residual"
            or now - cohort.last_tick > DEBRIS_TTL
        )

    def drain(self, cohort_ids: tuple[str, ...], tick: int) -> tuple[DebrisCohort, ...]:
        """Destructive #27 removal; unknown/double-drained ids raise."""
        if not isinstance(cohort_ids, tuple):
            raise TypeError("cohort_ids must be a tuple")
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        if len(set(cohort_ids)) != len(cohort_ids):
            raise ValueError("cohort_ids must be unique")
        by_id = {cohort.cohort_id: cohort for cohort in self._entries}
        missing = [cohort_id for cohort_id in cohort_ids if cohort_id not in by_id]
        if missing:
            raise ValueError(f"cannot drain unknown/expired cohorts {missing!r}")
        removed = tuple(by_id[cohort_id] for cohort_id in cohort_ids)
        removed_ids = set(cohort_ids)
        self._entries = deque(
            cohort
            for cohort in self._entries
            if cohort.cohort_id not in removed_ids
        )
        for cohort in removed:
            self.drains.append(cohort)
        self._drained += len(removed)
        self._assert_balance()
        return removed

    def promote(self, cohort_id: str, tick: int) -> None:
        """Mark a resident cohort promoted; INV-PROMOTE-ONCE (non-destructive)."""
        if type(tick) is not int or isinstance(tick, bool) or tick < 0:
            raise ValueError(f"tick must be a non-negative int, got {tick!r}")
        if not any(cohort.cohort_id == cohort_id for cohort in self._entries):
            raise ValueError(f"cannot promote unknown cohort {cohort_id!r}")
        if cohort_id in self._promoted:
            raise ValueError(
                f"INV-PROMOTE-ONCE: cohort {cohort_id!r} was already promoted"
            )
        self._promoted.add(cohort_id)
        self.promotions.append((cohort_id, tick))

    # ------------------------------------------------------------- accounting
    def _assert_balance(self) -> None:
        """The conservation identity, from the records as well as the counters.

        The counters (``_deposits/_drained/_evicted``) and the record lists
        (``_entries/drains/evictions``) must agree, otherwise a derived
        account would disagree with the asserted one.
        """
        resident = len(self._entries)
        drained = len(self.drains)
        evicted = len(self.evictions)
        assert (self._drained, self._evicted) == (drained, evicted), (
            "D3 record lists out of step with their counters: "
            f"drained counter={self._drained} records={drained}, "
            f"evicted counter={self._evicted} records={evicted}"
        )
        assert self._deposits == resident + drained + evicted, (
            "D3 mass balance broken: "
            f"deposits={self._deposits}, resident={resident}, "
            f"drained={drained}, evicted={evicted}"
        )

    def conservation_account(self) -> dict:
        """Canonical conservation identity: `deposits = resident + drained + evicted`.

        Reads are **not** a term.  A non-destructive read (#20/#21/#26/#27 all
        read the same cohort store without removing anything) can never sit on
        the outflow side of a conservation law without double counting; the
        ruling-F form `in = out_reads + drained + delta_resident` therefore
        over-counts by exactly the mass the reads handed out.  The terms below
        are the three states a deposited cohort can be in, each counted once.
        """
        deposits = self._deposits
        resident = len(self._entries)
        drained = len(self.drains)
        evicted = len(self.evictions)
        return {
            "deposits": deposits,
            "resident": resident,
            "drained": drained,
            "evicted": evicted,
            "ceiling_evicted": self._ceiling_evicted,
            "identity": "deposits = resident + drained + evicted",
            "holds": deposits == resident + drained + evicted,
        }

    def issuance_account(self) -> dict:
        """Ruling-F issuance account: every deposit issued to exactly one fate.

        The four terms **partition** ``deposits``, so the account closes:

        * ``promoted`` -- cohorts whose id was ever promoted for a consumer
          (INV-PROMOTE-ONCE makes this monotone: a cohort is counted once
          here whatever its later fate, so it is not also counted below);
        * ``resident_unpromoted`` -- still resident, not yet promoted;
        * ``drained_unpromoted`` -- removed by the destructive #27 drain
          before promotion;
        * ``evicted`` -- removed by capacity FIFO or the byte-ceiling backstop
          before promotion.

        The ``evicted`` term is deliberately the *unpromoted* evictions.
        ``_evicted`` counts **every** eviction and ``_ceiling_evicted`` is a
        subset of it (``_record_evicted`` increments both), not an additional
        term; taking ``_evicted`` whole would count a promoted-then-evicted
        cohort twice -- once under ``promoted``, once under ``evicted``.  That
        slack is reported as ``evicted_promoted`` instead of hidden.

        This is the ruling's intended *cohort issuance* account (which cohort
        was issued to which reader, and which was promoted), not a
        conservation law; the conservation identity is
        :meth:`conservation_account`.
        """
        promoted = len(self._promoted)
        resident_unpromoted = sum(
            cohort.cohort_id not in self._promoted for cohort in self._entries
        )
        drained_unpromoted = sum(
            cohort.cohort_id not in self._promoted for cohort in self.drains
        )
        evicted_promoted = sum(
            cohort.cohort_id in self._promoted for cohort in self.evictions
        )
        evicted = len(self.evictions) - evicted_promoted
        issued = promoted + resident_unpromoted + drained_unpromoted + evicted
        return {
            "deposits": self._deposits,
            "promoted": promoted,
            "resident_unpromoted": resident_unpromoted,
            "drained_unpromoted": drained_unpromoted,
            "evicted": evicted,
            "evicted_promoted": evicted_promoted,
            "issued": issued,
            "formula": (
                "deposits = promoted + resident_unpromoted "
                "+ drained_unpromoted + evicted"
            ),
            "holds": issued == self._deposits,
        }

    def mass_balance(self) -> dict:
        """Both named accounts; the flat keys are Round-11 compatibility.

        ``run.py`` stores this dict on every tick, so the flat
        ``deposits/resident/drained/evicted/ceiling_evicted/promoted/
        reads_by_kind`` keys stay exactly as they were; the two accounts are
        added beside them (``conservation_account``, ``issuance_account``).
        ``issuance`` is the Round-11 short name and now carries the full
        issuance account.  Neither account is ever substituted for the other.
        """
        self._assert_balance()
        conservation = self.conservation_account()
        issuance = self.issuance_account()
        return {
            "deposits": conservation["deposits"],
            "resident": conservation["resident"],
            "drained": conservation["drained"],
            "evicted": conservation["evicted"],
            "ceiling_evicted": conservation["ceiling_evicted"],
            "promoted": issuance["promoted"],
            "reads_by_kind": dict(self._reads_by_kind),
            "issuance": issuance,
            "conservation_account": conservation,
            "issuance_account": issuance,
        }

    def over_ceiling(self) -> bool:
        return self.byte_size() > self.max_bytes

    def byte_size(self) -> int:
        return sum(_cohort_bytes(cohort) for cohort in self._entries)

    def bodies(self) -> tuple[DebrisCohort, ...]:
        """Resident cohorts, oldest-first (cohort-level storage)."""
        return tuple(self._entries)

    def newest(self) -> DebrisCohort | None:
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class BoardSlot:
    """One resident D1 context slot: identity and represented size only.

    The slot retains a content digest (``body_id``) and the represented
    byte count, not the context vector itself (Lead adjudication Q2).  D1
    is operational, not parametric geometry, so this record is never a
    geometry payload.
    """

    slot_id: str
    body_id: str
    nbytes: int
    born_tick: int

    def __post_init__(self) -> None:
        for name in ("slot_id", "body_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.nbytes) is not int or isinstance(self.nbytes, bool):
            raise TypeError(f"nbytes must be an int, got {self.nbytes!r}")
        if self.nbytes <= 0:
            raise ValueError("nbytes must be positive for a written slot")
        if type(self.born_tick) is not int or isinstance(self.born_tick, bool):
            raise TypeError(f"born_tick must be an int, got {self.born_tick!r}")
        if self.born_tick < 0:
            raise ValueError("born_tick must be non-negative")


class BoardStore:
    """D1 Elastic Board, minimal slice: slot identity + eviction state.

    This store is *not* the full ``ElasticBoard`` contract: #11 (active
    context read) and #15 (evicted clusters) remain unimplemented, so no
    content vector is retained and no eviction consumer exists yet
    (Lead adjudication Q2/Q9).  ``BoardSlot.nbytes`` is the *represented*
    bytes of the #12 context record the write stands for (``8 *
    len(delta)``), not this store's own metadata footprint; #16's ``cost``
    uses exactly that number.

    Single writer (§9.11c): ``apply`` is the only public method that can
    mutate the board.  It is the executor-side, post-arrival commit of a
    #13 ``BoardOpBody``; #12 handling (P4's input) must never touch D1
    (§9.10).  All preconditions are checked before the first mutation, so
    a refused operation leaves the state digest byte-identical, records no
    eviction, and leaves ``last_commit`` unchanged (§9.11a).

    Commits are monotone (§9.14.5): ``apply`` refuses ``op.tick <=
    last_commit``, so the resident slots are appended in strictly
    increasing born-tick order.  ``slot_state`` therefore publishes
    ``slot_ids``/``oldest_slot_id`` in born-tick order by construction, not
    by dict-insertion accident.
    """

    def __init__(self, capacity: int = BOARD_CAPACITY) -> None:
        if (
            type(capacity) is not int
            or isinstance(capacity, bool)
            or capacity < 1
        ):
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        self._capacity = capacity
        self._slots: dict[str, BoardSlot] = {}
        self._evictions: list[tuple[str, int]] = []
        self._last_provenance: Provenance | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def last_commit(self) -> int | None:
        """The tick of the last committed #13, or None before the first."""
        if self._last_provenance is None:
            return None
        return self._last_provenance.tick_born

    @property
    def state_provenance(self) -> Provenance | None:
        """The provenance of the last committed #13 (the #14 read source)."""
        return self._last_provenance

    def apply(self, op: BoardOpBody, provenance: Provenance) -> None:
        """Commit exactly one #13 board operation, atomically.

        A ``write`` appends a slot, ``evict`` removes one and records the
        eviction, ``erase`` removes one without an eviction record, and
        ``resize`` changes capacity (never below occupancy).  Overflow,
        duplicate ids, unknown ids, an unchanged resize, and a provenance
        that is not ``(13, op.tick)`` all raise before any mutation.
        """
        from .provenance import Provenance as _Provenance

        if type(op) is not BoardOpBody:
            raise TypeError(
                f"BoardStore accepts BoardOpBody, got {type(op).__name__}"
            )
        if type(provenance) is not _Provenance:
            raise TypeError(
                "BoardStore needs an exact Provenance, got "
                f"{type(provenance).__name__}"
            )
        if provenance != _Provenance(13, op.tick):
            raise ValueError(
                f"#13 provenance must be (13, {op.tick}), got {provenance!r}"
            )

        # Monotone commit order (§9.14.5): the ledger has one #13 per tick,
        # so a stale or equal tick would make "oldest" an insertion accident
        # rather than a born-tick fact.  Refuse before any mutation.
        last = self.last_commit
        if last is not None and op.tick <= last:
            raise ValueError(
                "#13 commits must strictly increase; the ledger has one #13 "
                f"per tick (op tick {op.tick} <= last_commit {last})"
            )

        # Validate every precondition before mutating anything (atomicity).
        if op.op == "write":
            if len(self._slots) >= self._capacity:
                raise ValueError(
                    f"D1 is full ({self._capacity} slots); the policy must "
                    "evict before writing"
                )
            if op.slot_id in self._slots:
                raise ValueError(f"duplicate slot_id {op.slot_id!r}")
        elif op.op in ("evict", "erase"):
            if op.slot_id not in self._slots:
                raise ValueError(f"unknown slot_id {op.slot_id!r}")
        elif op.op == "resize":
            if op.nbytes == self._capacity:
                raise ValueError("resize must change the capacity")
            if op.nbytes < len(self._slots):
                raise ValueError(
                    f"resize to {op.nbytes} is below occupancy "
                    f"{len(self._slots)}"
                )
        else:  # pragma: no cover - BoardOpBody validates the verb
            raise ValueError(f"unsupported op {op.op!r}")

        if op.op == "write":
            assert op.body_id is not None  # BoardOpBody validation
            self._slots[op.slot_id] = BoardSlot(
                op.slot_id, op.body_id, op.nbytes, op.tick
            )
        elif op.op == "evict":
            del self._slots[op.slot_id]
            self._evictions.append((op.slot_id, op.tick))
        elif op.op == "erase":
            del self._slots[op.slot_id]
        else:  # resize
            self._capacity = op.nbytes
        self._last_provenance = provenance

    def slots(self) -> tuple[BoardSlot, ...]:
        """The resident slots, oldest -> newest."""
        return tuple(self._slots.values())

    def slot_state(self) -> SlotStateBody | None:
        """The #14 state snapshot, or None before the first committed #13.

        A board emptied by erasures/evictions still reports its capacity and
        history (it is a non-empty *state*, just an empty resident set);
        "empty" here means *never committed*, not zero residents (§9.14.2).

        ``slot_ids`` order is the resident born-tick order.  That is a
        structural fact, not an assertion: ``apply`` is the only mutator, it
        refuses ``op.tick <= last_commit``, and a ``write`` only appends, so
        resident insertion order and born-tick order cannot diverge through
        this class's public surface (§9.14.5).  The ordering is pinned by
        tests rather than by dead defensive code here.
        """
        if self._last_provenance is None:
            return None
        slot_ids = tuple(self._slots)
        return SlotStateBody(
            capacity=self._capacity,
            occupied=len(slot_ids),
            nbytes=sum(slot.nbytes for slot in self._slots.values()),
            slot_ids=slot_ids,
            oldest_slot_id=slot_ids[0] if slot_ids else None,
        )

    def evictions(self) -> tuple[tuple[str, int], ...]:
        """Evicted (slot_id, tick) pairs; #15's future evidence, not read."""
        return tuple(self._evictions)

    def state_digest(self) -> str:
        """Deterministic digest of the whole D1 state (atomicity checks)."""
        digest = hashlib.blake2b(digest_size=16)
        digest.update(repr(self._capacity).encode("utf-8"))
        for slot in self._slots.values():
            digest.update(
                repr(
                    (slot.slot_id, slot.body_id, slot.nbytes, slot.born_tick)
                ).encode("utf-8")
            )
        for eviction in self._evictions:
            digest.update(repr(eviction).encode("utf-8"))
        provenance = self._last_provenance
        digest.update(
            repr(
                None
                if provenance is None
                else (provenance.source_fid, provenance.tick_born)
            ).encode("utf-8")
        )
        return digest.hexdigest()


def allowed_payload(fid: int, payload: object) -> bool:
    """INV-EMPTINESS whitelist: exact type membership, no subclassing."""
    return type(payload) in PAYLOAD_SCHEMA[fid]


def require_payload(fid: int, payload: object) -> None:
    if not allowed_payload(fid, payload):
        raise TypeError(
            f"flow #{fid} rejects payload {type(payload).__name__}; "
            f"allows {sorted(t.__name__ for t in PAYLOAD_SCHEMA[fid])}"
        )
