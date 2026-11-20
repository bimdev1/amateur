"""Typed payloads for the geometry stores, schema-enforced.

INV-EMPTINESS by construction: the D4 (Base Core) write channel accepts
only `BiasPrimitive` records -- a whitelist, not a blacklist -- so a
trajectory, state-delta, or context payload is not merely forbidden, it is
unrepresentable on that channel. INV-NO-OSSIFY: D2 (Multi-Layer Mat)
payloads carry epochs, and the `MatBoard` lifecycle guarantees no resident
overlay's epoch falls more than one tick behind the present.

Schema (user-adjudicated):
- D2 payloads: (epoch, overlay_id, status in {active, stale});
- D4 payloads: (bias_primitive, confidence in [0, 1]); trajectory,
  state-delta, and context types are excluded by the whitelist;
- the phase-4/5 real flows carry frozen, hashable body records (see the
  ``*Body`` classes below); every other non-store flow stays a placeholder
  (payload None).  ``None`` remains allowed on every non-store channel, so
  the synthetic plumbing source and the still-placeholder flows keep
  working unchanged.

Physics-epoch bodies are *content*, never provenance: the ``Envelope``
carries ``(source_fid, tick_born)`` and the exact-type gate compares body
types only.  All bodies are frozen dataclasses whose fields are tuples and
scalars, so they are hashable exactly when the envelope needs them to be.

Accounting surfaces added for phase 4-5b (task-6):
- ``DebrisPool``: D3, capacity + byte ceiling, provenance-preserving.
- ``MatBoard`` patch table: D2 holds the predictor patch keyed by
  ``overlay_id`` (Q1-A); records stay ``MatOverlay``; patches are accounted
  in bytes against ``MAX_PATCH_BYTES`` per patch and ``MAX_PATCH_TOTAL_BYTES``
  per table.  The record lifecycle is unchanged (Lean mirror); the byte
  accounting is Python-asserted and marked *tested-not-proven*.
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

#: D3 production bounds: 8 bodies x 512 B centroid + bookkeeping, 64 KiB.
DEBRIS_CAPACITY = 8
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


# ---------------------------------------------------------------------------
# Geometry records (unchanged)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatOverlay:
    """One D2 mat overlay: epoch-stamped so staleness is checkable."""

    epoch: int
    overlay_id: str
    status: str

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {self.epoch}")
        if self.status not in ("active", "stale"):
            raise ValueError(f"status must be active|stale, got {self.status!r}")


@dataclass(frozen=True)
class BiasPrimitive:
    """One D4 sedimented bias: confidence-weighted, trajectory-free."""

    bias_primitive: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
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
    """#18 P3 -> P5: a candidate residual vector before digestion."""

    residual: tuple[float, ...]
    surprise: float
    tick: int

    def __post_init__(self) -> None:
        _require_vector("residual", self.residual)
        _require_finite("surprise", self.surprise)
        if self.tick < 0:
            raise ValueError("tick must be non-negative")


@dataclass(frozen=True)
class DigestedDebrisBody:
    """#19 P5 -> D3: a real residual cluster (the D3 body)."""

    debris_id: str
    centroid: tuple[float, ...]
    count: int
    spread: float
    first_tick: int
    last_tick: int
    peak_surprise: float

    def __post_init__(self) -> None:
        if not isinstance(self.debris_id, str):
            raise TypeError("debris_id must be a string")
        _require_vector("centroid", self.centroid)
        if self.count < 0:
            raise ValueError("count must be non-negative (0 marks the no-op)")
        _require_finite("spread", self.spread)
        _require_finite("peak_surprise", self.peak_surprise)
        if self.first_tick < 0 or self.last_tick < 0:
            raise ValueError("ticks must be non-negative")


@dataclass(frozen=True)
class StructuralMotifBody:
    """#21 P5 -> P6: a stable cluster extracted as a rule."""

    motif_id: str
    debris_id: str
    centroid: tuple[float, ...]
    count: int
    spread: float
    persistence: int
    drift: float
    first_tick: int
    last_tick: int

    def __post_init__(self) -> None:
        if not isinstance(self.motif_id, str) or not isinstance(self.debris_id, str):
            raise TypeError("motif_id/debris_id must be strings")
        _require_vector("centroid", self.centroid)
        if self.count < 0 or self.persistence < 0:
            raise ValueError("count/persistence must be non-negative")
        _require_finite("spread", self.spread)
        _require_finite("drift", self.drift)
        if self.first_tick < 0 or self.last_tick < 0:
            raise ValueError("ticks must be non-negative")


# ---------------------------------------------------------------------------
# Fixed-cadence no-op sentinels (exact-type, content-free)
# ---------------------------------------------------------------------------
def noop_debris() -> DigestedDebrisBody:
    """The #19 no-event body: a real body the D3 pool absorbs as a no-op."""
    return DigestedDebrisBody(
        debris_id=NO_DEBRIS_ID,
        centroid=(0.0,) * PAYLOAD_LATENT_DIM,
        count=0,
        spread=0.0,
        first_tick=0,
        last_tick=0,
        peak_surprise=0.0,
    )


def noop_motif() -> StructuralMotifBody:
    """The #21 no-event body: a real body P6 consumes as a no-op."""
    return StructuralMotifBody(
        motif_id=NO_MOTIF_ID,
        debris_id=NO_DEBRIS_ID,
        centroid=(0.0,) * PAYLOAD_LATENT_DIM,
        count=0,
        spread=0.0,
        persistence=0,
        drift=0.0,
        first_tick=0,
        last_tick=0,
    )


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
    """The #23/#24 no-event token: a real MatOverlay the mat ignores."""
    return MatOverlay(0, NOOP_OVERLAY_ID, "stale")


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
_MAT_READ = frozenset({MatOverlay, type(None)})
_MAT_WRITE = frozenset({MatOverlay})
_BIAS_READ = frozenset({BiasPrimitive, type(None)})
_BIAS_WRITE = frozenset({BiasPrimitive})

#: The phase-4/5 real-flow bodies.  None stays allowed on each channel so
#: the synthetic plumbing source and the still-placeholder flows are
#: unchanged; a physics derivation always returns the typed body.
REAL_BODIES: dict[int, type] = {
    3: FrameBody,
    6: PredictionErrorBody,
    10: LatentRolloutBody,
    12: StateDeltaBody,
    16: CostErrorBody,
    17: CollapseMandateBody,
    18: EmergentCrystalBody,
    19: DigestedDebrisBody,
    21: StructuralMotifBody,
    13: BoardOpBody,
    14: SlotStateBody,
}


def _schema() -> dict[int, frozenset[type]]:
    schema: dict[int, frozenset[type]] = {}
    for f in FLOW_LEDGER:
        if f.fid in D2_WRITES:
            schema[f.fid] = _MAT_WRITE
        elif f.fid in D2_FLOWS:
            schema[f.fid] = _MAT_READ
        elif f.fid in D4_WRITES:
            schema[f.fid] = _BIAS_WRITE
        elif f.fid in D4_FLOWS:
            schema[f.fid] = _BIAS_READ
        elif f.fid in REAL_BODIES:
            schema[f.fid] = frozenset({REAL_BODIES[f.fid], type(None)})
        else:
            schema[f.fid] = _PLACEHOLDER
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
    plumbing; in the physics epoch #23/#24 are organ-owned and sentinel
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

    # ------------------------------------------------------------- D4 core
    def sediment(self, bias: BiasPrimitive) -> None:
        self.d4.append(bias)

    def latest_bias(self) -> BiasPrimitive | None:
        return self.d4[-1] if self.d4 else None

    # ---------------------------------------------------------- D2 records
    def head(self) -> tuple[str, int]:
        return self._residents[0]

    def birth(self, overlay: MatOverlay) -> None:
        if overlay.overlay_id in self.resident_ids():
            raise ValueError(
                f"duplicate D2 birth for overlay_id {overlay.overlay_id!r}; "
                "a second record would share one patch entry"
            )
        self._residents.append((overlay.overlay_id, overlay.epoch))

    def revalidate(self, overlay: MatOverlay) -> None:
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
        return sum(_patch_attr(patch, "nbytes") for patch in self._patches.values())

    def patch_params(self) -> int:
        return sum(
            _patch_attr(patch, "param_count") for patch in self._patches.values()
        )

    def enforce_patch_ceiling(self, tick: int) -> None:
        """Evict resident records (and their patches) until the byte bound holds."""
        while self.patch_bytes() > self.max_patch_total_bytes:
            victims = [
                overlay_id
                for overlay_id, _epoch in self._residents
                if overlay_id in self._patches
            ]
            if not victims:
                break
            self._remove(victims[0], tick, cause="patch-bytes")

    def assert_patch_bounds(self, *, require_complete: bool = False) -> None:
        """Python-asserted substance bound (tested-not-proven; Lean: records).

        Raises ``AssertionError`` when the patch table and the resident
        records are out of step, a patch exceeds the per-patch bound, the
        total exceeds the table bound, or (physics mode) a resident record
        has no patch.
        """
        residents = self.resident_ids()
        patch_ids = frozenset(self._patches)
        assert patch_ids <= residents, (
            f"orphan patches {sorted(patch_ids - residents)} have no resident"
        )
        assert len(residents) <= MAT_CAPACITY, "record capacity exceeded"
        assert self.patch_bytes() <= self.max_patch_total_bytes, (
            f"patch table {self.patch_bytes()} > {self.max_patch_total_bytes} bytes"
        )
        for overlay_id, patch in self._patches.items():
            nbytes = _patch_attr(patch, "nbytes")
            assert nbytes <= self.max_patch_bytes, (
                f"patch {overlay_id!r} {nbytes} > {self.max_patch_bytes} bytes"
            )
        if require_complete:
            missing = sorted(residents - patch_ids)
            assert not missing, f"resident overlays without a patch: {missing}"


# ---------------------------------------------------------------------------
# D3 debris pool
# ---------------------------------------------------------------------------
def _debris_bytes(body: DigestedDebrisBody) -> int:
    """Deterministic body size: centroid floats + scalar fields + id."""
    return (
        len(body.centroid) * 8
        + 8 * 5  # count, spread, first_tick, last_tick, peak_surprise
        + len(body.debris_id.encode("utf-8"))
    )


@dataclass(frozen=True)
class DebrisEntry:
    """One stored D3 body plus the provenance of the #19 write that made it."""

    body: DigestedDebrisBody
    provenance: object


class DebrisPool:
    """D3 Debris & Scaffolding Pool: bounded by capacity and bytes.

    ``deposit`` appends a #19 body (never the fixed-cadence no-op) and
    evicts the oldest entries until both ceilings hold.  Reads hand back
    the stored body *and* its provenance, so a D3 read inherits the
    producing write's provenance exactly as the D2/D4 reads do.
    """

    def __init__(
        self,
        capacity: int = DEBRIS_CAPACITY,
        max_bytes: int = DEBRIS_MAX_BYTES,
    ) -> None:
        if capacity < 1 or max_bytes < 1:
            raise ValueError("capacity and max_bytes must be positive")
        self.capacity = int(capacity)
        self.max_bytes = int(max_bytes)
        self._entries: deque[DebrisEntry] = deque()
        self.evictions: list[tuple[DigestedDebrisBody, object, int]] = []

    def deposit(
        self,
        body: DigestedDebrisBody,
        provenance: object,
        tick: int,
    ) -> tuple[DebrisEntry, ...]:
        """Store one body; returns the entries evicted to respect the bounds."""
        if type(body) is not DigestedDebrisBody:
            raise TypeError(f"DebrisPool accepts DigestedDebrisBody, got {type(body).__name__}")
        if body.debris_id == NO_DEBRIS_ID:
            raise ValueError("the fixed-cadence no-op is not deposited")
        self._entries.append(DebrisEntry(body, provenance))
        evicted: list[DebrisEntry] = []
        while self._entries and (
            len(self._entries) > self.capacity or self.byte_size() > self.max_bytes
        ):
            entry = self._entries.popleft()
            self.evictions.append((entry.body, entry.provenance, tick))
            evicted.append(entry)
        return tuple(evicted)

    def byte_size(self) -> int:
        return sum(_debris_bytes(entry.body) for entry in self._entries)

    def bodies(self) -> tuple[DigestedDebrisBody, ...]:
        return tuple(entry.body for entry in self._entries)

    def newest(self) -> DebrisEntry | None:
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
        history (it is a non-empty *state*, just an empty resident set).
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
