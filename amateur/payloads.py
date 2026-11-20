"""Typed payloads for the geometry stores, schema-enforced.

INV-EMPTINESS by construction: the D4 (Base Core) write channel accepts
only `BiasPrimitive` records \u2014 a whitelist, not a blacklist \u2014 so a
trajectory, state-delta, or context payload is not merely forbidden, it is
unrepresentable on that channel. INV-NO-OSSIFY: D2 (Multi-Layer Mat)
payloads carry epochs, and the `MatBoard` lifecycle guarantees no resident
overlay's epoch falls more than one tick behind the present.

Schema (user-adjudicated):
- D2 payloads: (epoch, overlay_id, status in {active, stale});
- D4 payloads: (bias_primitive, confidence in [0, 1]); trajectory,
  state-delta, and context types are excluded by the whitelist;
- every other flow stays a placeholder (payload None) until physics.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .ledger import FLOW_LEDGER, STORES

_D2 = next(s for s in STORES if s.startswith("D2"))
_D4 = next(s for s in STORES if s.startswith("D4"))


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

_PLACEHOLDER = frozenset({type(None)})
_MAT_READ = frozenset({MatOverlay, type(None)})
_MAT_WRITE = frozenset({MatOverlay})
_BIAS_READ = frozenset({BiasPrimitive, type(None)})
_BIAS_WRITE = frozenset({BiasPrimitive})


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
        else:
            schema[f.fid] = _PLACEHOLDER
    return schema


PAYLOAD_SCHEMA: dict[int, frozenset[type]] = _schema()

MAT_CAPACITY = 4
MAT_SEED = ("ovr-0", 0)


class MatBoard:
    """D2 lifecycle, mirrors the formal matTick policy exactly.

    Per tick: one birth (append, from surviving crystals), one
    revalidation (head popped and re-appended at the new epoch, from the
    stale read), then one eviction iff the board exceeds MAT_CAPACITY.
    Seeded with a single epoch-0 overlay so the #25 -> #24 pairing has a
    candidate from tick 1.

    Cadence honesty: the one-birth-per-tick minting schedule is dry-run
    plumbing; in the physics epoch #23/#24 are trigger-gated. The Lean
    theorems that consume this policy quantify over every stream obeying
    the same update law (`mat_*_policy`), so a different schedule is a
    different law, never a silent assumption.
    """

    def __init__(self) -> None:
        self._residents: deque[tuple[str, int]] = deque([MAT_SEED])
        self.evictions: list[tuple[str, int, int]] = []
        self.d4: list[BiasPrimitive] = []

    def sediment(self, bias: BiasPrimitive) -> None:
        self.d4.append(bias)

    def latest_bias(self) -> BiasPrimitive | None:
        return self.d4[-1] if self.d4 else None

    def head(self) -> tuple[str, int]:
        return self._residents[0]

    def birth(self, overlay: MatOverlay) -> None:
        self._residents.append((overlay.overlay_id, overlay.epoch))

    def revalidate(self, overlay: MatOverlay) -> None:
        stale_id, _epoch = self._residents.popleft()
        if stale_id != overlay.overlay_id:
            raise RuntimeError(
                f"revalidation pairing broken: head {stale_id!r}, "
                f"write {overlay.overlay_id!r}"
            )
        self._residents.append((stale_id, overlay.epoch))

    def evict_if_needed(self, tick: int) -> None:
        if len(self._residents) > MAT_CAPACITY:
            overlay_id, epoch = self._residents.popleft()
            self.evictions.append((overlay_id, epoch, tick))

    def residents(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._residents)


def allowed_payload(fid: int, payload: object) -> bool:
    """INV-EMPTINESS whitelist: exact type membership, no subclassing."""
    return type(payload) in PAYLOAD_SCHEMA[fid]


def require_payload(fid: int, payload: object) -> None:
    if not allowed_payload(fid, payload):
        raise TypeError(
            f"flow #{fid} rejects payload {type(payload).__name__}; "
            f"allows {sorted(t.__name__ for t in PAYLOAD_SCHEMA[fid])}"
        )
