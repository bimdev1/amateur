"""Provenance carrier for the typed payload schema (A2, adjudicated).

The user adjudicated option A2: a generic ``Envelope`` placed around every
accepted body at the channel boundary. The typed payload schema is *not*
reopened:

- ``PAYLOAD_SCHEMA`` remains the exact-type body whitelist; the envelope
  gate checks ``type(envelope.body) in PAYLOAD_SCHEMA[fid]`` and never
  ``isinstance``, so ``Sneaky(MatOverlay)`` is still rejected.
- ``Provenance`` is content-blind: two non-negative integers naming the
  producing flow id and the tick at which the value was born. It carries
  no trajectory, state-delta, or context information, so D4's
  trajectory-free guarantee is untouched (``BiasPrimitive`` itself is
  unchanged).
- ``require_provenance`` enforces the emission invariant
  ``tick_born <= emit_tick``; ``require_channel_provenance`` adds the
  adjudicated per-flow rule: a store read with a non-``None`` body must
  have ``tick_born < emit_tick``, every other traversal ``==``.
- ``provenance_of`` / ``derived_at`` are option-agnostic accessors: they
  read an envelope, a bare ``Provenance``, or per-type provenance fields.

Honesty note (also in the memo): ``tick_born <= emit_tick`` is necessary
but not sufficient to prove "no minting at emit time". A value fabricated
at emission can honestly claim ``tick_born == emit_tick``; the per-flow
gate catches that for reads but not for writes. The full argument is
structural (bodies are built in ``DryRun._derive``, not in ``_arrive``)
plus continuity (a store read's provenance equals the stored record's).
The tests pin the ordering tripwire and the exact-type interaction; they
do not pretend the ordering check alone closes the gate.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ledger import FLOW_LEDGER, STORES
from .payloads import PAYLOAD_SCHEMA

#: The ledger's flow ids. ``source_fid`` names one of these edges, never a
#: node: a node owns several outgoing edges (P1 owns #2, #4, #5, #6), so a
#: node attribution would not identify which typed carrier produced the
#: value. Derived from the ledger so the two cannot drift.
FLOW_IDS: frozenset[int] = frozenset(flow.fid for flow in FLOW_LEDGER)

#: Flows whose source is a store, i.e. reads of stored state.
STORE_READ_FIDS: frozenset[int] = frozenset(
    flow.fid for flow in FLOW_LEDGER if flow.source in STORES
)

#: Flows whose target is a store, i.e. writes into stored state.
STORE_WRITE_FIDS: frozenset[int] = frozenset(
    flow.fid for flow in FLOW_LEDGER if flow.target in STORES
)


@dataclass(frozen=True)
class Provenance:
    """Where one payload value came from.

    ``source_fid`` is the *producing flow id*: for a value written by a
    process it is the emitting flow (e.g. ``22``); for a value read out of
    a store it is the historical write edge that created the stored
    record (e.g. ``23`` or ``24`` for a D2 read); for an exogenous boundary
    input it is the boundary edge itself (``1``/``3``). It is deliberately
    not the carrier's own flow id: ``FlowEvent.FLOW_ID`` already names the
    channel that is carrying the value now, and read payloads must point
    back past it.

    ``tick_born`` is the tick at which the value came into existence.
    Tick 0 is the pre-run seed (the D2 ``MAT_SEED`` overlay); live
    emissions are born at tick >= 1.
    """

    source_fid: int
    tick_born: int

    def __post_init__(self) -> None:
        if type(self.source_fid) is not int or self.source_fid not in FLOW_IDS:
            raise ValueError(
                "source_fid must be a ledger flow id in 1..26, "
                f"got {self.source_fid!r}"
            )
        if type(self.tick_born) is not int or self.tick_born < 0:
            raise ValueError(
                f"tick_born must be an int >= 0, got {self.tick_born!r}"
            )


@dataclass(frozen=True)
class Envelope:
    """A payload body plus the provenance of the value it carries.

    Envelopes are frozen and hashable exactly when their body is: the
    body's own equality, hash, and exact type are untouched. ``body`` may
    be ``None`` on the placeholder channels, which still carry provenance
    for the traversal but claim no derived value.
    """

    provenance: Provenance
    body: object

    def __post_init__(self) -> None:
        if type(self.provenance) is not Provenance:
            raise TypeError(
                "envelope provenance must be a Provenance, got "
                f"{type(self.provenance).__name__}"
            )


def provenance_of(value: object) -> Provenance:
    """Extract provenance from an envelope, a bare record, or fields.

    The field branch keeps the accessor usable under option B (per-type
    provenance fields) and during a staged migration.
    """
    if type(value) is Provenance:
        return value
    if type(value) is Envelope:
        return value.provenance
    source_fid = getattr(value, "source_fid", None)
    tick_born = getattr(value, "tick_born", None)
    if source_fid is not None and tick_born is not None:
        return Provenance(source_fid, tick_born)
    raise TypeError(
        f"value of type {type(value).__name__} carries no provenance; "
        "wrap it in Envelope(Provenance(source_fid, tick_born), body)"
    )


def require_provenance(value: object, emit_tick: int) -> Provenance:
    """Return provenance after enforcing ``tick_born <= emit_tick``.

    Raises ``TypeError`` when the value carries no provenance (so a raw
    body cannot pass the channel gate) and ``ValueError`` when the value
    claims to have been born after the tick that emits it.
    """
    if type(emit_tick) is not int or emit_tick < 0:
        raise ValueError(f"emit_tick must be an int >= 0, got {emit_tick!r}")
    provenance = provenance_of(value)
    if provenance.tick_born > emit_tick:
        raise ValueError(
            f"future-dated provenance {provenance!r}: tick_born "
            f"{provenance.tick_born} > emit tick {emit_tick}; a payload "
            "cannot be born after the emission that carries it"
        )
    return provenance


def derived_at(payload: object, tick: int) -> Provenance:
    """Assert the payload pre-dates ``tick`` and return its provenance.

    Equivalent to ``require_provenance`` under the memo's naming; kept as
    a separate name because the integration tests read as "this value was
    derived at or before tick t".
    """
    return require_provenance(payload, tick)


def allowed_body(fid: int, body: object) -> bool:
    """Exact-type body membership: the typed payload whitelist, unchanged."""
    return type(body) in PAYLOAD_SCHEMA[fid]


def allowed_envelope(fid: int, value: object) -> bool:
    """Channel gate: an exact ``Envelope`` whose body is an exact member.

    Subclass rejection is preserved twice over: ``type(value) is Envelope``
    rejects a subclassed envelope, and ``allowed_body`` uses exact type
    membership, so ``Sneaky(MatOverlay)`` inside a correct envelope is
    still rejected.
    """
    if type(value) is not Envelope:
        return False
    if type(value.provenance) is not Provenance:
        return False
    return allowed_body(fid, value.body)


def require_envelope(fid: int, value: object, emit_tick: int) -> Envelope:
    """Accept only a provenanced, exact-type body on flow ``fid``.

    Order of checks: envelope type -> body whitelist -> tick ordering.
    Every accepted payload therefore carries provenance and satisfies
    ``tick_born <= emit_tick`` by construction of the gate.
    """
    if type(value) is not Envelope:
        raise TypeError(
            f"flow #{fid} requires an Envelope carrying provenance, got "
            f"{type(value).__name__}; raw payloads are not accepted"
        )
    if not allowed_body(fid, value.body):
        allowed = sorted(t.__name__ for t in PAYLOAD_SCHEMA[fid])
        raise TypeError(
            f"flow #{fid} rejects body {type(value.body).__name__}; "
            f"allows {allowed}"
        )
    require_provenance(value.provenance, emit_tick)
    return value


def require_channel_provenance(
    fid: int, value: object, emit_tick: int
) -> Envelope:
    """The adjudicated per-flow birth rule, on top of ``require_envelope``.

    - store read with a non-``None`` body: ``tick_born < emit_tick`` (a
      value cannot be read before it was stored; reads precede this tick's
      writes in the stage order);
    - every other accepted traversal (write, placeholder, or empty read):
      ``tick_born == emit_tick``.
    """
    envelope = require_envelope(fid, value, emit_tick)
    born = envelope.provenance.tick_born
    if fid in STORE_READ_FIDS and envelope.body is not None:
        if born >= emit_tick:
            raise ValueError(
                f"flow #{fid} is a store read: a non-None body must be born "
                f"before the read, got tick_born {born} >= emit tick "
                f"{emit_tick}"
            )
    elif born != emit_tick:
        raise ValueError(
            f"flow #{fid}: this channel requires tick_born == emit tick "
            f"({emit_tick}), got {born}"
        )
    return envelope
