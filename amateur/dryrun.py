"""Dry run: pure event plumbing along the ledger spine.

No physics — every stage consumes its ledger inputs and emits its ledger
outputs. The geometry channels carry typed payloads (see payloads.py):
D2 writes (#23, #24) carry epoch-stamped overlays, D4 writes (#22) carry
bias primitives — INV-EMPTINESS holds by construction. Every other
payload stays a placeholder until physics. The driving gradient (#1, #3)
and the geometry echoes (#7, #8, #11, #14, #15, #20, #25, #26) are
injected every tick; read-consumes expect data (a deficit charge), store
absorbs are best-effort; the two in-tick cycles (#10 Resolved Trajectory,
#17 Collapse Mandate) are one-tick-delayed mailboxes.

Provenance (A2, user-adjudicated): every accepted payload is an
`Envelope(Provenance(source_fid, tick_born), body)` (see
`amateur/provenance.py`). `PAYLOAD_SCHEMA` remains the exact-type *body*
whitelist; the envelope is the transport gate at the channel boundary.
`_derive()` runs once at the start of every tick and fills
`self.derived` for all 26 fids from live node/store state; `_arrive()`
only reads that map, applies `require_channel_provenance`, and transports
the envelope. `_payload_for` (the old emit-time minter) is gone. Store
reads inherit the provenance of the stored record (D2 head, D4 sediment);
writes are born at their emission tick; `MAT_SEED` is `(23, 0)`. The
derivation pass does not make the synthetic bodies honest: #22's rotating
confidence and #23's `cry-{tick}` remain synthetic (gate recorded
PARTIAL) until physics derives them from #21/#25.

Channel semantics (user-adjudicated): reject-with-charge at capacity-1
mailboxes. An arrival into an empty box is accepted; an arrival onto a
non-empty box is refused: the event object is discarded at the channel
and a scalar excess charge is recorded (refusals are never retained as
events, never logged as payloads, never replayable). Acceptance is
first-arrival-wins with ledger order as tie-break: injections and stage
emissions proceed in ascending fid within their phase, in spine order.
Deficit (expected, absent) and excess (arrived, refused) charges are
distinct kinds with separate counters; they are never summed. The two
per-tick identities: attempted = accepted + refused, and accepted =
consumed + delta-sunk + delta-depth.

Cadence honesty: the uniform per-tick emission schedule and the one-#22
/one-#23-per-tick geometry writes are plumbing, not design — the always-on
spine's cadence only. In the physics epoch the sedimentation flows
(#22, #23, #24) become trigger-gated (6.0 is an interference-checked
gate, not a metronome), so theorems stated against this schedule are
registry-marked plumbing-only in docs/TESTING.md; the mat theorems carry
the schedule as an explicit hypothesis (`mat_*_policy`).

Verification: `flow_ops` and `step_flow` define a pure, per-flow
independent recurrence over mailbox depths; `formal/DryRun.lean` proves
its long-run shape (boundedness, cycle delays, sink rates, miss freezing)
and `tests/test_lean_lockstep.py` proves the generated Lean op table
matches this module's ledger-derived table.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Protocol

from .events import FlowEvent, carrier_for
from .ledger import E1, E2, FLOW_LEDGER, STORES
from .payloads import (
    MAT_SEED,
    NON_READ_FLOWS,
    NO_DEBRIS_ID,
    NOOP_OVERLAY_ID,
    BiasPrimitive,
    DebrisPool,
    DigestedDebrisBody,
    MatBoard,
    MatOverlay,
)
from .provenance import (
    Envelope,
    Provenance,
    require_channel_provenance,
)


@dataclass(frozen=True)
class Stage:
    """One spine stage: flows consumed, flows emitted, both by ledger id."""

    name: str
    consumes: tuple[int, ...]
    emits: tuple[int, ...]


SPINE: tuple[Stage, ...] = (
    Stage("1.0 Transduce Markov Blanket", (1, 3, 10), (2, 4, 5, 6)),
    Stage("2.0 Route Without Ossification", (6, 7, 8, 26), (9,)),
    Stage("3.0 Execute Observation Loop", (9, 11), (10, 12, 18)),
    Stage("4.0 Allocate & Regulate Whiteboard", (12, 14, 17), (13, 16)),
    Stage("5.0 Digest & Collapse Ceiling-Bound Units", (15, 16, 18, 20), (17, 19, 21)),
    Stage("6.0 Ratchet & Sediment Grammar", (21, 25), (22, 23, 24)),
)

PER_TICK_INJECTIONS: tuple[int, ...] = (1, 3, 7, 8, 11, 14, 15, 20, 25, 26)

SINK_FLOWS: tuple[int, ...] = tuple(
    f.fid for f in FLOW_LEDGER if f.target in (E1, E2)
)


def _write_fids() -> tuple[int, ...]:
    return tuple(
        f.fid for f in FLOW_LEDGER if f.target in STORES and "(read)" not in f.flow
    )


WRITE_FLOWS: tuple[int, ...] = _write_fids()

#: Non-read flows that stay placeholders this round (Lead decision f).
#: ``NON_READ_FLOWS`` is derived from the ledger; the executor never asks
#: the payload source for these, and the source's coverage ledger marks
#: them ``placeholder``.  Fixed dispatch, not a branch on the ledger.
PLACEHOLDER_FLOWS: frozenset[int] = frozenset({1, 2, 4, 5, 9, 13, 16})

#: Non-read flows the payload source must supply (content-only hook, Q2).
SOURCE_FLOWS: frozenset[int] = NON_READ_FLOWS - PLACEHOLDER_FLOWS


def flow_ops() -> dict[int, tuple[str, ...]]:
    """Per-flow op words, derived from the ledger — the model of record.

    Each word is the ordered sequence of mailbox operations one flow sees
    during a single tick. Pointwise independence is by construction.
    """
    ops: dict[int, list[str]] = {f.fid: [] for f in FLOW_LEDGER}
    for fid in PER_TICK_INJECTIONS:
        ops[fid].append("emit")
    for stage in SPINE:
        for fid in stage.consumes:
            ops[fid].append("consume")
        for fid in stage.emits:
            ops[fid].append("emit")
    for fid in WRITE_FLOWS:
        ops[fid].append("absorb")
    for fid in SINK_FLOWS:
        ops[fid].append("sink")
    return {fid: tuple(seq) for fid, seq in ops.items()}


FLOW_OPS: dict[int, tuple[str, ...]] = flow_ops()

CYCLE_FLOWS: tuple[int, ...] = tuple(
    fid for fid, ops in FLOW_OPS.items() if ops and ops[0] == "consume"
)


class FlowDelta(NamedTuple):
    """One flow's channel after one scalar tick.

    INV-BOUNDS (intra-tick, adjudicated): `box <= 1` whenever the
    incoming depth is -- the arrival op refuses onto a full box, so no
    event sequence can push the depth past one.
    """

    box: int
    miss: int
    refused: int
    sunk: int
    accepted: int
    consumed: int


def step_flow(fid: int, box: int) -> FlowDelta:
    """Pure step of one flow's channel (reject-with-charge semantics)."""
    miss = refused = sunk = accepted = consumed = 0
    for op in FLOW_OPS[fid]:
        if op == "emit":
            if box == 0:
                box, accepted = 1, accepted + 1
            else:
                refused += 1
        elif op in ("consume", "absorb"):
            if box > 0:
                box, consumed = box - 1, consumed + 1
            else:
                miss += 1
        else:
            sunk, box = sunk + box, 0
    return FlowDelta(box, miss, refused, sunk, accepted, consumed)


def prefix_depth(fid: int, box: int, n: int) -> int:
    """Depth after the first n ops of the word -- one event boundary.

    The Python mirror of the Lean `fold_box_bounded` theorem: for a
    compliant incoming depth, every prefix -- hence every event boundary
    within the tick -- keeps the depth at most one.
    """
    if not 0 <= n <= len(FLOW_OPS[fid]):
        raise ValueError(f"prefix length {n} out of range for #{fid}")
    depth = box
    for op in FLOW_OPS[fid][:n]:
        if op == "emit":
            depth = 1 if depth == 0 else depth
        elif op in ("consume", "absorb"):
            if depth > 0:
                depth -= 1
        else:
            depth = 0
    return depth


_BIAS_STEPS = (0.4, 0.6, 0.8)


@dataclass(frozen=True)
class TickStat:
    attempted: dict[int, int]
    accepted: dict[int, int]
    refused: dict[int, int]
    consumed: dict[int, int]
    sunk: dict[int, int]
    missed: dict[int, int]


@dataclass
class Report:
    ticks: int
    attempted: dict[int, int]
    accepted: dict[int, int]
    refused: dict[int, int]
    consumed: dict[int, int]
    missed: dict[int, int]
    sunk: dict[int, int]
    history: tuple[TickStat, ...]



class PayloadSource(Protocol):
    """Content-only payload source hook (Q2, adjudicated A).

    The source supplies *bodies* for the non-read flows.  It cannot
    construct envelopes, choose provenance, influence the schedule/stage
    order/channel semantics, or write back into the executor.  Store reads
    stay executor-derived.  A failure is an assertion in the derivation
    pass, never a channel charge.

    ``coverage`` is a plain data ledger (``fid -> derived | synthetic |
    placeholder``) read by tests only; no behaviour branches on it.  For a
    physics source, ``derive_body`` asserts for any fid it is asked for
    that is not in its derived dispatch.
    """

    mode: str
    coverage: Mapping[int, str]

    def derive_body(
        self, fid: int, tick: int, *, board: MatBoard, debris: DebrisPool
    ) -> object: ...

    def on_commit(
        self,
        fid: int,
        envelope: Envelope,
        *,
        board: MatBoard,
        debris: DebrisPool,
    ) -> None: ...


class SyntheticSource:
    """The explicit plumbing sentinel: synthetic bodies, no physics.

    Reachable only by explicit construction; ``DryRun()`` defaults to it so
    the plumbing tests keep their historical synthetic behaviour.  Every
    non-read flow is covered as ``synthetic`` in the coverage ledger.
    ``PhysicsRun`` never uses this class.
    """

    mode = "synthetic"

    def __init__(self) -> None:
        self.coverage: dict[int, str] = {
            fid: "synthetic" for fid in sorted(NON_READ_FLOWS)
        }

    def derive_body(
        self, fid: int, tick: int, *, board: MatBoard, debris: DebrisPool
    ) -> object:
        if fid == 22:
            slot = (tick - 1) % 3
            return BiasPrimitive(f"bias-{slot}", _BIAS_STEPS[slot])
        if fid == 23:
            return MatOverlay(tick, f"cry-{tick}", "active")
        if fid == 24:
            head_id = board.head()[0]
            return MatOverlay(tick, head_id, "active")
        return None

    def on_commit(
        self,
        fid: int,
        envelope: Envelope,
        *,
        board: MatBoard,
        debris: DebrisPool,
    ) -> None:
        return None


class DryRun:
    """Mailbox-based executor of the ledger spine, plumbing only."""

    def __init__(
        self,
        source: PayloadSource | None = None,
        *,
        seed_board: bool = True,
    ) -> None:
        self.source: PayloadSource = (
            SyntheticSource() if source is None else source
        )
        self.mailboxes: dict[int, deque[FlowEvent]] = {
            f.fid: deque() for f in FLOW_LEDGER
        }
        self.board = MatBoard(seed=seed_board)
        self.debris = DebrisPool()
        self.derived: dict[int, Envelope] = {}
        self.emissions: list[tuple[int, int, Envelope]] = []
        self._head_provenance: dict[str, Provenance] = (
            {MAT_SEED[0]: Provenance(23, 0)} if seed_board else {}
        )
        self._d4_provenance: list[Provenance] = []
        #: Confidence-0 #22 writes absorbed as consumed no-ops (Q4).
        self.zero_confidence_absorbed: Counter[int] = Counter()
        self._tick_no = 0
        self.attempted: Counter[int] = Counter()
        self.accepted: Counter[int] = Counter()
        self.refused: Counter[int] = Counter()
        self.consumed: Counter[int] = Counter()
        self.missed: Counter[int] = Counter()
        self.sunk: Counter[int] = Counter()
        self.max_depth: Counter[int] = Counter()

    @property
    def tick_no(self) -> int:
        """The current tick index (0 before the first ``tick``)."""
        return self._tick_no

    def _touch(self, fid: int) -> None:
        depth = len(self.mailboxes[fid])
        if depth > self.max_depth[fid]:
            self.max_depth[fid] = depth

    def _derive(self) -> None:
        """Derivation pass: fill every flow's envelope from live state.

        Runs once at the start of the tick, after the clock advances and
        before any arrival. Bodies and provenance are built here, never in
        `_arrive`: store reads inherit the stored record's provenance, every
        non-read flow is born at this tick. The channel gate is deliberately
        not applied here; `_arrive` validates against the receiving channel.

        Source hook (Q2, adjudicated A): the executor keeps the store reads
        (#7 D4, #8/#25 D2, and the unimplemented D1/D3 reads) and delegates
        the non-read flows to `self.source.derive_body`.  A fixed
        ``PLACEHOLDER_FLOWS`` set is emitted as ``None`` by the executor and
        never asked of the source; that dispatch is the executor's contract,
        not a branch on the source's coverage ledger.  A source that cannot
        derive a body it is asked for raises in this pass, before any
        arrival, so the failure is an assertion and never a channel charge.

        Physics bodies are real values; the fixed cadence is preserved by
        exact-type no-op sentinels for the trigger-gated flows (#17, #19,
        #21, #23, #24) and the Q4 zero-confidence #22 gate body.
        """
        tick = self._tick_no
        envelopes: dict[int, Envelope] = {}

        def put(fid: int, body: object, provenance: Provenance) -> None:
            envelopes[fid] = Envelope(provenance, body)

        # D2 head reads (#8, #25) inherit the head record's provenance.
        # A physics-mode board may be empty between evictions; a read with
        # no resident is an empty body with self-origin provenance.
        residents = self.board.residents()
        if residents:
            head_id, epoch = residents[0]
            try:
                head_provenance = self._head_provenance[head_id]
            except KeyError:
                raise RuntimeError(
                    f"no stored provenance for D2 head {head_id!r}"
                ) from None
            put(8, MatOverlay(epoch, head_id, "stale"), head_provenance)
            put(25, MatOverlay(epoch, head_id, "stale"), head_provenance)
        else:
            put(8, None, Provenance(8, tick))
            put(25, None, Provenance(25, tick))

        # D4 read (#7) hands back the exact stored object, provenance and
        # all.  The D4 core is empty until a real sediment passes the gate.
        latest = self.board.latest_bias()
        if latest is None:
            put(7, None, Provenance(7, tick))
        else:
            put(7, latest, self._d4_provenance[-1])

        # Unimplemented D1/D3 reads: empty body, self-origin provenance.
        for fid in (11, 14, 15, 20, 26):
            put(fid, None, Provenance(fid, tick))

        # Non-read flows: fixed executor dispatch.  Placeholders are never
        # asked of the source; every other flow is a content-only source
        # derivation born at this tick.
        for fid in sorted(NON_READ_FLOWS):
            if fid in PLACEHOLDER_FLOWS:
                body: object = None
            else:
                body = self.source.derive_body(
                    fid, tick, board=self.board, debris=self.debris
                )
            put(fid, body, Provenance(fid, tick))

        self.derived = envelopes

    def _prune_head_provenance(self) -> None:
        """Drop provenance for overlays evicted from the mat."""
        resident_ids = {oid for oid, _epoch in self.board.residents()}
        for oid in [oid for oid in self._head_provenance
                    if oid not in resident_ids]:
            del self._head_provenance[oid]

    def _arrive(self, fid: int, *, copies: int = 1) -> None:
        """Transport the derived envelope to a capacity-1 channel.

        First arrival in stage order wins; ledger order is the tie-break
        (callers proceed in ascending fid within their phase). The body and
        its provenance were built by `_derive` -- this method constructs
        neither. A refused copy is discarded here (the same envelope object
        never enters a second box) and leaves a scalar excess charge,
        distinct from the miss (deficit) charge and never summed with it.
        O(1) per copy, exactly like a sink.
        """
        if copies < 1:
            raise ValueError(
                "starvation (k = 0) is deferred to the physics epoch"
            )
        envelope = self.derived[fid]
        require_channel_provenance(fid, envelope, self._tick_no)
        carrier = carrier_for(fid)
        for _ in range(copies):
            box = self.mailboxes[fid]
            self.attempted[fid] += 1
            if box:
                self.refused[fid] += 1
                continue
            box.append(carrier(payload=envelope))
            self.emissions.append((self._tick_no, fid, envelope))
            self.accepted[fid] += 1
            self._touch(fid)

    def _consume(self, fid: int, *, silent: bool = False) -> None:
        box = self.mailboxes[fid]
        if box:
            box.popleft()
            self.consumed[fid] += 1
            self._touch(fid)
        elif not silent:
            self.missed[fid] += 1

    def _absorb_writes(self) -> None:
        """Absorb committed writes; notify the source one-way, post-commit.

        Store effects: #19 deposits a real D3 body (the fixed-cadence no-op
        body is not deposited), #23 births / #24 revalidates a D2 record,
        and a confidence-0 #22 is a **consumed no-op** (Q4): the event is
        drained with the other write mailboxes, but no D4 state changes, so
        the identity ``B: 1 = 1 + 0 + 0`` holds on the absorbed path rather
        than by skipping the event.

        Patch attach (Q1-A) happens inside the source's one-way
        ``on_commit`` notification, after the record commit and before the
        capacity/byte evictions; eviction removes the record and its patch
        together.  A `require_complete` patch-bounds assertion runs in
        physics mode so a resident record without a patch fails loudly.
        """
        for fid in WRITE_FLOWS:
            self._consume(fid, silent=True)
        envelopes = {
            fid: envelope
            for tick, fid, envelope in self.emissions
            if tick == self._tick_no
        }

        # D3: deposit the real #19 body; the no-op sentinel is not stored.
        deposit = envelopes.get(19)
        if (
            deposit is not None
            and type(deposit.body) is DigestedDebrisBody
            and deposit.body.debris_id != NO_DEBRIS_ID
        ):
            self.debris.deposit(deposit.body, deposit.provenance, self._tick_no)

        # D2 record commits (#23 birth, #24 revalidation).  The fixed-cadence
        # no-op token leaves the store untouched.
        overlay = envelopes.get(23)
        if (
            overlay is not None
            and type(overlay.body) is MatOverlay
            and overlay.body.overlay_id != NOOP_OVERLAY_ID
        ):
            self.board.birth(overlay.body)
            self._head_provenance[overlay.body.overlay_id] = overlay.provenance
        overlay = envelopes.get(24)
        if (
            overlay is not None
            and type(overlay.body) is MatOverlay
            and overlay.body.overlay_id != NOOP_OVERLAY_ID
        ):
            self.board.revalidate(overlay.body)
            self._head_provenance[overlay.body.overlay_id] = overlay.provenance

        # One-way post-commit notification: the source attaches the patch
        # for the committed records (Q1-A).
        for fid in (23, 24):
            envelope = envelopes.get(fid)
            if (
                envelope is not None
                and type(envelope.body) is MatOverlay
                and envelope.body.overlay_id != NOOP_OVERLAY_ID
            ):
                self.source.on_commit(
                    fid, envelope, board=self.board, debris=self.debris
                )

        self.board.evict_if_needed(self._tick_no)
        self.board.enforce_patch_ceiling(self._tick_no)
        self._prune_head_provenance()

        # D4 sediment: confidence > 0 only.  A confidence-0 sentinel is the
        # real ratchet gate's derived output and is absorbed as a no-op.
        bias = envelopes.get(22)
        if bias is not None and type(bias.body) is BiasPrimitive:
            if bias.body.confidence > 0.0:
                self.board.sediment(bias.body)
                self._d4_provenance.append(bias.provenance)
                assert len(self._d4_provenance) == len(self.board.d4), (
                    "D4 provenance list out of step with board.d4"
                )
            else:
                self.zero_confidence_absorbed[22] += 1

        # Post-commit notifications for the remaining writes, after the
        # evictions, so a physics source can prune non-resident overlays and
        # the executor can align the store with the organ in the same tick.
        for fid in (19, 22):
            envelope = envelopes.get(fid)
            if envelope is not None:
                self.source.on_commit(
                    fid, envelope, board=self.board, debris=self.debris
                )
        self.board.assert_patch_bounds(
            require_complete=(self.source.mode == "physics")
        )

    def _drain_sinks(self) -> None:
        for fid in SINK_FLOWS:
            box = self.mailboxes[fid]
            self.sunk[fid] += len(box)
            box.clear()

    def tick(self, injections: Mapping[int, int] | None = None) -> None:
        """One tick: derivation pass, injections (with adversarial
        multiplicity), spine, write absorption, sink drain.

        `_derive()` runs first, after the clock advances and before any
        arrival: every flow's envelope is built from live state once.
        `injections[fid] = k` overrides that flow's canonical single
        arrival with k arrival attempts (k >= 1; starvation k = 0 is
        deferred to the physics epoch). All arrivals proceed in ledger
        fid order; later arrivals onto a full box are refused.
        """
        self._tick_no += 1
        self._derive()
        boosts = dict(injections or {})
        unknown = sorted(set(boosts) - set(self.mailboxes))
        if unknown:
            raise KeyError(f"unknown flow ids: {unknown}")
        plan = {fid: boosts.pop(fid, 1) for fid in PER_TICK_INJECTIONS}
        for fid, k in sorted(boosts.items()):
            plan[fid] = k
        for fid in sorted(plan):
            self._arrive(fid, copies=plan[fid])
        for stage in SPINE:
            for fid in stage.consumes:
                self._consume(fid)
            for fid in stage.emits:
                self._arrive(fid)
        self._absorb_writes()
        self._drain_sinks()

    def run(self, ticks: int) -> Report:
        if ticks < 1:
            raise ValueError("ticks must be >= 1")
        history: list[TickStat] = []
        for _ in range(ticks):
            before = (
                Counter(self.attempted),
                Counter(self.accepted),
                Counter(self.refused),
                Counter(self.consumed),
                Counter(self.sunk),
                Counter(self.missed),
            )
            self.tick()
            history.append(
                TickStat(
                    attempted=dict(self.attempted - before[0]),
                    accepted=dict(self.accepted - before[1]),
                    refused=dict(self.refused - before[2]),
                    consumed=dict(self.consumed - before[3]),
                    sunk=dict(self.sunk - before[4]),
                    missed=dict(self.missed - before[5]),
                )
            )
        return Report(
            ticks=ticks,
            attempted=dict(self.attempted),
            accepted=dict(self.accepted),
            refused=dict(self.refused),
            consumed=dict(self.consumed),
            missed=dict(self.missed),
            sunk=dict(self.sunk),
            history=tuple(history),
        )

    def in_transit(self) -> dict[int, int]:
        """Leftovers: events emitted but not yet consumed.

        Only the one-tick-delayed cycles (#10, #17) should be non-empty
        immediately after a whole number of ticks.
        """
        return {fid: len(box) for fid, box in self.mailboxes.items() if box}


def main(argv: list[str] | None = None) -> None:
    import sys

    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run = DryRun()
    report = run.run(ticks)
    labels = {f.fid: f.flow for f in FLOW_LEDGER}
    print(f"dry run: {ticks} ticks, {len(FLOW_LEDGER)} flows")
    print(
        f"{'#':>3}  {'flow':<38} {'accepted':>8} {'refused':>8} "
        f"{'consumed':>8} {'missed':>7}"
    )
    for fid in sorted(labels):
        print(
            f"{fid:>3}  {labels[fid]:<38} "
            f"{report.accepted.get(fid, 0):>8} "
            f"{report.refused.get(fid, 0):>8} "
            f"{report.consumed.get(fid, 0):>8} "
            f"{report.missed.get(fid, 0):>7}"
        )
    print("sink drain:", {labels[k]: v for k, v in sorted(report.sunk.items())})
    live = all(report.accepted.get(fid) == ticks for fid in labels)
    print("liveness (every flow, every tick):", live)
    transit = {labels[fid]: n for fid, n in sorted(run.in_transit().items())}
    print("in transit:", transit)


if __name__ == "__main__":
    main()
