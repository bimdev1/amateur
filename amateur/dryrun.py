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
from typing import Mapping, NamedTuple

from .events import FlowEvent, carrier_for
from .ledger import E1, E2, FLOW_LEDGER, STORES
from .payloads import (
    BiasPrimitive,
    D2_FLOWS,
    D2_WRITES,
    D4_FLOWS,
    D4_WRITES,
    MatBoard,
    MatOverlay,
    require_payload,
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


def _payload_for(fid: int, tick: int, board: MatBoard) -> object:
    """Synthetic payload for the typed geometry channels.

    D2: the stale read (#25) delivers the board head; the ratchet's
    revalidation (#24) and surviving-crystal birth (#23) carry the current
    epoch. D4: the base-ops read (#7) delivers the latest bias; the
    sedimentation (#22) mints a rotating confidence primitive. Sparse
    reads may be None when the store has not yet filled.
    """
    if fid in D2_WRITES:
        stale_id, _ = board.head()
        if fid == 24:
            return MatOverlay(tick, stale_id, "active")
        return MatOverlay(tick, f"cry-{tick}", "active")
    if fid in D2_FLOWS:
        overlay_id, epoch = board.head()
        return MatOverlay(epoch, overlay_id, "stale")
    if fid in D4_WRITES:
        slot = (tick - 1) % 3
        return BiasPrimitive(f"bias-{slot}", _BIAS_STEPS[slot])
    if fid in D4_FLOWS:
        latest = board.latest_bias()
        return latest
    return None


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


class DryRun:
    """Mailbox-based executor of the ledger spine, plumbing only."""

    def __init__(self) -> None:
        self.mailboxes: dict[int, deque[FlowEvent]] = {
            f.fid: deque() for f in FLOW_LEDGER
        }
        self.board = MatBoard()
        self.emissions: list[tuple[int, int, object]] = []
        self._tick_no = 0
        self.attempted: Counter[int] = Counter()
        self.accepted: Counter[int] = Counter()
        self.refused: Counter[int] = Counter()
        self.consumed: Counter[int] = Counter()
        self.missed: Counter[int] = Counter()
        self.sunk: Counter[int] = Counter()
        self.max_depth: Counter[int] = Counter()

    def _touch(self, fid: int) -> None:
        depth = len(self.mailboxes[fid])
        if depth > self.max_depth[fid]:
            self.max_depth[fid] = depth

    def _arrive(self, fid: int, *, copies: int = 1) -> None:
        """Reject-with-charge arrival at a capacity-1 channel (adjudicated).

        First arrival in stage order wins; ledger order is the tie-break
        (callers proceed in ascending fid within their phase). A refused
        copy is discarded here -- the event object never materializes, no
        payload is minted -- and leaves a scalar excess charge, distinct
        from the miss (deficit) charge and never summed with it. O(1)
        per copy, exactly like a sink.
        """
        if copies < 1:
            raise ValueError(
                "starvation (k = 0) is deferred to the physics epoch"
            )
        carrier = carrier_for(fid)
        for _ in range(copies):
            box = self.mailboxes[fid]
            self.attempted[fid] += 1
            if box:
                self.refused[fid] += 1
                continue
            payload = _payload_for(fid, self._tick_no, self.board)
            require_payload(fid, payload)
            box.append(carrier(payload=payload))
            self.emissions.append((self._tick_no, fid, payload))
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
        for fid in WRITE_FLOWS:
            self._consume(fid, silent=True)
        payloads = {
            fid: payload
            for tick, fid, payload in self.emissions
            if tick == self._tick_no
        }
        overlay = payloads.get(23)
        if isinstance(overlay, MatOverlay):
            self.board.birth(overlay)
        overlay = payloads.get(24)
        if isinstance(overlay, MatOverlay):
            self.board.revalidate(overlay)
        self.board.evict_if_needed(self._tick_no)
        bias = payloads.get(22)
        if isinstance(bias, BiasPrimitive):
            self.board.sediment(bias)

    def _drain_sinks(self) -> None:
        for fid in SINK_FLOWS:
            box = self.mailboxes[fid]
            self.sunk[fid] += len(box)
            box.clear()

    def tick(self, injections: Mapping[int, int] | None = None) -> None:
        """One tick: injections (with adversarial multiplicity), spine,
        write absorption, sink drain.

        `injections[fid] = k` overrides that flow's canonical single
        arrival with k arrival attempts (k >= 1; starvation k = 0 is
        deferred to the physics epoch). All arrivals proceed in ledger
        fid order; later arrivals onto a full box are refused.
        """
        self._tick_no += 1
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
