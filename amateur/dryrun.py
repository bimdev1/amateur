"""Dry run: pure event plumbing along the ledger spine.

No physics — every stage consumes its ledger inputs and emits its ledger
outputs with placeholder payloads. The driving gradient (#1, #3) and the
geometry echoes (#7, #8, #11, #14, #15, #20, #25, #26) are injected every
tick; read-consumes expect data (a miss is counted), store absorbs are
best-effort; the two in-tick cycles (#10 Resolved Trajectory, #17 Collapse
Mandate) are one-tick-delayed mailboxes.

Verification: `flow_ops` and `step_flow` define a pure, per-flow
independent recurrence over mailbox depths; `formal/DryRun.lean` proves
its long-run shape (boundedness, cycle delays, sink rates, miss freezing)
and `tests/test_lean_lockstep.py` proves the generated Lean op table
matches this module's ledger-derived table.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from .events import FlowEvent, carrier_for
from .ledger import E1, E2, FLOW_LEDGER, STORES


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


def step_flow(fid: int, box: int) -> tuple[int, int, int]:
    """Pure step of one flow's mailbox. Returns (box', missed_delta, sunk_delta)."""
    missed = sunk = 0
    for op in FLOW_OPS[fid]:
        if op == "emit":
            box += 1
        elif op in ("consume", "absorb"):
            if box > 0:
                box -= 1
            else:
                missed += 1
        else:
            sunk += box
            box = 0
    return box, missed, sunk


@dataclass(frozen=True)
class TickStat:
    emitted: dict[int, int]
    consumed: dict[int, int]
    sunk: dict[int, int]
    missed: dict[int, int]


@dataclass
class Report:
    ticks: int
    emitted: dict[int, int]
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
        self.emitted: Counter[int] = Counter()
        self.consumed: Counter[int] = Counter()
        self.missed: Counter[int] = Counter()
        self.sunk: Counter[int] = Counter()

    def _emit(self, fid: int) -> None:
        carrier = carrier_for(fid)
        self.mailboxes[fid].append(carrier(payload=None))
        self.emitted[fid] += 1

    def _consume(self, fid: int, *, silent: bool = False) -> None:
        box = self.mailboxes[fid]
        if box:
            box.popleft()
            self.consumed[fid] += 1
        elif not silent:
            self.missed[fid] += 1

    def _absorb_writes(self) -> None:
        for fid in WRITE_FLOWS:
            self._consume(fid, silent=True)

    def _drain_sinks(self) -> None:
        for fid in SINK_FLOWS:
            box = self.mailboxes[fid]
            self.sunk[fid] += len(box)
            box.clear()

    def tick(self) -> None:
        for fid in PER_TICK_INJECTIONS:
            self._emit(fid)
        for stage in SPINE:
            for fid in stage.consumes:
                self._consume(fid)
            for fid in stage.emits:
                self._emit(fid)
        self._absorb_writes()
        self._drain_sinks()

    def run(self, ticks: int) -> Report:
        if ticks < 1:
            raise ValueError("ticks must be >= 1")
        history: list[TickStat] = []
        for _ in range(ticks):
            before = (
                Counter(self.emitted),
                Counter(self.consumed),
                Counter(self.sunk),
                Counter(self.missed),
            )
            self.tick()
            history.append(
                TickStat(
                    emitted=dict(self.emitted - before[0]),
                    consumed=dict(self.consumed - before[1]),
                    sunk=dict(self.sunk - before[2]),
                    missed=dict(self.missed - before[3]),
                )
            )
        return Report(
            ticks=ticks,
            emitted=dict(self.emitted),
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
    print(f"{'#':>3}  {'flow':<38} {'emitted':>7} {'consumed':>8} {'missed':>7}")
    for fid in sorted(labels):
        print(
            f"{fid:>3}  {labels[fid]:<38} "
            f"{report.emitted.get(fid, 0):>7} "
            f"{report.consumed.get(fid, 0):>8} "
            f"{report.missed.get(fid, 0):>7}"
        )
    print("sink drain:", {labels[k]: v for k, v in sorted(report.sunk.items())})
    live = all(report.emitted.get(fid) == ticks for fid in labels)
    print("liveness (every flow, every tick):", live)
    transit = {labels[fid]: n for fid, n in sorted(run.in_transit().items())}
    print("in transit:", transit)


if __name__ == "__main__":
    main()
