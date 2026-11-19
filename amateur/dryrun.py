"""Dry run: pure event plumbing along the ledger spine.

No physics — every stage consumes its ledger inputs and emits its ledger
outputs with placeholder payloads. The driving gradient (#1, #3) and the
geometry echoes (#7, #8, #11, #14, #15, #20, #25, #26) are injected every
tick; the two in-tick cycles (#10 Resolved Trajectory, #17 Collapse
Mandate) are one-tick-delayed mailboxes.

The dry run exists to prove the spine is deadlock-free, every flow is
traversed, and degraded throughput reaches only the terminal sink.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterator  # noqa: F401

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

_WRITE_FLOWS: tuple[int, ...] = tuple(
    f.fid for f in FLOW_LEDGER if f.target in STORES and "(read)" not in f.flow
)

_SINK_FLOWS: tuple[int, ...] = tuple(
    f.fid for f in FLOW_LEDGER if f.target in (E1, E2)
)


@dataclass
class Report:
    ticks: int
    emitted: dict[int, int]
    missed: dict[int, int]
    sunk: dict[int, int]

    @property
    def traversed_flows(self) -> set[int]:
        return {fid for fid, n in self.emitted.items() if n}


class DryRun:
    """Mailbox-based executor of the ledger spine, plumbing only."""

    def __init__(self) -> None:
        self.mailboxes: dict[int, deque[FlowEvent]] = {
            f.fid: deque() for f in FLOW_LEDGER
        }
        self.emitted: Counter[int] = Counter()
        self.missed: Counter[int] = Counter()
        self.sunk: Counter[int] = Counter()

    def _emit(self, fid: int) -> None:
        carrier = carrier_for(fid)
        self.mailboxes[fid].append(carrier(payload=None))
        self.emitted[fid] += 1

    def _consume(self, fid: int) -> None:
        box = self.mailboxes[fid]
        if box:
            box.popleft()
        else:
            self.missed[fid] += 1

    def _drain_sinks(self) -> None:
        for fid in _SINK_FLOWS:
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
        for fid in _WRITE_FLOWS:
            self._consume(fid)
        self._drain_sinks()

    def run(self, ticks: int) -> Report:
        if ticks < 1:
            raise ValueError("ticks must be >= 1")
        for _ in range(ticks):
            self.tick()
        return Report(
            ticks=ticks,
            emitted=dict(self.emitted),
            missed=dict(self.missed),
            sunk=dict(self.sunk),
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
    print(f"{'#':>3}  {'flow':<38} {'emitted':>7} {'missed':>7}")
    for fid in sorted(labels):
        print(
            f"{fid:>3}  {labels[fid]:<38} "
            f"{report.emitted.get(fid, 0):>7} {report.missed.get(fid, 0):>7}"
        )
    print("sunks:", {labels[k]: v for k, v in sorted(report.sunk.items())})
    live = all(report.emitted.get(fid) == ticks for fid in labels)
    print("liveness (every flow, every tick):", live)
    transit = {labels[fid]: n for fid, n in sorted(run.in_transit().items())}
    print("in transit:", transit)


if __name__ == "__main__":
    main()
