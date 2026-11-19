"""Bookkeeping laws of the dry-run executor over the mailbox model.

- Conservation: every tick, emitted = consumed + missed + sunk + delta
  in-transit. Nothing is created off-ledger or lost between mailboxes.
- Liveness: every one of the 26 flows fires exactly once per tick.
- Stationarity: from tick 2 on, the per-tick tick stats are identical;
  tick 1 differs from tick 2 only in the two cycle warm-up misses.
- Master invariants, empirically: steady depth profile {10: 1, 17: 1},
  cumulative misses frozen at one per cycle, sinks draining exactly one
  event per flow per tick, absorbed writes never miss, and the pure
  pointwise recurrence (step_flow) tracks the executor tick for tick.
"""
from __future__ import annotations

import unittest

from amateur.dryrun import (
    FLOW_OPS,
    PER_TICK_INJECTIONS,
    WRITE_FLOWS,
    DryRun,
    step_flow,
)
from amateur.ledger import FLOW_LEDGER

SINKS = (2, 4, 5)


class TestConservation(unittest.TestCase):
    def test_conservation_identity_per_run(self) -> None:
        for ticks in (1, 2, 8, 64):
            run = DryRun()
            transit_before = sum(run.in_transit().values())
            report = run.run(ticks)
            transit_after = sum(run.in_transit().values())
            delta = transit_after - transit_before
            # Misses remove nothing from the mailboxes: the balance is
            # emitted = popped(consumed) + sunk + delta in-transit.
            self.assertEqual(
                sum(report.emitted.values()),
                sum(report.consumed.values())
                + sum(report.sunk.values())
                + delta,
                msg=f"ticks={ticks}: events leak or are invented",
            )

    def test_every_tick_is_fully_live(self) -> None:
        for tick_stat in DryRun().run(8).history:
            self.assertEqual(sum(tick_stat.emitted.values()), len(FLOW_LEDGER))
            for flow in FLOW_LEDGER:
                self.assertEqual(tick_stat.emitted.get(flow.fid, 0), 1)


class TestStationarity(unittest.TestCase):
    def test_tick_stats_identical_from_tick_two(self) -> None:
        history = DryRun().run(16).history
        for tick_stat in history[1:]:
            self.assertEqual(tick_stat, history[1])

    def test_warmup_differs_only_in_cycle_warmup(self) -> None:
        history = DryRun().run(16).history
        self.assertEqual(history[0].missed, {10: 1, 17: 1})
        self.assertEqual(history[1].missed, {})
        for field in ("emitted", "sunk"):
            self.assertEqual(getattr(history[0], field), getattr(history[1], field))
        steady = history[1].consumed
        fids = sorted(FLOW_OPS)
        warm = {
            fid: steady.get(fid, 0) - (1 if fid in (10, 17) else 0)
            for fid in fids
        }
        self.assertEqual(
            {fid: history[0].consumed.get(fid, 0) for fid in fids}, warm
        )


class TestMasterInvariants(unittest.TestCase):
    def test_steady_state_shape_from_tick_one(self) -> None:
        for ticks in (1, 2, 3, 5, 64):
            run = DryRun()
            report = run.run(ticks)
            self.assertEqual(
                {fid: n for fid, n in run.in_transit().items() if n},
                {10: 1, 17: 1},
                msg=f"ticks={ticks}: in-transit profile",
            )
            self.assertEqual(
                {fid: n for fid, n in report.missed.items() if n},
                {10: 1, 17: 1},
                msg=f"ticks={ticks}: cumulative misses",
            )
            self.assertEqual(
                {fid: n for fid, n in report.sunk.items() if n},
                {fid: ticks for fid in SINKS},
                msg=f"ticks={ticks}: sink drain",
            )

    def test_absorbed_writes_never_miss(self) -> None:
        report = DryRun().run(64)
        for fid in WRITE_FLOWS:
            self.assertNotIn(fid, report.missed, msg=f"write flow #{fid} missed")
            self.assertEqual(report.consumed.get(fid, 0), 64, msg=f"#{fid}")

    def test_scalar_recurrence_matches_executor_and_theorems(self) -> None:
        boxes = {fid: 0 for fid in FLOW_OPS}
        missed = {fid: 0 for fid in FLOW_OPS}
        sunk = {fid: 0 for fid in FLOW_OPS}
        run = DryRun()
        for tick in range(1, 33):
            for fid, ops in FLOW_OPS.items():
                boxes[fid], missed_delta, sunk_delta = step_flow(fid, boxes[fid])
                missed[fid] += missed_delta
                sunk[fid] += sunk_delta
            report = run.run(1)
            self.assertEqual({fid: n for fid, n in boxes.items() if n}, {10: 1, 17: 1})
            self.assertEqual({fid: n for fid, n in missed.items() if n}, {10: 1, 17: 1})
            self.assertEqual(
                {fid: n for fid, n in sunk.items() if n},
                {fid: tick for fid in SINKS},
            )
            self.assertEqual(
                {fid: report.emitted.get(fid, 0) for fid in FLOW_OPS},
                {fid: tick for fid in FLOW_OPS},
            )

    def test_injections_partition_the_live_flows(self) -> None:
        self.assertTrue(set(PER_TICK_INJECTIONS).issubset(FLOW_OPS.keys()))


if __name__ == "__main__":
    unittest.main()
