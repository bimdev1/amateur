"""Bookkeeping laws of the dry-run executor over the mailbox model.

- Conservation (adjudicated identities, per flow per tick):
  attempted = accepted + refused, and accepted = consumed + delta-sunk
  + delta-depth. Nothing is created off-ledger or lost between
  mailboxes; refused copies never enter identity B, misses are deficit
  charges on the expectation side and remove nothing.
- Liveness: every one of the 27 flows fires exactly once per tick, and
  the canonical schedule refuses nothing.
- Stationarity: from tick 2 on, the per-tick tick stats are identical;
  tick 1 differs from tick 2 only in the two cycle warm-up misses.
- Master invariants, empirically: steady depth profile {10: 1, 17: 1},
  cumulative misses frozen at one per cycle, sinks draining exactly one
  event per flow per tick, absorbed writes never miss, and the pure
  pointwise recurrence (step_flow) tracks the executor tick for tick.
- Intra-tick bounds: prefix_depth mirrors the Lean `fold_box_bounded`
  theorem -- from any compliant depth, every event boundary of the word
  keeps depth <= 1.
"""
from __future__ import annotations

import unittest

from amateur.dryrun import (
    FLOW_OPS,
    PER_TICK_INJECTIONS,
    WRITE_FLOWS,
    DryRun,
    prefix_depth,
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
            # Refused copies never enter the balance; misses remove
            # nothing: accepted = popped(consumed) + sunk + delta.
            self.assertEqual(
                sum(report.accepted.values()),
                sum(report.consumed.values())
                + sum(report.sunk.values())
                + delta,
                msg=f"ticks={ticks}: events leak or are invented",
            )
            self.assertEqual(
                sum(report.attempted.values()),
                sum(report.accepted.values()) + sum(report.refused.values()),
                msg=f"ticks={ticks}: identity A broken over the run",
            )
            self.assertEqual(
                sum(report.refused.values()), 0,
                msg=f"ticks={ticks}: canonical schedule must not refuse",
            )

    def test_every_tick_is_fully_live(self) -> None:
        for tick_stat in DryRun().run(8).history:
            self.assertEqual(sum(tick_stat.accepted.values()), len(FLOW_LEDGER))
            for flow in FLOW_LEDGER:
                self.assertEqual(tick_stat.accepted.get(flow.fid, 0), 1)
                self.assertEqual(tick_stat.attempted.get(flow.fid, 0), 1)
                self.assertEqual(tick_stat.refused.get(flow.fid, 0), 0)


class TestRejectWithCharge(unittest.TestCase):
    def test_channel_identities_per_flow_per_tick(self) -> None:
        run = DryRun()
        fids = [f.fid for f in FLOW_LEDGER]
        depth = dict.fromkeys(fids, 0)
        for _ in range(16):
            before = {
                fid: (
                    run.attempted[fid],
                    run.accepted[fid],
                    run.refused[fid],
                    run.consumed[fid],
                    run.sunk[fid],
                )
                for fid in fids
            }
            depth_before = depth
            run.tick()
            depth = {fid: len(run.mailboxes[fid]) for fid in fids}
            for fid in fids:
                attempt = run.attempted[fid] - before[fid][0]
                accept = run.accepted[fid] - before[fid][1]
                refuse = run.refused[fid] - before[fid][2]
                consume = run.consumed[fid] - before[fid][3]
                sunk = run.sunk[fid] - before[fid][4]
                self.assertEqual(attempt, accept + refuse, msg=f"#{fid}: A")
                self.assertEqual(
                    accept,
                    consume + sunk + depth[fid] - depth_before[fid],
                    msg=f"#{fid}: B",
                )
                self.assertLessEqual(depth[fid], 1, msg=f"#{fid}: bounds")

    def test_prefix_depth_never_exceeds_one(self) -> None:
        for fid in FLOW_OPS:
            for start in (0, 1):
                for n in range(len(FLOW_OPS[fid]) + 1):
                    self.assertLessEqual(
                        prefix_depth(fid, start, n),
                        1,
                        msg=f"#{fid} from {start} after {n} events",
                    )

    def test_prefix_depth_agrees_with_step_flow(self) -> None:
        for fid in FLOW_OPS:
            self.assertEqual(
                prefix_depth(fid, 0, len(FLOW_OPS[fid])),
                step_flow(fid, 0).box,
                msg=f"#{fid}",
            )

    def test_prefix_length_validation(self) -> None:
        for bad in (-1, len(FLOW_OPS[1]) + 1):
            with self.assertRaises(ValueError):
                prefix_depth(1, 0, bad)


class TestStationarity(unittest.TestCase):
    def test_tick_stats_identical_from_tick_two(self) -> None:
        history = DryRun().run(16).history
        for tick_stat in history[1:]:
            self.assertEqual(tick_stat, history[1])

    def test_warmup_differs_only_in_cycle_warmup(self) -> None:
        history = DryRun().run(16).history
        self.assertEqual(history[0].missed, {10: 1, 17: 1})
        self.assertEqual(history[1].missed, {})
        for field in ("accepted", "sunk"):
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
        refused = {fid: 0 for fid in FLOW_OPS}
        sunk = {fid: 0 for fid in FLOW_OPS}
        run = DryRun()
        for tick in range(1, 33):
            for fid, ops in FLOW_OPS.items():
                delta = step_flow(fid, boxes[fid])
                boxes[fid] = delta.box
                missed[fid] += delta.miss
                refused[fid] += delta.refused
                sunk[fid] += delta.sunk
            report = run.run(1)
            self.assertEqual({fid: n for fid, n in boxes.items() if n}, {10: 1, 17: 1})
            self.assertEqual({fid: n for fid, n in missed.items() if n}, {10: 1, 17: 1})
            self.assertEqual(
                {fid: n for fid, n in refused.items() if n},
                {},
                msg=f"tick {tick}: canonical refusal",
            )
            self.assertEqual(
                {fid: n for fid, n in sunk.items() if n},
                {fid: tick for fid in SINKS},
            )
            self.assertEqual(
                {fid: report.accepted.get(fid, 0) for fid in FLOW_OPS},
                {fid: tick for fid in FLOW_OPS},
            )

    def test_injections_partition_the_live_flows(self) -> None:
        self.assertTrue(set(PER_TICK_INJECTIONS).issubset(FLOW_OPS.keys()))


if __name__ == "__main__":
    unittest.main()
