"""Dry-run contract tests: plumbing must be live, acyclic-safe, and
thermodynamically terminal.
"""
from __future__ import annotations

import unittest

from amateur.dryrun import (
    PER_TICK_INJECTIONS,
    SPINE,
    DryRun,
)
from amateur.ledger import FLOW_LEDGER


class TestSpineConsistency(unittest.TestCase):
    def test_spine_covers_every_flow_exactly_once(self) -> None:
        stage_emits = [fid for stage in SPINE for fid in stage.emits]
        self.assertEqual(
            sorted(stage_emits + list(PER_TICK_INJECTIONS)),
            [f.fid for f in FLOW_LEDGER],
            msg="spine emits + injections must partition the ledger",
        )

    def test_spine_stage_names_match_ledger_processes(self) -> None:
        process_names = {f.source for f in FLOW_LEDGER if f.source.startswith(("0", "1", "2", "3", "4", "5", "6"))}
        stage_names = {stage.name for stage in SPINE}
        self.assertEqual(stage_names, process_names)

    def test_every_consumed_flow_is_emitted_somewhere(self) -> None:
        emits = {fid for stage in SPINE for fid in stage.emits}
        emits |= set(PER_TICK_INJECTIONS)
        for stage in SPINE:
            for fid in stage.consumes:
                self.assertIn(fid, emits, msg=f"stage {stage.name} consumes unsourced #{fid}")

    def test_cycle_edges_are_delayed_not_impossible(self) -> None:
        # A cycle edge can be served same-tick only if its emitter runs
        # before its consumer. #10 (P3 -> P1) and #17 (P5 -> P4) must be
        # emitted by later stages than their consumers: one-tick delay,
        # never same-tick deadlock.
        for fid in (10, 17):
            emits_at = [i for i, s in enumerate(SPINE) if fid in s.emits]
            consumes_at = [i for i, s in enumerate(SPINE) if fid in s.consumes]
            self.assertEqual(len(emits_at), 1, msg=f"#{fid}")
            self.assertEqual(len(consumes_at), 1, msg=f"#{fid}")
            self.assertGreater(
                emits_at[0], consumes_at[0], msg=f"#{fid} would deadlock"
            )


class TestDryRun(unittest.TestCase):
    def test_every_flow_traversed_every_tick(self) -> None:
        report = DryRun().run(8)
        for fid in (f.fid for f in FLOW_LEDGER):
            self.assertEqual(report.emitted.get(fid, 0), 8, msg=f"flow #{fid} starved")

    def test_first_tick_misses_are_exactly_the_cycles(self) -> None:
        report = DryRun().run(8)
        self.assertEqual(report.missed, {10: 1, 17: 1})

    def test_sinks_receive_only_and_exactly_their_flows(self) -> None:
        report = DryRun().run(8)
        self.assertEqual(report.sunk, {2: 8, 4: 8, 5: 8})

    def test_only_cycles_left_in_transit(self) -> None:
        run = DryRun()
        run.run(8)
        transit = {fid for fid, n in run.in_transit().items() if n}
        self.assertEqual(transit, {10, 17})

    def test_deadlock_free_and_bounded_over_many_ticks(self) -> None:
        report = DryRun().run(200)
        self.assertEqual(sum(report.emitted.values()), 200 * len(FLOW_LEDGER))
        for fid, n in report.missed.items():
            self.assertEqual(n, 1, msg=f"misses grow for #{fid}")

    def test_rejects_non_positive_ticks(self) -> None:
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                DryRun().run(bad)


if __name__ == "__main__":
    unittest.main()
