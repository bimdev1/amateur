"""Dry-run contract tests: plumbing must be live, acyclic-safe, and
thermodynamically terminal.
"""
from __future__ import annotations

import unittest

from amateur.dryrun import (
    PER_TICK_INJECTIONS,
    PLACEHOLDER_FLOWS,
    SPINE,
    SyntheticSource,
    DryRun,
)
from amateur.ledger import FLOW_LEDGER
from amateur.payloads import (
    COHORT_KINDS,
    CostErrorBody,
    BiasPrimitive,
    DissipatedEntropyBody,
    EmergentCrystalBody,
    MatOverlay,
    noop_crystal,
    noop_overlay,
)


def run_max_depth_max(run: DryRun) -> int:
    return max(run.max_depth.values())


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
            self.assertEqual(
                report.accepted.get(fid, 0), 8, msg=f"flow #{fid} starved"
            )
            self.assertEqual(
                report.attempted.get(fid, 0), 8, msg=f"flow #{fid} unattempted"
            )
            self.assertEqual(
                report.refused.get(fid, 0), 0,
                msg=f"canonical schedule refuses #{fid}",
            )

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
        run = DryRun()
        report = run.run(200)
        self.assertEqual(sum(report.accepted.values()), 200 * len(FLOW_LEDGER))
        self.assertEqual(sum(report.refused.values()), 0)
        self.assertEqual(run_max_depth_max(run), 1)
        for fid, n in report.missed.items():
            self.assertEqual(n, 1, msg=f"misses grow for #{fid}")

    def test_starvation_and_unknown_flow_rejected(self) -> None:
        run = DryRun()
        run.tick()
        with self.assertRaises(ValueError):
            run.tick(injections={1: 0})
        with self.assertRaises(KeyError):
            run.tick(injections={99: 2})

    def test_rejects_non_positive_ticks(self) -> None:
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                DryRun().run(bad)


class _ShadowSource(SyntheticSource):
    """Synthetic source with a candidate + harmful shadow evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.observed_ticks = 0

    def derive_body(self, fid, tick, *, board, debris):
        if fid == 18:
            if tick == 1:
                return EmergentCrystalBody(
                    "cand-shadow", (0.0,) * 64, 0.0, 1
                )
            return noop_crystal()
        return super().derive_body(fid, tick, board=board, debris=debris)

    def shadow_evidence(self, tick):
        if 2 <= tick <= 5:
            self.observed_ticks += 1
            return {"cand-shadow": (0.6, 0.4)}  # ratio 1.5, harmful
        return {}


class _PhysicsSource(SyntheticSource):
    mode = "physics"


class TestRound11Executor(unittest.TestCase):
    def test_16_is_source_derived_never_placeholder(self) -> None:
        run = DryRun()
        run.run(4)
        self.assertNotIn(16, PLACEHOLDER_FLOWS)
        bodies = [
            env.body for _tick, fid, env in run.emissions if fid == 16
        ]
        self.assertEqual(len(bodies), 4)
        for body in bodies:
            self.assertIsInstance(body, CostErrorBody)
        self.assertEqual(run.source.coverage[16], "synthetic")

    def test_d3_read_bodies_and_destructive_27(self) -> None:
        run = DryRun()
        run.run(6)
        emitted = {
            fid: env.body for _tick, fid, env in run.emissions if _tick == 6
        }
        self.assertIsNotNone(emitted[20])
        self.assertIsNotNone(emitted[21])
        self.assertIsNotNone(emitted[26])
        self.assertIsInstance(emitted[27], DissipatedEntropyBody)
        self.assertGreater(len(emitted[27].cohort_ids), 0)
        self.assertGreater(len(run.debris.drains), 0)
        resident = {c.cohort_id for c in run.debris.bodies()}
        self.assertFalse(
            resident & {c.cohort_id for c in run.debris.drains}
        )

    def test_harmful_survivor_is_surfaced_not_prefiltered(self) -> None:
        source = _ShadowSource()
        run = DryRun(source=source)
        run.run(5)
        survivors = [
            env.body for _tick, fid, env in run.emissions
            if fid == 23 and env.body is not None
        ]
        self.assertTrue(survivors, "the harmful survivor never surfaced")
        self.assertFalse(survivors[-1].qualified)
        self.assertGreaterEqual(survivors[-1].ratio, 1.0)

    def test_refused_27_copies_never_drain(self) -> None:
        run = DryRun()
        run.run(2)
        before_drains = len(run.debris.drains)
        run.tick(injections={27: 3})
        accepted = [
            env for tick, fid, env in run.emissions
            if tick == run.tick_no and fid == 27
        ]
        self.assertEqual(len(accepted), 1)  # first arrival wins
        self.assertEqual(run.refused[27], 2)
        body = accepted[0].body
        self.assertIsInstance(body, DissipatedEntropyBody)
        self.assertEqual(
            len(run.debris.drains) - before_drains, 1
        )
        self.assertEqual(
            {c.cohort_id for c in run.debris.drains[-1:]}
            & set(body.cohort_ids),
            set(body.cohort_ids),
        )

    def test_physics_24_lineage_gate_raises(self) -> None:
        run = DryRun(source=_PhysicsSource())
        with self.assertRaises(AssertionError):
            run.tick()

    def test_physics_22_lineage_gate_raises(self) -> None:
        class Source(_PhysicsSource):
            def derive_body(self, fid, tick, *, board, debris):
                if fid == 24:
                    if tick == 1:
                        return noop_overlay()
                    return MatOverlay(
                        tick, board.head()[0], "active",
                        motif_lineage=f"motif-{tick - 1}",
                    )
                if fid == 22:
                    if tick == 1:
                        return BiasPrimitive("b", 0.0)
                    return BiasPrimitive("b", 0.5, "no-such-candidate")
                return super().derive_body(fid, tick, board=board, debris=debris)

            def on_commit(self, fid, envelope, *, board, debris):
                if fid == 18:
                    from types import SimpleNamespace

                    board.attach_candidate_patch(
                        envelope.body.candidate_id,
                        SimpleNamespace(
                            overlay_id=envelope.body.candidate_id,
                            nbytes=1,
                            param_count=1,
                        ),
                    )

        run = DryRun(source=Source())
        run.tick()  # #24 noop, Q4 zero-confidence: accepted wiring
        with self.assertRaises(AssertionError):
            run.tick()

    def test_22_gate_refuses_unqualified_survivor(self) -> None:
        class Source(_ShadowSource):
            mode = "physics"

            def derive_body(self, fid, tick, *, board, debris):
                if fid == 24:
                    return noop_overlay()
                if fid == 22:
                    if tick < 5:
                        return BiasPrimitive("b", 0.0)
                    return BiasPrimitive("b", 0.5, "cand-shadow")
                return super().derive_body(fid, tick, board=board, debris=debris)

            def on_commit(self, fid, envelope, *, board, debris):
                return None

        run = DryRun(source=Source())
        with self.assertRaises(AssertionError):
            for _ in range(5):
                run.tick()

    def test_1000_tick_profile(self) -> None:
        run = DryRun()
        report = run.run(1000)
        self.assertEqual(sum(report.accepted.values()), 27_000)
        self.assertEqual(sum(report.refused.values()), 0)
        self.assertEqual(report.sunk, {2: 1000, 4: 1000, 5: 1000})
        self.assertEqual(report.missed, {10: 1, 17: 1})
        self.assertEqual(
            {fid: n for fid, n in run.in_transit().items() if n},
            {10: 1, 17: 1},
        )
        self.assertEqual(max(run.max_depth.values()), 1)
        balance = run.debris.mass_balance()
        self.assertEqual(
            balance["deposits"],
            balance["resident"] + balance["drained"] + balance["evicted"],
        )
        # Both named accounts (prereg §7) close on the real run.  The flat
        # keys above are the Round-11 surface `run.py` stores every tick and
        # are unchanged; the two accounts are reported beside them.
        conservation = balance["conservation_account"]
        issuance = balance["issuance_account"]
        self.assertTrue(conservation["holds"], msg=conservation)
        self.assertTrue(issuance["holds"], msg=issuance)
        self.assertEqual(
            conservation["identity"],
            "deposits = resident + drained + evicted",
        )
        self.assertEqual(
            issuance["formula"],
            "deposits = promoted + resident_unpromoted "
            "+ drained_unpromoted + evicted",
        )
        # Measured 1000-tick profile, pinned: 32 cohort capacity, 4 cohorts
        # per #19 deposit, TTL drainage, FIFO capacity eviction.
        self.assertEqual(
            (conservation["deposits"], conservation["resident"],
             conservation["drained"], conservation["evicted"],
             conservation["ceiling_evicted"]),
            (4000, 32, 999, 2969, 0),
        )
        self.assertEqual(
            (issuance["promoted"], issuance["resident_unpromoted"],
             issuance["drained_unpromoted"], issuance["evicted"],
             issuance["evicted_promoted"], issuance["issued"]),
            (0, 32, 999, 2969, 0, 4000),
        )
        self.assertIs(balance["issuance"], issuance)

    def test_mass_balance_flat_keys_are_the_round11_surface(self) -> None:
        """`run.py` stores this dict per tick: no existing key may move."""
        run = DryRun()
        run.run(3)
        balance = run.debris.mass_balance()
        for key in (
            "deposits",
            "resident",
            "drained",
            "evicted",
            "ceiling_evicted",
            "promoted",
            "reads_by_kind",
            "issuance",
        ):
            self.assertIn(key, balance, msg=key)
        self.assertEqual(set(balance["reads_by_kind"]), set(COHORT_KINDS))
        # The two named accounts are additive, not a replacement.
        self.assertIsInstance(balance["conservation_account"], dict)
        self.assertIsInstance(balance["issuance_account"], dict)
        self.assertIs(balance["issuance"], balance["issuance_account"])


if __name__ == "__main__":
    unittest.main()
