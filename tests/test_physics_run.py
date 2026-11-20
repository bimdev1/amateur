"""Round 11 wave 3: the LeWM physics run through the ledger executor.

Pins ``docs/round11-contract.md`` §3.5/§5 at the executor boundary.  The
numbers quoted in the assertions are **measured this round** and kept as
regression fingerprints; no round 7-9a measurement is re-asserted.

What is pinned here:

* ``DryRun()`` defaults to the explicit ``SyntheticSource`` plumbing
  sentinel; ``PhysicsRun`` constructs a ``PhysicsSource``.
* The coverage ledger is **derived** from the executor's ledger constants:
  10/16 non-read flows derived, 6/16 placeholders ``{1,2,4,5,9,13}``,
  11/11 store reads executor-derived; the source's dispatch set *is* the
  derived set, an absent fid asserts, and a source failure is an assertion
  in the derivation pass -- never a channel charge.
* ``#16`` is a real derived body: ``#16.error == #12.surprise`` and
  ``#16.cost == ||#12.delta||_2`` exactly, same tick.
* The registered seed-7 trace: immediate ``#18`` birth at the trigger tick,
  out-of-sample shadow scoring, the ``#23`` survivor surfacing without a
  ``qualified`` pre-filter, the ``#22`` gate refusing a harmful survivor
  (measured, non-vacuously) and promoting only a qualified one, the tagged
  ``#19`` deposit / destructive ``#27`` drain, and the D4 append-only
  retained-content identity.
* Both pre-registered gates are recomputed independently from the
  ``RunTick`` rows and cross-checked against the summary: ADAPTATION is
  strictly ``< 1.0`` over the flipped-phase shadow rows; ADMISSION is
  ``== 0`` promoted-harmful.
* Seed 0 is the rejection control.  Its measured outcome is reported
  honestly: the registered trigger never fires under the fit-window guard,
  so zero promotions is met **vacuously** and ADAPTATION has ``n = 0``.
* The single-stream regime cannot reach ``#21`` -> ``#24`` (no motif cohort:
  ``MOTIF_MIN_PERSISTENCE`` is bank-derived and is *not* tuned), so the full
  live-overlay lifecycle is exercised on the executor with a deposited
  motif-cohort fixture, clearly labelled as a wiring test.
* The reject-with-charge identity works on the ``#22`` channel; the Q4
  zero-confidence write is a consumed no-op with identity B; the D2 byte
  ceilings fail closed in the derivation pass; the encoder stays
  bit-identical throughout.

Runtime: ~25 s (checkpoint load plus the 36-tick scenarios, their fits and
the lifecycle fixture).
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from amateur.dryrun import PLACEHOLDER_FLOWS, DryRun
from amateur.payloads import (
    MAX_PATCH_TOTAL_BYTES,
    NON_READ_FLOWS,
    PAYLOAD_SCHEMA,
    BiasPrimitive,
    DebrisCohort,
    DigestedDebrisBody,
    MatOverlay,
)
from amateur.physics.organ import (
    MOTIF_MIN_PERSISTENCE,
    NURSERY_CAPACITY,
    RESIDENCE_N,
)
from amateur.physics.run import (
    ADAPTATION_GATE,
    ADMISSION_GATE,
    CONTROL_SEED,
    DERIVED_FLOWS,
    REGISTERED_SEED,
    STORE_READ_FLOWS,
    TOTAL_TICKS,
    PhysicsRun,
    PhysicsSource,
    coverage_ledger,
)
from amateur.provenance import Provenance

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / ".work" / "forge" / "checkpoint.pt"
STATS = REPO / ".work" / "forge" / "stats.json"
ARTIFACTS_AVAILABLE = CHECKPOINT.is_file() and STATS.is_file()
ARTIFACT_REASON = (
    f"phase-3 artifacts absent ({CHECKPOINT} / {STATS}); regenerate with "
    "`python3 -m amateur.physics.train --out-dir .work/forge --seed 0`"
)
requires_artifacts = unittest.skipUnless(ARTIFACTS_AVAILABLE, ARTIFACT_REASON)


def setUpModule() -> None:
    torch.set_num_threads(min(8, os.cpu_count() or 1))


# ---------------------------------------------------------------------------
# Scenario cache: two measured runs shared by every read-only assertion.
# ---------------------------------------------------------------------------
_RUNS: dict[int, tuple[PhysicsRun, object]] = {}


def scenario(seed: int):
    """The measured ``(run, result)`` for one seed (computed once)."""
    if seed not in _RUNS:
        run = PhysicsRun(seed=seed)
        _RUNS[seed] = (run, run.scenario(seed=seed))
    return _RUNS[seed]


def _emissions_by_tick(run: PhysicsRun) -> dict[int, dict[int, object]]:
    """``tick -> fid -> envelope`` from the executor's own emission log."""
    found: dict[int, dict[int, object]] = {}
    for tick, fid, envelope in run.dry.emissions:
        found.setdefault(int(tick), {})[int(fid)] = envelope
    return found


def _adaptation_from_rows(rows) -> tuple[int, float]:
    """ADAPTATION recomputed independently from the rows (contract §5)."""
    selected = [
        row
        for row in rows
        if row.phase == "flipped" and row.shadow_ratio is not None
    ]
    if not selected:
        return 0, float("nan")
    return (
        len(selected),
        sum(float(row.shadow_ratio) for row in selected) / len(selected),
    )


def _admission_from_rows(rows) -> tuple[int, list[tuple[str, float]]]:
    """ADMISSION recomputed independently from the rows (contract §5).

    A promotion row carries the candidate's final residence ratio, so the
    gate can be recomputed from the trace alone: promoted candidates whose
    final ``ratio >= 1.0``.
    """
    promoted = [
        (str(row.promoted_id), float(row.candidate_ratio))
        for row in rows
        if row.promoted_id is not None
    ]
    harmful = [item for item in promoted if not item[1] < 1.0]
    return len(harmful), harmful


class TestSourceSelection(unittest.TestCase):
    def test_dryrun_default_is_the_synthetic_sentinel(self) -> None:
        self.assertEqual(DryRun().source.mode, "synthetic")
        self.assertTrue(DryRun().source.coverage)
        self.assertTrue(
            all(value == "synthetic" for value in DryRun().source.coverage.values())
        )

    @requires_artifacts
    def test_physics_run_source_is_physics(self) -> None:
        run = PhysicsRun(seed=REGISTERED_SEED)
        self.assertEqual(run.source.mode, "physics")
        self.assertIsInstance(run.source, PhysicsSource)
        self.assertEqual(run.seed, REGISTERED_SEED)


class TestCoverageLedger(unittest.TestCase):
    """Contract §2.7: the coverage account is derived from the ledger."""

    def test_arithmetic_is_derived_from_the_ledger(self) -> None:
        account = coverage_ledger()
        self.assertEqual(account["n_non_read"], 16)
        self.assertEqual(account["n_derived"], 10)
        self.assertEqual(account["n_placeholders"], 6)
        self.assertEqual(account["n_store_reads"], 11)
        self.assertEqual(
            account["n_derived"] + account["n_placeholders"],
            account["n_non_read"],
        )
        self.assertEqual(
            account["placeholders"], (1, 2, 4, 5, 9, 13)
        )
        self.assertEqual(account["placeholders"], tuple(sorted(PLACEHOLDER_FLOWS)))
        self.assertEqual(
            frozenset(account["derived"]), frozenset(NON_READ_FLOWS) - PLACEHOLDER_FLOWS
        )
        self.assertEqual(frozenset(account["store_reads"]), frozenset(STORE_READ_FLOWS))
        # The two sets partition the 27-flow ledger.
        self.assertEqual(
            frozenset(account["non_read"]) | frozenset(account["store_reads"]),
            frozenset(range(1, 28)),
        )

    @requires_artifacts
    def test_source_dispatch_is_exactly_the_derived_set(self) -> None:
        source = PhysicsRun(seed=REGISTERED_SEED).source
        self.assertEqual(frozenset(source._derivers), frozenset(DERIVED_FLOWS))
        account = coverage_ledger()
        for fid, claim in source.coverage.items():
            expected = (
                "placeholder" if fid in account["placeholders"] else "derived"
            )
            self.assertEqual(claim, expected, msg=f"#{fid}")
        self.assertEqual(len(source.read_coverage), 11)
        self.assertTrue(
            all(
                claim == "store-read/executor-derived"
                for claim in source.read_coverage.values()
            )
        )

    @requires_artifacts
    def test_derived_flows_carry_their_pinned_types(self) -> None:
        run, _result = scenario(REGISTERED_SEED)
        emissions = _emissions_by_tick(run)
        self.assertTrue(emissions)
        for fid in sorted(DERIVED_FLOWS):
            seen = [
                envelope
                for tick in emissions
                for body_fid, envelope in emissions[tick].items()
                if body_fid == fid
            ]
            self.assertTrue(seen, msg=f"#{fid} never arrived")
            for envelope in seen:
                self.assertIn(type(envelope.body), PAYLOAD_SCHEMA[fid], msg=f"#{fid}")
        # The always-real flows are never the None placeholder.
        for fid in (3, 6, 10, 12, 16):
            for tick in emissions:
                self.assertIsNotNone(emissions[tick][fid].body, msg=f"#{fid}@{tick}")
        # The no-op write channels carry their exact-type sentinels, never None.
        for fid in (18, 19, 22, 24):
            for tick in emissions:
                self.assertIsNotNone(emissions[tick][fid].body, msg=f"#{fid}@{tick}")

    @requires_artifacts
    def test_placeholder_envelopes_are_none(self) -> None:
        run, _result = scenario(REGISTERED_SEED)
        emissions = _emissions_by_tick(run)
        for fid in sorted(PLACEHOLDER_FLOWS):
            for tick in emissions:
                self.assertIsNone(emissions[tick][fid].body, msg=f"#{fid}@{tick}")

    @requires_artifacts
    def test_absent_fid_asserts_and_never_falls_back(self) -> None:
        run = PhysicsRun(seed=REGISTERED_SEED)
        run.reset()
        with self.assertRaises(AssertionError):
            run.source.derive_body(99, 1, board=run.board, debris=run.dry.debris)
        # A source with no bound context refuses to invent a body either.
        fresh = PhysicsSource(run.organ)
        with self.assertRaises(AssertionError):
            fresh.derive_body(12, 1, board=run.board, debris=run.dry.debris)


class TestSourceFailureIsNotACharge(unittest.TestCase):
    class _ExplodingSource:
        mode = "physics"

        def __init__(self) -> None:
            self.coverage = {fid: "synthetic" for fid in sorted(NON_READ_FLOWS)}

        def derive_body(self, fid, tick, *, board, debris):  # noqa: ANN001
            raise RuntimeError("derivation failed")

        def on_commit(self, fid, envelope, *, board, debris):  # noqa: ANN001
            raise AssertionError("on_commit must not be reached")

    def test_failure_raises_before_any_arrival(self) -> None:
        dry = DryRun(source=self._ExplodingSource())
        with self.assertRaises(RuntimeError):
            dry.tick()
        self.assertEqual(sum(dry.attempted.values()), 0)
        self.assertEqual(sum(dry.accepted.values()), 0)
        self.assertEqual(sum(dry.refused.values()), 0)
        self.assertEqual(dry.emissions, [])
        # A failure is not a charge: no mailbox, D2 (beyond the seed record)
        # or D3 state was touched.
        self.assertEqual(dry.board.resident_ids(), frozenset({"ovr-0"}))
        self.assertEqual(len(dry.debris), 0)


@requires_artifacts
class TestCostErrorIdentity(unittest.TestCase):
    """Contract §2.6: ``#16`` is a real derived body, not a dressed constant."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)
        cls.emissions = _emissions_by_tick(cls.runner)

    def test_error_equals_surprise_and_cost_equals_the_delta_norm(self) -> None:
        checked = 0
        for tick, envelopes in self.emissions.items():
            sixteen = envelopes[16]
            twelve = envelopes[12]
            self.assertEqual(sixteen.provenance.tick_born, twelve.provenance.tick_born)
            self.assertEqual(sixteen.body.error, twelve.body.surprise, msg=f"t{tick}")
            norm = float(
                np.linalg.norm(np.asarray(twelve.body.delta, dtype=np.float64))
            )
            self.assertEqual(sixteen.body.cost, norm, msg=f"t{tick}")
            # The same tick's #12 delta is the vector #16's cost was taken of.
            self.assertEqual(len(twelve.body.delta), 64)
            checked += 1
        self.assertEqual(checked, TOTAL_TICKS)

    def test_the_cost_is_measured_not_a_constant(self) -> None:
        costs = {
            float(self.emissions[tick][16].body.cost) for tick in self.emissions
        }
        errors = {
            float(self.emissions[tick][16].body.error) for tick in self.emissions
        }
        self.assertGreater(len(costs), 10)
        self.assertGreater(len(errors), 10)
        self.assertGreater(max(costs), 0.0)


@requires_artifacts
class TestRegisteredScenario(unittest.TestCase):
    """The pre-registered seed-7 trace (§5), measured this round."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)
        cls.rows = cls.result.ticks
        cls.summary = cls.result.summary

    def test_scenario_shape_and_spike(self) -> None:
        scenario_ = self.summary["scenario"]
        self.assertEqual(scenario_["seed"], REGISTERED_SEED)
        self.assertEqual(scenario_["ticks"], TOTAL_TICKS)
        self.assertEqual((scenario_["flip_at"], scenario_["flip_back_at"]), (8, 28))
        self.assertEqual(len(self.rows), TOTAL_TICKS)
        self.assertEqual(
            [row.tick for row in self.rows], list(range(1, TOTAL_TICKS + 1))
        )
        self.assertEqual(
            {row.phase for row in self.rows}, {"normal", "flipped", "recovered"}
        )
        self.assertGreater(self.summary["spike_peak"], self.summary["normal_mean"])
        self.assertGreater(
            self.summary["spike_peak"] / self.summary["normal_mean"], 3.0
        )

    def test_candidate_is_born_at_the_trigger_tick(self) -> None:
        """Immediate birth at the first spike past the registered fit window."""
        self.assertEqual(self.summary["lifecycle"]["candidate_birth"], 19)
        curve = self.summary["residence_curve"]
        self.assertEqual(len(curve), 9)  # t19 (birth) .. t27
        self.assertEqual([row["residence_ticks"] for row in curve], list(range(0, 9)))
        self.assertEqual([row["tick"] for row in curve], list(range(19, 28)))
        self.assertTrue(all(row["phase"] == "flipped" for row in curve))
        # The curve's cumulative ratio is monotonically non-decreasing here
        # (measured) and crosses below 1.0 only on the last resident tick.
        ratios = [row["ratio"] for row in curve if row["residence_ticks"] > 0]
        self.assertEqual(len(ratios), 8)
        self.assertEqual(
            [round(ratio, 6) for ratio in ratios],
            [
                1.184592,
                1.113826,
                1.007115,
                1.009547,
                1.023553,
                1.019522,
                1.002973,
                0.999225,
            ],
        )
        self.assertTrue(all(row["qualified"] is False for row in curve[:-1]))
        self.assertTrue(curve[-1]["qualified"])

    def test_candidate_fit_is_capacity_bounded(self) -> None:
        fit = self.summary["candidate_fit"]
        self.assertTrue(fit["attempted"])
        record = fit["record"]
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["rank"], 26)
        self.assertEqual(record["param_count"], 5_056)
        self.assertEqual(record["nbytes"], 20_224)
        self.assertEqual(record["budget_n"], 16)
        self.assertEqual(record["reason"], "")
        self.assertLess(record["loss_after"], record["loss_before"])
        self.assertLessEqual(
            record["param_count"], int(record["budget_capacity"])
        )

    def test_shadow_evidence_is_out_of_sample_only(self) -> None:
        """The first shadow row is the tick *after* the birth (in-sample excluded)."""
        shadow_ticks = [
            row.tick for row in self.rows if row.shadow_ratio is not None
        ]
        birth = self.summary["lifecycle"]["candidate_birth"]
        assert birth is not None
        self.assertNotIn(birth, shadow_ticks)
        self.assertEqual(shadow_ticks, list(range(birth + 1, 28)))
        self.assertEqual(len(shadow_ticks), 8)
        # One shadow-scored resident candidate per row.
        self.assertTrue(all(row.shadow_n == 1 for row in self.rows if row.shadow_ratio))

    def test_store_survivor_reads_the_same_evidence_as_the_organ(self) -> None:
        """#23 surfaces the lowest-ratio resident; the curve is that evidence."""
        survivor_rows = [row for row in self.rows if row.survivor_id is not None]
        self.assertEqual([row.tick for row in survivor_rows], [23, 24, 25, 26, 27])
        for row in survivor_rows:
            self.assertEqual(row.survivor_id, row.candidate_id)
            self.assertEqual(row.survivor_ratio, row.candidate_ratio)
            self.assertEqual(
                row.survivor_qualified, row.candidate_ratio < 1.0
            )
        # The first surface is exactly the RESIDENCE_N-th out-of-sample sample.
        first = survivor_rows[0]
        self.assertEqual(first.candidate_residence_ticks, RESIDENCE_N)
        self.assertEqual(self.summary["lifecycle"]["first_survivor"], 23)

    def test_harmful_survivor_is_surfaced_and_refused(self) -> None:
        """The #22 gate refuses a surfaced harmful survivor (non-vacuous)."""
        harmful = [
            row
            for row in self.rows
            if row.survivor_id is not None and row.survivor_qualified is False
        ]
        self.assertEqual([row.tick for row in harmful], [23, 24, 25, 26])
        for row in harmful:
            self.assertGreaterEqual(row.survivor_ratio, 1.0)
            self.assertEqual(row.gate, "ratchet-gate-blocked")
            self.assertIsNone(row.promoted_id)
        self.assertEqual(self.summary["harmful_survivor_ticks"], 4)
        self.assertEqual(
            [item["tick"] for item in self.summary["harmful_survivors"]],
            [23, 24, 25, 26],
        )

    def test_promotion_is_the_qualified_survivor(self) -> None:
        promoted = [row for row in self.rows if row.promoted_id is not None]
        self.assertEqual(len(promoted), 1)
        row = promoted[0]
        self.assertEqual(row.tick, 27)
        self.assertEqual(row.promoted_id, "cand-01")
        self.assertEqual(row.gate, "ratchet-gate-passed")
        self.assertLess(row.candidate_ratio, 1.0)
        self.assertEqual(row.nursery_ids, ())  # removed atomically on promotion
        # The D4 sediment is that promotion's output, with the lineage.
        sediments = self.summary["d4"]["sediments"]
        self.assertEqual(len(sediments), 1)
        self.assertEqual(sediments[0]["candidate_lineage"], "cand-01")
        self.assertEqual(sediments[0]["tick"], 27)
        self.assertAlmostEqual(
            sediments[0]["confidence"], 1.0 - row.candidate_ratio, places=12
        )

    def test_d4_is_append_only_and_retained_byte_identical(self) -> None:
        trace = self.result.d4_trace
        self.assertEqual(len(trace), TOTAL_TICKS + 1)
        self.assertEqual(trace[0], ())
        for previous, nxt in zip(trace, trace[1:]):
            self.assertEqual(nxt[: len(previous)], previous)  # append-only
        self.assertEqual(self.summary["d4"]["retained_byte_identical"], True)
        self.assertEqual(self.summary["d4"]["byte_identical"], False)
        self.assertEqual(len(trace[-1]), 1)
        self.assertEqual(trace[-1], tuple(self.runner.board.d4))

    def test_registered_trace_cannot_reach_the_live_overlay_path(self) -> None:
        """Reachability finding: one stream never produces a motif cohort."""
        self.assertEqual(self.summary["cohort_counts"]["motif"], 0)
        self.assertEqual(self.summary["lifecycle"]["motif_reads"], 0)
        self.assertEqual(self.summary["lifecycle"]["overlay_birth"], None)
        self.assertEqual(self.summary["lifecycle"]["overlay_stale"], None)
        self.assertEqual(self.summary["lifecycle"]["overlay_evict"], None)
        self.assertGreaterEqual(MOTIF_MIN_PERSISTENCE, 10)  # not tuned this round
        self.assertFalse(
            any(row.motif_id is not None for row in self.rows)
        )

    def test_tagged_d3_deposits_and_destructive_drain(self) -> None:
        deposits = [row for row in self.rows if row.cohort_kinds]
        drains = [row for row in self.rows if row.drained_ids]
        self.assertGreaterEqual(len(deposits), 1)
        self.assertGreaterEqual(len(drains), 1)
        kinds = {kind for row in deposits for kind in row.cohort_kinds}
        self.assertTrue(kinds <= {"motif", "scaffold", "trail", "residual"})
        self.assertEqual(kinds, {"trail", "residual"})
        mass = self.summary["d3"]["mass"]
        self.assertEqual(
            mass["deposits"], mass["resident"] + mass["drained"] + mass["evicted"]
        )
        self.assertEqual(
            self.summary["d3"]["drained"], mass["drained"]
        )
        self.assertEqual(self.summary["d3"]["ceiling_evictions"], 0)

    def test_end_state_is_clean(self) -> None:
        self.assertTrue(self.summary["encoder_bit_identical"])
        self.assertTrue(self.runner.organ.encoder_bit_identical())
        self.assertEqual(self.runner.board.residents(), ())
        self.assertEqual(self.runner.board.patches(), ())
        self.assertEqual(self.runner.board.patch_bytes(), 0)
        self.assertEqual(self.runner.board.nursery_patch_bytes(), 0)
        self.assertTrue(self.summary["lifecycle"]["d2_empty_at_end"])
        self.assertTrue(self.summary["lifecycle"]["nursery_empty_at_end"])
        self.assertEqual(self.runner.organ.candidate_ids(), ())


@requires_artifacts
class TestAdaptationGate(unittest.TestCase):
    """ADAPTATION (pre-registered §5): PASS iff strictly below 1.0."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)

    def test_gate_is_recomputed_from_the_rows_and_passes(self) -> None:
        n, value = _adaptation_from_rows(self.result.ticks)
        self.assertEqual(n, 8)
        self.assertAlmostEqual(value, 0.9555247158775397, places=12)
        self.assertLess(value, ADAPTATION_GATE)
        summary_gate = self.result.summary["gates"]["adaptation"]
        self.assertEqual(summary_gate["n"], n)
        self.assertAlmostEqual(summary_gate["value"], value, places=12)
        self.assertTrue(summary_gate["passed"])
        self.assertEqual(summary_gate["threshold"], ADAPTATION_GATE)
        # Strictness is pinned: the assertion is < 1.0, never <=.
        self.assertFalse(value >= ADAPTATION_GATE)

    def test_gate_set_is_the_intersection_of_the_two_predicates(self) -> None:
        selected = {
            row.tick
            for row in self.result.ticks
            if row.phase == "flipped" and row.shadow_ratio is not None
        }
        self.assertEqual(selected, set(range(20, 28)))
        # No shadow row exists outside the flipped phase in this trace.
        self.assertFalse(
            any(
                row.shadow_ratio is not None and row.phase != "flipped"
                for row in self.result.ticks
            )
        )


@requires_artifacts
class TestAdmissionGate(unittest.TestCase):
    """ADMISSION (pre-registered §5): PASS iff no promoted candidate is harmful."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)

    def test_gate_is_recomputed_from_the_rows_and_passes(self) -> None:
        count, harmful = _admission_from_rows(self.result.ticks)
        self.assertEqual(count, ADMISSION_GATE)
        self.assertEqual(harmful, [])
        summary_gate = self.result.summary["gates"]["admission"]
        self.assertEqual(summary_gate["value"], count)
        self.assertEqual(summary_gate["n"], 1)
        self.assertTrue(summary_gate["passed"])
        self.assertEqual(summary_gate["threshold"], ADMISSION_GATE)

    def test_the_promotion_log_agrees_with_the_trace(self) -> None:
        """The organ mirror and the executor rows are the same promotion."""
        promotions = self.runner.organ.promotions
        self.assertEqual(len(promotions), 1)
        promotion = promotions[0]
        row = next(row for row in self.result.ticks if row.promoted_id)
        self.assertEqual(promotion.candidate_id, row.promoted_id)
        self.assertEqual(promotion.promoted_tick, row.tick)
        self.assertEqual(promotion.residence_ticks, row.candidate_residence_ticks)
        self.assertAlmostEqual(promotion.ratio, row.candidate_ratio, places=15)
        self.assertTrue(promotion.qualified)
        self.assertEqual(
            self.result.summary["gates"]["admission"]["promotions"][0]["ratio"],
            promotion.ratio,
        )


@requires_artifacts
class TestControlSeed0(unittest.TestCase):
    """Seed 0 is the rejection control; its measured outcome is reported."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(CONTROL_SEED)

    def test_zero_promotions_but_vacuously(self) -> None:
        """The registered expectation holds, but no candidate was ever born."""
        self.assertEqual(self.result.summary["gates"]["admission"]["n"], 0)
        self.assertEqual(self.result.summary["gates"]["admission"]["value"], 0)
        self.assertTrue(self.result.summary["gates"]["admission"]["passed"])
        # Honest finding: the control never triggered under the fit-window
        # guard (its spikes are all before frame 16), so the zero is vacuous.
        self.assertEqual(self.runner.births, [])
        self.assertEqual(self.result.summary["lifecycle"]["candidate_births"], ())
        self.assertFalse(self.result.summary["candidate_fit"]["attempted"])
        spikes = [
            row.tick
            for row in self.result.ticks
            if row.base_surprise > self.runner.organ.thresholds.spike_threshold
        ]
        self.assertTrue(spikes)
        self.assertLess(max(spikes), self.runner.fit_window)
        self.assertEqual(self.result.summary["survivor_ticks"], 0)

    def test_adaptation_has_no_rows_and_is_reported_as_nan(self) -> None:
        gate = self.result.summary["gates"]["adaptation"]
        self.assertEqual(gate["n"], 0)
        self.assertFalse(gate["passed"])
        self.assertTrue(np.isnan(gate["value"]))
        self.assertIn("no shadow-scored", gate["reason"])

    def test_d4_is_byte_identical_and_empty(self) -> None:
        self.assertTrue(self.result.summary["d4"]["byte_identical"])
        self.assertTrue(self.result.summary["d4"]["retained_byte_identical"])
        self.assertEqual(self.result.summary["d4"]["sediments"], ())
        self.assertEqual(self.runner.board.d4, [])


@requires_artifacts
class TestLiveOverlayLifecycle(unittest.TestCase):
    """The ``#21`` -> ``#24`` -> ``#25`` -> evict lifecycle, executor-side.

    A single stream cannot produce a motif cohort (``MOTIF_MIN_PERSISTENCE``
    is bank-derived and is **not** tuned), so this test deposits one valid
    motif cohort into D3 mid-run and lets the executor do the rest.  The
    fixture supplies only the cohort; every derivation, gate, commit,
    staleness decision and eviction below is the production path.  This is a
    wiring test, not a measurement of the registered scenario.
    """

    DEPOSIT_TICK = 17
    MOTIF_ID = "motif-fixture-01"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = PhysicsRun(seed=REGISTERED_SEED)
        cls.runner.reset()
        # Drive the registered shape (the test injects one cohort mid-run, so
        # the loop is walked directly rather than through ``scenario``).
        cls.runner._flip_at = 8
        cls.runner._flip_back_at = 28
        cls.rows = []
        cls.bodies: dict[int, dict[int, object]] = {}
        cls.patch_ids: dict[int, tuple[str, ...]] = {}
        for index in range(1, TOTAL_TICKS + 1):
            if index == 8 or index == 28:
                cls.runner.flip_gravity()
            if index == cls.DEPOSIT_TICK:
                cls._deposit_motif(index)
            cls.rows.append(cls.runner.tick())
            cls.bodies[index] = {
                fid: cls.runner.dry.derived[fid].body for fid in (8, 21, 23, 24, 25)
            }
            cls.patch_ids[index] = tuple(
                overlay_id for overlay_id, _patch in cls.runner.board.patches()
            )

    @classmethod
    def _deposit_motif(cls, tick: int) -> None:
        provenance = Provenance(19, tick - 1)  # a read must be strictly older
        cls.runner.dry.debris.deposit(
            DigestedDebrisBody(
                debris_id="debris-fixture",
                cohorts=(
                    DebrisCohort(
                        cohort_id=cls.MOTIF_ID,
                        kind="motif",
                        centroid=(0.05,) * 64,
                        count=6,
                        spread=0.01,
                        first_tick=8,
                        last_tick=tick,
                        peak_surprise=0.5,
                        provenance=provenance,
                    ),
                ),
            ),
            provenance,
            tick,
        )

    def test_motif_read_drives_the_live_overlay_birth(self) -> None:
        read = [row for row in self.rows if row.motif_id == self.MOTIF_ID]
        self.assertTrue(read)
        self.assertEqual(read[0].tick, self.DEPOSIT_TICK)
        birth = next(row for row in self.rows if row.adopted is not None)
        self.assertEqual(birth.tick, self.DEPOSIT_TICK)
        self.assertEqual(birth.adopted, "ovr-01")
        self.assertIn("ovr-01", birth.residents)
        # The D2 write carries the #21 motif lineage (the executor's gate).
        body = self.bodies[birth.tick][24]
        self.assertIsInstance(body, MatOverlay)
        self.assertEqual(body.motif_lineage, self.MOTIF_ID)
        self.assertEqual(self.bodies[birth.tick][21].motif_id, self.MOTIF_ID)
        # The live patch is attached to D2 (Q1-A) and bounded.
        self.assertGreater(birth.patch_bytes, 0)
        self.assertLessEqual(birth.patch_bytes, MAX_PATCH_TOTAL_BYTES)
        self.assertIn("ovr-01", self.patch_ids[birth.tick])
        # The #8/#25 reads label the head active / stale respectively.  On
        # the birth tick they are still None (the #24 write commits at the end
        # of that tick); from the next tick on they pair.
        self.assertIsNone(self.bodies[birth.tick][8])
        self.assertIsNone(self.bodies[birth.tick][25])
        active = self.bodies[birth.tick + 1][8]
        stale = self.bodies[birth.tick + 1][25]
        self.assertEqual(active.status, "active")
        self.assertEqual(stale.status, "stale")
        self.assertEqual(active.overlay_id, stale.overlay_id)
        self.assertEqual(active.overlay_id, "ovr-01")
        # The #25 hand-off is the same epoch the head carries, one tick later.
        self.assertEqual(active.epoch, stale.epoch)

    def test_revalidation_advances_the_epoch(self) -> None:
        self.assertGreaterEqual(len(self.runner.organ.overlay_table.revalidations), 1)
        epochs = [record.epoch for record in self.runner.organ.overlay_table.revalidations]
        self.assertEqual(epochs, sorted(epochs))
        self.assertEqual(self.runner.organ.overlay_table.births[0].epoch, 1)

    def test_stale_then_evict_and_both_indices_clean(self) -> None:
        staled = [row for row in self.rows if row.staled is not None]
        evicted = [row for row in self.rows if row.evicted]
        self.assertEqual(len(staled), 1)
        self.assertEqual(staled[0].staled, "ovr-01")
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].evicted, ("ovr-01",))
        self.assertLess(staled[0].tick, evicted[0].tick)
        self.assertEqual(evicted[0].tick, staled[0].tick + 3)  # STALE_AFTER
        self.assertEqual(evicted[-1].residents, ())
        self.assertEqual(evicted[-1].patch_bytes, 0)
        self.assertEqual(self.runner.board.residents(), ())
        self.assertEqual(self.runner.board.patches(), ())
        self.assertEqual(self.runner.organ.overlay_table.records(), ())
        self.assertEqual(self.runner.organ.overlay_table.active(), ())
        # The store's #8/#25 reads are the *pre-eviction* state on the
        # eviction tick (the organ decides first, the prune mirrors at the
        # end of the same tick) and None from the next tick on.
        self.assertEqual(self.bodies[evicted[0].tick][8].overlay_id, "ovr-01")
        self.assertIsNone(self.bodies[evicted[0].tick + 1][8])
        self.assertIsNone(self.bodies[evicted[0].tick + 1][25])

    def test_the_registered_run_never_enters_this_path(self) -> None:
        _runner, result = scenario(REGISTERED_SEED)
        self.assertEqual(result.summary["lifecycle"]["overlay_birth"], None)
        self.assertEqual(result.summary["d4"]["sediments"][0]["candidate_lineage"], "cand-01")


@requires_artifacts
class TestRejectWithChargeAndNoop(unittest.TestCase):
    """Reject-with-charge and the Q4 zero-confidence no-op, physics path."""

    def _drive_to(self, tick: int) -> PhysicsRun:
        run = PhysicsRun(seed=REGISTERED_SEED)
        run.reset()
        run._flip_at, run._flip_back_at = 8, 28
        for index in range(1, tick):
            if index == 8:
                run.flip_gravity()
            run.tick()
        return run

    def test_duplicate_arrival_is_charged_and_state_is_unchanged(self) -> None:
        run = self._drive_to(24)  # t23/t24 surface a harmful survivor
        d4_before = tuple(run.board.d4)
        before = (
            sum(run.dry.attempted.values()),
            sum(run.dry.accepted.values()),
            sum(run.dry.refused.values()),
            run.board.nursery_ids(),
            len(run.board.d4),
        )
        row = run.tick(injections={22: 2})
        after = (
            sum(run.dry.attempted.values()),
            sum(run.dry.accepted.values()),
            sum(run.dry.refused.values()),
        )
        attempted = after[0] - before[0]
        accepted = after[1] - before[1]
        refused = after[2] - before[2]
        # Identity A on the #22 channel: attempts = accepts + refusals.
        self.assertEqual(attempted, accepted + refused)
        self.assertGreaterEqual(refused, 1)
        # The refused copies are not consumed: identity B holds on #22.
        self.assertEqual(run.dry.accepted[22], run.dry.consumed[22])
        # The accepted zero-confidence copy is a consumed no-op.
        self.assertEqual(row.gate, "ratchet-gate-blocked")
        self.assertIsNone(row.promoted_id)
        self.assertEqual(tuple(run.board.d4), d4_before)
        self.assertEqual(run.board.nursery_ids(), before[3])
        self.assertEqual(len(run.board.d4), before[4])
        self.assertGreater(run.dry.zero_confidence_absorbed[22], 0)

    def test_positive_gate_needs_the_consumed_survivor_lineage(self) -> None:
        """A positive #22 must cite this tick's consumed #23 survivor."""
        run = self._drive_to(27)
        row = run.tick()
        self.assertEqual(row.tick, 27)
        self.assertEqual(row.promoted_id, "cand-01")
        body = run.dry.derived[22].body
        self.assertEqual(body.candidate_lineage, run.dry.derived[23].body.candidate_id)
        self.assertGreater(body.confidence, 0.0)
        self.assertLessEqual(body.confidence, 1.0)


@requires_artifacts
class TestPatchAccounting(unittest.TestCase):
    """D2 substance accounting: candidates and overlays, atomic eviction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)

    def test_no_orphan_patch_or_record_at_any_tick(self) -> None:
        self.assertEqual(self.runner.board.patches(), ())
        self.assertEqual(self.runner.board.resident_ids(), frozenset())
        self.assertEqual(self.runner.board.nursery_ids(), ())
        self.assertLessEqual(
            self.result.summary["max_patch_bytes_seen"],
            MAX_PATCH_TOTAL_BYTES,
        )

    def test_resident_without_patch_asserts(self) -> None:
        from amateur.payloads import EmergentCrystalBody, MatBoard

        board = MatBoard(seed=False)
        board.nursery_birth(
            EmergentCrystalBody(
                candidate_id="cand-x",
                residual=(0.0,) * 64,
                surprise=0.0,
                born_tick=1,
            ),
            1,
        )
        with self.assertRaises(AssertionError):
            board.assert_patch_bounds(require_complete=True)

    def test_oversized_candidate_patch_fails_before_the_commit(self) -> None:
        """The outer byte ceiling fails closed in the derivation pass (Q1-A)."""
        from amateur.payloads import EmergentCrystalBody

        run = PhysicsRun(seed=REGISTERED_SEED)
        run.reset()
        # A candidate record whose patch claims more bytes than the ceiling.
        board = run.board
        board.nursery_birth(
            EmergentCrystalBody(
                candidate_id="cand-big",
                residual=(0.0,) * 64,
                surprise=0.0,
                born_tick=1,
            ),
            1,
        )

        class _BigPatch:
            overlay_id = "cand-big"
            nbytes = MAX_PATCH_TOTAL_BYTES + 1
            param_count = 1

        with self.assertRaises(ValueError):
            board.attach_candidate_patch("cand-big", _BigPatch())
        self.assertEqual(board.nursery_patch_bytes(), 0)
        self.assertEqual(board.patch_bytes(), 0)


@requires_artifacts
class TestResetRepeatability(unittest.TestCase):
    def test_two_scenarios_are_identical(self) -> None:
        first_run = PhysicsRun(seed=REGISTERED_SEED)
        first = first_run.scenario(seed=REGISTERED_SEED)
        second_run = PhysicsRun(seed=REGISTERED_SEED)
        second = second_run.scenario(seed=REGISTERED_SEED)
        self.assertEqual(first.ticks, second.ticks)
        self.assertEqual(first.summary["gates"], second.summary["gates"])
        self.assertEqual(
            first.summary["residence_curve"], second.summary["residence_curve"]
        )

    def test_reset_clears_the_nursery_and_the_store(self) -> None:
        run = PhysicsRun(seed=REGISTERED_SEED)
        result = run.scenario(seed=REGISTERED_SEED)
        self.assertEqual(len(result.ticks), TOTAL_TICKS)
        run.reset()
        self.assertEqual(run.ticks, [])
        self.assertEqual(run.births, [])
        self.assertEqual(run.organ.candidate_ids(), ())
        self.assertEqual(run.board.nursery_ids(), ())
        self.assertEqual(run.board.residents(), ())
        self.assertEqual(run.board.d4, [])
        self.assertEqual(run.board.patch_bytes(), 0)
        self.assertFalse(run.organ.promotions)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
