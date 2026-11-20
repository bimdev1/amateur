"""Round 12a: the LeWM physics run through the ledger executor.

Pins the round-12a pre-registration (§1, §2, §4) at the executor boundary.
The numbers quoted in the assertions are **measured this round** and kept as
regression fingerprints; the Round-11 trace (birth t19, K=5,056) is *not*
the 12a registered trace and is reproduced only by the labelled historical
diagnostic mode (task-26), never by the registered path.

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
* Evidence-sufficiency trigger (prereg §4): ``N_MIN = 2`` replaces the old
  absolute ``tick >= fit_window`` guard, and the fit consumes
  ``min(fit_window, tick)`` real transitions.  Measured: seed 7 births at
  its first spike t12 (N=12, rank 19, K=3,712); seed 0 births at its first
  spike t9 (N=9, rank 14, K=2,752).
* ADAPTATION is the **ratio-of-means** over the declared flipped-phase
  out-of-sample tick set (prereg §2), with the secondary mean-of-ratios
  always reported and labelled; both tick sets are recomputed from the rows
  and compared with the summary.  Measured seed 7: primary 1.013378 over
  t13..t27 -> the gate **FAILS** (the candidate is harmful).  The
  with-in-sample sample is report-only and comes from the organ.
* The registered seed-7 trace: immediate ``#18`` birth at the trigger tick,
  out-of-sample shadow scoring, the ``#23`` survivor surfaced without a
  ``qualified`` pre-filter, the ``#22`` gate refusing the harmful survivor
  (measured, non-vacuous), and the bounded, non-steering candidate remaining
  resident at end (prereg §4).  No promotion, D4 empty.
* Seed 0 is the rejection control.  Measured pre-floor behaviour is reported
  honestly: it births the small candidate at t9 and the candidate is
  qualified at t14 (ratio 0.983203 < 1.0), so the **pre-floor** gate
  promotes it; the round-12a ``MDL_FLOOR`` (task-26) is the condition
  expected to refuse it.
* The ``#22`` gate is floored (prereg §3): a positive sediment requires
  ``#23`` qualified **and** ``MDL_FLOOR.admitted(record)``.  Measured: the
  registered seed 7 has 0 promotions (21 harmful floor refusals, 0
  qualified refusals); the floor-ablated seed 0 reproduces the pre-floor
  t14 promotion, which the floor refuses ("uneconomic": ~168 bits vs
  88,064 bits).  A refused candidate keeps scoring, so ADAPTATION's tick set
  depends on the floor on the control (seed 0: n=18 floor-on vs n=5
  ablated) -- both are reported, neither is tuned.
* The historical falsifier (``trigger="round11"``, floor ablated) reproduces
  the Round-11 candidate (birth t19, K=5,056, promotion t27);
  ``MDL_FLOOR.verdict`` refuses it (dbar 1.8972e-4, gain 3.547 bits vs cost
  161,792 bits, "uneconomic").  The registered trigger is asserted to be
  ``N_MIN``-based and the historical mode to be opt-in only.
* The single-flip regime cannot reach ``#21`` -> ``#24`` (no motif cohort
  for seed 7 under the derived thresholds), so the full live-overlay
  lifecycle is exercised on the executor with a deposited motif-cohort
  fixture, clearly labelled as a wiring test.
* The reject-with-charge identity works on the ``#22`` channel; the Q4
  zero-confidence write is a consumed no-op with identity B; the D2 byte
  ceilings fail closed in the derivation pass; the encoder stays
  bit-identical throughout.

Runtime: ~1 min (checkpoint load plus the 36-tick scenarios, their fits and
the lifecycle fixture).
"""
from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import amateur.physics.run as run_module
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
from amateur.physics.organ import MDL_FLOOR, NURSERY_CAPACITY, RESIDENCE_N
from amateur.physics.run import (
    ADAPTATION_GATE,
    ADMISSION_GATE,
    CONTROL_SEED,
    DEFAULT_GAIN_FLOOR,
    DERIVED_FLOWS,
    N_MIN,
    REGISTERED_SEED,
    STORE_READ_FLOWS,
    TOTAL_TICKS,
    TRIGGERS,
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
# Scenario cache: measured runs shared by every read-only assertion.
# ---------------------------------------------------------------------------
_RUNS: dict[tuple, tuple[PhysicsRun, object]] = {}


def scenario(
    seed: int,
    *,
    trigger: str = "registered",
    gain_floor: str = "on",
    patch_mode: str = "fitted",
    no_flip: bool = False,
):
    """The measured ``(run, result)`` for one configuration (computed once)."""
    key = (int(seed), str(trigger), str(gain_floor), str(patch_mode), bool(no_flip))
    if key not in _RUNS:
        run = PhysicsRun(
            seed=seed,
            trigger=trigger,
            gain_floor=gain_floor,
            patch_mode=patch_mode,
        )
        result = (
            run.scenario(seed=seed, flip_at=None)
            if no_flip
            else run.scenario(seed=seed)
        )
        _RUNS[key] = (run, result)
    return _RUNS[key]


def _emissions_by_tick(run: PhysicsRun) -> dict[int, dict[int, object]]:
    """``tick -> fid -> envelope`` from the executor's own emission log."""
    found: dict[int, dict[int, object]] = {}
    for tick, fid, envelope in run.dry.emissions:
        found.setdefault(int(tick), {})[int(fid)] = envelope
    return found


def _adaptation_sums(rows) -> tuple[int, float, float, list[int]]:
    """``(n, adapted_sum, base_sum, ticks)`` over the ADAPTATION_SET.

    The tick set is the pre-registered predicate, recomputed from the rows:
    ``phase == "flipped"`` and ``shadow_adapted_sum is not None`` (prereg §2).
    """
    selected = [
        row
        for row in rows
        if row.phase == "flipped" and row.shadow_adapted_sum is not None
    ]
    return (
        len(selected),
        float(sum(float(row.shadow_adapted_sum) for row in selected)),
        float(sum(float(row.shadow_base_sum) for row in selected)),
        [int(row.tick) for row in selected],
    )


def _adaptation_primary_from_rows(rows) -> tuple[int, float, list[int]]:
    """The ratio-of-means primary, recomputed independently (prereg §2)."""
    n, adapted_sum, base_sum, ticks = _adaptation_sums(rows)
    if n == 0:
        return 0, float("nan"), ticks
    return n, adapted_sum / base_sum, ticks


def _adaptation_secondary_from_rows(rows) -> float:
    """The pre-registered diagnostic ``mean_t(adapted/base)`` (prereg §2)."""
    selected = [
        row
        for row in rows
        if row.phase == "flipped" and row.shadow_adapted_sum is not None
    ]
    if not selected:
        return float("nan")
    return float(np.mean([float(row.shadow_ratio) for row in selected]))


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
        """Immediate birth at the first spike past ``N_MIN`` (prereg §4)."""
        self.assertEqual(self.summary["lifecycle"]["candidate_birth"], 12)
        curve = self.summary["residence_curve"]
        self.assertEqual(len(curve), 25)  # t12 (birth) .. t36
        self.assertEqual(
            [row["residence_ticks"] for row in curve], list(range(0, 25))
        )
        self.assertEqual([row["tick"] for row in curve], list(range(12, 37)))
        self.assertEqual(
            [row["phase"] for row in curve[:16]], ["flipped"] * 16
        )
        self.assertEqual(
            [row["phase"] for row in curve[16:]], ["recovered"] * 9
        )
        # The cumulative ratio-of-means curve (measured).  It dips below 1.0
        # for the first two out-of-sample ticks and then stays above 1.0: the
        # candidate is harmful, and the #22 gate refuses it on every tick.
        ratios = [row["ratio"] for row in curve if row["residence_ticks"] > 0]
        self.assertEqual(len(ratios), 24)
        self.assertEqual(
            [round(ratio, 6) for ratio in ratios],
            [
                0.558813,
                0.671137,
                1.024420,
                1.116961,
                1.076961,
                1.074951,
                1.035313,
                1.018495,
                1.013410,
                1.024524,
                1.031208,
                1.033788,
                1.023817,
                1.011789,
                1.013378,
                1.022438,
                1.011043,
                1.017925,
                1.047557,
                1.052642,
                1.055078,
                1.058536,
                1.059913,
                1.060769,
            ],
        )
        self.assertTrue(all(row["qualified"] is False for row in curve))

    def test_candidate_fit_is_capacity_bounded(self) -> None:
        fit = self.summary["candidate_fit"]
        self.assertTrue(fit["attempted"])
        record = fit["record"]
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["rank"], 19)
        self.assertEqual(record["param_count"], 3_712)
        self.assertEqual(record["nbytes"], 14_848)
        self.assertEqual(record["budget_n"], 12)
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
        # Scoring continues through the recovery phase (the candidate stays
        # resident): the residence set is *not* a subset of the flipped phase
        # in the 12a trace -- a measured difference from the Round-11 trace.
        self.assertEqual(shadow_ticks, list(range(birth + 1, TOTAL_TICKS + 1)))
        self.assertEqual(len(shadow_ticks), 24)
        # One shadow-scored resident candidate per row.
        self.assertTrue(all(row.shadow_n == 1 for row in self.rows if row.shadow_ratio))

    def test_store_survivor_reads_the_same_evidence_as_the_organ(self) -> None:
        """#23 surfaces the lowest-ratio resident; the curve is that evidence."""
        survivor_rows = [row for row in self.rows if row.survivor_id is not None]
        self.assertEqual(
            [row.tick for row in survivor_rows], list(range(16, TOTAL_TICKS + 1))
        )
        for row in survivor_rows:
            self.assertEqual(row.survivor_id, row.candidate_id)
            self.assertEqual(row.survivor_ratio, row.candidate_ratio)
            self.assertEqual(
                row.survivor_qualified, row.candidate_ratio < 1.0
            )
        # The first surface is exactly the RESIDENCE_N-th out-of-sample sample.
        first = survivor_rows[0]
        self.assertEqual(first.candidate_residence_ticks, RESIDENCE_N)
        self.assertEqual(self.summary["lifecycle"]["first_survivor"], 16)

    def test_harmful_survivor_is_surfaced_and_refused(self) -> None:
        """The #22 gate refuses the surfaced harmful survivor (non-vacuous)."""
        harmful = [
            row
            for row in self.rows
            if row.survivor_id is not None and row.survivor_qualified is False
        ]
        self.assertEqual(
            [row.tick for row in harmful], list(range(16, TOTAL_TICKS + 1))
        )
        for row in harmful:
            self.assertGreaterEqual(row.survivor_ratio, 1.0)
            self.assertEqual(row.gate, "ratchet-gate-blocked")
            self.assertIsNone(row.promoted_id)
        self.assertEqual(
            self.summary["harmful_survivor_ticks"], len(harmful)
        )
        self.assertEqual(
            [item["tick"] for item in self.summary["harmful_survivors"]],
            [row.tick for row in harmful],
        )

    def test_registered_candidate_is_never_qualified_and_writes_no_d4(self) -> None:
        """prereg §4: the registered candidate is harmful; ``#22`` never passes.

        Every cumulative residence ratio after the two early help-ticks is
        ``>= 1.0``, so ``#22`` is the zero-confidence sentinel on every
        survivor tick and D4 is never written.  The qualified-survivor
        promotion path is exercised by the seed-0 control below; the
        floor-on refusal of the registered run is added in task-26.
        """
        self.assertEqual(self.runner.organ.promotions, ())
        self.assertFalse([row for row in self.rows if row.promoted_id is not None])
        self.assertEqual(self.summary["d4"]["sediments"], ())
        self.assertEqual(self.runner.board.d4, [])
        self.assertGreater(
            self.summary["gates"]["adaptation"]["primary"], ADAPTATION_GATE
        )

    def test_d4_is_append_only_and_retained_byte_identical(self) -> None:
        trace = self.result.d4_trace
        self.assertEqual(len(trace), TOTAL_TICKS + 1)
        self.assertEqual(trace[0], ())
        for previous, nxt in zip(trace, trace[1:]):
            self.assertEqual(nxt[: len(previous)], previous)  # append-only
        self.assertEqual(self.summary["d4"]["retained_byte_identical"], True)
        # No sediment: the harmful candidate is never promoted.
        self.assertEqual(self.summary["d4"]["byte_identical"], True)
        self.assertEqual(trace[-1], ())
        self.assertEqual(tuple(self.runner.board.d4), ())

    def test_registered_trace_cannot_reach_the_live_overlay_path(self) -> None:
        """Reachability finding: one stream never produces a motif cohort.

        The motif thresholds are the derived ``(2, 2)`` pair of prereg §5
        (task-23); the measured fact reported here is that the registered
        seed 7 still produces no motif cohort and no live-overlay event.
        """
        self.assertEqual(self.summary["cohort_counts"]["motif"], 0)
        self.assertEqual(self.summary["lifecycle"]["motif_reads"], 0)
        self.assertEqual(self.summary["lifecycle"]["overlay_birth"], None)
        self.assertEqual(self.summary["lifecycle"]["overlay_stale"], None)
        self.assertEqual(self.summary["lifecycle"]["overlay_evict"], None)
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

    def test_end_state_is_measured_and_bounded(self) -> None:
        """prereg §4: the refused candidate stays resident; no live overlay."""
        self.assertTrue(self.summary["encoder_bit_identical"])
        self.assertTrue(self.runner.organ.encoder_bit_identical())
        # No live overlay ever existed, so the D2 overlay index is empty and
        # carries no patch bytes.  The candidate's substance patch remains --
        # attached to the nursery record, bounded and non-steering.
        self.assertEqual(self.runner.board.residents(), ())
        self.assertEqual(self.runner.board.patches(), ())
        self.assertEqual(
            self.runner.board.patch_bytes(),
            self.runner.board.nursery_patch_bytes(),
        )
        self.assertEqual(self.runner.board.nursery_ids(), ("cand-01",))
        self.assertEqual(self.runner.organ.candidate_ids(), ("cand-01",))
        self.assertEqual(self.runner.board.nursery_patch_bytes(), 14_848)
        self.assertLessEqual(
            self.runner.board.nursery_patch_bytes(), MAX_PATCH_TOTAL_BYTES
        )
        self.assertTrue(self.summary["lifecycle"]["d2_empty_at_end"])
        self.assertFalse(self.summary["lifecycle"]["nursery_empty_at_end"])
        self.assertEqual(self.runner.board.d4, [])


@requires_artifacts
class TestAdaptationGate(unittest.TestCase):
    """ADAPTATION (prereg §1/§2): ratio-of-means primary, strict ``< 1.0``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)

    def test_primary_is_the_ratio_of_row_sums_and_fails(self) -> None:
        n, primary, ticks = _adaptation_primary_from_rows(self.result.ticks)
        self.assertEqual(n, 15)
        self.assertEqual(ticks, list(range(13, 28)))
        self.assertAlmostEqual(primary, 1.013377976345238, places=12)
        summary_gate = self.result.summary["gates"]["adaptation"]
        self.assertEqual(summary_gate["n"], n)
        self.assertAlmostEqual(summary_gate["primary"], primary, places=15)
        self.assertAlmostEqual(summary_gate["value"], primary, places=15)
        # Measured finding, not tuned: the candidate is harmful, so the
        # registered trace FAILS ADAPTATION under the pre-registered statistic.
        self.assertGreater(primary, ADAPTATION_GATE)
        self.assertFalse(summary_gate["passed"])
        self.assertEqual(summary_gate["threshold"], ADAPTATION_GATE)
        # Strictness is pinned: the gate is < 1.0, never <=.
        self.assertFalse(primary < ADAPTATION_GATE)

    def test_secondary_is_the_mean_of_per_tick_ratios(self) -> None:
        secondary = _adaptation_secondary_from_rows(self.result.ticks)
        self.assertAlmostEqual(secondary, 1.0211688667826473, places=12)
        summary_gate = self.result.summary["gates"]["adaptation"]
        self.assertAlmostEqual(
            summary_gate["secondary_mean_of_ratios"], secondary, places=15
        )
        # The two statistics are different numbers: that is the aggregation
        # convention the pre-registration fixes (ratio-of-means is the gate).
        self.assertNotAlmostEqual(
            summary_gate["primary"],
            summary_gate["secondary_mean_of_ratios"],
            places=6,
        )

    def test_the_gate_set_is_the_declared_predicate(self) -> None:
        selected = [
            row.tick
            for row in self.result.ticks
            if row.phase == "flipped" and row.shadow_adapted_sum is not None
        ]
        self.assertEqual(selected, list(range(13, 28)))
        self.assertEqual(self.result.summary["adaptation_set"], selected)
        # The rows sum to the declared statistic: no re-weighting by frame.
        n, adapted_sum, base_sum, _ticks = _adaptation_sums(self.result.ticks)
        self.assertEqual(
            self.result.summary["gates"]["adaptation"]["primary"],
            adapted_sum / base_sum,
        )
        # Shadow rows outside the flipped phase are scored (the candidate
        # stays resident through recovery) but are NOT in the gate set; the
        # residence set carries them (prereg §2: the sets are declared, not
        # assumed to coincide with the phase).
        outside = [
            row.tick
            for row in self.result.ticks
            if row.shadow_adapted_sum is not None and row.phase != "flipped"
        ]
        self.assertEqual(outside, list(range(28, TOTAL_TICKS + 1)))
        self.assertFalse(set(outside) & set(selected))


@requires_artifacts
class TestAdaptationConvention(unittest.TestCase):
    """prereg §1/§2: both numbers, both tick sets, report-only birth sample."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)
        cls.gate = cls.result.summary["gates"]["adaptation"]

    def test_tick_sets_are_the_declared_predicate_intersections(self) -> None:
        rows = self.result.ticks
        adaptation = [
            row.tick
            for row in rows
            if row.phase == "flipped" and row.shadow_adapted_sum is not None
        ]
        record = self.runner._tracked_candidate
        self.assertIsNotNone(record)
        residence = [
            int(sample[0])
            for sample in (getattr(record, "evidence", ()) or ())
        ]
        self.assertEqual(self.gate["tick_sets"]["adaptation"], adaptation)
        self.assertEqual(self.gate["tick_sets"]["residence"], residence)
        self.assertEqual(self.result.summary["adaptation_set"], adaptation)
        self.assertEqual(self.result.summary["residence_set"], residence)
        self.assertEqual(
            self.result.summary["with_in_sample_set"],
            self.gate["tick_sets"]["with_in_sample"],
        )
        self.assertEqual(
            self.result.summary["residence_ratio_of_means"],
            self.gate["residence_ratio_of_means"],
        )

    def test_with_in_sample_adds_exactly_the_birth_sample(self) -> None:
        record = self.runner._tracked_candidate
        self.assertIsNotNone(record)
        sample = getattr(record, "in_sample", None)
        if sample is None:
            # The organ-side ``in_sample`` field lands in task-23.  Until it
            # is there the sets coincide and the summary must say so; the
            # runner never fabricates the birth sample (prereg §1).
            self.assertEqual(self.gate["with_in_sample"], self.gate["primary"])
            self.assertEqual(self.gate["n_with_in_sample"], self.gate["n"])
            self.assertEqual(
                self.gate["tick_sets"]["with_in_sample"],
                self.gate["tick_sets"]["adaptation"],
            )
            self.assertEqual(
                self.gate["tick_sets"]["adaptation"],
                self.result.summary["adaptation_set"],
            )
            return
        birth_tick, birth_adapted, birth_base = sample
        n, adapted_sum, base_sum, ticks = _adaptation_sums(self.result.ticks)
        birth_row = next(
            row for row in self.result.ticks
            if int(row.frame_tick) == int(birth_tick)
        )
        self.assertEqual(birth_row.phase, "flipped")
        expected = (adapted_sum + float(birth_adapted)) / (
            base_sum + float(birth_base)
        )
        self.assertAlmostEqual(self.gate["with_in_sample"], expected, places=15)
        self.assertEqual(self.gate["n_with_in_sample"], n + 1)
        self.assertEqual(
            self.gate["tick_sets"]["with_in_sample"],
            sorted(ticks + [int(birth_tick)]),
        )
        # The birth sample is report-only: it is not in the evidence set, so
        # the residence / adaptation numbers cannot have used it.
        self.assertNotIn(int(birth_tick), ticks)
        self.assertFalse(
            any(int(item[0]) == int(birth_tick) for item in record.evidence)
        )

    def test_residence_ratio_of_means_is_the_evidence_sums(self) -> None:
        record = self.runner._tracked_candidate
        evidence = tuple(record.evidence)
        self.assertEqual(len(evidence), 24)
        adapted_sum = float(sum(item[1] for item in evidence))
        base_sum = float(sum(item[2] for item in evidence))
        self.assertAlmostEqual(
            self.gate["residence_ratio_of_means"], adapted_sum / base_sum,
            places=12,
        )
        self.assertAlmostEqual(
            self.gate["residence_ratio_of_means"], 1.060768745410438,
            places=12,
        )
        # The residence set is *not* a subset of the flipped phase in 12a:
        # scoring runs through the recovery phase (measured fact, prereg §2).
        self.assertEqual(
            self.gate["tick_sets"]["residence"],
            list(range(13, TOTAL_TICKS + 1)),
        )
        self.assertGreater(
            self.gate["tick_sets"]["residence"][-1], 27
        )


@requires_artifacts
class TestAdmissionGate(unittest.TestCase):
    """ADMISSION (prereg §5): PASS iff no promoted candidate is harmful."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)

    def test_registered_trace_has_no_promotion_and_the_gate_is_vacuous(self) -> None:
        count, harmful = _admission_from_rows(self.result.ticks)
        self.assertEqual(count, ADMISSION_GATE)
        self.assertEqual(harmful, [])
        summary_gate = self.result.summary["gates"]["admission"]
        self.assertEqual(summary_gate["value"], count)
        self.assertEqual(summary_gate["n"], 0)
        self.assertTrue(summary_gate["passed"])
        self.assertEqual(summary_gate["threshold"], ADMISSION_GATE)
        self.assertEqual(self.runner.organ.promotions, ())

    def test_the_promotion_log_mirrors_the_ablated_seed0_promotion(self) -> None:
        """The qualified-survivor path is measured on the floor-ablated run."""
        run, result = scenario(CONTROL_SEED, gain_floor="ablate")
        promotions = run.organ.promotions
        self.assertEqual(len(promotions), 1)
        promotion = promotions[0]
        row = next(row for row in result.ticks if row.promoted_id)
        self.assertEqual(promotion.candidate_id, row.promoted_id)
        self.assertEqual(promotion.promoted_tick, row.tick)
        self.assertEqual(promotion.residence_ticks, row.candidate_residence_ticks)
        self.assertAlmostEqual(promotion.ratio, row.candidate_ratio, places=15)
        self.assertTrue(promotion.qualified)
        self.assertLess(promotion.ratio, 1.0)
        self.assertEqual(
            result.summary["gates"]["admission"]["promotions"][0]["ratio"],
            promotion.ratio,
        )
        # With the registered floor on, the same qualified survivor is
        # refused, so the registered control promotes nothing.
        _on_run, on_result = scenario(CONTROL_SEED)
        self.assertEqual(on_result.summary["gates"]["admission"]["n"], 0)


@requires_artifacts
class TestControlSeed0(unittest.TestCase):
    """Seed 0 is the rejection control; the floor is what refuses it.

    prereg §4: seed 0 births at its first spike (t9) with the small
    capacity-bounded candidate (N=9, rank 14, K=2,752).  Its residence
    evidence qualifies at t14 (ratio 0.983203 < 1.0), so the pre-floor gate
    would promote it -- and does, in the ablated diagnostic.  Under the
    registered ``MDL_FLOOR`` the t14 promotion is refused ("uneconomic":
    gain ~168 bits vs cost 88,064 bits), so the registered control has
    **zero promotions** and the refused candidate keeps scoring.  The
    end state is therefore reported, not asserted clean (Lead ruling).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(CONTROL_SEED)
        cls.ablated_runner, cls.ablated_result = scenario(
            CONTROL_SEED, gain_floor="ablate"
        )

    def test_births_the_small_candidate_at_its_first_spike(self) -> None:
        self.assertEqual(self.result.summary["lifecycle"]["candidate_birth"], 9)
        self.assertEqual(self.runner.births, [("cand-01", 9)])
        record = self.result.summary["candidate_fit"]["record"]
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["rank"], 14)
        self.assertEqual(record["param_count"], 2_752)
        self.assertEqual(record["nbytes"], 11_008)
        self.assertEqual(record["budget_n"], 9)
        self.assertEqual(record["budget_capacity"], 2_880.0)
        self.assertEqual(record["reason"], "")

    def test_floor_refuses_the_pre_floor_promotion(self) -> None:
        self.assertEqual(self.runner.gain_floor, "on")
        self.assertEqual(self.runner.organ.promotions, ())
        self.assertEqual(self.result.summary["gates"]["admission"]["n"], 0)
        self.assertEqual(self.result.summary["gates"]["admission"]["value"], 0)
        self.assertTrue(self.result.summary["gates"]["admission"]["passed"])
        qualified = self.result.summary["floor"]["qualified_refusals"]
        self.assertTrue(qualified)
        first = qualified[0]
        self.assertEqual((first["tick"], first["candidate_id"]), (14, "cand-01"))
        self.assertTrue(first["qualified"])
        self.assertFalse(first["admitted"])
        self.assertEqual(first["reason"], "uneconomic")
        self.assertGreater(first["dbar"], 0.0)
        self.assertLess(first["gain_bits"], first["cost_bits"])
        # The floor is a reading of the record's own out-of-sample evidence AT
        # THE DECISION; the same record keeps scoring after the refusal, and
        # by end of run its accumulated evidence is harmful.  Both readings
        # are reported: the refusal decision is "uneconomic", the end-state
        # verdict is a refusal too (class "harmful").
        record = self.runner._tracked_candidate
        verdict = MDL_FLOOR.verdict(record)
        self.assertFalse(verdict["admitted"])
        self.assertLess(verdict["gain_bits"], verdict["cost_bits"])
        self.assertIn(verdict["reason"], {"harmful", "uneconomic"})
        self.assertEqual(verdict["reason"], "harmful")
        # Refused, not promoted: the candidate stays resident and D4 is clean.
        self.assertEqual(self.runner.board.nursery_ids(), ("cand-01",))
        self.assertEqual(self.runner.board.d4, [])

    def test_ablated_control_reproduces_the_pre_floor_promotion(self) -> None:
        """The floor-ablated diagnostic reproduces the round-11 gate exactly."""
        self.assertEqual(self.ablated_runner.gain_floor, "ablate")
        promotions = self.ablated_runner.organ.promotions
        self.assertEqual(len(promotions), 1)
        promotion = promotions[0]
        self.assertEqual(
            (promotion.candidate_id, promotion.promoted_tick), ("cand-01", 14)
        )
        self.assertAlmostEqual(promotion.ratio, 0.9832027173109578, places=12)
        self.assertTrue(promotion.qualified)
        # D4 gets exactly one sediment, append-only, with the lineage.
        self.assertEqual(len(self.ablated_runner.board.d4), 1)
        self.assertEqual(
            self.ablated_runner.board.d4[0].candidate_lineage, "cand-01"
        )
        self.assertFalse(self.ablated_result.summary["d4"]["byte_identical"])
        self.assertTrue(
            self.ablated_result.summary["d4"]["retained_byte_identical"]
        )
        sediments = self.ablated_result.summary["d4"]["sediments"]
        self.assertEqual(len(sediments), 1)
        self.assertEqual(sediments[0]["tick"], 14)
        self.assertAlmostEqual(
            sediments[0]["confidence"], 1.0 - 0.9832027173109578, places=12
        )
        # Promotion removes the record and its patch atomically.
        self.assertEqual(self.ablated_runner.board.nursery_ids(), ())
        self.assertEqual(self.ablated_runner.board.nursery_patch_bytes(), 0)
        self.assertEqual(self.ablated_runner.board.patch_bytes(), 0)

    def test_adaptation_set_grows_when_the_promotion_is_refused(self) -> None:
        """Floor-on vs ablated: the ADAPTATION set depends on the promotion.

        This is a measured consequence, not a tuning: a refused candidate
        keeps being shadow-scored, so the flipped-phase evidence set grows
        and the gate reading changes.  Both numbers are reported.
        """
        n, primary, ticks = _adaptation_primary_from_rows(self.result.ticks)
        self.assertEqual(n, 18)
        self.assertEqual(ticks, list(range(10, 28)))
        self.assertAlmostEqual(primary, 1.032106685155439, places=12)
        gate = self.result.summary["gates"]["adaptation"]
        self.assertAlmostEqual(gate["primary"], primary, places=15)
        self.assertFalse(gate["passed"])
        self.assertGreater(primary, ADAPTATION_GATE)
        self.assertAlmostEqual(
            gate["secondary_mean_of_ratios"], 1.0761264980702896, places=12
        )
        kn, kprimary, kticks = _adaptation_primary_from_rows(
            self.ablated_result.ticks
        )
        self.assertEqual(kn, 5)
        self.assertEqual(kticks, [10, 11, 12, 13, 14])
        self.assertAlmostEqual(kprimary, 0.9832027173109578, places=12)
        self.assertTrue(self.ablated_result.summary["gates"]["adaptation"]["passed"])
        self.assertEqual(self.result.summary["residence_set"], list(range(10, 37)))
        self.assertEqual(
            self.ablated_result.summary["residence_set"], [10, 11, 12, 13, 14]
        )

    def test_floor_on_end_state_is_measured_not_asserted_clean(self) -> None:
        """Do NOT assert a clean end state for the control (Lead ruling)."""
        self.assertEqual(self.runner.board.nursery_ids(), ("cand-01",))
        self.assertEqual(self.runner.board.nursery_patch_bytes(), 11_008)
        self.assertEqual(self.runner.board.d4, [])
        self.assertTrue(self.result.summary["d4"]["byte_identical"])
        self.assertTrue(self.result.summary["d4"]["retained_byte_identical"])
        # The control does reach the motif path (prereg §5 measured
        # consequence); the registered seed 7 does not.
        self.assertGreaterEqual(
            self.result.summary["lifecycle"]["motif_reads"], 1
        )


@requires_artifacts
class TestLiveOverlayLifecycle(unittest.TestCase):
    """The ``#21`` -> ``#24`` -> ``#25`` -> evict lifecycle, executor-side.

    The derived motif thresholds (prereg §5) give the registered seed 7 no
    motif cohort, so this test deposits one valid motif cohort into D3
    mid-run and lets the executor do the rest.  The fixture supplies only the
    cohort; every derivation, gate, commit, staleness decision and eviction
    below is the production path.  This is a wiring test, not a measurement
    of the registered scenario.  Note the overlay id is read from the organ
    (the candidate fit consumes an overlay serial before the live overlay is
    born), never hardcoded.
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
        overlay_id = self.runner.organ.overlay_table.births[0].overlay_id
        read = [row for row in self.rows if row.motif_id == self.MOTIF_ID]
        self.assertTrue(read)
        self.assertEqual(read[0].tick, self.DEPOSIT_TICK)
        birth = next(row for row in self.rows if row.adopted is not None)
        self.assertEqual(birth.tick, self.DEPOSIT_TICK)
        self.assertEqual(birth.adopted, overlay_id)
        self.assertIn(overlay_id, birth.residents)
        # The D2 write carries the #21 motif lineage (the executor's gate).
        body = self.bodies[birth.tick][24]
        self.assertIsInstance(body, MatOverlay)
        self.assertEqual(body.motif_lineage, self.MOTIF_ID)
        self.assertEqual(self.bodies[birth.tick][21].motif_id, self.MOTIF_ID)
        # The live patch is attached to D2 (Q1-A) and bounded.
        self.assertGreater(birth.patch_bytes, 0)
        self.assertLessEqual(birth.patch_bytes, MAX_PATCH_TOTAL_BYTES)
        self.assertIn(overlay_id, self.patch_ids[birth.tick])
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
        self.assertEqual(active.overlay_id, overlay_id)
        # The #25 hand-off is the same epoch the head carries, one tick later.
        self.assertEqual(active.epoch, stale.epoch)

    def test_revalidation_advances_the_epoch(self) -> None:
        self.assertGreaterEqual(len(self.runner.organ.overlay_table.revalidations), 1)
        epochs = [record.epoch for record in self.runner.organ.overlay_table.revalidations]
        self.assertEqual(epochs, sorted(epochs))
        self.assertEqual(self.runner.organ.overlay_table.births[0].epoch, 1)

    def test_stale_then_evict_and_both_indices_clean(self) -> None:
        overlay_id = self.runner.organ.overlay_table.births[0].overlay_id
        staled = [row for row in self.rows if row.staled is not None]
        evicted = [row for row in self.rows if row.evicted]
        self.assertEqual(len(staled), 1)
        self.assertEqual(staled[0].staled, overlay_id)
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].evicted, (overlay_id,))
        self.assertLess(staled[0].tick, evicted[0].tick)
        self.assertEqual(evicted[0].tick, staled[0].tick + 3)  # STALE_AFTER
        self.assertEqual(evicted[-1].residents, ())
        # The overlay patch is gone; the remaining bytes are the resident
        # candidate's nursery substance (the reflected candidate is not
        # evicted with the live overlay).
        self.assertEqual(self.runner.board.nursery_patch_bytes(), 14_848)
        self.assertEqual(evicted[-1].patch_bytes, 14_848)
        self.assertEqual(self.runner.board.residents(), ())
        self.assertEqual(self.runner.board.patches(), ())
        self.assertEqual(self.runner.organ.overlay_table.records(), ())
        self.assertEqual(self.runner.organ.overlay_table.active(), ())
        # The store's #8/#25 reads are the *pre-eviction* state on the
        # eviction tick (the organ decides first, the prune mirrors at the
        # end of the same tick) and None from the next tick on.
        self.assertEqual(self.bodies[evicted[0].tick][8].overlay_id, overlay_id)
        self.assertIsNone(self.bodies[evicted[0].tick + 1][8])
        self.assertIsNone(self.bodies[evicted[0].tick + 1][25])

    def test_the_registered_run_never_enters_this_path(self) -> None:
        _runner, result = scenario(REGISTERED_SEED)
        self.assertEqual(result.summary["lifecycle"]["overlay_birth"], None)
        self.assertEqual(result.summary["cohort_counts"]["motif"], 0)
        # The registered trace has no promotion either (the candidate is
        # harmful); D4 stays empty.
        self.assertEqual(result.summary["d4"]["sediments"], ())


@requires_artifacts
class TestRejectWithChargeAndNoop(unittest.TestCase):
    """Reject-with-charge and the Q4 zero-confidence no-op, physics path."""

    def _drive_to(
        self,
        tick: int,
        *,
        seed: int = REGISTERED_SEED,
        gain_floor: str = "on",
    ) -> PhysicsRun:
        run = PhysicsRun(seed=seed, gain_floor=gain_floor)
        run.reset()
        run._flip_at, run._flip_back_at = 8, 28
        for index in range(1, tick):
            if index == 8:
                run.flip_gravity()
            run.tick()
        return run

    def test_duplicate_arrival_is_charged_and_state_is_unchanged(self) -> None:
        run = self._drive_to(20)  # t16+ surface the harmful survivor
        d4_before = tuple(run.board.d4)
        before = (
            sum(run.dry.attempted.values()),
            sum(run.dry.accepted.values()),
            sum(run.dry.refused.values()),
            run.board.nursery_ids(),
            len(run.board.d4),
        )
        row = run.tick(injections={22: 2})
        self.assertEqual(row.tick, 20)
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
        """A positive #22 must cite this tick's consumed #23 survivor.

        Seed 7 never produces a qualified survivor in 12a (it is harmful), so
        the positive path is measured on the seed-0 control with the floor
        ablated (the pre-floor gate), where the candidate qualifies and is
        promoted at t14.
        """
        run = self._drive_to(14, seed=CONTROL_SEED, gain_floor="ablate")
        row = run.tick()
        self.assertEqual(row.tick, 14)
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
        # No live overlay ever existed: the overlay index and its patches are
        # empty.  The refused candidate remains resident with its substance
        # patch, which is accounted as nursery bytes (bounded).
        self.assertEqual(self.runner.board.patches(), ())
        self.assertEqual(self.runner.board.resident_ids(), frozenset())
        self.assertEqual(self.runner.board.nursery_ids(), ("cand-01",))
        self.assertEqual(self.runner.board.nursery_patch_bytes(), 14_848)
        self.assertEqual(
            self.runner.board.patch_bytes(),
            self.runner.board.nursery_patch_bytes(),
        )
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
        self.assertIsNone(run._tracked_candidate)


@requires_artifacts
class TestTrigger(unittest.TestCase):
    """prereg §4: ``N_MIN`` replaces the absolute tick guard; N caps the fit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.run7, cls.result7 = scenario(REGISTERED_SEED)
        cls.run0, cls.result0 = scenario(CONTROL_SEED)

    def test_seed7_births_at_its_first_spike_with_n_min(self) -> None:
        threshold = float(self.run7.organ.thresholds.spike_threshold)
        spikes = [
            row.tick
            for row in self.result7.ticks
            if row.surprise > threshold and row.tick >= N_MIN
        ]
        self.assertEqual(spikes[0], 12)
        self.assertEqual(
            self.result7.summary["lifecycle"]["candidate_birth"], spikes[0]
        )
        # The old ``tick >= fit_window`` guard skipped t12 (the largest spike
        # of the trace); prereg §4 removes it, and the Lead amended the
        # prereg to record this consequence.  The task prose's "t19" was the
        # Round-11 fingerprint, superseded by the prereg.
        self.assertEqual(
            self.result7.summary["candidate_fit"]["record"]["budget_n"], 12
        )

    def test_seed0_births_at_its_first_spike_with_a_smaller_candidate(self) -> None:
        threshold = float(self.run0.organ.thresholds.spike_threshold)
        spikes = [
            row.tick
            for row in self.result0.ticks
            if row.surprise > threshold and row.tick >= N_MIN
        ]
        self.assertEqual(spikes[0], 9)
        self.assertEqual(
            self.result0.summary["lifecycle"]["candidate_birth"], spikes[0]
        )
        record0 = self.result0.summary["candidate_fit"]["record"]
        record7 = self.result7.summary["candidate_fit"]["record"]
        self.assertEqual(record0["param_count"], 2_752)
        self.assertEqual(record7["param_count"], 3_712)
        # "Small candidate": strictly smaller than both the measured seed-7
        # candidate and the historical N=16 value (K=5,056).
        self.assertLess(record0["param_count"], record7["param_count"])
        self.assertLess(record0["param_count"], 5_056)

    def test_fit_window_is_capped_to_the_actual_evidence(self) -> None:
        for run, result, expected in (
            (self.run7, self.result7, 12),
            (self.run0, self.result0, 9),
        ):
            record = result.summary["candidate_fit"]["record"]
            self.assertEqual(record["budget_n"], expected)
            # The structural cap (16) is not the trigger: the evidence is
            # smaller, and the capacity bound uses ``min(fit_window, tick)``.
            self.assertLess(expected, run.fit_window)
        self.assertEqual(self.run7.fit_window, 16)

    def test_registered_trigger_uses_n_min_and_historical_is_opt_in(self) -> None:
        """The registered path uses N_MIN; round11 is a labelled opt-in mode.

        The source still contains the old ``tick >= fit_window`` guard, but
        only inside the ``trigger="round11"`` HISTORICAL DIAGNOSTIC branch,
        which is used to reproduce the Round-11 falsifier trace.  The default
        is ``"registered"`` and the tests prove it behaviourally and at the
        signature level.
        """
        source = Path(run_module.__file__).read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_should_birth"
        )
        names = {
            node.id for node in ast.walk(function) if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn("N_MIN", names)  # the registered branch
        self.assertIn("fit_window", attributes)  # the historical branch
        self.assertEqual(N_MIN, 2)
        self.assertEqual(set(TRIGGERS), {"registered", "round11"})
        self.assertEqual(run_module.DEFAULT_TRIGGER, "registered")

        runner = PhysicsRun(seed=REGISTERED_SEED)
        self.assertEqual(runner.trigger, "registered")
        runner._fit_attempted = False
        runner.birth_on_spike = True

        class _Observation:
            def __init__(self, tick: int, surprise: float) -> None:
                self.tick = tick
                self.surprise = surprise

        high = float(runner.organ.thresholds.spike_threshold) * 2.0
        # Registered: N_MIN is the floor, N_MIN-1 is refused.
        self.assertFalse(runner._should_birth(_Observation(N_MIN - 1, high)))
        self.assertTrue(runner._should_birth(_Observation(N_MIN, high)))
        with mock.patch.object(run_module, "N_MIN", 5):
            self.assertFalse(runner._should_birth(_Observation(4, high)))
            self.assertTrue(runner._should_birth(_Observation(5, high)))
        # Historical diagnostic: the old absolute fit-window guard.
        historical = PhysicsRun(seed=REGISTERED_SEED, trigger="round11")
        self.assertEqual(historical.trigger, "round11")
        historical._fit_attempted = False
        historical.birth_on_spike = True
        self.assertFalse(
            historical._should_birth(
                _Observation(historical.fit_window - 1, high)
            )
        )
        self.assertTrue(
            historical._should_birth(_Observation(historical.fit_window, high))
        )


@requires_artifacts
class TestGainFloorGate(unittest.TestCase):
    """prereg §3/§6: the MDL floor, its refusals and the retroactive falsifier."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner, cls.result = scenario(REGISTERED_SEED)

    def test_registered_default_is_gain_floor_on(self) -> None:
        import inspect

        default = inspect.signature(PhysicsRun.__init__).parameters[
            "gain_floor"
        ].default
        self.assertEqual(default, "on")
        self.assertEqual(DEFAULT_GAIN_FLOOR, "on")
        self.assertEqual(PhysicsRun(seed=REGISTERED_SEED).gain_floor, "on")
        self.assertEqual(
            self.result.summary["scenario"]["gain_floor"], "on"
        )
        self.assertEqual(self.result.summary["floor"]["mode"], "on")

    def test_seed7_floor_on_has_zero_promotions_and_harmful_refusals(self) -> None:
        gate = self.result.summary["gates"]["adaptation"]
        self.assertEqual(gate["n"], 15)
        self.assertAlmostEqual(gate["primary"], 1.013377976345238, places=12)
        # Pre-registered expectation, reported not tuned: the candidate is
        # harmful, so ADAPTATION FAILS and ADMISSION is vacuous (n=0).
        self.assertFalse(gate["passed"])
        self.assertEqual(self.result.summary["gates"]["admission"]["n"], 0)
        self.assertEqual(self.result.summary["gates"]["admission"]["value"], 0)
        self.assertTrue(self.result.summary["gates"]["admission"]["passed"])
        self.assertEqual(self.runner.organ.promotions, ())
        self.assertEqual(self.runner.board.d4, [])
        refusals = self.result.summary["floor"]["refusals"]
        self.assertTrue(refusals)
        self.assertTrue(all(item["reason"] == "harmful" for item in refusals))
        self.assertTrue(all(item["dbar"] <= 0.0 for item in refusals))
        # A harmful survivor is never qualified, so no qualified refusal.
        self.assertEqual(
            self.result.summary["floor"]["qualified_refusals"], ()
        )

    def test_seed7_ablated_still_has_no_promotion(self) -> None:
        """The new-trigger candidate is harmful: the floor is not the reason.

        Removing the floor does not rescue it, which is why the falsifier
        must use the historical (round-11) trigger instead.
        """
        _run, result = scenario(REGISTERED_SEED, gain_floor="ablate")
        self.assertEqual(result.summary["gates"]["admission"]["n"], 0)
        self.assertEqual(result.summary["gates"]["admission"]["value"], 0)
        self.assertTrue(result.summary["gates"]["admission"]["passed"])
        self.assertEqual(result.summary["floor"]["refusals"], ())

    def test_historical_falsifier_refuses_the_round11_candidate(self) -> None:
        """prereg §3.5: the floor rejects the Round-11 t27 promotion.

        ``trigger="round11"`` + ``gain_floor="ablate"`` reproduces the
        Round-11 candidate (birth t19, K=5,056, promotion t27); the floor is
        then applied retroactively to the promoted record.  Every number is
        asserted from the run, not hardcoded.
        """
        run, result = scenario(
            REGISTERED_SEED, trigger="round11", gain_floor="ablate"
        )
        self.assertEqual(result.summary["scenario"]["trigger"], "round11")
        self.assertEqual(result.summary["scenario"]["gain_floor"], "ablate")
        # Round-11 reproduction: immediate promotion of the qualified t27
        # survivor under the ablated floor.
        self.assertEqual(run.births, [("cand-01", 19)])
        fit = result.summary["candidate_fit"]["record"]
        self.assertEqual(
            (fit["rank"], fit["param_count"], fit["budget_n"]),
            (26, 5_056, 16),
        )
        promotions = run.organ.promotions
        self.assertEqual(len(promotions), 1)
        self.assertEqual(
            (promotions[0].candidate_id, promotions[0].promoted_tick),
            ("cand-01", 27),
        )
        # The retroactive verdict: the same record, read by the floor.
        verdict = result.summary["candidate_verdict"]
        self.assertIsNotNone(verdict)
        assert verdict is not None
        self.assertFalse(verdict["admitted"])
        self.assertEqual(verdict["reason"], "uneconomic")
        self.assertLess(verdict["gain_bits"], verdict["cost_bits"])
        self.assertEqual(verdict["cost_bits"], 32.0 * fit["param_count"])
        record = run._tracked_candidate
        independent = MDL_FLOOR.verdict(record)
        for key in ("dbar", "gain_bits", "cost_bits"):
            self.assertAlmostEqual(
                float(independent[key]), float(verdict[key]), places=12
            )
        self.assertFalse(independent["admitted"])
        # Under the registered floor the historical candidate is refused at
        # the same decision tick (the floor is the condition, not a redraw).
        _on_run, on_result = scenario(
            REGISTERED_SEED, trigger="round11", gain_floor="on"
        )
        self.assertEqual(on_result.summary["gates"]["admission"]["n"], 0)
        qualified = on_result.summary["floor"]["qualified_refusals"]
        self.assertTrue(qualified)
        self.assertEqual(
            (qualified[0]["tick"], qualified[0]["reason"]),
            (27, "uneconomic"),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
