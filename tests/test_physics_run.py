"""Phase 4-5b: executor wiring tests (PhysicsSource + PhysicsRun).

Pins the Q1-Q4 binding contract at the executor boundary:

* ``DryRun()`` defaults to the explicit ``SyntheticSource`` plumbing
  sentinel; ``PhysicsRun`` constructs a ``PhysicsSource`` (Q2).
* The coverage ledger is plain data covering every non-read flow; derived
  flows produce a non-None exact-type body, placeholders are ``None`` in
  the executor's derived envelope, and an absent fid asserts rather than
  falling back (no silent mint fallback).
* A source failure is an assertion in the derivation pass, never a channel
  charge.
* The D2 patch table is substance-accounted: the Round 9a capacity bound
  stores a rank-26 low-rank head.2 patch (5,056 params / 20,224 B) on the
  single-stream fit, the dense full-predictor request (210,920 params) is
  refused by the evidence budget *before* any fit or overlay, an
  over-ceiling synthetic patch is still refused in the derivation pass,
  record+patch eviction is atomic, and a resident record with no patch
  asserts.
* The #22 gate body is the source's derived output (not a literal in
  ``_derive``).  The gate is real: it consumes cumulative adapted-vs-base
  evidence that the organ records on applied ticks and publishes after
  ``stale_patience`` observations.  Round 9a changed the measured regime:
  the registered trace's patch now *passes* pre-adoption validation, so its
  cumulative evidence is beneficial and the gate really passes on 11 ticks
  (measuring 0.0245..0.0626 confidence) while the overlay is active, then
  blocks again once the overlay is evicted.  Those 11 sediments are the
  gate's own derived output, with no test double.  D4 retention is proven
  non-vacuously: the retained prefix is byte-identical and object-identical
  across the rest of the 36-tick scenario; every later sediment is an
  append.  The identity ``B: 1 = 1 + 0 + 0`` still holds on the
  zero-confidence path (25 consumed no-ops).
* Q1 atomicity: the outer byte ceilings remain storage bounds, so a
  synthetic over-ceiling patch is still refused in the derivation pass,
  before any arrival/commit, leaving the store, D4 and the organ clean.
  The Round 9a evidence bound is applied earlier (at fit time), so a
  ``patch="full"`` request never produces an overlay at all.
* P6 reset is explicit: two consecutive ``acceptance()`` calls (and two
  consecutive ``run()`` calls) on one runner produce identical traces.
* The single-stream acceptance reports the measured pre/post surprise
  honestly.  Round 9a's validation-gated candidate is born at tick 18 and
  the registered trace still does **not** cross the acceptance criterion
  over every active tick (measured 1.0461, improved from the pre-9a
  1.2614); over the active *flipped* ticks only it measures 0.9666.  Both
  numbers are reported; ``test_single_stream_surprise_drops`` stays an
  explicit recorded failure on the registered metric, not weakened into
  "no material harm".  The bank-fit positive result (0.8226x) is in
  ``tests/test_physics_organ.py``.  The pre-adoption validation path is
  pinned for both outcomes: a real birth (seed 7) and a real interference
  rejection (seed 0).
* Encoder bit-identity is asserted with ``torch.equal`` after the run.

Runtime: ~12 s on 8 CPU threads (checkpoint load plus the repeated
36-tick scenario runs and their fits).
"""
from __future__ import annotations

import os
import pickle
import unittest
from pathlib import Path

import numpy as np
import torch

from amateur.dryrun import DryRun, PLACEHOLDER_FLOWS
from amateur.payloads import (
    MAX_PATCH_BYTES,
    MAX_PATCH_TOTAL_BYTES,
    NON_READ_FLOWS,
    REAL_BODIES,
    BiasPrimitive,
    MatOverlay,
    PAYLOAD_SCHEMA,
)
from amateur.physics.organ import PhysicsOrgan
from amateur.physics.run import PhysicsRun, PhysicsSource

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


def make_run(seed: int = 7, **kwargs: object) -> PhysicsRun:
    return PhysicsRun(seed=seed, **kwargs)  # type: ignore[arg-type]


PROPOSAL_TICK = 16
BIRTH_TICK = 18


def _drive_to_proposal(run: PhysicsRun) -> None:
    """Drive normal -> flip up to the registered proposal tick (16)."""
    for index in range(1, PROPOSAL_TICK + 1):
        if index == 8:
            run.flip_gravity()
        run.tick()


def _drive_to_birth(run: PhysicsRun) -> None:
    """Drive up to and including the registered birth tick (18)."""
    row = None
    for index in range(1, BIRTH_TICK + 1):
        if index == 8:
            run.flip_gravity()
        row = run.tick()
    assert row is not None and row.adopted is not None, (
        "the registered birth did not fire; the scenario changed"
    )


class _OversizedPatch:
    """A synthetic patch over the per-patch ceiling (storage bound only).

    Round 9a applies the evidence bound at fit time, so no admissible fit
    can produce an over-ceiling patch; the outer ``MatBoard`` byte ceiling
    is still enforced as a storage bound and is exercised with this
    explicitly synthetic object.
    """

    def __init__(self, overlay_id: str, *, nbytes: int = MAX_PATCH_BYTES + 1) -> None:
        self.overlay_id = overlay_id
        self._nbytes = int(nbytes)
        self.param_count = self._nbytes // 4

    @property
    def nbytes(self) -> int:
        return self._nbytes


def _body_type(fid: int) -> type:
    if fid in REAL_BODIES:
        return REAL_BODIES[fid]
    if fid == 22:
        return BiasPrimitive
    if fid in (23, 24):
        return MatOverlay
    raise AssertionError(f"no body type for #{fid}")


# ---------------------------------------------------------------------------
# Source selection and coverage ledger
# ---------------------------------------------------------------------------
class TestSourceSelection(unittest.TestCase):
    def test_dryrun_default_is_the_synthetic_sentinel(self) -> None:
        dry = DryRun()
        self.assertIsInstance(dry.source, type(dry.source))
        self.assertEqual(dry.source.mode, "synthetic")
        self.assertEqual(
            set(dry.source.coverage), set(NON_READ_FLOWS)
        )
        self.assertEqual(set(dry.source.coverage.values()), {"synthetic"})

    @requires_artifacts
    def test_physics_run_source_is_physics(self) -> None:
        run = make_run()
        self.assertIsInstance(run.source, PhysicsSource)
        self.assertEqual(run.source.mode, "physics")


@requires_artifacts
class TestCoverageLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = make_run()
        cls.runner.tick()

    def test_ledger_covers_every_non_read_flow(self) -> None:
        coverage = self.runner.source.coverage
        self.assertEqual(set(coverage), set(NON_READ_FLOWS))
        self.assertLessEqual(set(coverage.values()), {"derived", "placeholder"})
        placeholders = {fid for fid, mode in coverage.items() if mode == "placeholder"}
        derived = {fid for fid, mode in coverage.items() if mode == "derived"}
        self.assertEqual(placeholders, PLACEHOLDER_FLOWS)
        self.assertEqual(derived, set(NON_READ_FLOWS) - PLACEHOLDER_FLOWS)

    def test_derived_claim_holds_for_every_derived_fid(self) -> None:
        tick = self.runner.dry.tick_no
        for fid, mode in self.runner.source.coverage.items():
            if mode != "derived":
                continue
            body = self.runner.source.derive_body(
                fid, tick, board=self.runner.board, debris=self.runner.dry.debris
            )
            self.assertIsNotNone(body, msg=f"#{fid} derived but None")
            self.assertIs(type(body), _body_type(fid), msg=f"#{fid}")
            self.assertIn(type(body), PAYLOAD_SCHEMA[fid], msg=f"#{fid}")

    def test_placeholder_envelopes_are_none(self) -> None:
        for fid in PLACEHOLDER_FLOWS:
            self.assertIsNone(
                self.runner.dry.derived[fid].body, msg=f"#{fid} placeholder not None"
            )

    def test_absent_fid_asserts_and_never_falls_back(self) -> None:
        with self.assertRaises(AssertionError):
            self.runner.source.derive_body(
                99, 1, board=self.runner.board, debris=self.runner.dry.debris
            )
        # A placeholder fid is not in the derived dispatch either.
        for fid in PLACEHOLDER_FLOWS:
            with self.assertRaises(AssertionError):
                self.runner.source.derive_body(
                    fid, 1, board=self.runner.board, debris=self.runner.dry.debris
                )

    def test_source_without_context_asserts(self) -> None:
        source = PhysicsSource(self.runner.organ)
        with self.assertRaises(AssertionError):
            source.derive_body(6, 1, board=self.runner.board, debris=self.runner.dry.debris)


class TestSourceFailureIsNotACharge(unittest.TestCase):
    class _ExplodingSource:
        mode = "synthetic"
        coverage = {1: "synthetic", 22: "synthetic"}

        def derive_body(self, fid: int, tick: int, *, board: object, debris: object) -> object:
            raise AssertionError(f"cannot derive #{fid}")

        def on_commit(self, fid: int, envelope: object, *, board: object, debris: object) -> None:
            return None

    def test_failure_raises_before_any_arrival(self) -> None:
        dry = DryRun(source=self._ExplodingSource())
        with self.assertRaises(AssertionError):
            dry.tick()
        self.assertEqual(sum(dry.attempted.values()), 0)
        self.assertEqual(sum(dry.accepted.values()), 0)
        self.assertEqual(sum(dry.refused.values()), 0)
        self.assertEqual(sum(dry.missed.values()), 0)
        self.assertEqual(sum(dry.consumed.values()), 0)


# ---------------------------------------------------------------------------
# #22 gate and no-op semantics
# ---------------------------------------------------------------------------
@requires_artifacts
class TestZeroConfidenceGate(unittest.TestCase):
    """Q4 + Round 9a: the #22 gate is real in *both* directions.

    Round 8 measured only the blocked branch (the dense patch was harmful).
    Round 9a's capacity-bounded patch passes pre-adoption validation, so the
    gate really passes on the active ticks and really blocks once the
    overlay is gone.  No test double is used anywhere below.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = make_run(seed=7)
        cls.result = cls.runner.acceptance()

    def test_d4_sediments_are_the_real_gate_output(self) -> None:
        """The registered trace passes the gate on 11 witnessed ticks."""
        summary = self.result.summary
        self.assertEqual(summary["d4_sediments"], 11)
        self.assertFalse(summary["d4_byte_identical"])
        self.assertEqual(len(self.runner.board.d4), 11)
        for body in self.runner.board.d4:
            self.assertIs(type(body), BiasPrimitive)
            self.assertEqual(body.bias_primitive, "ratchet-gate-passed")
            self.assertGreater(body.confidence, 0.0)
            self.assertLessEqual(body.confidence, 1.0)
        stages = []
        previous = 0
        for row in self.result.ticks:
            if row.d4_size > previous:
                stages.append(row.frame_tick)
            previous = row.d4_size
        # Sediments appear only once the window has 2 applied ticks (19, 20
        # -> first publication at 21) and stop when the overlay goes stale.
        self.assertEqual(stages, [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

    def test_zero_confidence_write_is_a_consumed_noop_identity_b(self) -> None:
        summary = self.result.summary
        ticks = len(self.result.ticks)
        self.assertEqual(summary["accepted_22"], ticks)
        self.assertEqual(summary["consumed_22"], ticks)
        self.assertEqual(summary["refused_22"], 0)
        self.assertEqual(summary["sink_depth_22"], 0)
        self.assertEqual(summary["zero_confidence_writes"], ticks - summary["d4_sediments"])
        self.assertEqual(summary["zero_confidence_writes"], 25)
        # 25 consumed no-ops + 11 real sediments = 36 settled writes.
        self.assertEqual(summary["zero_confidence_writes"] + summary["d4_sediments"], ticks)
        self.assertEqual(
            summary["accepted_22"],
            summary["consumed_22"] + summary["sink_depth_22"] + 0,
        )

    def test_gate_body_is_the_derived_output(self) -> None:
        body = self.runner.dry.derived[22].body
        self.assertIs(type(body), BiasPrimitive)
        # After the eviction there is no active overlay: blocked again.
        self.assertEqual(
            (body.bias_primitive, body.confidence),
            ("ratchet-gate-blocked", 0.0),
        )
        self.assertEqual(self.runner.organ.ratchet_gate_passed(), (False, 0.0))
        self.assertIsNone(self.runner.organ._gate_evidence)

    def test_gate_blocked_without_evidence(self) -> None:
        run = make_run(seed=7)
        run.tick()
        self.assertEqual(run.organ.ratchet_gate_passed(), (False, 0.0))
        self.assertEqual(
            (run.dry.derived[22].body.bias_primitive,
             run.dry.derived[22].body.confidence),
            ("ratchet-gate-blocked", 0.0),
        )
        self.assertEqual(run.board.d4, [])

    def test_gate_blocked_with_evidence_but_no_active_overlay(self) -> None:
        run = make_run(seed=7)
        run.organ._record_gate_evidence(0.2, 0.4)  # beneficial verdict, no overlay
        run.tick()
        self.assertEqual(run.organ.ratchet_gate_passed(), (False, 0.0))
        self.assertEqual(run.dry.derived[22].body.bias_primitive,
                         "ratchet-gate-blocked")
        self.assertEqual(run.board.d4, [])

    def test_gate_passes_on_published_beneficial_evidence(self) -> None:
        """The passing branch through a real tick, exercised by evidence.

        The birth fires at the registered tick; ``flush`` makes the next
        tick's derivation the only decision point, so the injected verdict
        survives to the gate.  This isolates the passing branch from the
        validation machinery; the *real* passing branch is measured in
        ``test_d4_sediments_are_the_real_gate_output`` above.
        """
        run = make_run(seed=7)
        _drive_to_birth(run)
        self.assertEqual(run.board.d4, [])
        run.organ.flush()
        run.organ._record_gate_evidence(0.2, 0.4)
        row = run.tick()
        self.assertEqual(row.tick, 19)
        body = run.dry.derived[22].body
        self.assertIs(type(body), BiasPrimitive)
        self.assertEqual(body.bias_primitive, "ratchet-gate-passed")
        self.assertGreater(body.confidence, 0.0)
        self.assertEqual(body.confidence, 0.5)
        self.assertEqual(len(run.board.d4), 1)
        self.assertIs(run.board.d4[0], body)

    def test_gate_blocked_on_harmful_evidence(self) -> None:
        run = make_run(seed=7)
        _drive_to_birth(run)
        run.organ.flush()
        run.organ._record_gate_evidence(0.4, 0.2)  # adapted worse than base
        run.tick()
        self.assertEqual(run.organ.ratchet_gate_passed(), (False, 0.0))
        self.assertEqual(run.dry.derived[22].body.bias_primitive,
                         "ratchet-gate-blocked")
        self.assertEqual(run.board.d4, [])

    def test_retained_invariant_survives_the_scenario(self) -> None:
        """Non-vacuous Q4 retention across the remaining flip scenario.

        The first real sediment (tick 21) is snapshotted while D4 is
        non-empty; the scenario then continues to tick 36 (flip back at 28,
        recover).  The retained prefix must stay byte-identical *and* be the
        same object, and every later write must be an append.  Every tick
        still settles exactly one #22 event (identity B).
        """
        run = make_run(seed=7)
        run.reset()
        run._flip_at = 8
        run._flip_back_at = 28
        retained = None
        retained_body = None
        for index in range(1, 37):
            if index in (8, 28):
                run.flip_gravity()
            row = run.tick()
            # Identity B on every tick: 1 accepted = 1 consumed + 0 residue.
            self.assertEqual(run.dry.accepted[22], index, msg=f"tick {index}")
            self.assertEqual(run.dry.consumed[22], index, msg=f"tick {index}")
            self.assertEqual(run.dry.refused[22], 0, msg=f"tick {index}")
            self.assertEqual(len(run.dry.mailboxes[22]), 0, msg=f"tick {index}")
            if retained is None and len(run.board.d4) == 1:
                retained = pickle.dumps(run.board.d4, protocol=4)
                retained_body = run.board.d4[0]
                self.assertEqual(row.frame_tick, 21)
        self.assertIsNotNone(retained)
        self.assertIsNotNone(retained_body)
        # Non-vacuous: the retained prefix is still exactly what it was.
        self.assertEqual(pickle.dumps(run.board.d4[:1], protocol=4), retained)
        self.assertIs(run.board.d4[0], retained_body)
        self.assertAlmostEqual(retained_body.confidence, 0.024529555468589148, places=12)
        self.assertEqual(len(run.board.d4), 11)
        self.assertEqual(
            [row.phase for row in run.ticks[-1:]], ["recovered"]
        )


# ---------------------------------------------------------------------------
# D2 patch accounting through the executor
# ---------------------------------------------------------------------------
@requires_artifacts
class TestPatchAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = make_run(seed=7)
        cls.result = cls.runner.acceptance()

    def test_head_patch_attached_and_then_removed_atomically(self) -> None:
        """Round 9a: the stored patch is the rank-26 low-rank head.2 delta."""
        summary = self.result.summary
        self.assertEqual(summary["fit_patch"], "head")
        self.assertEqual(summary["fit_rank"], 26)
        self.assertEqual(summary["fit_param_count"], 5_056)   # 192*26 + 64
        self.assertEqual(summary["fit_nbytes"], 20_224)       # 4 * 5056
        self.assertEqual(summary["fit_budget_n"], 16)
        self.assertEqual(summary["fit_budget_capacity"], 5_120.0)
        self.assertEqual(summary["fit_reason"], "")
        self.assertLessEqual(summary["max_patch_bytes_seen"], MAX_PATCH_BYTES)
        row = next(row for row in self.result.ticks if row.patch_bytes)
        self.assertIn("ovr-01", row.residents)
        self.assertEqual(row.patch_bytes, 20_224)
        # The eviction tick removes the record and the patch together.
        eviction = next(row for row in self.result.ticks if row.evicted)
        self.assertNotIn(eviction.evicted[0], eviction.residents)
        self.assertEqual(eviction.patch_bytes, 0)
        self.assertEqual(summary["residents_end"], ())
        self.assertEqual(summary["patches_end"], ())
        self.assertEqual(self.runner.board.patches(), ())

    def test_no_orphan_patch_or_record_at_any_tick(self) -> None:
        for row in self.result.ticks:
            if row.residents:
                self.assertEqual(row.patch_bytes, 20_224, msg=f"tick {row.tick}")
            else:
                self.assertEqual(row.patch_bytes, 0, msg=f"tick {row.tick}")

    def test_byte_ceilings(self) -> None:
        self.assertLessEqual(MAX_PATCH_BYTES, 262_144)
        self.assertEqual(MAX_PATCH_TOTAL_BYTES, 4 * MAX_PATCH_BYTES)
        self.assertLessEqual(self.runner.board.patch_bytes(), MAX_PATCH_TOTAL_BYTES)

    def test_resident_without_patch_asserts(self) -> None:
        from amateur.payloads import MatBoard

        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-x", "active"))
        with self.assertRaises(AssertionError):
            board.assert_patch_bounds(require_complete=True)

    @requires_artifacts
    def test_missing_patch_on_commit_asserts(self) -> None:
        run = make_run(seed=7)
        run.tick()
        run.board.birth(MatOverlay(9, "ghost", "active"))
        envelope = run.dry.derived[23]
        from amateur.provenance import Envelope, Provenance

        ghost = Envelope(Provenance(23, run.dry.tick_no),
                         MatOverlay(1, "ghost", "active"))
        with self.assertRaises(AssertionError):
            run.source.on_commit(23, ghost, board=run.board, debris=run.dry.debris)


# ---------------------------------------------------------------------------
# Q1 atomicity: capacity refuses dense patches, ceilings refuse oversized ones
# ---------------------------------------------------------------------------
@requires_artifacts
class TestFullPatchModeFailsClosed(unittest.TestCase):
    def test_full_patch_is_refused_by_capacity_before_any_fit(self) -> None:
        """The dense full-predictor request never reaches a fit or an overlay.

        Round 9a: ``patch="full"`` is 210,920 stored parameters, far over
        the evidence budget's admissible rank-ceilinged patch (5,056 params
        from the same 16 transitions).  The refusal is recorded with the
        budget and the scenario simply continues with no overlay -- there is
        nothing to attach and nothing is half-committed.
        """
        organ = PhysicsOrgan(store_frames=True, fit_streams={0}, patch="full")
        run = PhysicsRun(organ=organ, seed=7)
        result = run.acceptance()
        summary = result.summary
        self.assertIsNone(summary["decision"])
        self.assertIsNone(summary["adoption_tick"])
        self.assertEqual(summary["active_ticks"], 0)
        self.assertEqual(summary["fit_reason"], "capacity")
        self.assertEqual(summary["fit_param_count"], 0)
        self.assertEqual(organ.fit_history[-1].requested_params, 210_920)
        self.assertEqual(organ.refusals[-1]["reason"], "capacity")
        self.assertEqual(organ.refusals[-1]["max_params"], 5_056)
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())
        self.assertEqual(run.board.residents(), ())
        self.assertEqual(run.board.patches(), ())
        self.assertEqual(run.board.patch_bytes(), 0)
        self.assertEqual(run.board.d4, [])
        self.assertEqual(organ.fit_history[-1].reason, "capacity")

    def test_oversized_synthetic_patch_fails_in_the_derivation_pass(self) -> None:
        """The outer byte ceiling is a storage bound and still fails closed.

        A synthetic patch larger than ``MAX_PATCH_BYTES`` is refused by the
        #23 derivation (before any arrival, commit, or charge); the failing
        tick leaves the store, D2 and the organ clean.
        """
        run = make_run(seed=7)
        run.reset()
        organ = run.organ
        observation = organ.step(0)
        digest = organ.digest(observation)
        ratchet = organ.ratchet(digest)
        oversized = _OversizedPatch("ovr-big")
        organ.overlay_table.birth(oversized, motif_id="motif-big", tick=1)
        record = organ.overlay_table.record("ovr-big")
        assert record is not None
        run.source.begin_tick(
            observation.frame, observation, digest, ratchet, adoption=record
        )
        self.assertEqual(run.board.residents(), ())
        counters_before = (
            dict(run.dry.attempted),
            dict(run.dry.accepted),
            dict(run.dry.refused),
            dict(run.dry.consumed),
            dict(run.dry.missed),
        )
        with self.assertRaises(AssertionError) as caught:
            run.source._birth_token(2, run.board)
        message = str(caught.exception)
        self.assertIn(f"{oversized.nbytes}", message)
        self.assertIn("per-patch ceiling", message)
        counters_after = (
            dict(run.dry.attempted),
            dict(run.dry.accepted),
            dict(run.dry.refused),
            dict(run.dry.consumed),
            dict(run.dry.missed),
        )
        self.assertEqual(counters_after, counters_before)
        self.assertGreater(oversized.nbytes, MAX_PATCH_BYTES)
        self.assertEqual(run.board.patches(), ())
        self.assertEqual(run.board.patch_bytes(), 0)
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())
        self.assertEqual(organ.overlay_table.adapter("ovr-big"), None)

    def test_low_rank_patch_path_still_attaches(self) -> None:
        """The admissible low-rank patch (20,224 B) still commits and attaches."""
        run = make_run(seed=7)
        _drive_to_birth(run)
        self.assertEqual(run.board.residents(), (("ovr-01", 1),))
        patches = run.board.patches()
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0][0], "ovr-01")
        self.assertEqual(run.board.patch_bytes(), 20_224)
        run.board.assert_patch_bounds(require_complete=True)


# ---------------------------------------------------------------------------
# P6 reset: the registered acceptance is repeatable on one runner (C)
# ---------------------------------------------------------------------------
@requires_artifacts
class TestResetRepeatability(unittest.TestCase):
    def test_two_acceptance_calls_are_identical(self) -> None:
        run = make_run(seed=7)
        first = run.acceptance()
        second = run.acceptance()
        self.assertEqual(first.ticks, second.ticks)
        self.assertEqual(first.summary, second.summary)
        summary = second.summary
        self.assertEqual(summary["proposal_tick"], 16)
        self.assertEqual(summary["proposal_rank"], 26)
        self.assertEqual(summary["proposal_param_count"], 5_056)
        self.assertEqual(summary["decision"], "birth")
        self.assertEqual(summary["adoption_tick"], 18)
        self.assertEqual(summary["stale_tick"], 31)
        self.assertEqual(summary["eviction_tick"], 35)
        self.assertEqual(summary["active_ticks"], 14)
        self.assertAlmostEqual(summary["forward_ratio"], 1.0461437156029878, places=12)
        self.assertAlmostEqual(
            summary["forward_ratio_flip_only"], 0.9665724706513396, places=12
        )
        self.assertEqual(summary["d4_sediments"], 11)
        self.assertEqual(summary["residents_end"], ())
        self.assertEqual(summary["patches_end"], ())
        self.assertEqual(summary["patch_bytes_end"], 0)
        self.assertEqual(run.organ.overlay_table.records(), ())
        self.assertEqual(run.organ.overlay_table.active(), ())
        self.assertEqual([record.stale_tick for record in run.organ.overlay_table.stales], [31])
        self.assertEqual(
            [record.overlay_id for record in run.organ.overlay_table.evictions],
            ["ovr-01"],
        )

    def test_two_run_calls_are_identical(self) -> None:
        run = make_run(seed=7)
        first = run.run(36, flip_at=8, flip_back_at=28)
        second = run.run(36, flip_at=8, flip_back_at=28)
        self.assertEqual(first, second)
        self.assertEqual(run.board.residents(), ())
        self.assertEqual(run.board.patches(), ())
        self.assertEqual(run.board.patch_bytes(), 0)
        self.assertEqual(run.organ.overlay_table.records(), ())


# ---------------------------------------------------------------------------
# Single-stream acceptance: state lifecycle and the honest fit result
# ---------------------------------------------------------------------------
@requires_artifacts
class TestSingleStreamAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = make_run(seed=7)
        cls.before = pickle.dumps(cls.runner.board.d4, protocol=4)
        cls.result = cls.runner.acceptance()
        cls.after = pickle.dumps(cls.runner.board.d4, protocol=4)

    def test_spike_is_measured(self) -> None:
        summary = self.result.summary
        self.assertGreater(summary["spike_peak"], summary["normal_mean"])
        self.assertGreater(summary["spike_over_normal"], 3.0)

    def test_state_lifecycle_birth_stale_evict(self) -> None:
        """A validation-passed patch that later harms is staleness-evicted."""
        summary = self.result.summary
        birth = summary["adoption_tick"]
        stale = summary["stale_tick"]
        eviction = summary["eviction_tick"]
        self.assertIsNotNone(birth)
        self.assertIsNotNone(stale)
        self.assertIsNotNone(eviction)
        assert birth is not None and stale is not None and eviction is not None
        self.assertEqual(birth, 18)  # the decision tick, after 17-18 validation
        self.assertLess(birth, stale)
        self.assertLess(stale, eviction)
        self.assertGreater(summary["active_ticks"], 0)
        # The harm that triggers staleness is measured on the flip-back ticks:
        # the patch is beneficial over the flipped window (0.9666) and harmful
        # on the recovered phase (the all-active ratio 1.0461 is above 1).
        self.assertLess(summary["forward_ratio_flip_only"], 1.0)
        self.assertGreater(summary["forward_ratio"], 1.0)
        self.assertEqual(self.runner.organ.overlay_table.records(), ())
        self.assertEqual(self.runner.organ.overlay_table.active(), ())
        self.assertEqual(self.runner.board.residents(), ())
        self.assertEqual(self.runner.board.patches(), ())
        self.assertEqual(self.runner.board.patch_bytes(), 0)

    def test_d4_moves_only_by_real_gate_sediments(self) -> None:
        """Round 9a: the literal empty-D4 snapshot no longer applies.

        The passed patch makes the #22 gate fire 11 times, so D4 grows by 11
        real, gate-derived bodies.  What must still hold is that the retained
        contents are byte-identical (see
        ``TestZeroConfidenceGate.test_retained_invariant_survives_the_scenario``)
        and that nothing but the gate's own output is retained.
        """
        self.assertNotEqual(self.before, self.after)
        self.assertFalse(self.result.summary["d4_byte_identical"])
        self.assertEqual(self.result.summary["d4_sediments"], 11)
        self.assertEqual(len(self.runner.board.d4), 11)
        self.assertEqual(self.after, pickle.dumps(self.runner.board.d4, protocol=4))

    def test_encoder_bit_identical_after_fit(self) -> None:
        self.assertTrue(self.result.summary["encoder_bit_identical"])
        self.assertTrue(self.runner.organ.encoder_bit_identical())
        self.assertIn("fit_nbytes", self.result.summary)

    def test_d3_pool_bounded(self) -> None:
        from amateur.payloads import DEBRIS_CAPACITY, DEBRIS_MAX_BYTES

        self.assertGreaterEqual(self.result.summary["debris_bodies"], 1)
        self.assertLessEqual(self.runner.dry.debris.__len__(), DEBRIS_CAPACITY)
        self.assertLessEqual(self.runner.dry.debris.byte_size(), DEBRIS_MAX_BYTES)

    @unittest.expectedFailure
    def test_single_stream_surprise_drops(self) -> None:
        """RECORDED FAILURE: the registered trace still misses the criterion.

        Round 9a re-measured this on the registered seed-7 trace: the
        validation-gated rank-26 patch is born at tick 18 (validation
        0.133868 vs base 0.147455, ratio 0.9079) and the adapted/base ratio
        over every active tick is 1.0461 (improved from 1.2614 with the
        dense patch, but still above 1.0).  Over the active *flipped* ticks
        only it is 0.9666, i.e. the patch helps during the failure window
        and hurts after the flip-back, where the ratchet stales and evicts
        it.  The assertion is deliberately kept strict on the registered
        metric: breaking 1.0 over every active tick has not been measured
        on the registered trace, and selecting a different seed after seeing
        the results would be seed-shopping.  Do not weaken this to "no
        material harm"; if the policy changes, re-measure.
        """
        self.assertLess(self.result.summary["forward_ratio"], 1.0)


# ---------------------------------------------------------------------------
# Round 9a: pre-adoption validation through the executor
# ---------------------------------------------------------------------------
@requires_artifacts
class TestPreAdoptionValidation(unittest.TestCase):
    """Proposal at 16, held-out validation on 17-18, decision at 18.

    Both outcomes are measured on real traces: seed 7 passes validation and
    is later staleness-evicted; seed 0's candidate loses the paired
    comparison and is rejected before ever being born.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.birth_runner = make_run(seed=7)
        cls.birth = cls.birth_runner.acceptance()
        cls.reject_runner = make_run(seed=0)
        cls.reject = cls.reject_runner.acceptance()

    def test_proposal_is_capacity_bounded_and_measured(self) -> None:
        summary = self.birth.summary
        self.assertEqual(summary["proposal_tick"], 16)
        self.assertEqual(summary["proposal_rank"], 26)
        self.assertEqual(summary["proposal_param_count"], 5_056)
        self.assertEqual(summary["proposal_capacity"], 5_056)
        self.assertEqual(summary["proposal_budget_n"], 16)
        self.assertEqual(summary["validation_ticks"], (17, 18))
        self.assertEqual(summary["validation_n"], 2)
        self.assertAlmostEqual(
            summary["validation_candidate_mean"], 0.13386843726038933, places=12
        )
        self.assertAlmostEqual(
            summary["validation_base_mean"], 0.1474553719162941, places=12
        )
        self.assertAlmostEqual(
            summary["validation_ratio"], 0.9078573097790045, places=12
        )
        self.assertEqual(summary["decision"], "birth")
        self.assertEqual(summary["adoption_tick"], 18)
        self.assertIsNone(summary["rejection_reason"])

    def test_validation_trace_is_visible_per_tick(self) -> None:
        rows = {row.frame_tick: row for row in self.birth.ticks}
        self.assertEqual(rows[16].proposal_tick, 16)
        self.assertEqual(rows[16].validation_n, 0)
        self.assertEqual(rows[16].validation_ticks, ())
        self.assertIsNone(rows[16].candidate_mean)
        self.assertEqual(rows[17].validation_n, 1)
        self.assertEqual(rows[17].validation_ticks, (17,))
        self.assertGreater(rows[17].candidate_mean, 0.0)
        self.assertEqual(rows[18].validation_n, 2)
        self.assertEqual(rows[18].validation_ticks, (17, 18))
        self.assertEqual(rows[18].decision, "birth")
        self.assertIsNone(rows[18].rejection)
        self.assertEqual(rows[19].decision, None)
        self.assertEqual(rows[19].candidate_mean, rows[18].candidate_mean)

    def test_no_peek_in_the_per_tick_trace(self) -> None:
        for row in self.birth.ticks:
            self.assertEqual(len(row.validation_ticks), row.validation_n)
            for tick in row.validation_ticks:
                self.assertIsNotNone(row.proposal_tick)
                assert row.proposal_tick is not None
                self.assertGreater(tick, row.proposal_tick, msg=f"row {row.frame_tick}")
                self.assertLessEqual(tick, row.frame_tick)
        # No validation evidence exists before the proposal tick.
        for row in self.birth.ticks:
            if row.frame_tick < PROPOSAL_TICK:
                self.assertEqual(row.validation_n, 0)
                self.assertIsNone(row.proposal_tick)

    def test_rejection_trace_has_no_birth_no_activation(self) -> None:
        summary = self.reject.summary
        self.assertEqual(summary["proposal_tick"], 16)
        self.assertEqual(summary["proposal_rank"], 26)
        self.assertEqual(summary["validation_ticks"], (17, 18))
        self.assertEqual(summary["decision"], "interference")
        self.assertEqual(summary["rejection_tick"], 18)
        self.assertEqual(summary["rejection_reason"], "interference")
        self.assertIsNone(summary["adoption_tick"])
        self.assertEqual(summary["active_ticks"], 0)
        self.assertGreater(
            summary["validation_candidate_mean"], summary["validation_base_mean"]
        )
        self.assertAlmostEqual(
            summary["validation_candidate_mean"], 0.1909084, places=6
        )
        self.assertAlmostEqual(
            summary["validation_base_mean"], 0.1338746, places=6
        )
        # Nothing was born, activated or attached, and the end state is clean.
        self.assertEqual(self.reject_runner.organ.overlay_table.all_ids(), frozenset())
        self.assertEqual(self.reject_runner.organ.overlay_table.active(), ())
        self.assertEqual(self.reject_runner.board.residents(), ())
        self.assertEqual(self.reject_runner.board.patches(), ())
        self.assertEqual(self.reject_runner.board.patch_bytes(), 0)
        self.assertEqual(self.reject_runner.board.d4, [])
        self.assertTrue(summary["d4_byte_identical"])
        self.assertEqual(summary["d4_sediments"], 0)
        self.assertTrue(summary["encoder_bit_identical"])
        self.assertTrue(self.reject_runner.organ.encoder_bit_identical())
        # The fit itself was admissible: the refusal is the validation's.
        self.assertEqual(summary["fit_reason"], "")
        self.assertEqual(summary["fit_nbytes"], 20_224)

    def test_rejection_is_publicly_readable(self) -> None:
        rejections = self.reject_runner.organ.rejections
        self.assertEqual(len(rejections), 1)
        record = rejections[0]
        self.assertEqual(record.tick, 18)
        self.assertEqual(record.reason, "interference")
        self.assertEqual(record.n, 2)
        self.assertGreater(record.candidate_mean, record.base_mean)
        self.assertEqual(record.budget.n_transitions, 16)
        self.assertEqual(record.budget.max_rank, 26)
        self.assertEqual(record.budget.max_params, 5_056)
        # The per-tick trace shows the rejection at the decision tick.
        rows = {row.frame_tick: row for row in self.reject.ticks}
        self.assertEqual(rows[18].decision, "interference")
        self.assertEqual(rows[18].rejection, "interference")
        self.assertIsNone(rows[18].adopted)
        self.assertGreater(rows[18].candidate_mean, rows[18].base_mean)


if __name__ == "__main__":
    unittest.main()
