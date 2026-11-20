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
* The D2 patch table is substance-accounted: the head patch (136,960 B)
  attaches, the full patch (843,680 B) is rejected by the 256 KiB
  per-patch ceiling, record+patch eviction is atomic, and a resident
  record with no patch asserts.
* The #22 zero-confidence gate body is the source's derived output (not a
  literal in ``_derive``).  The gate is real: it consumes cumulative
  adapted-vs-base evidence that the organ records on applied ticks and
  publishes after ``stale_patience`` observations.  The registered
  acceptance's cumulative evidence is harmful, so the gate stays blocked
  with no test double (36 confidence-0 no-ops); the passing branch is
  exercised only by injected evidence and is labelled as such.  The gate
  is absorbed as a consumed no-op, and the identity ``B: 1 = 1 + 0 + 0``
  holds on that path.  D4 retention is proven non-vacuously: one real
  sediment passes the gate, then survives the registered 36-tick flip
  scenario byte-identically, as the same object, with no second sediment.
  The acceptance's empty-vs-empty snapshot is kept but renamed to say only
  that.
* Q1 atomicity: a patch that cannot be attached (the supported
  ``patch="full"`` mode is 843,680 B, over the 262,144 B per-patch
  ceiling) fails in the derivation pass, before any arrival/commit; the
  failing tick leaves the store, D4 and the organ clean.  The head patch
  (136,960 B) still attaches.
* P6 reset is explicit: two consecutive ``acceptance()`` calls (and two
  consecutive ``run()`` calls) on one runner produce identical traces.
* The single-stream acceptance reports the measured pre/post surprise
  honestly.  The pre-registered bounded same-stream replay (last 16
  transitions) does **not** produce a forward drop on this checkpoint;
  ``test_single_stream_surprise_drops`` is an explicit recorded failure
  (not weakened into "no material harm").  The bank-fit positive result is
  in ``tests/test_physics_organ.py``.
* Encoder bit-identity is asserted with ``torch.equal`` after the run.

Runtime: ~9–11 s on 8 CPU threads (checkpoint load plus the repeated
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


ADOPTION_TICK = 16


def _drive_to_adoption(run: PhysicsRun) -> None:
    """Drive normal -> flip up to the registered adoption tick (16)."""
    row = None
    for index in range(1, ADOPTION_TICK + 1):
        if index == 8:
            run.flip_gravity()
        row = run.tick()
    assert row is not None and row.adopted is not None, (
        "the registered adoption did not fire; the scenario changed"
    )


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
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = make_run(seed=7)
        cls.result = cls.runner.acceptance()

    def test_empty_d4_stays_empty_byte_identical(self) -> None:
        """Pins only empty-vs-empty across the registered acceptance.

        ``acceptance()`` calls ``reset()``, which rebuilds the board, so the
        empty D4 snapshotted before the run is a *different list object*
        from the empty D4 after it.  This test therefore proves only that
        the acceptance leaves no stray D4 writes; it cannot prove retention
        (``pickle.dumps([]) == pickle.dumps([])`` is vacuous).  The
        non-vacuous retention check is
        ``test_retained_invariant_survives_the_scenario``.
        """
        before = pickle.dumps(self.runner.board.d4, protocol=4)
        self.assertEqual(before, pickle.dumps([], protocol=4))
        self.assertTrue(self.result.summary["d4_byte_identical"])
        self.assertEqual(self.runner.board.d4, [])
        self.assertEqual(self.result.summary["d4_bytes_before"],
                         self.result.summary["d4_bytes_after"])

    def test_retained_invariant_survives_the_scenario(self) -> None:
        """Non-vacuous Q4 retention: one real sediment survives the flip.

        Tick 1 is seeded with a **test-double gate override** so a real
        ``BiasPrimitive`` is committed by ``_absorb_writes`` and sedimented
        into D4 by the store.  The override is a state-seeding device, not
        a claim about gate derivation -- the real gate's evidence path is
        tested by ``test_gate_passes_on_published_beneficial_evidence`` and
        its blocked paths above; at tick 1 there is no active overlay, so
        the real gate cannot pass yet.  The gate is then restored to
        blocked and the remaining 35 ticks of the registered 36-tick
        scenario (normal, flip at 8, flip back at 28, recover) are driven
        with ``tick()`` / ``flip_gravity()`` directly.  ``run()`` and
        ``acceptance()`` are deliberately not used: both call ``reset()``,
        which rebuilds the board and would erase the seed, so neither can
        observe retention at all.

        The check is non-vacuous because D4 is non-empty before and after:
        byte equality of the serialized retained list, plus object identity
        of the retained body, means the retained contents really were
        retained.  The 35 blocked ticks are all absorbed as claimed no-ops
        (identity B each tick, no second sediment).
        """
        run = make_run(seed=7)
        run.reset()
        run.organ.ratchet_gate_passed = lambda: (True, 0.5)  # type: ignore[assignment]
        rows = [run.tick()]
        self.assertEqual(len(run.board.d4), 1)
        retained_body = run.board.d4[0]
        self.assertEqual(
            (retained_body.bias_primitive, retained_body.confidence),
            ("ratchet-gate-passed", 0.5),
        )
        # The retained object is the one that passed the gate, not a copy.
        self.assertIs(run.dry.derived[22].body, retained_body)
        retained = pickle.dumps(run.board.d4, protocol=4)
        self.assertNotEqual(retained, pickle.dumps([], protocol=4))
        self.assertEqual(rows[0].d4_size, 1)

        # Restore the real gate (blocked) for the rest of the scenario.
        run.organ.ratchet_gate_passed = lambda: (False, 0.0)  # type: ignore[assignment]
        # Mirror run()'s phase bookkeeping without its reset().
        run._flip_at = 8
        run._flip_back_at = 28
        for index in range(2, 37):
            if index in (8, 28):
                run.flip_gravity()
            accepted = run.dry.accepted[22]
            consumed = run.dry.consumed[22]
            refused = run.dry.refused[22]
            row = run.tick()
            rows.append(row)
            # Identity B on every tick: 1 = 1 + 0 + 0, no refusal, no residue.
            self.assertEqual(run.dry.accepted[22] - accepted, 1, msg=f"tick {index}")
            self.assertEqual(run.dry.consumed[22] - consumed, 1, msg=f"tick {index}")
            self.assertEqual(run.dry.refused[22] - refused, 0, msg=f"tick {index}")
            self.assertEqual(len(run.dry.mailboxes[22]), 0, msg=f"tick {index}")
            self.assertEqual(row.d4_size, 1, msg=f"tick {index}")

        self.assertEqual(len(rows), 36)
        self.assertEqual(
            {row.phase for row in rows}, {"normal", "flipped", "recovered"}
        )
        # Retention measured on non-empty contents.
        self.assertEqual(pickle.dumps(run.board.d4, protocol=4), retained)
        self.assertEqual(len(run.board.d4), 1)
        self.assertIs(run.board.d4[0], retained_body)
        # The 35 blocked ticks were absorbed, not skipped: one positive
        # sediment + 35 claimed no-ops = 36 settled writes.
        self.assertEqual(run.dry.zero_confidence_absorbed[22], 35)
        self.assertEqual(run.dry.accepted[22], 36)
        self.assertEqual(run.dry.consumed[22], 36)
        self.assertEqual(run.dry.refused[22], 0)
        self.assertEqual(
            run.dry.accepted[22],
            run.dry.consumed[22] + len(run.dry.mailboxes[22]) + 0,
        )

    def test_zero_confidence_write_is_a_consumed_noop_identity_b(self) -> None:
        summary = self.result.summary
        ticks = len(self.result.ticks)
        self.assertEqual(summary["accepted_22"], ticks)
        self.assertEqual(summary["consumed_22"], ticks)
        self.assertEqual(summary["refused_22"], 0)
        self.assertEqual(summary["sink_depth_22"], 0)
        self.assertEqual(summary["zero_confidence_writes"], ticks)
        # Identity B on the absorbed no-op path: 1 = 1 + 0 + 0 per tick.
        self.assertEqual(
            summary["accepted_22"],
            summary["consumed_22"] + summary["sink_depth_22"] + 0,
        )

    def test_gate_body_is_the_derived_output(self) -> None:
        body = self.runner.dry.derived[22].body
        self.assertIs(type(body), BiasPrimitive)
        self.assertEqual(
            (body.bias_primitive, body.confidence),
            ("ratchet-gate-blocked", 0.0),
        )
        self.assertEqual(self.runner.organ.ratchet_gate_passed(), (False, 0.0))

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

        The adoption fires at the registered tick; ``flush`` makes the next
        tick's derivation the only decision point, so the injected verdict
        survives to the gate.  This is the only way to reach the passing
        branch in this slice (no adaptation is beneficial over
        ``stale_patience`` applied ticks), and it is labelled as injected.
        """
        run = make_run(seed=7)
        _drive_to_adoption(run)
        run.organ.flush()
        run.organ._record_gate_evidence(0.2, 0.4)
        row = run.tick()
        self.assertEqual(row.tick, 17)
        body = run.dry.derived[22].body
        self.assertIs(type(body), BiasPrimitive)
        self.assertEqual(body.bias_primitive, "ratchet-gate-passed")
        self.assertGreater(body.confidence, 0.0)
        self.assertEqual(body.confidence, 0.5)
        self.assertEqual(len(run.board.d4), 1)
        self.assertIs(run.board.d4[0], body)

    def test_gate_blocked_on_harmful_evidence(self) -> None:
        run = make_run(seed=7)
        _drive_to_adoption(run)
        run.organ.flush()
        run.organ._record_gate_evidence(0.4, 0.2)  # adapted worse than base
        run.tick()
        self.assertEqual(run.organ.ratchet_gate_passed(), (False, 0.0))
        self.assertEqual(run.dry.derived[22].body.bias_primitive,
                         "ratchet-gate-blocked")
        self.assertEqual(run.board.d4, [])

    def test_acceptance_gate_blocked_by_measured_harm(self) -> None:
        """The registered trace must stay blocked with no test double."""
        self.assertEqual(self.runner.organ.ratchet_gate_passed(), (False, 0.0))
        self.assertEqual(self.runner.organ._gate_evidence, None)
        self.assertEqual(self.result.summary["zero_confidence_writes"],
                         len(self.result.ticks))
        self.assertEqual(self.runner.board.d4, [])


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
        summary = self.result.summary
        self.assertEqual(summary["fit_patch"], "head")
        self.assertEqual(summary["fit_param_count"], 34_240)
        self.assertEqual(summary["fit_nbytes"], 136_960)
        self.assertLessEqual(summary["max_patch_bytes_seen"], MAX_PATCH_BYTES)
        row = next(row for row in self.result.ticks if row.patch_bytes)
        self.assertIn("ovr-01", row.residents)
        self.assertEqual(row.patch_bytes, 136_960)
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
                self.assertEqual(row.patch_bytes, 136_960, msg=f"tick {row.tick}")
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
# Q1 atomicity: an unattachable patch fails in the derivation pass (A)
# ---------------------------------------------------------------------------
@requires_artifacts
class TestFullPatchModeFailsClosed(unittest.TestCase):
    def test_full_patch_birth_asserts_before_any_commit(self) -> None:
        """A full-predictor patch (843,680 B) is refused before the commit.

        ``_absorb_writes`` commits the #23 birth before ``on_commit``
        attaches the patch, so the ceiling check must live in the derivation
        pass: the failing tick must add no arrivals, no charges, no D2
        record, no D2 patch, and no organ-side overlay retention.
        """
        organ = PhysicsOrgan(store_frames=True, fit_streams={0}, patch="full")
        run = PhysicsRun(organ=organ, seed=7)
        run.reset()
        for index in range(1, ADOPTION_TICK):
            if index == 8:
                run.flip_gravity()
            run.tick()
        counters_before = (
            dict(run.dry.attempted),
            dict(run.dry.accepted),
            dict(run.dry.refused),
            dict(run.dry.consumed),
            dict(run.dry.missed),
        )
        with self.assertRaises(AssertionError) as caught:
            run.tick()
        message = str(caught.exception)
        self.assertIn("843680", message)
        self.assertIn("per-patch ceiling", message)
        counters_after = (
            dict(run.dry.attempted),
            dict(run.dry.accepted),
            dict(run.dry.refused),
            dict(run.dry.consumed),
            dict(run.dry.missed),
        )
        self.assertEqual(counters_after, counters_before)
        self.assertEqual(run.board.residents(), ())
        self.assertEqual(run.board.patches(), ())
        self.assertEqual(run.board.patch_bytes(), 0)
        self.assertEqual(run.board.d4, [])
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())
        self.assertEqual(organ.overlay_table.active(), ())
        self.assertEqual(organ.overlay_table.adapter("ovr-01"), None)
        self.assertEqual(len(organ.fit_history), 1)
        self.assertEqual(organ.fit_history[0].patch, "full")
        self.assertEqual(organ.fit_history[0].nbytes, 843_680)

    def test_head_patch_path_still_attaches(self) -> None:
        """The accepted head patch (136,960 B) still commits and attaches."""
        run = make_run(seed=7)
        _drive_to_adoption(run)
        self.assertEqual(run.board.residents(), (("ovr-01", 1),))
        patches = run.board.patches()
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0][0], "ovr-01")
        self.assertEqual(run.board.patch_bytes(), 136_960)
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
        self.assertEqual(summary["adoption_tick"], 16)
        self.assertEqual(summary["stale_tick"], 19)
        self.assertEqual(summary["eviction_tick"], 23)
        self.assertEqual(summary["active_ticks"], 4)
        self.assertAlmostEqual(summary["forward_ratio"], 1.2613833722904746, places=12)
        self.assertEqual(summary["residents_end"], ())
        self.assertEqual(summary["patches_end"], ())
        self.assertEqual(summary["patch_bytes_end"], 0)
        self.assertEqual(run.organ.overlay_table.records(), ())
        self.assertEqual(run.organ.overlay_table.active(), ())
        self.assertEqual([record.stale_tick for record in run.organ.overlay_table.stales], [19])
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

    def test_state_lifecycle_adopt_stale_evict(self) -> None:
        summary = self.result.summary
        adoption = summary["adoption_tick"]
        stale = summary["stale_tick"]
        eviction = summary["eviction_tick"]
        self.assertIsNotNone(adoption)
        self.assertIsNotNone(stale)
        self.assertIsNotNone(eviction)
        assert adoption is not None and stale is not None and eviction is not None
        self.assertLess(adoption, stale)
        self.assertLessEqual(stale, eviction)
        self.assertGreater(summary["active_ticks"], 0)
        self.assertEqual(self.runner.organ.overlay_table.records(), ())
        self.assertEqual(self.runner.organ.overlay_table.active(), ())
        self.assertEqual(self.runner.board.residents(), ())
        self.assertEqual(self.runner.board.patches(), ())

    def test_d4_serialized_bytes_unchanged(self) -> None:
        self.assertEqual(self.before, self.after)
        self.assertEqual(self.before, pickle.dumps([], protocol=4))
        self.assertTrue(self.result.summary["d4_byte_identical"])

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
        """RECORDED FAILURE: the pre-registered same-stream replay does not drop.

        Measured: adapted mean 0.3264 vs base 0.2588 over the 4 active ticks
        (forward ratio 1.2614, no-drop).  The assertion is deliberately kept
        strict: a single-stream predictor-only fit memorises its replay
        window and does not transfer forward on this checkpoint.  The
        positive measured result is the bank/held-out fit in
        tests/test_physics_organ.py (0.690x on unseen episodes).  Do not
        weaken this to "no material harm"; if the policy changes, re-measure.
        """
        self.assertLess(self.result.summary["forward_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
