"""Tests for the physics organ (``amateur/physics/organ.py``), Round 11.

Pins the *new* architecture (contract §3.1-§3.4) rather than the round-9a
pre-adoption scaffolding it replaced:

* thresholds are *derived* from the measured phase-3 statistics and separate
  real normal residuals from real flipped residuals;
* each observation carries the real residual, the latent state delta, the
  3-frame causal window and the resolved latent rollout;
* the P5 digest is a bounded pool whose every item is **tagged into one of
  the four cohorts** (``motif`` / ``residual`` / ``scaffold`` / ``trail``),
  with deterministic unique ids; a non-ingest tick emits no cohort;
* the **candidate** (``cand-NN``) is fitted under the unchanged capacity bound
  (``K <= gamma*N*D``, ``gamma = 5.0``, last head layer only) and is
  **substance, never steering**: it is stored in the nursery, applied only to
  a shadow copy of the predictor, and the encoder, the base predictor and the
  live working model are asserted bit-identical after every shadow score;
* residence evidence is exactly one paired ``(tick, adapted, base)`` sample
  per resident tick, and ``qualified`` is ``ratio < 1.0`` over at least
  ``RESIDENCE_N`` samples;
* the live patch arrives only through ``adopt_motif`` (the ``#21`` motif
  cohort -> ``#24``), with the round-9a birth / revalidate / stale / cooldown
  / evict semantics unchanged;
* the round-9a **live-overlay D4 gate is gone** (ruling C): the decision path
  is candidate residence evidence, and the assertions that pinned the old
  path are replaced by stronger nursery-evidence assertions (see
  :class:`TestGateRemoval` and :meth:`TestResidenceEvidence` -- never
  deleted);
* the organ-level nursery probe exercises the whole lifecycle on a real flip
  scenario and makes **zero D4 writes**.

The artifact-dependent tests skip (with an explicit reason) if
``.work/forge/checkpoint.pt`` / ``stats.json`` are absent; they are
regenerable with ``python3 -m amateur.physics.train``.

Runtime: ~40 s on 8 CPU threads (the nursery probe and the held-out fit
probe dominate).
"""
from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path

import numpy as np
import torch

from amateur.physics.data import generate_rollouts
from amateur.physics.env import PhysicsEnv, WorldConfig
from amateur.physics.micro_lewm import MicroLeWM
from amateur.physics.organ import (
    COHORT_KINDS,
    DEFAULT_FIT_WINDOW,
    DEFAULT_LATENT_DIM,
    NURSERY_CAPACITY,
    RESIDENCE_N,
    CandidateRecord,
    CandidateTable,
    CohortRecord,
    CollapseMandate,
    Observation,
    OverlayTable,
    Phase3Stats,
    PhysicsOrgan,
    PredictorOverlay,
    ResidualDigester,
    StructuralMotif,
    derive_thresholds,
    frames_tensor_from,
)

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


def make_organ(**kwargs: object) -> PhysicsOrgan:
    return PhysicsOrgan(
        checkpoint_path=CHECKPOINT,
        stats_path=STATS,
        **kwargs,  # type: ignore[arg-type]
    )


def observation(
    *,
    tick: int,
    residual: np.ndarray,
    surprise: float | None = None,
    stream: int = 0,
    ingest: bool = True,
) -> Observation:
    residual = np.asarray(residual, dtype=np.float64)
    latent = np.zeros(DEFAULT_LATENT_DIM, dtype=np.float64)
    if surprise is None:
        surprise = float(np.mean(residual * residual))
    return Observation(
        tick=tick,
        stream=stream,
        action=0,
        latent=latent,
        latent_delta=latent.copy(),
        surprise=surprise,
        residual=residual,
        rollout=np.zeros((4, DEFAULT_LATENT_DIM), dtype=np.float64),
        window=latent[None].copy(),
        overlay_id=None,
        base_surprise=surprise,
        ingest=ingest,
    )


def dummy_overlay(overlay_id: str, *, rank: int = 1, fill: float = 0.0) -> PredictorOverlay:
    """A tiny low-rank patch for lifecycle-only tests (no fit involved).

    Constructed in the *model's* shape (``u: (64, r)``, ``v: (128, r)``,
    ``bias: (64,)``) so it can be applied to a real predictor copy.
    """
    return PredictorOverlay(
        overlay_id=overlay_id,
        u=torch.full((DEFAULT_LATENT_DIM, rank), fill),
        v=torch.full((128, rank), fill),
        bias=torch.zeros(DEFAULT_LATENT_DIM),
        fit_transitions=1,
        fit_streams=1,
        loss_before=1.0,
        loss_after=0.5,
        patch="head",
    )


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
@requires_artifacts
class TestThresholdDerivation(unittest.TestCase):
    def test_thresholds_come_from_the_phase3_artifact(self) -> None:
        stats = Phase3Stats.from_stats(STATS)
        thresholds = derive_thresholds(stats)
        self.assertAlmostEqual(thresholds.debris_threshold, stats.normal_q99, places=12)
        self.assertAlmostEqual(thresholds.spike_threshold, stats.normal_q99, places=12)
        self.assertAlmostEqual(
            thresholds.merge_radius,
            float(np.sqrt(stats.latent_dim * stats.post_flip_q95)),
            places=12,
        )
        self.assertAlmostEqual(thresholds.normal_baseline, stats.normal_mean, places=12)
        # Recorded phase-3 values (the numbers the memo cites).
        self.assertAlmostEqual(stats.normal_mean, 0.04938631785656374, places=12)
        self.assertAlmostEqual(stats.normal_q99, 0.4130416682362567, places=12)
        self.assertAlmostEqual(stats.post_flip_q95, 0.7105834186077118, places=12)
        self.assertAlmostEqual(thresholds.merge_radius, 6.743688811836853, places=9)

    def test_organ_uses_the_same_thresholds(self) -> None:
        organ = make_organ()
        expected = derive_thresholds(Phase3Stats.from_stats(STATS))
        self.assertEqual(organ.thresholds, expected)

    def test_nursery_design_constants_are_the_contract_values(self) -> None:
        """§1.8: NURSERY_CAPACITY=4, RESIDENCE_N=4 and the four cohort kinds."""
        self.assertEqual(NURSERY_CAPACITY, 4)
        self.assertEqual(RESIDENCE_N, 4)
        thresholds = make_organ().thresholds
        self.assertEqual(thresholds.nursery_capacity, NURSERY_CAPACITY)
        self.assertEqual(thresholds.residence_n, RESIDENCE_N)
        self.assertEqual(tuple(COHORT_KINDS), ("motif", "scaffold", "trail", "residual"))
        try:  # the payload layer is the other owner of these constants
            from amateur.payloads import NURSERY_CAPACITY as P_CAP
            from amateur.payloads import RESIDENCE_N as P_RES
        except ImportError:  # wave-1 schema not landed yet
            return
        self.assertEqual(NURSERY_CAPACITY, P_CAP)
        self.assertEqual(RESIDENCE_N, P_RES)


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------
@requires_artifacts
class TestObservation(unittest.TestCase):
    def test_reset_frame_has_no_prediction(self) -> None:
        organ = make_organ()
        obs = organ.reset(seed=3)
        self.assertEqual(obs.tick, 0)
        self.assertTrue(np.isnan(obs.surprise))
        self.assertEqual(obs.latent.shape, (64,))
        self.assertEqual(obs.rollout.shape, (4, 64))
        self.assertEqual(obs.window.shape, (1, 64))
        self.assertTrue(np.allclose(obs.latent_delta, 0.0))
        self.assertIsNone(obs.overlay_id)

    def test_surprise_matches_batched_reference(self) -> None:
        organ = make_organ()
        rng = np.random.default_rng(11)
        actions = [int(rng.integers(0, 5)) for _ in range(10)]
        obs = organ.reset(seed=5)
        observations = [obs]
        env = PhysicsEnv(WorldConfig(frame_size=96))
        frames = [env.reset(seed=5)]
        for action in actions:
            obs = organ.step(action)
            observations.append(obs)
            frames.append(env.step(action))
            self.assertEqual(obs.rollout.shape, (4, 64))
            self.assertLessEqual(obs.window.shape[0], 3)
            if np.isfinite(obs.surprise):
                self.assertAlmostEqual(
                    obs.surprise,
                    float(np.mean(obs.residual * obs.residual)),
                    places=3,
                )
        with torch.no_grad():
            z = organ.base_model.encode(frames_tensor_from(np.stack(frames)[None]))
            prediction = organ.base_model.predict(z, torch.tensor(actions)[None])[0]
            reference = (prediction - z[0, 1:]).pow(2).mean(dim=-1).numpy()
        measured = np.array([item.surprise for item in observations[1:]])
        np.testing.assert_allclose(measured, reference, rtol=1e-5, atol=1e-7)

    def test_deterministic_for_fixed_seed_and_actions(self) -> None:
        organ_a = make_organ()
        organ_b = make_organ()
        rng = np.random.default_rng(19)
        actions = [int(rng.integers(0, 5)) for _ in range(6)]
        organ_a.reset(seed=9)
        organ_b.reset(seed=9)
        surprises_a = []
        surprises_b = []
        for action in actions:
            surprises_a.append(organ_a.step(action).surprise)
            surprises_b.append(organ_b.step(action).surprise)
        np.testing.assert_array_equal(np.asarray(surprises_a), np.asarray(surprises_b))
        self.assertEqual(
            [o.tick for o in organ_a.observations],
            [o.tick for o in organ_b.observations],
        )

    def test_gravity_flip_changes_dynamics_not_rendering(self) -> None:
        organ = make_organ()
        organ.reset(seed=2)
        sign = organ.flip_gravity()
        self.assertIn(sign, (-1, 1))
        obs = organ.step(0)
        self.assertEqual(obs.latent.shape, (64,))
        self.assertTrue(np.isfinite(obs.surprise))

    def test_transition_probe_is_recorded_for_shadow_scoring(self) -> None:
        """Round 11 (E): the exact GRU input a shadow score must replay."""
        organ = make_organ()
        organ.reset(seed=1)
        organ.step(2)
        organ.step(3)
        state = organ._state(0)
        self.assertEqual(len(state.transitions), 2)
        frame_tick, prev_latent, action, hidden_before, latent = state.transitions[0]
        self.assertEqual(frame_tick, 1)
        self.assertEqual(action, 2)
        # The first prediction starts from the GRU's zero hidden state.
        self.assertIsNone(hidden_before)
        self.assertEqual(tuple(prev_latent.shape), (64,))
        self.assertEqual(tuple(latent.shape), (64,))
        second = state.transitions[1]
        self.assertEqual(second[0], 2)
        self.assertIsNotNone(second[3])
        # An action-less reset contributes no transition.
        organ.reset(seed=1)
        self.assertEqual(len(organ._state(0).transitions), 0)


# ---------------------------------------------------------------------------
# Digest (P5): separation, boundedness, tagged cohorts
# ---------------------------------------------------------------------------
@requires_artifacts
class TestDigestSeparation(unittest.TestCase):
    """The derived q99 gate separates real normal from real flipped residuals."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        cls.model = MicroLeWM.load(CHECKPOINT)
        cls.env = PhysicsEnv(WorldConfig(frame_size=96))

    def _ingest_rates(self) -> tuple[int, int, int, int]:
        """Phase-matched post-flip window (transitions 8..23) for both sets."""
        normal = generate_rollouts(self.env, 16, 24, seed=31, gravity_sign=1)
        flipped = generate_rollouts(self.env, 16, 24, seed=31, gravity_sign=1, flip_at=8)
        counts = []
        for batch in (normal, flipped):
            digester = ResidualDigester(derive_thresholds(Phase3Stats.from_stats(STATS)))
            with torch.no_grad():
                frames = frames_tensor_from(batch.frames)
                z = self.model.encode(frames)
                prediction = self.model.predict(z, torch.from_numpy(batch.actions).long())
                error = (prediction - z[:, 1:]).cpu().numpy()
                surprise = (error**2).mean(axis=-1)
            ingested = 0
            for stream in range(len(batch)):
                for transition in range(8, batch.horizon):
                    result = digester.observe(
                        observation(
                            tick=transition,
                            residual=error[stream, transition],
                            surprise=float(surprise[stream, transition]),
                            stream=stream,
                        )
                    )
                    ingested += int(result.ingested)
            counts.append((ingested, len(batch) * (batch.horizon - 8)))
        return counts[0][0], counts[0][1], counts[1][0], counts[1][1]

    def test_q99_gate_separates_flipped_from_normal(self) -> None:
        normal_hits, normal_total, flipped_hits, flipped_total = self._ingest_rates()
        normal_rate = normal_hits / normal_total
        flipped_rate = flipped_hits / flipped_total
        # Measured on this checkpoint over the phase-matched window: normal
        # 0/256, flipped 57/256 (22.27 %).  The gate is the measured normal
        # q99, not a tuned constant.
        self.assertLessEqual(normal_rate, 0.03)
        self.assertGreaterEqual(flipped_rate, 0.15)
        self.assertGreaterEqual(flipped_rate, 5.0 * normal_rate)
        self.assertGreater(flipped_hits, normal_hits)

    def test_pool_capacity_and_ceiling_mandate(self) -> None:
        thresholds = derive_thresholds(Phase3Stats.from_stats(STATS))
        digester = ResidualDigester(thresholds)
        with torch.no_grad():
            batch = generate_rollouts(self.env, 2, 24, seed=31, gravity_sign=1, flip_at=8)
            frames = frames_tensor_from(batch.frames)
            z = self.model.encode(frames)
            prediction = self.model.predict(z, torch.from_numpy(batch.actions).long())
            error = (prediction - z[:, 1:]).cpu().numpy()
            surprise = (error**2).mean(axis=-1)
        base = error.reshape(-1, 64)[np.argmax(surprise.reshape(-1))]
        mandates: list[CollapseMandate] = []
        scaffolds = 0
        for index in range(thresholds.debris_capacity + 5):
            residual = base.copy()
            if index < 64:
                residual[index] += 2.5 * thresholds.merge_radius
            result = digester.observe(
                observation(tick=index, residual=residual, surprise=float(np.mean(residual**2)))
            )
            if result.mandate is not None:
                mandates.append(result.mandate)
            scaffolds += sum(
                1 for cohort in result.cohorts if cohort.kind == "scaffold"
            )
            self.assertLessEqual(len(result.pool), thresholds.debris_capacity)
        self.assertGreaterEqual(len(mandates), 1)
        self.assertGreaterEqual(digester.mandates, 1)
        self.assertGreaterEqual(digester.merges + digester.evictions, 1)
        for body in digester.snapshot():
            self.assertGreaterEqual(body.count, 1)
            self.assertEqual(body.centroid.shape, (64,))
        # Round 11: every consumed structure (merge source / capacity victim)
        # is surfaced as a scaffold cohort -- the support machinery P4
        # recycles.  The pool is bounded, so these must exist.
        self.assertGreaterEqual(scaffolds, 1)

    def test_motif_requires_persistence(self) -> None:
        thresholds = derive_thresholds(Phase3Stats.from_stats(STATS))
        digester = ResidualDigester(thresholds)
        residual = np.full(64, 0.0)
        residual[0] = 4.0  # surprise = 16/64 = 0.25 (below q99): not ingested
        result = digester.observe(observation(tick=0, residual=residual, surprise=0.25))
        self.assertFalse(result.ingested)
        self.assertEqual(result.cohorts, ())
        # A real-ingestion vector, repeated on consecutive ticks, forms a motif
        # only once the persistence gate is reached -- and the cohort kind
        # flips from ``residual`` to ``motif`` at exactly that tick.
        residual = np.zeros(64)
        residual[0] = np.sqrt(64 * 0.5)  # surprise 0.5 > q99
        seen_motif = False
        for tick in range(thresholds.motif_min_persistence + 2):
            result = digester.observe(observation(tick=tick, residual=residual))
            kinds = {cohort.kind for cohort in result.cohorts}
            self.assertIn("trail", kinds)
            if result.motif is not None:
                seen_motif = True
                self.assertIn("motif", kinds)
                self.assertNotIn("residual", kinds)
                self.assertGreaterEqual(
                    result.motif.persistence, thresholds.motif_min_persistence
                )
            elif not seen_motif:
                self.assertNotIn("motif", kinds)
        self.assertTrue(seen_motif)

    def test_cohort_ids_are_deterministic_and_unique(self) -> None:
        thresholds = derive_thresholds(Phase3Stats.from_stats(STATS))

        def run() -> list[str]:
            digester = ResidualDigester(thresholds)
            ids: list[str] = []
            for tick in range(12):
                residual = np.zeros(64)
                residual[tick % 5] = np.sqrt(64 * 0.6)
                result = digester.observe(
                    observation(tick=tick, residual=residual, stream=tick % 3)
                )
                for cohort in result.cohorts:
                    self.assertIn(cohort.kind, COHORT_KINDS)
                    self.assertEqual(cohort.centroid.shape, (64,))
                    self.assertGreaterEqual(cohort.count, 1)
                ids.extend(cohort.cohort_id for cohort in result.cohorts)
            return ids

        first = run()
        second = run()
        self.assertEqual(first, second)             # deterministic
        self.assertEqual(len(set(first)), len(first))  # unique per (kind, body, tick)

    def test_flip_run_tags_every_ingest_tick(self) -> None:
        """Every ingested tick yields a motif-or-residual body plus a trail."""
        thresholds = derive_thresholds(Phase3Stats.from_stats(STATS))
        digester = ResidualDigester(thresholds)
        with torch.no_grad():
            batch = generate_rollouts(self.env, 8, 24, seed=31, gravity_sign=1, flip_at=8)
            frames = frames_tensor_from(batch.frames)
            z = self.model.encode(frames)
            prediction = self.model.predict(z, torch.from_numpy(batch.actions).long())
            error = (prediction - z[:, 1:]).cpu().numpy()
            surprise = (error**2).mean(axis=-1)
        counts = {kind: 0 for kind in COHORT_KINDS}
        ingest_ticks = 0
        for stream in range(len(batch)):
            for transition in range(8, batch.horizon):
                result = digester.observe(
                    observation(
                        tick=transition,
                        residual=error[stream, transition],
                        surprise=float(surprise[stream, transition]),
                        stream=stream,
                    )
                )
                if not result.ingested:
                    self.assertEqual(result.cohorts, ())
                    continue
                ingest_ticks += 1
                kinds = [cohort.kind for cohort in result.cohorts]
                self.assertEqual(kinds.count("trail"), 1)
                self.assertEqual(
                    sum(1 for kind in kinds if kind in ("motif", "residual")), 1
                )
                # The tag is the stability verdict, not a guess: a ``motif``
                # cohort's body satisfies the motif gates, a ``residual`` one
                # does not.
                motif = result.motif
                for cohort in result.cohorts:
                    if cohort.kind == "motif":
                        self.assertIsNotNone(motif)
                        self.assertGreaterEqual(cohort.count, thresholds.motif_min_count)
                        self.assertEqual(cohort.body_id, motif.debris_id)
                    if cohort.kind == "residual":
                        self.assertIsNone(motif)
                    counts[cohort.kind] += 1
        self.assertGreater(ingest_ticks, 0)
        self.assertGreater(counts["residual"], 0)
        self.assertGreater(counts["trail"], 0)
        self.assertEqual(counts["trail"], ingest_ticks)


# ---------------------------------------------------------------------------
# CandidateTable (no artifacts)
# ---------------------------------------------------------------------------
class TestCandidateTable(unittest.TestCase):
    def _record(self, name: str) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=name,
            overlay=dummy_overlay(name),
            born_tick=1,
        )

    def test_birth_capacity_is_fifo_and_logged(self) -> None:
        table = CandidateTable(capacity=2)
        table.birth(self._record("cand-01"))
        table.birth(self._record("cand-02"))
        table.birth(self._record("cand-03"))
        self.assertEqual(table.ids(), ("cand-02", "cand-03"))
        self.assertEqual(table.evictions, [("cand-01", 1, 1, "capacity")])
        with self.assertRaises(ValueError):
            table.birth(self._record("cand-03"))  # duplicate id is loud

    def test_observe_requires_a_resident_and_appends_one_sample(self) -> None:
        table = CandidateTable()
        table.birth(self._record("cand-01"))
        table.observe("cand-01", 2, 0.5, 1.0)
        self.assertEqual(table.evidence("cand-01"), ((2, 0.5, 1.0),))
        with self.assertRaises(KeyError):
            table.observe("cand-99", 2, 0.5, 1.0)  # unknown id: loud, no upsert
        self.assertEqual(table.ids(), ("cand-01",))

    def test_residence_arithmetic_and_qualification(self) -> None:
        table = CandidateTable()
        record = table.birth(self._record("cand-01"))
        self.assertEqual(record.residence_ticks, 0)
        self.assertTrue(np.isnan(record.adapted_mean))
        self.assertEqual(record.ratio, float("inf"))
        self.assertFalse(record.qualified(2))
        table.observe("cand-01", 2, 0.5, 1.0)
        table.observe("cand-01", 3, 0.7, 1.0)
        self.assertEqual(record.residence_ticks, 2)
        self.assertAlmostEqual(record.adapted_mean, 0.6, places=12)
        self.assertAlmostEqual(record.base_mean, 1.0, places=12)
        self.assertAlmostEqual(record.ratio, 0.6, places=12)
        self.assertFalse(record.qualified(3))   # residence not yet deep enough
        self.assertTrue(record.qualified(2))    # ratio < 1.0 with 2 samples
        # A harmful candidate never qualifies, however long it resides.
        for tick in range(2, RESIDENCE_N):
            table.observe("cand-01", tick, 0.5, 1.0)
        self.assertTrue(record.qualified(RESIDENCE_N))
        harmful = table.birth(self._record("cand-02"))
        for tick in range(10):
            table.observe("cand-02", tick, 2.0, 1.0)
        self.assertAlmostEqual(harmful.ratio, 2.0, places=12)
        self.assertFalse(harmful.qualified(RESIDENCE_N))
        self.assertEqual(table.survivor().candidate_id, "cand-01")

    def test_survivor_selection_and_tie_break(self) -> None:
        table = CandidateTable()
        table.birth(self._record("cand-01"))
        table.birth(self._record("cand-02"))
        for tick in range(RESIDENCE_N):
            table.observe("cand-01", tick, 0.5, 1.0)
            table.observe("cand-02", tick, 0.5, 1.0)
        # Identical ratios: the oldest birth wins.
        self.assertEqual(table.survivor().candidate_id, "cand-01")
        table.observe("cand-02", 99, 0.1, 1.0)
        self.assertEqual(table.survivor().candidate_id, "cand-02")

    def test_promote_retire_prune_and_reset(self) -> None:
        table = CandidateTable()
        table.birth(self._record("cand-01"))
        table.birth(self._record("cand-02"))
        table.observe("cand-01", 2, 0.4, 1.0)
        table.observe("cand-01", 3, 0.6, 1.0)
        promotion = table.promote("cand-01", 4)
        self.assertEqual(promotion.candidate_id, "cand-01")
        self.assertEqual(promotion.promoted_tick, 4)
        self.assertEqual(promotion.residence_ticks, 2)
        self.assertAlmostEqual(promotion.ratio, 0.5, places=12)
        self.assertEqual(table.ids(), ("cand-02",))
        self.assertEqual(table.promotions, [promotion])  # the log survives
        with self.assertRaises(KeyError):
            table.promote("cand-01", 5)  # already promoted: loud
        retired = table.retire("cand-02", 6)
        self.assertIsNone(table.record("cand-02"))
        self.assertEqual(table.evictions[-1], ("cand-02", 1, 6, "retired"))
        self.assertEqual(table.ids(), ())
        # prune drops everything the store no longer holds
        table.birth(self._record("cand-03"))
        self.assertEqual(len(table.prune_to(frozenset())), 1)
        self.assertEqual(table.ids(), ())
        table.birth(self._record("cand-04"))
        table.reset()
        self.assertEqual(table.ids(), ())
        self.assertEqual(table.promotions, [])
        self.assertEqual(table.evictions, [])
        self.assertIsNone(table.survivor())


# ---------------------------------------------------------------------------
# Overlay live-lifecycle table (unchanged by Round 11 -- kept, not replaced)
# ---------------------------------------------------------------------------
class TestOverlayTable(unittest.TestCase):
    """The overlay index policy is unchanged; ``adopt_motif`` drives it.

    These two assertions predate Round 11 and are kept verbatim: the round
    changed *who* decides to birth an overlay (``#21`` -> ``#24``), not how
    the table behaves.
    """

    def test_birth_revalidate_stale_evict(self) -> None:
        table = OverlayTable()
        record = table.birth(dummy_overlay("ovr-a"), motif_id="motif-a", tick=5)
        self.assertEqual(record.token(), (1, "ovr-a", "active"))
        self.assertIsNotNone(table.resolve("ovr-a"))
        record = table.revalidate("ovr-a", 6)
        self.assertEqual(record.epoch, 2)
        self.assertEqual(record.token(), (2, "ovr-a", "active"))
        record = table.mark_stale("ovr-a", 7)
        self.assertEqual(record.token(), (2, "ovr-a", "stale"))
        self.assertIsNone(table.resolve("ovr-a"))  # stale stops affecting predictions
        self.assertEqual(table.resolve("nope"), None)
        self.assertEqual(table.evict_due(9), ())
        evicted = table.evict_due(10)
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].token(), (2, "ovr-a", "stale"))
        self.assertEqual(len(table), 0)
        self.assertIsNone(table.record("ovr-a"))

    def test_capacity_four_evicts_stale_first(self) -> None:
        table = OverlayTable()
        for index in range(4):
            table.birth(dummy_overlay(f"ovr-{index}"), motif_id=f"m{index}", tick=index)
        table.mark_stale("ovr-0", 4)
        table.birth(dummy_overlay("ovr-4"), motif_id="m4", tick=5)
        self.assertEqual(len(table), 4)
        self.assertIsNone(table.record("ovr-0"))  # stale evicted first
        self.assertIsNotNone(table.record("ovr-1"))
        self.assertEqual(table.records()[0][0], 1)


# ---------------------------------------------------------------------------
# Candidate fit: capacity bound, no live application (#18)
# ---------------------------------------------------------------------------
#: One deterministic stream with a gravity flip at frame tick 8.
STREAM_SEED = 0


def _stream_batch(horizon: int = 20, *, seed: int = STREAM_SEED):
    env = PhysicsEnv(WorldConfig(frame_size=96))
    return generate_rollouts(env, 1, horizon, seed=seed, gravity_sign=1, flip_at=8)


def _prime_stream(organ: PhysicsOrgan, batch, *, horizon: int = 16) -> None:
    """Feed one stream ``horizon`` real transitions (flip at frame tick 8)."""
    organ.begin_stream(batch.frames[0, 0], stream=0)
    for transition in range(horizon):
        organ.observe(
            batch.frames[0, transition + 1],
            int(batch.actions[0, transition]),
            stream=0,
        )


def _window_motif(first_tick: int, last_tick: int) -> StructuralMotif:
    return StructuralMotif(
        motif_id="motif-window",
        debris_id="debris-window",
        centroid=np.zeros(64),
        count=max(1, last_tick - first_tick + 1),
        spread=0.0,
        persistence=max(1, last_tick - first_tick + 1),
        drift=0.0,
        first_tick=first_tick,
        last_tick=last_tick,
    )


@requires_artifacts
class TestCandidateFit(unittest.TestCase):
    """The nursery birth is the capacity-bounded fit, without application."""

    def _organ_with_16(self) -> PhysicsOrgan:
        organ = make_organ(fit_streams={0})
        _prime_stream(organ, _stream_batch())
        return organ

    def test_birth_is_capacity_bounded_and_never_steers(self) -> None:
        organ = self._organ_with_16()
        working_before = [
            parameter.detach().clone() for parameter in organ.working_model.parameters()
        ]
        candidate = organ.birth_candidate(16, fit_window=16)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.candidate_id, "cand-01")
        self.assertEqual(candidate.overlay.overlay_id, "cand-01")
        self.assertEqual(candidate.born_tick, 16)
        self.assertEqual(candidate.evidence, [])
        # The unchanged Round-9a capacity bound: 16 transitions x 1 stream.
        self.assertEqual(candidate.overlay.rank, 26)          # floor((5*16*64-64)/192)
        self.assertEqual(candidate.overlay.param_count, 5_056)
        self.assertEqual(candidate.overlay.budget_n, 16)
        self.assertLessEqual(candidate.overlay.param_count, candidate.overlay.capacity)
        self.assertFalse(candidate.overlay.changes_gru)
        self.assertLess(candidate.overlay.loss_after, candidate.overlay.loss_before)
        self.assertEqual(organ.fit_history[-1].reason, "")
        # Nothing was applied: no live overlay, no working-model change.
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())
        self.assertEqual(organ.overlay_table.active(), ())
        self.assertIsNone(organ._active_overlay_id)
        for before, parameter in zip(
            working_before, organ.working_model.parameters()
        ):
            self.assertTrue(torch.equal(before, parameter.detach()))
        self.assertTrue(organ.encoder_bit_identical())
        self.assertEqual(organ.candidate_ids(), ("cand-01",))
        # The store's attach hook reads the patch by candidate id.
        patch = organ.candidate_patch("cand-01")
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch.rank, 26)

    def test_full_patch_is_refused_by_capacity(self) -> None:
        organ = make_organ(patch="full", fit_streams={0})
        _prime_stream(organ, _stream_batch())
        self.assertIsNone(organ.birth_candidate(16, fit_window=16))
        record = organ.fit_history[-1]
        self.assertEqual(record.reason, "capacity")
        self.assertEqual(record.patch, "full")
        self.assertEqual(record.requested_params, 210_920)
        self.assertEqual(record.budget_n, 16)
        self.assertEqual(record.budget_capacity, 5_120.0)
        self.assertEqual(record.budget_max_params, 5_056)
        self.assertEqual(organ.candidate_ids(), ())
        self.assertEqual(organ.refusals[-1]["reason"], "capacity")

    def test_no_evidence_is_recorded_not_silent(self) -> None:
        organ = make_organ(fit_streams={0})
        self.assertIsNone(organ.birth_candidate(1, fit_window=16))
        record = organ.fit_history[-1]
        self.assertEqual(record.reason, "no-evidence")
        self.assertEqual(record.budget_n, 0)
        self.assertEqual(organ.candidate_ids(), ())

    def test_no_peek_is_structurally_enforced(self) -> None:
        """The birth fit can never see a transition after the birth tick."""
        organ = self._organ_with_16()
        with self.assertRaises(AssertionError):
            organ._self_guard_fit_boundary(_window_motif(1, 17), 16)

    def test_candidate_capacity_evicts_oldest(self) -> None:
        organ = self._organ_with_16()
        born = []
        for _ in range(NURSERY_CAPACITY + 1):
            candidate = organ.birth_candidate(16, fit_window=16)
            if candidate is None:  # pragma: no cover - fit is deterministic
                self.fail("the fit refused unexpectedly")
            born.append(candidate.candidate_id)
        self.assertEqual(len(organ.candidate_ids()), NURSERY_CAPACITY)
        self.assertNotIn(born[0], organ.candidate_ids())
        self.assertEqual(
            organ.nursery.evictions[-1], (born[0], 16, 16, "capacity")
        )


# ---------------------------------------------------------------------------
# Shadow scoring: the candidate is evidence, never control
# ---------------------------------------------------------------------------
@requires_artifacts
class TestShadowScoring(unittest.TestCase):
    def _organ_with_candidate(self) -> tuple[PhysicsOrgan, object, CandidateRecord]:
        organ = make_organ(fit_streams={0})
        batch = _stream_batch()
        _prime_stream(organ, batch)
        candidate = organ.birth_candidate(16, fit_window=16)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        return organ, batch, candidate

    def test_shadow_scoring_never_touches_the_live_model(self) -> None:
        organ, batch, candidate = self._organ_with_candidate()
        encoder_before = [
            parameter.detach().clone()
            for parameter in organ.base_model.encoder_parameters()
        ]
        base_before = [
            parameter.detach().clone()
            for parameter in organ.base_model.predictor.parameters()
        ]
        working_before = [
            parameter.detach().clone() for parameter in organ.working_model.parameters()
        ]
        organ.observe(batch.frames[0, 17], int(batch.actions[0, 16]), stream=0)
        scores = organ.shadow_scores(17)
        self.assertEqual(set(scores), {candidate.candidate_id})
        for before, parameter in zip(
            encoder_before, organ.base_model.encoder_parameters()
        ):
            self.assertTrue(torch.equal(before, parameter.detach()))
        for before, parameter in zip(
            base_before, organ.base_model.predictor.parameters()
        ):
            self.assertTrue(torch.equal(before, parameter.detach()))
        for before, parameter in zip(
            working_before, organ.working_model.parameters()
        ):
            self.assertTrue(torch.equal(before, parameter.detach()))
        self.assertTrue(organ.encoder_bit_identical())

    def test_one_sample_per_resident_tick_and_idempotent(self) -> None:
        organ, batch, candidate = self._organ_with_candidate()
        for frame_tick, transition in ((17, 16), (18, 17), (19, 18)):
            organ.observe(batch.frames[0, frame_tick], int(batch.actions[0, transition]), stream=0)
            first = organ.shadow_scores(frame_tick)
            second = organ.shadow_scores(frame_tick)  # re-scoring is idempotent
            self.assertEqual(first, second)
            self.assertEqual(
                candidate.residence_ticks, frame_tick - 16
            )
        self.assertEqual(
            [sample[0] for sample in candidate.evidence], [17, 18, 19]
        )
        self.assertEqual(
            organ.residence_evidence(candidate.candidate_id)[0], 3
        )
        # A tick with no recorded transition scores nothing and records nothing.
        self.assertEqual(organ.shadow_scores(999), {})
        self.assertEqual(candidate.residence_ticks, 3)

    def test_shadow_score_equals_an_independent_replay(self) -> None:
        """The method is auditable: recompute the pair without the organ."""
        organ, batch, candidate = self._organ_with_candidate()
        for frame_tick, transition in ((17, 16), (18, 17)):
            organ.observe(
                batch.frames[0, frame_tick], int(batch.actions[0, transition]), stream=0
            )
        organ.shadow_scores(17)
        organ.shadow_scores(18)
        # Independent replay: the same two transitions, the same hidden states.
        state = organ._state(0)
        pairs = []
        for frame_tick in (17, 18):
            item = next(t for t in state.transitions if t[0] == frame_tick)
            _, prev_latent, action, hidden_before, latent = item
            probe = copy.deepcopy(organ.base_model)
            candidate.overlay.apply_to(probe)
            with torch.no_grad():
                adapted_prediction, _ = probe.predict_step(prev_latent, int(action), hidden_before)
                base_prediction, _ = organ.base_model.predict_step(
                    prev_latent, int(action), hidden_before
                )
                adapted = float(
                    torch.mean((adapted_prediction - latent).pow(2))
                )
                base = float(torch.mean((base_prediction - latent).pow(2)))
            pairs.append((adapted, base))
        for measured, expected in zip(candidate.evidence, pairs):
            self.assertAlmostEqual(measured[1], expected[0], places=6)
            self.assertAlmostEqual(measured[2], expected[1], places=6)
        # Each pair is a real paired measurement on the same transition, so
        # both sides are strictly positive finite surprises.
        for adapted, base in pairs:
            self.assertTrue(np.isfinite(adapted) and adapted >= 0.0)
            self.assertTrue(np.isfinite(base) and base >= 0.0)

    def test_residence_evidence_is_the_cycle_evidence(self) -> None:
        """The measured residence curve: one sample per tick, mean ratio."""
        organ, batch, candidate = self._organ_with_candidate()
        for frame_tick, transition in ((17, 16), (18, 17), (19, 18), (20, 19)):
            organ.observe(
                batch.frames[0, frame_tick], int(batch.actions[0, transition]), stream=0
            )
            organ.shadow_scores(frame_tick)
        ticks, adapted, base, ratio = organ.residence_evidence(candidate.candidate_id)
        self.assertEqual(ticks, RESIDENCE_N)
        self.assertAlmostEqual(
            adapted, float(np.mean([sample[1] for sample in candidate.evidence])), places=9
        )
        self.assertAlmostEqual(
            base, float(np.mean([sample[2] for sample in candidate.evidence])), places=9
        )
        self.assertAlmostEqual(ratio, adapted / base, places=9)
        # Measured on this stream: the candidate wins on residence (mean ratio
        # < 1.0), though individual ticks are mixed -- the pre-registered gate
        # reads the mean, so the per-tick spread is reported, not hidden.
        self.assertLess(ratio, 1.0)
        self.assertEqual(organ.qualified(candidate.candidate_id), ratio < 1.0)
        self.assertEqual(
            organ.survivor().candidate_id, candidate.candidate_id
        )

    def test_harmful_survivor_is_surfaced_with_its_evidence(self) -> None:
        """The selection does NOT pre-filter on ``qualified`` (ruling, §1.9).

        A harmful resident is surfaced by ``#23`` with ``qualified is False``
        so that the **gate** is the thing that refuses it; if the selection
        filtered it out, ``#22``'s refusal and the ADMISSION metric would be
        vacuous.  This is the organ-side mirror of
        ``MatBoard.nursery_survivor``; the executor test
        (``tests/test_physics_run.py``) asserts the surfaced-and-refused
        store path.
        """
        organ, batch, candidate = self._organ_with_candidate()
        # Drive the candidate's evidence negative by hand (its record is the
        # organ's own bookkeeping; this is the falsifier's fixture, not a
        # measurement).
        for tick in range(RESIDENCE_N):
            organ.nursery.observe(candidate.candidate_id, 17 + tick, 2.0, 1.0)
        ticks, adapted, base, ratio = organ.residence_evidence(
            candidate.candidate_id
        )
        self.assertEqual(ticks, RESIDENCE_N)
        self.assertGreaterEqual(ratio, 1.0)
        self.assertAlmostEqual(ratio, 2.0, places=12)
        self.assertFalse(organ.qualified(candidate.candidate_id))
        survivor = organ.survivor()
        self.assertIsNotNone(survivor)
        assert survivor is not None
        self.assertEqual(survivor.candidate_id, candidate.candidate_id)
        self.assertIs(survivor, organ.nursery.require(candidate.candidate_id))
        # The evidence, not a filter, is what the gate reads.
        self.assertGreaterEqual(survivor.ratio, 1.0)
        self.assertEqual(
            survivor.qualified(organ.thresholds.residence_n), False
        )
        # Promotion is a bookkeeping mirror and stays honest: the record shows
        # the harmful final ratio, which is exactly what ADMISSION counts.
        promotion = organ.promote_candidate(candidate.candidate_id, 21)
        self.assertEqual(promotion.ratio, 2.0)
        self.assertFalse(promotion.qualified)
        self.assertIsNone(organ.candidate_patch(candidate.candidate_id))

    def test_promote_and_prune_drop_the_shadow_predictor(self) -> None:
        organ, batch, candidate = self._organ_with_candidate()
        self.assertIn(candidate.candidate_id, organ._shadow_predictors)
        organ.promote_candidate(candidate.candidate_id, 17)
        self.assertNotIn(candidate.candidate_id, organ._shadow_predictors)
        self.assertEqual(organ.promotions[0].candidate_id, candidate.candidate_id)
        second = organ.birth_candidate(18, fit_window=16)
        self.assertIsNotNone(second)
        assert second is not None
        organ.prune_candidates(frozenset())
        self.assertNotIn(second.candidate_id, organ._shadow_predictors)
        self.assertEqual(organ.candidate_ids(), ())
        with self.assertRaises(KeyError):
            organ.residence_evidence(candidate.candidate_id)


# ---------------------------------------------------------------------------
# Motif-driven adoption (#21 -> #24)
# ---------------------------------------------------------------------------
def motif_cohort(
    *,
    cohort_id: str = "motif-debris-001-s0-0018",
    first_tick: int = 10,
    last_tick: int = 18,
    body_id: str = "debris-001",
) -> CohortRecord:
    return CohortRecord(
        cohort_id=cohort_id,
        kind="motif",
        centroid=np.full(DEFAULT_LATENT_DIM, 0.5),
        count=12,
        spread=1.0,
        first_tick=first_tick,
        last_tick=last_tick,
        peak_surprise=0.9,
        body_id=body_id,
    )


@requires_artifacts
class TestAdoptMotif(unittest.TestCase):
    def _organ_with_16(self) -> PhysicsOrgan:
        organ = make_organ(fit_streams={0})
        _prime_stream(organ, _stream_batch())
        return organ

    def test_adopt_births_then_revalidates_the_live_overlay(self) -> None:
        organ = self._organ_with_16()
        cohort = motif_cohort()
        record = organ.adopt_motif(cohort, 18)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.overlay_id, "ovr-01")
        self.assertEqual(record.token(), (1, "ovr-01", "active"))
        self.assertEqual(record.motif_id, cohort.cohort_id)
        self.assertEqual(organ._active_overlay_id, "ovr-01")
        self.assertEqual(organ.overlay_table.births, [record])
        self.assertEqual(record.epoch, 1)
        # The live patch is active: the working predictor now carries it.
        self.assertTrue(organ.encoder_bit_identical())
        self.assertEqual(organ.d4.writes, 0)
        # The same motif revalidates (epoch advances) and re-activates.
        again = organ.adopt_motif(cohort, 19)
        assert again is not None
        self.assertEqual(again.token(), (2, "ovr-01", "active"))
        self.assertEqual(organ.overlay_table.births, [record])
        self.assertEqual(organ.overlay_table.revalidations[-1].epoch, 2)

    def test_single_active_ratchet_refuses_a_different_motif(self) -> None:
        organ = self._organ_with_16()
        self.assertIsNotNone(organ.adopt_motif(motif_cohort(), 18))
        other = organ.adopt_motif(
            motif_cohort(cohort_id="motif-debris-002-s0-0018", body_id="debris-002"),
            19,
        )
        self.assertIsNone(other)
        self.assertEqual(organ.overlay_table.all_ids(), frozenset({"ovr-01"}))
        self.assertEqual(organ._active_overlay_id, "ovr-01")

    def test_cooldown_blocks_adoption(self) -> None:
        organ = self._organ_with_16()
        organ._cooldown_until = 22  # set by the staleness rule in production
        self.assertIsNone(organ.adopt_motif(motif_cohort(), 20))
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())
        self.assertIsNotNone(organ.adopt_motif(motif_cohort(), 22))

    def test_a_refused_fit_is_recorded_and_adoption_returns_none(self) -> None:
        organ = self._organ_with_16()
        # A motif window that ends at tick 0 has no evidence transition at all
        # (each transition predicts frame t >= 1).
        refused = organ.adopt_motif(motif_cohort(first_tick=0, last_tick=0), 0)
        self.assertIsNone(refused)
        self.assertEqual(organ.fit_history[-1].reason, "no-evidence")
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())

    def test_adopt_accepts_a_payload_shaped_motif_body(self) -> None:
        """#21 is an executor store read: the body may be a payload dataclass."""
        organ = self._organ_with_16()

        class PayloadLikeMotif:
            motif_id = "motif-debris-007"
            debris_id = "debris-007"
            centroid = tuple([0.25] * DEFAULT_LATENT_DIM)
            count = 11
            spread = 1.5
            persistence = 10
            drift = 0.5
            first_tick = 9
            last_tick = 18

        record = organ.adopt_motif(PayloadLikeMotif(), 18)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.motif_id, "motif-debris-007")
        self.assertEqual(record.overlay_id, "ovr-01")

    def test_stale_and_evict_semantics_are_unchanged(self) -> None:
        organ = self._organ_with_16()
        record = organ.adopt_motif(motif_cohort(), 18)
        assert record is not None
        stale = organ.overlay_table.mark_stale(record.overlay_id, 20)
        self.assertEqual(stale.status, "stale")
        self.assertIsNone(organ.overlay_table.resolve(record.overlay_id))
        # The patch object survives (it is D2's substance) but no longer
        # resolves to the prediction path.
        self.assertIsNotNone(organ.overlay_table.adapter(record.overlay_id))
        self.assertEqual(organ.overlay_table.evict_due(22), ())
        evicted = organ.overlay_table.evict_due(23)
        self.assertEqual([item.token() for item in evicted], [(1, "ovr-01", "stale")])
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())


# ---------------------------------------------------------------------------
# The live-overlay D4 gate is gone (ruling C) -- replaced by residence evidence
# ---------------------------------------------------------------------------
@requires_artifacts
class TestGateRemoval(unittest.TestCase):
    """Round 9a's live-overlay gate is deleted as a D4 decision path.

    The assertions that used to pin ``ratchet_gate_passed`` /
    ``_accrue_gate_evidence`` / ``_record_gate_evidence`` are replaced by
    stronger nursery-evidence assertions: the decision input is now the
    candidate's residence evidence, and the deleted path must not come back.
    """

    def test_the_live_overlay_gate_is_not_a_decision_path(self) -> None:
        organ = make_organ()
        for name in (
            "ratchet_gate_passed",
            "_accrue_gate_evidence",
            "_record_gate_evidence",
        ):
            self.assertFalse(hasattr(organ, name), f"{name} must be removed (ruling C)")
        for name in ("_gate_evidence", "_gate_adapted_sum", "_gate_base_sum"):
            self.assertFalse(hasattr(organ, name))
        # The decision path that replaced it exists and is public.
        self.assertTrue(callable(organ.residence_evidence))
        self.assertTrue(callable(organ.qualified))
        self.assertTrue(callable(organ.survivor))

    def test_a_live_overlay_alone_cannot_reach_d4(self) -> None:
        """Nothing in the ratchet writes D4, however long the overlay is active."""
        organ = make_organ(fit_streams={0})
        batch = _stream_batch(24)
        _prime_stream(organ, batch, horizon=18)
        record = organ.adopt_motif(motif_cohort(), 18)
        self.assertIsNotNone(record)
        for frame_tick in range(19, 24):
            organ.observe(
                batch.frames[0, frame_tick], int(batch.actions[0, frame_tick - 1]), stream=0
            )
            organ.ratchet(organ.digest(organ.observations[-1]))
        organ.flush()
        self.assertEqual(organ.d4.writes, 0)
        self.assertEqual(organ.d4.payloads, [])
        self.assertEqual(organ.candidate_ids(), ())  # no candidate, no survivor
        self.assertIsNone(organ.survivor())


@requires_artifacts
class TestReset(unittest.TestCase):
    def test_reset_clears_the_nursery_as_well_as_p6(self) -> None:
        organ = make_organ(fit_streams={0})
        batch = _stream_batch()
        _prime_stream(organ, batch)
        candidate = organ.birth_candidate(16, fit_window=16)
        self.assertIsNotNone(candidate)
        organ.observe(batch.frames[0, 17], int(batch.actions[0, 16]), stream=0)
        organ.shadow_scores(17)
        organ.adopt_motif(motif_cohort(), 17)
        self.assertGreater(len(organ.candidate_ids()), 0)
        self.assertGreater(len(organ.overlay_table), 0)
        self.assertGreater(len(organ.fit_history), 0)
        organ.reset_p6_state()
        self.assertEqual(organ.candidate_ids(), ())
        self.assertEqual(organ._shadow_predictors, {})
        self.assertEqual(organ.promotions, ())
        self.assertEqual(organ.overlay_table.all_ids(), frozenset())
        self.assertIsNone(organ._active_overlay_id)
        self.assertEqual(organ.fit_history, [])
        self.assertEqual(organ.observations, [])
        self.assertEqual(organ.refusals, [])
        self.assertEqual(organ.d4.writes, 0)
        self.assertTrue(organ.encoder_bit_identical())


# ---------------------------------------------------------------------------
# The organ-level nursery probe: the whole lifecycle on a real flip
# ---------------------------------------------------------------------------
@requires_artifacts
class TestNurseryScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        cls.organ = make_organ()
        cls.trace = cls.organ.nursery_scenario(
            seed=0,
            n_streams=16,
            horizon=36,
            flip_at=8,
            flip_back_at=28,
            fit_window=DEFAULT_FIT_WINDOW,
        )

    def test_every_cohort_class_appears_over_the_flip(self) -> None:
        counts = self.trace.cohort_counts
        for kind in COHORT_KINDS:
            self.assertGreater(counts[kind], 0, f"no {kind} cohort in the run")
        # The tags are not decorative: a motif cohort is rare (the persistence
        # gate is 10 consecutive supported ticks) and residuals dominate.
        self.assertGreater(counts["residual"], counts["motif"])
        self.assertGreater(counts["trail"], 0)

    def test_candidate_is_born_shadow_scored_and_promoted(self) -> None:
        self.assertTrue(self.trace.born)
        born = self.trace.born[0]
        self.assertEqual(born["candidate_id"], "cand-01")
        self.assertEqual(born["fit_transitions"], 160)     # 16 streams x 10
        self.assertLess(born["loss_after"], born["loss_before"])
        self.assertTrue(self.trace.promotions)
        promotion = self.trace.promotions[0]
        self.assertEqual(promotion.candidate_id, born["candidate_id"])
        self.assertEqual(promotion.residence_ticks, RESIDENCE_N)
        self.assertTrue(promotion.qualified)
        self.assertLess(promotion.ratio, 1.0)
        self.assertLess(born["born_tick"], promotion.promoted_tick)
        # Shadow rows are exactly the candidate's out-of-sample residence
        # ticks: the birth tick is excluded by construction (the fit window
        # includes it, so that sample would be in-sample evidence), which is
        # also the only evidence set the executor's store can carry.
        self.assertEqual(
            [row[0] for row in self.trace.shadow_rows()],
            list(range(born["born_tick"] + 1, promotion.promoted_tick + 1)),
        )

    def test_shadow_adaptation_is_below_one(self) -> None:
        """Organ-side ADAPTATION: mean shadow ratio over the flipped rows."""
        ratio = self.trace.adaptation_ratio()
        self.assertTrue(np.isfinite(ratio))
        self.assertLess(ratio, 1.0)
        # The per-tick curve is reported, not hidden: it is not monotone.
        ratios = [row[2] for row in self.trace.shadow_rows()]
        self.assertGreater(max(ratios), 1.0)
        self.assertGreater(min(ratios), 0.0)

    def test_live_overlay_lifecycle_from_the_motif_cohort(self) -> None:
        lifecycle = self.trace.lifecycle()
        self.assertTrue(self.trace.overlay_births)
        # Measured on this trace: the motif cohort appears at tick 24 (the
        # overlay persists as a body for ~10 supported ticks).  The pinned
        # claim is structural -- adopted during the flip, before the flip
        # back -- not the exact tick.
        self.assertGreaterEqual(lifecycle["overlay_birth"], 9)
        self.assertLess(lifecycle["overlay_birth"], self.trace.flip_back_at)
        birth = self.trace.overlay_births[0]
        self.assertEqual(birth.token(), (1, "ovr-02", "active"))
        self.assertEqual(birth.motif_id, "motif-debris-001-s6-0024")
        self.assertTrue(self.trace.overlay_revalidations)
        self.assertEqual(self.trace.overlay_revalidations[0].epoch, 2)
        stale = lifecycle["overlay_stale"]
        eviction = lifecycle["overlay_evict"]
        self.assertIsNotNone(stale)
        self.assertIsNotNone(eviction)
        assert stale is not None and eviction is not None
        self.assertGreater(stale, birth.birth_tick)
        self.assertGreater(eviction, stale)
        # The patch went stale in the recovered phase, not during the flip: the
        # staleness rule reads the predictor's own adapted-vs-base comparison.
        self.assertGreaterEqual(stale, self.trace.flip_back_at)
        self.assertEqual(
            [item.token() for item in self.trace.overlay_evictions],
            [(8, "ovr-02", "stale")],
        )

    def test_live_overlay_helps_in_its_regime_and_hurts_out_of_it(self) -> None:
        """Measured live state (contract §5(iii)): the patch is regime-bound.

        While the flipped dynamics it was fitted for persist, the live overlay
        beats the base predictor; after the flip back it hurts, which is
        exactly what the staleness rule must catch (and it does: stale at 31,
        evicted at 34).  Reported as measured state, never as a gate.
        """
        flipped = self.trace.live_ratio(phase="flipped")
        recovered = self.trace.live_ratio(phase="recovered")
        self.assertTrue(np.isfinite(flipped))
        self.assertTrue(np.isfinite(recovered))
        self.assertLess(flipped, 1.0)
        self.assertGreater(recovered, 1.0)
        self.assertTrue(self.trace.live_rows)
        # The steered ticks lie between the overlay birth and its staleness.
        # The stale *decision* for tick t is finalised when t+1's digest
        # arrives (the round-9a lifecycle cadence, unchanged), so the patch
        # steers at most one tick past the recorded stale tick.
        birth = self.trace.adoption_tick
        stale = self.trace.stale_tick
        assert birth is not None and stale is not None
        for tick, _, _ in self.trace.live_rows:
            self.assertGreater(tick, birth)
            self.assertLessEqual(tick, stale + 1)

    def test_zero_d4_writes_and_encoder_bit_identical(self) -> None:
        self.assertEqual(self.trace.d4_writes, 0)
        self.assertEqual(self.organ.d4.writes, 0)
        self.assertEqual(self.organ.d4.payloads, [])
        self.assertTrue(self.trace.encoder_bit_identical)
        self.assertTrue(self.organ.encoder_bit_identical())

    def test_end_state_is_clean(self) -> None:
        # The promoted candidate left the nursery; the live overlay was
        # evicted.  Nothing is retained on the physics side.
        self.assertEqual(self.organ.candidate_ids(), ())
        self.assertEqual(self.organ.overlay_table.all_ids(), frozenset())
        self.assertIsNone(self.organ._active_overlay_id)


# ---------------------------------------------------------------------------
# Overlay fit (P6): held-out recovery + encoder bit-identity (unchanged)
# ---------------------------------------------------------------------------
@requires_artifacts
class TestOverlayFitHeldOut(unittest.TestCase):
    """Fit on episodes 0..15, evaluate on unseen seed-77 episodes.

    The fit machinery is unchanged in Round 11 (same capacity bound, same
    replay path); this probe pins that it still produces a patch that helps
    on streams it never saw -- a property, not a re-asserted 9a number.
    """

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        cls.organ = make_organ(patch="head", fit_steps=40)
        env = PhysicsEnv(WorldConfig(frame_size=96))
        cls.fit_batch = generate_rollouts(env, 16, 24, seed=31, gravity_sign=1, flip_at=8)
        cls.heldout_batch = generate_rollouts(env, 16, 24, seed=77, gravity_sign=1, flip_at=8)
        for stream in range(len(cls.fit_batch)):
            cls.organ.begin_stream(cls.fit_batch.frames[stream, 0], stream=stream)
        for transition in range(cls.fit_batch.horizon):
            for stream in range(len(cls.fit_batch)):
                cls.organ.observe(
                    cls.fit_batch.frames[stream, transition + 1],
                    int(cls.fit_batch.actions[stream, transition]),
                    stream=stream,
                )
        cls.overlay = cls.organ.fit_overlay(
            _window_motif(9, 23), tick=24
        )

    def test_overlay_is_capacity_bounded_and_predictor_only(self) -> None:
        self.assertIsNotNone(self.overlay)
        overlay = self.overlay
        assert overlay is not None
        self.assertEqual(overlay.patch, "head")
        self.assertFalse(overlay.changes_gru)
        self.assertEqual(overlay.fit_streams, 16)
        self.assertEqual(overlay.fit_transitions, 256)
        self.assertEqual(overlay.budget_n, 256)
        self.assertEqual(overlay.rank, 64)            # min(D, H), not r_max=640
        self.assertEqual(overlay.param_count, 12_352)
        self.assertEqual(overlay.nbytes, 49_408)
        self.assertLessEqual(overlay.param_count, overlay.capacity)
        self.assertLess(overlay.loss_after, overlay.loss_before)
        probe = copy.deepcopy(self.organ.base_model)
        before = {
            name: parameter.detach().clone()
            for name, parameter in probe.predictor.named_parameters()
        }
        overlay.apply_to(probe)
        changed = {
            name
            for name, parameter in probe.predictor.named_parameters()
            if not torch.equal(parameter, before[name])
        }
        self.assertEqual(changed, {"head.2.weight", "head.2.bias"})
        self.assertEqual(tuple(overlay.materialise().shape), (64, 128))

    def test_encoder_is_bit_identical_after_fit(self) -> None:
        self.assertTrue(self.organ.encoder_bit_identical())
        with torch.no_grad():
            for parameter in self.organ.base_model.encoder_parameters():
                self.assertFalse(parameter.requires_grad)

    def test_heldout_flipped_surprise_improves(self) -> None:
        overlay = self.overlay
        assert overlay is not None
        with torch.no_grad():
            frames = frames_tensor_from(self.heldout_batch.frames)
            actions = torch.from_numpy(self.heldout_batch.actions).long()
            latents = self.organ.base_model.encode(frames)
            base = (self.organ.base_model.predict(latents, actions) - latents[:, 1:]).pow(2)
            base = base.mean(dim=-1).numpy()
            adapted_model = copy.deepcopy(self.organ.base_model)
            overlay.apply_to(adapted_model)
            adapted = (adapted_model.predict(latents, actions) - latents[:, 1:]).pow(2)
            adapted = adapted.mean(dim=-1).numpy()
        post = slice(8, 24)
        ratio = float(adapted[:, post].mean() / base[:, post].mean())
        self.assertLess(ratio, 0.85)
        self.assertGreater(ratio, 0.0)
        per_transition = adapted[:, post].mean(axis=0) / base[:, post].mean(axis=0)
        self.assertLess(float(per_transition.max()), 1.05)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
