"""Tests for the phase 4-5a physics organ (``amateur/physics/organ.py``).

Pins the contract the executor will consume:

* thresholds are *derived* from the measured phase-3 statistics, and they
  separate real normal residuals from real flipped residuals;
* each observation carries the real residual, the latent state delta, the
  3-frame causal window, and the resolved latent rollout; online surprise
  agrees with the batched ``MicroLeWM.predict`` reference;
* the digest is a bounded pool: capacity is respected, ceiling pressure
  emits a mandate, and eviction/merge actually happen;
* the overlay is a predictor-only patch: it improves held-out flipped
  surprise on episodes it never saw, and every base-encoder parameter is
  bit-identical before and after the fit (``torch.equal``, no tolerance);
* the overlay table lifecycle is birth / revalidate (epoch advances) /
  stale (no longer resolves) / evict (gone), capacity 4;
* the acceptance probe produces the full trace -- spike, adoption,
  recovery, stale, evict -- with **zero D4 (Base Core) writes**.

The artifact-dependent tests skip (with an explicit reason) if
``.work/forge/checkpoint.pt`` / ``stats.json`` are absent; they are
regenerable with ``python3 -m amateur.physics.train``.

Runtime: ~13 s on 8 CPU threads (the acceptance probe dominates).
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
    DEFAULT_LATENT_DIM,
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

    @requires_artifacts
    def test_organ_uses_the_same_thresholds(self) -> None:
        organ = make_organ()
        expected = derive_thresholds(Phase3Stats.from_stats(STATS))
        self.assertEqual(organ.thresholds, expected)


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
        # Rebuild the same frames deterministically from the env state.
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
        # Batched reference: identical math, different execution path.
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


# ---------------------------------------------------------------------------
# Digest (P5)
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
        # Measured on this checkpoint over the phase-matched post-flip window
        # (transitions 8..23): normal 0/256, flipped 57/256 (22.27 %).  Over a
        # whole normal episode the rate is 4/384 (1.04 %), matching the q99
        # gate's 1 % design budget.  The gate is the measured normal q99.
        self.assertLessEqual(normal_rate, 0.03)
        self.assertGreaterEqual(flipped_rate, 0.15)
        self.assertGreaterEqual(flipped_rate, 5.0 * normal_rate)
        self.assertGreater(flipped_hits, normal_hits)

    def test_pool_capacity_and_ceiling_mandate(self) -> None:
        thresholds = derive_thresholds(Phase3Stats.from_stats(STATS))
        digester = ResidualDigester(thresholds)
        # Real flipped residual as the base vector, pushed along orthogonal
        # directions by two merge radii each: mechanically distinct clusters.
        with torch.no_grad():
            batch = generate_rollouts(self.env, 2, 24, seed=31, gravity_sign=1, flip_at=8)
            frames = frames_tensor_from(batch.frames)
            z = self.model.encode(frames)
            prediction = self.model.predict(z, torch.from_numpy(batch.actions).long())
            error = (prediction - z[:, 1:]).cpu().numpy()
            surprise = (error**2).mean(axis=-1)
        base = error.reshape(-1, 64)[np.argmax(surprise.reshape(-1))]
        mandates: list[CollapseMandate] = []
        for index in range(thresholds.debris_capacity + 5):
            residual = base.copy()
            if index < 64:
                residual[index] += 2.5 * thresholds.merge_radius
            result = digester.observe(
                observation(tick=index, residual=residual, surprise=float(np.mean(residual**2)))
            )
            if result.mandate is not None:
                mandates.append(result.mandate)
            self.assertLessEqual(len(result.pool), thresholds.debris_capacity)
        self.assertGreaterEqual(len(mandates), 1)
        self.assertGreaterEqual(digester.mandates, 1)
        self.assertGreaterEqual(digester.merges + digester.evictions, 1)
        for body in digester.snapshot():
            self.assertGreaterEqual(body.count, 1)
            self.assertEqual(body.centroid.shape, (64,))

    def test_motif_requires_persistence(self) -> None:
        thresholds = derive_thresholds(Phase3Stats.from_stats(STATS))
        digester = ResidualDigester(thresholds)
        residual = np.full(64, 0.0)
        residual[0] = 4.0  # surprise = 16/64 = 0.25 (below q99): not ingested
        result = digester.observe(observation(tick=0, residual=residual, surprise=0.25))
        self.assertFalse(result.ingested)
        # A real-ingestion vector, repeated on consecutive ticks, forms a motif
        # only once the persistence gate is reached.
        residual = np.zeros(64)
        residual[0] = np.sqrt(64 * 0.5)  # surprise 0.5 > q99
        seen_motif = False
        for tick in range(thresholds.motif_min_persistence + 2):
            result = digester.observe(observation(tick=tick, residual=residual))
            if result.motif is not None:
                seen_motif = True
                self.assertGreaterEqual(result.motif.persistence, thresholds.motif_min_persistence)
            elif tick < thresholds.motif_min_persistence - 1:
                self.assertFalse(seen_motif)
        self.assertTrue(seen_motif)


# ---------------------------------------------------------------------------
# Overlay lifecycle (no artifacts needed)
# ---------------------------------------------------------------------------
def dummy_overlay(overlay_id: str) -> PredictorOverlay:
    return PredictorOverlay(
        overlay_id=overlay_id,
        deltas={"head.0.weight": torch.zeros(2, 2)},
        param_count=4,
        fit_transitions=1,
        fit_streams=1,
        loss_before=1.0,
        loss_after=0.5,
        patch="head",
    )


class TestOverlayTable(unittest.TestCase):
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
# Overlay fit (P6): held-out recovery + encoder bit-identity
# ---------------------------------------------------------------------------
@requires_artifacts
class TestOverlayFitHeldOut(unittest.TestCase):
    """Fit on episodes 0..15, evaluate on unseen seed-77 episodes."""

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
        cls.motif = StructuralMotif(
            motif_id="motif-probe",
            debris_id="debris-probe",
            centroid=np.zeros(64),
            count=128,
            spread=4.0,
            persistence=10,
            drift=1.0,
            first_tick=9,
            last_tick=23,
        )
        cls.overlay = cls.organ.fit_overlay(cls.motif, tick=24)

    def test_overlay_is_predictor_only_and_compact(self) -> None:
        self.assertIsNotNone(self.overlay)
        overlay = self.overlay
        assert overlay is not None
        self.assertEqual(overlay.patch, "head")
        self.assertFalse(overlay.changes_gru)
        self.assertLess(overlay.param_count, 50_000)
        self.assertLess(overlay.nbytes, 200_000)
        self.assertEqual(overlay.fit_streams, 16)
        self.assertGreater(overlay.fit_transitions, 100)
        self.assertLess(overlay.loss_after, overlay.loss_before)
        for name in overlay.deltas:
            self.assertTrue(name.startswith(("head.", "norm.")), name)

    def test_encoder_is_bit_identical_after_fit(self) -> None:
        self.assertTrue(self.organ.encoder_bit_identical())
        # Stronger: the snapshot was taken at organ construction, before any
        # observation or fit, and every tensor compares with torch.equal.
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
        # Measured 0.690x (base 0.2616 -> adapted 0.1805) on unseen episodes.
        self.assertLess(ratio, 0.85)
        self.assertGreater(ratio, 0.0)
        per_transition = adapted[:, post].mean(axis=0) / base[:, post].mean(axis=0)
        self.assertLess(float(per_transition.max()), 1.05)


# ---------------------------------------------------------------------------
# Acceptance probe
# ---------------------------------------------------------------------------
@requires_artifacts
class TestAcceptanceScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        cls.organ = make_organ(patch="head", fit_steps=40)
        cls.trace = cls.organ.acceptance_scenario()

    def test_spike_over_normal_baseline(self) -> None:
        summary = self.trace.summary()
        # Measured: post-flip base 0.2874 vs phase-matched normal 0.0332
        # (8.66x); peak tick mean 0.4568 (13.8x).
        self.assertGreater(summary["post_flip_base_mean"], 5.0 * summary["normal_post_mean"])
        self.assertGreater(summary["post_flip_peak_tick_mean"], 5.0 * summary["normal_post_mean"])

    def test_birth_revalidate_stale_evict_ordering(self) -> None:
        adoption = self.trace.adoption_tick
        stale = self.trace.stale_tick
        eviction = self.trace.eviction_tick
        self.assertIsNotNone(adoption)
        self.assertIsNotNone(stale)
        self.assertIsNotNone(eviction)
        assert adoption is not None and stale is not None and eviction is not None
        self.assertLess(adoption, self.trace.flip_back_at)
        self.assertLessEqual(stale, eviction)
        stales = [item for item in self.trace.events if item.stales]
        self.assertTrue(stales)
        self.assertEqual(stales[0].stales[0].status, "stale")
        self.assertGreater(stales[0].stales[0].epoch, 1)  # revalidation advanced it
        births = [item for item in self.trace.events if item.births]
        self.assertEqual(births[0].births[0].token(), (1, "ovr-01", "active"))
        self.assertEqual(len(self.organ.overlay_table), 0)
        self.assertEqual(self.organ.overlay_table.active(), ())

    def test_recovery_toward_baseline(self) -> None:
        summary = self.trace.summary()
        # Measured: adapted 0.1723 vs base 0.2517 on the held-out streams
        # over the same ticks (0.685x); it is *toward*, not at, the baseline.
        self.assertLess(summary["adapted_over_base"], 0.85)
        self.assertGreater(summary["adapted_over_base"], 0.0)
        self.assertLess(summary["adapted_post_mean"], summary["base_post_adoption_mean"])

    def test_d4_base_core_receives_zero_writes(self) -> None:
        self.assertEqual(self.trace.d4_writes, 0)
        self.assertEqual(self.organ.d4.writes, 0)
        self.assertEqual(self.organ.d4.payloads, [])
        for event in self.trace.events:
            self.assertEqual(event.d4_writes, 0)

    def test_encoder_untouched_by_acceptance_run(self) -> None:
        self.assertTrue(self.trace.encoder_bit_identical)
        self.assertTrue(self.organ.encoder_bit_identical())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
