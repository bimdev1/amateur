"""Boundary tests for the micro-LeWM that phases 4-5 will consume.

Pins the contract:

* encode/predict/predict_step shapes and acceptance of ``uint8`` frames;
* finite, non-degenerate SIGReg;
* disjoint encoder/predictor parameter sets (overlays adapt the predictor);
* an encoder freeze that is enforced across ``train()`` and gives encoder
  gradients of ``None`` while the predictor still trains;
* checkpoint round-trip in a *subprocess* (fresh interpreter, fresh module
  state) including the frozen flag and config;
* a short overfit sanity check (loss falls on a fixed tiny batch);
* seeded reproducibility of model initialisation and window sampling.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from amateur.physics.data import (
    RolloutBatch,
    generate_rollouts,
    sample_window_batch,
)
from amateur.physics.env import PhysicsEnv, WorldConfig
from amateur.physics.micro_lewm import MicroLeWM, MicroLeWMConfig, SIGReg

REPO = Path(__file__).resolve().parents[1]


def tiny_config(**overrides) -> MicroLeWMConfig:
    values = dict(
        frame_size=64,
        latent_dim=16,
        encoder_widths=(8, 16, 16),
        predictor_hidden=32,
        predictor_layers=1,
        act_emb_dim=4,
        sigreg_knots=9,
        sigreg_projections=32,
        sigreg_weight=0.09,
        lr=3e-3,
        seed=0,
    )
    values.update(overrides)
    return MicroLeWMConfig(**values)


def tiny_rollouts(episodes: int = 4, horizon: int = 8, seed: int = 5) -> RolloutBatch:
    env = PhysicsEnv(WorldConfig(frame_size=64))
    return generate_rollouts(env, episodes, horizon, seed=seed, gravity_sign=1)


class TestSIGReg(unittest.TestCase):
    def test_finite_and_positive(self) -> None:
        sigreg = SIGReg(knots=9, num_proj=32)
        value = sigreg(torch.randn(4, 8, 16))
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(value), 0.0)

    def test_collapsed_latents_are_penalised_more(self) -> None:
        sigreg = SIGReg(knots=9, num_proj=32)
        torch.manual_seed(0)
        spread = sigreg(torch.randn(6, 32, 16))
        collapsed = sigreg(torch.zeros(6, 32, 16))
        self.assertGreater(float(collapsed), float(spread))

    def test_rejects_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            SIGReg(knots=5, num_proj=8)(torch.randn(8, 16))


class TestMicroLeWMModel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = MicroLeWM(tiny_config())

    def test_parameter_budget_and_disjoint_sets(self) -> None:
        counts = self.model.param_counts()
        self.assertLess(counts["total"], 2_000_000)
        self.assertEqual(counts["total"], counts["encoder"] + counts["predictor"])
        encoder_ids = {id(p) for p in self.model.encoder_parameters()}
        predictor_ids = {id(p) for p in self.model.predictor_parameters()}
        self.assertTrue(encoder_ids.isdisjoint(predictor_ids))
        self.assertEqual(len(encoder_ids) + len(predictor_ids), len({id(p) for p in self.model.parameters()}))

    def test_encode_shapes_uint8_and_float(self) -> None:
        frames = torch.randint(0, 256, (2, 4, 3, 64, 64), dtype=torch.uint8)
        latents = self.model.encode(frames)
        self.assertEqual(latents.shape, (2, 4, 16))
        single = self.model.encode(frames[:, 0])
        self.assertEqual(single.shape, (2, 16))
        # numpy channels-last is accepted and normalised identically
        np_frames = frames[0, 0].permute(1, 2, 0).numpy()[None]
        np_latent = self.model.encode(np_frames)
        self.assertTrue(
            torch.allclose(np_latent[0], self.model.encode(frames[0:1, 0].float() / 255.0)[0])
        )

    def test_predict_shapes_and_action_alignment(self) -> None:
        latents = torch.randn(2, 5, 16)
        actions = torch.randint(0, 5, (2, 4))
        prediction = self.model.predict(latents, actions)
        self.assertEqual(prediction.shape, (2, 4, 16))
        # A trailing action is accepted and ignored.
        longer = self.model.predict(latents, torch.randint(0, 5, (2, 5)))
        self.assertTrue(torch.equal(prediction, longer))
        with self.assertRaises(ValueError):
            self.model.predict(latents, torch.randint(0, 5, (2, 2)))
        with self.assertRaises(ValueError):
            self.model.predict(latents, torch.full((2, 4), 9))
        with self.assertRaises(TypeError):
            self.model.predict(latents, torch.rand(2, 4))

    def test_predict_equals_step_by_step(self) -> None:
        latents = torch.randn(2, 5, 16)
        actions = torch.randint(0, 5, (2, 5))
        batched = self.model.predict(latents, actions)
        hidden = self.model.init_hidden(2)
        stepped = []
        for step in range(4):
            z_hat, hidden = self.model.predict_step(latents[:, step], actions[:, step], hidden)
            stepped.append(z_hat)
        self.assertTrue(torch.allclose(batched, torch.stack(stepped, dim=1), atol=1e-6))

    def test_surprise_shapes_and_reduction(self) -> None:
        frames = torch.randint(0, 256, (2, 4, 3, 64, 64), dtype=torch.uint8)
        actions = torch.randint(0, 5, (2, 3))
        per_step = self.model.surprise(frames, actions)
        self.assertEqual(per_step.shape, (2, 3))
        self.assertTrue(torch.isfinite(per_step).all())
        self.assertGreaterEqual(float(per_step.min()), 0.0)
        reduced = self.model.surprise(frames, actions, reduce=True)
        self.assertAlmostEqual(float(reduced), float(per_step.mean()), places=6)

    def test_loss_terms_are_finite(self) -> None:
        frames = torch.randint(0, 256, (2, 5, 3, 64, 64), dtype=torch.uint8)
        actions = torch.randint(0, 5, (2, 4))
        terms = self.model.loss_terms(frames, actions)
        for key in ("loss", "pred_mse", "sigreg"):
            self.assertTrue(torch.isfinite(terms[key]), key)
        expected = terms["pred_mse"].detach() + self.model.config.sigreg_weight * terms["sigreg"].detach()
        self.assertAlmostEqual(float(terms["loss"].detach()), float(expected), places=5)

    def test_seeded_initialisation_is_reproducible(self) -> None:
        first = MicroLeWM(tiny_config(seed=3))
        second = MicroLeWM(tiny_config(seed=3))
        other = MicroLeWM(tiny_config(seed=4))
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))
        differences = [
            not torch.equal(a, b) for a, b in zip(first.parameters(), other.parameters())
        ]
        self.assertTrue(any(differences))


class TestFreezing(unittest.TestCase):
    def test_freeze_disables_encoder_gradients_but_not_predictor(self) -> None:
        model = MicroLeWM(tiny_config())
        model.freeze_encoder()
        self.assertTrue(model.encoder_frozen)
        self.assertTrue(all(not p.requires_grad for p in model.encoder_parameters()))
        self.assertTrue(all(p.requires_grad for p in model.predictor_parameters()))

        # Explicit .train() must not put the frozen encoder back into training.
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.predictor.training)

        frames = torch.randint(0, 256, (2, 5, 3, 64, 64), dtype=torch.uint8)
        actions = torch.randint(0, 5, (2, 4))
        model.configure_optimizer()
        terms = model.loss_terms(frames, actions)
        terms["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.encoder_parameters()))
        self.assertTrue(any(p.grad is not None for p in model.predictor_parameters()))
        model.zero_grad(set_to_none=True)

    def test_predictor_only_optimizer_excludes_encoder(self) -> None:
        model = MicroLeWM(tiny_config())
        optimizer = model.predictor_optimizer()
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        encoder_ids = {id(p) for p in model.encoder_parameters()}
        predictor_ids = {id(p) for p in model.predictor_parameters()}
        self.assertTrue(optimized.isdisjoint(encoder_ids))
        self.assertEqual(optimized, predictor_ids)

    def test_frozen_model_still_trains_predictor(self) -> None:
        model = MicroLeWM(tiny_config(lr=1e-2))
        rollouts = tiny_rollouts()
        frames, actions = sample_window_batch(
            rollouts, min(6, rollouts.horizon), 4, np.random.default_rng(0)
        )
        model.fit(rollouts, steps=15, seq_len=min(6, rollouts.horizon), batch_size=4, log=False)
        self.assertTrue(model.encoder_frozen)
        model.configure_optimizer()
        before = [p.detach().clone() for p in model.predictor_parameters()]
        for _ in range(3):
            model.train_step(frames, actions)
        after = list(model.predictor_parameters())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))


class TestOverfit(unittest.TestCase):
    def test_loss_decreases_on_a_fixed_tiny_batch(self) -> None:
        # A default-size micro-LeWM (still ~0.52M parameters) at 64x64 for
        # speed.  The earlier 4-window/32-projection variant was not a valid
        # overfit probe: with so few samples the SIGReg statistic starts at a
        # near-flat point and its gradient (~1e-4) loses to the prediction
        # collapse force, so the loss barely moves.  The check below is
        # therefore stricter (50% drop required) and additionally requires
        # the latent distribution to actually spread.
        model = MicroLeWM(MicroLeWMConfig(frame_size=64, seed=0))
        rollouts = generate_rollouts(
            PhysicsEnv(WorldConfig(frame_size=64)), 6, 10, seed=5, gravity_sign=1
        )
        frames, actions = sample_window_batch(rollouts, 8, 8, np.random.default_rng(0))
        model.configure_optimizer()
        with torch.no_grad():
            initial = {key: float(value) for key, value in model.loss_terms(frames, actions).items()}
            initial_std = float(model.encode(frames).std())
        for _ in range(80):
            model.train_step(frames, actions)
        with torch.no_grad():
            final = {key: float(value) for key, value in model.loss_terms(frames, actions).items()}
            final_std = float(model.encode(frames).std())
        self.assertLess(final["loss"], 0.5 * initial["loss"])
        self.assertGreater(final_std, 0.2)
        self.assertGreater(final_std, 5.0 * initial_std)
        self.assertTrue(np.isfinite(final["loss"]))

    def test_window_sampling_is_seeded(self) -> None:
        rollouts = tiny_rollouts()
        first = sample_window_batch(rollouts, 6, 4, np.random.default_rng(1))
        second = sample_window_batch(rollouts, 6, 4, np.random.default_rng(1))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_save_load_round_trip_in_subprocess(self) -> None:
        model = MicroLeWM(tiny_config(seed=7))
        model.fit(tiny_rollouts(), steps=5, seq_len=6, batch_size=4, log=False)
        self.assertTrue(model.encoder_frozen)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            inputs = Path(tmp) / "inputs.npz"
            outputs = Path(tmp) / "outputs.npz"
            model.save(checkpoint, extra={"purpose": "round-trip"})
            rng = np.random.default_rng(0)
            frames = rng.integers(0, 256, size=(2, 3, 64, 64, 3), dtype=np.uint8)
            np.savez(inputs, frames=frames)
            expected = model.encode(frames).detach().numpy()

            script = (
                "import json, sys\n"
                "import numpy as np, torch\n"
                "from amateur.physics.micro_lewm import MicroLeWM\n"
                "model = MicroLeWM.load(sys.argv[1])\n"
                "data = np.load(sys.argv[2])\n"
                "with torch.no_grad():\n"
                "    latent = model.encode(data['frames']).numpy()\n"
                "np.savez(sys.argv[3], latent=latent)\n"
                "print(json.dumps({'frozen': model.encoder_frozen, "
                "'config': model.config.to_dict(), 'meta': model.checkpoint_meta}))\n"
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
            completed = subprocess.run(
                [sys.executable, "-c", script, str(checkpoint), str(inputs), str(outputs)],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=180,
            )
            reported = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertTrue(reported["frozen"])
            self.assertEqual(MicroLeWMConfig.from_dict(reported["config"]), model.config)
            self.assertEqual(reported["meta"], {"purpose": "round-trip"})
            reloaded = np.load(outputs)["latent"]
            self.assertTrue(np.allclose(expected, reloaded, atol=1e-5))

    def test_load_rejects_foreign_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            torch.save({"format": "not-ours"}, path)
            with self.assertRaises(ValueError):
                MicroLeWM.load(path)


if __name__ == "__main__":
    unittest.main()
