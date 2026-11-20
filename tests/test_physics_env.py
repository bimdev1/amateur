"""Boundary tests for the flippable 2-D physics environment.

These tests pin the environment contract that phases 4-5 consume:

* seeded determinism (reset and step);
* a fixed frame shape/dtype/range;
* a gravity flip that is a *dynamics* change -- trajectories diverge,
  while two worlds in the same physical state render identical frames;
* bounded positions/velocities over long random rollouts;
* strict action validation and state-copy discipline.
"""
from __future__ import annotations

import unittest

import numpy as np

from amateur.physics.env import ACTIONS, ACTION_DELTAS, PhysicsEnv, WorldConfig


def _env(**overrides) -> PhysicsEnv:
    base = {"frame_size": 64}
    base.update(overrides)
    return PhysicsEnv(WorldConfig(**base))


class TestPhysicsEnvFrames(unittest.TestCase):
    def test_frame_shape_dtype_and_range(self) -> None:
        env = _env()
        frame = env.reset(seed=0)
        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreaterEqual(int(frame.min()), 0)
        self.assertLessEqual(int(frame.max()), 255)
        for action in range(env.n_actions):
            frame = env.step(action)
            self.assertEqual(frame.shape, (64, 64, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_action_space_matches_deltas(self) -> None:
        self.assertEqual(len(ACTIONS), len(ACTION_DELTAS))
        self.assertEqual(ACTIONS, ("noop", "left", "right", "down", "up"))
        self.assertTrue(np.array_equal(ACTION_DELTAS[0], [0.0, 0.0]))
        self.assertLess(ACTION_DELTAS[1][0], 0.0)  # left
        self.assertGreater(ACTION_DELTAS[2][0], 0.0)  # right
        self.assertLess(ACTION_DELTAS[3][1], 0.0)  # down
        self.assertGreater(ACTION_DELTAS[4][1], 0.0)  # up


class TestPhysicsEnvDeterminism(unittest.TestCase):
    def test_reset_is_seeded(self) -> None:
        left = _env().reset(seed=7)
        right = _env().reset(seed=7)
        self.assertTrue(np.array_equal(left, right))

    def test_reset_seeds_differ(self) -> None:
        left = _env()
        right = _env()
        left.reset(seed=7)
        right.reset(seed=8)
        self.assertFalse(np.array_equal(left.state()["agent_pos"], right.state()["agent_pos"]))

    def test_step_sequence_is_seeded(self) -> None:
        actions = [1, 4, 2, 0, 3] * 6
        one = _env()
        two = _env()
        one.reset(seed=11)
        two.reset(seed=11)
        for action in actions:
            frame_one = one.step(action)
            frame_two = two.step(action)
            self.assertTrue(np.array_equal(frame_one, frame_two))
        self.assertEqual(one.state()["tick"], two.state()["tick"])
        for key in ("agent_pos", "agent_vel", "object_pos", "object_vel"):
            self.assertTrue(np.array_equal(one.state()[key], two.state()[key]), key)

    def test_reset_restores_configured_sign(self) -> None:
        env = _env(gravity_sign=-1)
        self.assertEqual(env.reset(seed=0).shape, (64, 64, 3))
        self.assertEqual(env.gravity_sign, -1)
        env.flip_gravity()
        self.assertEqual(env.gravity_sign, 1)
        env.reset(seed=0)
        self.assertEqual(env.gravity_sign, -1)


class TestGravityFlip(unittest.TestCase):
    def test_gravity_sign_convention(self) -> None:
        # With zero initial velocity, +1 pulls down (y decreases) and -1 pulls
        # up (y increases).
        config = dict(frame_size=64, initial_speed=0.0)
        down = _env(**config)
        up = _env(**config)
        down.reset(seed=3, gravity_sign=1)
        up.reset(seed=3, gravity_sign=-1)
        y_down_before = float(down.state()["agent_pos"][1])
        y_up_before = float(up.state()["agent_pos"][1])
        down.step(0)
        up.step(0)
        self.assertLess(float(down.state()["agent_pos"][1]), y_down_before)
        self.assertGreater(float(up.state()["agent_pos"][1]), y_up_before)
        self.assertEqual(down.state()["signed_gravity"], -down.config.gravity)
        self.assertEqual(up.state()["signed_gravity"], up.config.gravity)

    def test_flip_changes_trajectory_quantitatively(self) -> None:
        normal = _env()
        flipped = _env()
        normal.reset(seed=11)
        flipped.reset(seed=11)
        # Identical pre-flip history, then diverge only because of the flip.
        for _ in range(5):
            normal.step(4)
            flipped.step(4)
        flipped.flip_gravity()
        for _ in range(12):
            normal.step(4)
            flipped.step(4)
        self.assertEqual(flipped.state()["gravity_sign"], -1)
        self.assertEqual(flipped.state()["gravity_flips"], 1)
        object_gap = float(
            np.linalg.norm(normal.state()["object_pos"] - flipped.state()["object_pos"])
        )
        agent_gap = float(
            np.linalg.norm(normal.state()["agent_pos"] - flipped.state()["agent_pos"])
        )
        self.assertGreater(object_gap, 0.05)
        self.assertGreater(agent_gap, 0.02)
        self.assertFalse(np.array_equal(normal.render(), flipped.render()))

    def test_flip_does_not_change_rendering_of_a_fixed_state(self) -> None:
        # A flip must be a dynamics change, not a visual trick: the same
        # physical state renders byte-identically under either sign.
        env = _env()
        env.reset(seed=5)
        for _ in range(4):
            env.step(1)
        before = env.render()
        positions = {key: env.state()[key].copy() for key in ("agent_pos", "object_pos")}
        env.flip_gravity()
        after = env.render()
        self.assertTrue(np.array_equal(before, after))
        for key, value in positions.items():
            self.assertTrue(np.array_equal(value, env.state()[key]), key)
        self.assertEqual(env.state()["tick"], 4)


class TestPhysicsEnvBounds(unittest.TestCase):
    def test_long_rollout_stays_bounded(self) -> None:
        env = _env()
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        for _ in range(300):
            frame = env.step(int(rng.integers(0, env.n_actions)))
            state = env.state()
            for key in ("agent_pos", "object_pos"):
                pos = state[key]
                self.assertTrue(np.all(pos >= 0.0) and np.all(pos <= 1.0))
            for key in ("agent_vel", "object_vel"):
                speed = float(np.linalg.norm(state[key]))
                self.assertLessEqual(speed, env.config.max_speed + 1e-9)
            self.assertGreaterEqual(int(frame.min()), 0)
            self.assertLessEqual(int(frame.max()), 255)

    def test_actions_are_validated(self) -> None:
        env = _env()
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step(5)
        with self.assertRaises(ValueError):
            env.step(-1)
        with self.assertRaises(TypeError):
            env.step(1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            env.step(True)  # type: ignore[arg-type]

    def test_state_is_a_defensive_copy(self) -> None:
        env = _env()
        env.reset(seed=1)
        state = env.state()
        state["agent_pos"][:] = 0.0
        self.assertFalse(np.array_equal(state["agent_pos"], env.state()["agent_pos"]))

    def test_state_contract(self) -> None:
        env = _env()
        env.reset(seed=2)
        state = env.state()
        for key in (
            "tick",
            "gravity",
            "gravity_sign",
            "signed_gravity",
            "gravity_flips",
            "agent_pos",
            "agent_vel",
            "object_pos",
            "object_vel",
            "agent_radius",
            "object_radius",
            "frame_size",
            "last_action",
        ):
            self.assertIn(key, state)
        self.assertEqual(state["agent_pos"].shape, (2,))
        self.assertEqual(state["object_vel"].shape, (2,))
        self.assertEqual(state["frame_size"], 64)
        self.assertIn(state["gravity_sign"], (-1, 1))


if __name__ == "__main__":
    unittest.main()
