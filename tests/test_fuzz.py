"""Adversarial fuzz sweep over the reject-with-charge channel.

Distribution (plumbing, not spec -- see amateur/fuzz.py): seeds 0..11,
1000 ticks per seed, injection multiplicities k in 1..8 on the two E1
channels. run_fuzz asserts per tick, per flow: identity A
(attempted = accepted + refused), identity B (accepted = consumed +
delta-sunk + delta-depth), INV-BOUNDS intra-tick (depth <= 1 at every
event boundary via the O(1) max_depth monitor), the schedule-shift pin
(flows outside the plan keep exactly one attempt), and scalar-recurrence
tracking (multiplicity cannot alter dynamics). Here we pin the
aggregate shape and cross-seed determinism.
"""
from __future__ import annotations

import unittest

from amateur.fuzz import run_fuzz
from amateur.ledger import FLOW_LEDGER

SEEDS = range(12)
TICKS = 1000
MAX_K = 8
FLOW_COUNT = len(FLOW_LEDGER)


class TestFuzzSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = [run_fuzz(seed, TICKS, MAX_K) for seed in SEEDS]

    def test_every_seed_survives_the_full_battery(self) -> None:
        for result in self.results:
            self.assertEqual(result.ticks, TICKS, msg=f"seed {result.seed}")
            self.assertEqual(result.max_k, MAX_K, msg=f"seed {result.seed}")

    def test_identity_a_over_each_run(self) -> None:
        for result in self.results:
            self.assertEqual(
                result.attempted,
                result.accepted + result.refused,
                msg=f"seed {result.seed}",
            )

    def test_aggregate_bounds_and_warmup_only_misses(self) -> None:
        for result in self.results:
            self.assertGreaterEqual(
                result.attempted, FLOW_COUNT * TICKS, msg=f"seed {result.seed}"
            )
            self.assertLessEqual(
                result.attempted,
                (2 * MAX_K + FLOW_COUNT - 2) * TICKS,
                msg=f"seed {result.seed}",
            )
            self.assertGreaterEqual(
                result.accepted, (FLOW_COUNT - 2) * TICKS, msg=f"seed {result.seed}"
            )
            self.assertLessEqual(
                result.refused, 2 * (MAX_K - 1) * TICKS, msg=f"seed {result.seed}"
            )
            self.assertEqual(
                result.missed, 2, msg=f"seed {result.seed}: misses grew"
            )
            self.assertTrue(
                all(1 <= k <= MAX_K for k in result.peak_k.values()),
                msg=f"seed {result.seed}",
            )

    def test_adversarial_pressure_actually_refuses(self) -> None:
        self.assertTrue(
            all(result.refused > 0 for result in self.results),
            msg="multiplicity pressure must register as excess charge",
        )

    def test_same_seed_is_deterministic(self) -> None:
        again = run_fuzz(7, TICKS, MAX_K)
        self.assertEqual(again, self.results[7])


if __name__ == "__main__":
    unittest.main()
