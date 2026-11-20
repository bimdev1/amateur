"""Tests for the Round 9a evidence-bounded capacity budget.

The user's binding condition: patch size/rank scales with the evidence that
justifies it ("four frames earns a sentence, not a manifesto").  These tests
pin the derivation itself -- pure arithmetic, no model needed:

* ``K <= gamma * N * D`` with ``gamma = 5.0``, ``D = 64`` and ``N`` the real
  evidence transitions (never the requested window or the motif count);
* the low-rank last-head-layer patch ``K(r) = 192 r + 64``,
  ``nbytes(r) = 4 K(r)``, ``r_max = min(floor((gamma N D - D) / 192), 64)``;
* the contract's worked examples: ``N=4 -> r=6, K=1,216``,
  ``N=16 -> r=26, K=5,056``;
* the dense forms are inadmissible at every reachable evidence size: the
  dense head+norm patch (34,240 params) and the full predictor (210,920
  params) exceed the rank-ceilinged patch class;
* ``N`` small enough to leave no rank is "no candidate", not a bias-only
  patch.
"""
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from amateur.physics.capacity import (
    DEFAULT_GAMMA,
    DEFAULT_HEAD_IN,
    DEFAULT_LATENT_DIM,
    EvidenceBudget,
)

#: The dense head+norm patch measured by the phase-4-5b fit path.
DENSE_HEAD_NORM_PARAMS = 34_240
#: The dense full-predictor patch measured by the phase-4-5b fit path.
DENSE_FULL_PARAMS = 210_920


class TestDerivation(unittest.TestCase):
    def test_contract_examples(self) -> None:
        """The two worked examples in the binding contract."""
        four = EvidenceBudget(4)
        self.assertEqual(four.max_rank, 6)
        self.assertEqual(four.max_params, 1_216)
        self.assertEqual(four.max_nbytes, 4_864)
        self.assertEqual(four.capacity_params, 5.0 * 4 * 64)

        sixteen = EvidenceBudget(16)
        self.assertEqual(sixteen.max_rank, 26)
        self.assertEqual(sixteen.max_params, 5_056)
        self.assertEqual(sixteen.max_nbytes, 20_224)
        self.assertEqual(sixteen.capacity_params, 5.0 * 16 * 64)

    def test_rank_formula(self) -> None:
        for n in (0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 63, 64, 65, 256, 1024):
            budget = EvidenceBudget(n)
            expected = min(64, max(0, (5 * n * 64 - 64) // 192))
            self.assertEqual(budget.max_rank, expected, msg=f"N={n}")

    def test_parameter_bound_holds(self) -> None:
        for n in (1, 2, 4, 8, 16, 32, 64, 256):
            budget = EvidenceBudget(n)
            self.assertGreaterEqual(budget.max_rank, 1, msg=f"N={n}")
            self.assertLessEqual(
                budget.max_params, budget.capacity_params, msg=f"N={n}"
            )
            self.assertTrue(budget.admit_params(budget.max_params), msg=f"N={n}")
            self.assertEqual(
                budget.max_params, 192 * budget.max_rank + 64, msg=f"N={n}"
            )
            self.assertEqual(
                budget.max_nbytes, 4 * budget.max_params, msg=f"N={n}"
            )

    def test_rank_is_monotone_and_capped(self) -> None:
        ranks = [EvidenceBudget(n).max_rank for n in range(0, 300)]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(max(ranks), 64)
        self.assertEqual(EvidenceBudget(10_000).max_rank, 64)

    def test_no_rank_means_no_candidate(self) -> None:
        empty = EvidenceBudget(0)
        self.assertEqual(empty.max_rank, 0)
        self.assertEqual(empty.max_params, 0)
        self.assertEqual(empty.max_nbytes, 0)
        self.assertFalse(empty.admit_params(1))
        # A bias-only patch is not a motif adaptation.
        one = EvidenceBudget(1)
        self.assertEqual(one.max_rank, 1)
        self.assertEqual(one.max_params, 256)

    def test_latent_and_head_widths_are_parameters(self) -> None:
        self.assertEqual(DEFAULT_LATENT_DIM, 64)
        self.assertEqual(DEFAULT_HEAD_IN, 128)
        self.assertEqual(DEFAULT_GAMMA, 5.0)
        budget = EvidenceBudget(16, latent_dim=32, head_in=256)
        self.assertEqual(budget.per_rank_params, 288)
        self.assertEqual(budget.max_params, 288 * budget.max_rank + 32)

    def test_budget_is_frozen(self) -> None:
        budget = EvidenceBudget(16)
        with self.assertRaises(FrozenInstanceError):
            budget.n_transitions = 17  # type: ignore[misc]

    def test_invalid_budgets_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceBudget(-1)
        with self.assertRaises(ValueError):
            EvidenceBudget(1, latent_dim=0)
        with self.assertRaises(ValueError):
            EvidenceBudget(1, gamma=0.0)
        with self.assertRaises(ValueError):
            EvidenceBudget(1, gamma=float("nan"))
        with self.assertRaises(ValueError):
            EvidenceBudget(1).params_for_rank(-1)
        with self.assertRaises(ValueError):
            EvidenceBudget(1).admit_params(-1)


class TestAdmission(unittest.TestCase):
    def test_dense_forms_are_inadmissible(self) -> None:
        """Both dense patches are a different patch class, at every N."""
        for n in (4, 16, 64, 256, 1024, 100_000):
            budget = EvidenceBudget(n)
            self.assertFalse(
                budget.admit_params(DENSE_HEAD_NORM_PARAMS), msg=f"head+norm N={n}"
            )
            self.assertFalse(
                budget.admit_params(DENSE_FULL_PARAMS), msg=f"full N={n}"
            )

    def test_admission_tracks_the_rank_ceiling(self) -> None:
        budget = EvidenceBudget(16)
        self.assertTrue(budget.admit_rank(26))
        self.assertFalse(budget.admit_rank(27))
        self.assertFalse(budget.admit_rank(64))
        self.assertTrue(budget.admit_params(5_056))
        self.assertFalse(budget.admit_params(5_057))
        self.assertFalse(budget.admit_params(5_120))  # K(27) > r_max but <= capacity

    def test_describe_is_plain_numbers(self) -> None:
        described = EvidenceBudget(16).describe()
        self.assertEqual(described["n_transitions"], 16.0)
        self.assertEqual(described["max_rank"], 26.0)
        self.assertEqual(described["max_params"], 5_056.0)
        self.assertEqual(described["capacity_params"], 5_120.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
