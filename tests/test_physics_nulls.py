"""Round 12a §6: the pre-registered null controls, executor path.

Pins the pre-registered null expectation: **zero promotions for every null
with the gain floor on**, through the identical pipeline (same trigger,
organ, executor, evidence set and gate).  The four nulls are:

* ``zero``     -- patch factors and bias exactly zero (neutral by
  construction: shadow scoring must return ``adapted == base`` exactly);
* ``random``   -- random factors at the fitted norms (same ``K``);
* ``shuffled`` -- the standard fit with the fit-window targets permuted;
* ``no_flip``  -- the registered environment with no gravity flip at all.

The secondary diagnostic (clearly labelled, never a gate) is the same four
nulls with the floor ablated: it shows whether the pre-floor gate would have
admitted noise.  Measured this round: **no null promotes in either mode**,
and the zero candidate is never even qualified (its ratio is exactly 1.0,
so it cannot satisfy ``ratio < 1.0``).

Every number here is a measurement of this round's tree.
"""
from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path

import torch

from amateur.physics.nulls import (
    NULL_NAMES,
    NULL_PATCH_MODES,
    null_suite,
    run_null,
    to_jsonable,
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


# ---------------------------------------------------------------------------
# One measured null suite shared by every read-only assertion.
# ---------------------------------------------------------------------------
_SUITE: dict[str, dict] = {}


def suite() -> dict[str, dict]:
    if not _SUITE:
        _SUITE.update(null_suite())
    return _SUITE


@requires_artifacts
class TestNullSuiteShape(unittest.TestCase):
    def test_the_suite_covers_every_null_in_both_gate_modes(self) -> None:
        rows = suite()
        expected = {
            f"null_{name}{suffix}"
            for name in NULL_NAMES
            for suffix in ("", "_ablated")
        }
        self.assertEqual(set(rows), expected)
        for key, row in rows.items():
            self.assertIn(row["gain_floor"], {"on", "ablate"}, msg=key)
            self.assertEqual(row["trigger"], "registered", msg=key)


@requires_artifacts
class TestNullPromotions(unittest.TestCase):
    """The pre-registered expectation: zero promotions with the floor on."""

    def test_no_null_promotes_with_the_gate_floor_on(self) -> None:
        for name in NULL_NAMES:
            row = suite()[f"null_{name}"]
            with self.subTest(null=name):
                self.assertEqual(row["promotions"], [])
                self.assertEqual(row["admission"]["n"], 0)
                self.assertEqual(row["admission"]["value"], 0)
                self.assertTrue(row["admission"]["passed"])
                self.assertEqual(row["d4"]["sediments"], [])

    def test_zero_promotions_are_not_vacuous_for_the_candidate_nulls(self) -> None:
        """Each of zero/random/shuffled actually births and accumulates evidence."""
        for name in NULL_PATCH_MODES:
            row = suite()[f"null_{name}"]
            with self.subTest(null=name):
                self.assertTrue(row["births"], msg=f"{name} never birthed")
                self.assertEqual(row["births"][0][1], 12)  # the registered spike
                self.assertTrue(row["evidence"], msg=f"{name} has no evidence")
                self.assertEqual(len(row["evidence"]), 24)

    def test_ablated_nulls_are_reported_side_by_side(self) -> None:
        """Secondary diagnostic (prereg §6): the old gate also admitted nothing.

        Measured: no null promotes even with the floor ablated, on this
        trace.  The prereg conditional ("if a null promotes there while the
        real candidate also promotes, the old pass was noise") is therefore
        not triggered: the real candidate does not promote either.
        """
        for name in NULL_NAMES:
            on = suite()[f"null_{name}"]
            off = suite()[f"null_{name}_ablated"]
            with self.subTest(null=name):
                self.assertEqual(on["gain_floor"], "on")
                self.assertEqual(off["gain_floor"], "ablate")
                self.assertEqual(on["promotions"], [])
                self.assertEqual(off["promotions"], [])


@requires_artifacts
class TestZeroNull(unittest.TestCase):
    """``zero`` is neutral by construction: adapted must equal base exactly."""

    def test_zero_patch_scores_exactly_like_the_base(self) -> None:
        row = suite()["null_zero"]
        self.assertTrue(row["evidence"])
        for tick, adapted, base in row["evidence"]:
            self.assertEqual(adapted, base, msg=f"t{tick}")
        self.assertEqual(row["adaptation"]["primary"], 1.0)
        self.assertEqual(row["adaptation"]["secondary_mean_of_ratios"], 1.0)
        verdict = row["candidate_verdict"]
        self.assertIsNotNone(verdict)
        assert verdict is not None
        self.assertEqual(verdict["dbar"], 0.0)
        self.assertEqual(verdict["gain_bits"], 0.0)
        self.assertFalse(verdict["admitted"])
        self.assertEqual(verdict["reason"], "harmful")

    def test_zero_patch_gate_refusal_is_harmful(self) -> None:
        row = suite()["null_zero"]
        refusals = row["floor_refusals"]
        self.assertTrue(refusals)
        self.assertTrue(
            all(item["reason"] in {"harmful", "uneconomic"} for item in refusals)
        )
        self.assertTrue(all(item["dbar"] <= 0.0 for item in refusals))
        # Neutrality also means it can never satisfy the strict qualified test.
        self.assertEqual(row["qualified_floor_refusals"], [])
        self.assertEqual(row["promotions"], [])


@requires_artifacts
class TestNullIdentities(unittest.TestCase):
    def test_random_and_shuffled_are_deterministic(self) -> None:
        for name in ("random", "shuffled"):
            row = suite()[f"null_{name}"]
            again = run_null(name)
            with self.subTest(null=name):
                self.assertEqual(row["evidence"], again["evidence"])
                self.assertEqual(row["candidate_verdict"], again["candidate_verdict"])
                self.assertEqual(row["births"], again["births"])

    def test_the_birth_decision_is_mode_invariant(self) -> None:
        """Ratification: the no-improvement check runs on the FITTED patch.

        The null substance is substituted only after that check, so every
        mode births at the same tick with the same rank/K.  Measured: t12,
        rank 19, K=3,712 for all three candidate nulls.
        """
        fitted = suite()["null_zero"]["candidate_fit"]["record"]
        self.assertIsNotNone(fitted)
        assert fitted is not None
        for name in NULL_PATCH_MODES:
            row = suite()[f"null_{name}"]
            with self.subTest(null=name):
                record = row["candidate_fit"]["record"]
                self.assertIsNotNone(record)
                assert record is not None
                self.assertEqual(record["rank"], 19)
                self.assertEqual(record["param_count"], 3_712)
                self.assertEqual(record["budget_n"], 12)
                self.assertEqual(record["param_count"], fitted["param_count"])


@requires_artifacts
class TestNoFlipNull(unittest.TestCase):
    """``no_flip`` is reported honestly: it is vacuous on this trace."""

    def test_no_flip_is_vacuous_and_reported(self) -> None:
        for suffix in ("", "_ablated"):
            row = suite()[f"null_no_flip{suffix}"]
            with self.subTest(mode=row["gain_floor"]):
                self.assertEqual(row["births"], [])
                self.assertEqual(row["evidence"], [])
                self.assertEqual(row["promotions"], [])
                self.assertEqual(row["admission"]["n"], 0)
                self.assertEqual(row["admission"]["value"], 0)
                self.assertEqual(row["adaptation"]["n"], 0)
                self.assertTrue(
                    math.isnan(row["adaptation"]["primary"])
                )
                # No spike means no candidate and no floor reading at all.
                self.assertIsNone(row["candidate_verdict"])
                self.assertEqual(row["floor_refusals"], [])


class TestResultsJsonShape(unittest.TestCase):
    """The artifact coercion is strict JSON (finite numbers only)."""

    def test_to_jsonable_turns_tuples_and_nan_into_json_safe_values(self) -> None:
        payload = {
            "tick_sets": {"adaptation": (13, 14, 15)},
            "primary": float("nan"),
            "secondary": float("inf"),
            "nested": [{"ratio": 0.5}, (1, 2)],
        }
        coerced = to_jsonable(payload)
        self.assertEqual(coerced["tick_sets"]["adaptation"], [13, 14, 15])
        self.assertIsNone(coerced["primary"])
        self.assertIsNone(coerced["secondary"])
        self.assertEqual(coerced["nested"], [{"ratio": 0.5}, [1, 2]])
        text = json.dumps(coerced, allow_nan=False)
        self.assertIn("tick_sets", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
