"""Lean/Python lockstep: the checked-in Lean model must be the Python one.

Two directions:
1. Table parity — the `opsOf` arms parsed out of formal/DryRun.lean are
   exactly amateur.dryrun.flow_ops() (apart from the deliberate
   absorb -> consume compilation), and the Lean profile markers name the
   same cycle/sink flows.
2. Independent replay — the scalar recurrence run over the parsed Lean
   table alone (no amateur.dryrun machinery) reproduces the master
   invariants empirically for 64 ticks; the theorems in DryRun.lean prove
   the same shape for all ticks.

When a lean binary is available (env LEAN_BIN), the theorem file itself is
compiled and must succeed with no `sorry`.
"""
from __future__ import annotations

import os
import re
import subprocess
import unittest
from collections import Counter
from pathlib import Path

from amateur.dryrun import CYCLE_FLOWS, FLOW_OPS, SINK_FLOWS
from amateur.ledger import FLOW_LEDGER

REPO = Path(__file__).resolve().parents[1]
FORMAL = REPO / "formal" / "DryRun.lean"
LEAN_BIN = os.environ.get("LEAN_BIN", "/tmp/lean4/lean-4.21.0-linux/bin/lean")

_LEAN_TO_WORD = {"Op.emit": "emit", "Op.consume": "consume", "Op.sink": "sink"}


def parse_ops_of(text: str) -> dict[int, tuple[str, ...]]:
    section = text.split("def opsOf", 1)[1].split("| _ => []", 1)[0]
    arms: dict[int, tuple[str, ...]] = {}
    for match in re.finditer(r"^\s*\| (\d+) => \[([^\]]*)\]", section, re.M):
        words = tuple(
            _LEAN_TO_WORD[w.strip()]
            for w in match.group(2).split(",")
            if w.strip()
        )
        arms[int(match.group(1))] = words
    return arms


def step(ops: tuple[str, ...], box: int) -> tuple[int, int, int, int, int, int]:
    """Executor-free replay of the reject-with-charge channel semantics.

    Returns (box, missed, refused, sunk, accepted, consumed) -- the same
    accounting as amateur.dryrun.step_flow, re-implemented here over the
    parsed Lean table only.
    """
    missed = refused = sunk = accepted = consumed = 0
    for op in ops:
        if op == "emit":
            if box == 0:
                box, accepted = 1, accepted + 1
            else:
                refused += 1
        elif op == "consume":
            if box > 0:
                box, consumed = box - 1, consumed + 1
            else:
                missed += 1
        elif op == "sink":
            sunk += box
            box = 0
        else:
            raise ValueError(f"unknown op word {op!r}")
    return box, missed, refused, sunk, accepted, consumed


class TestTableParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arms = parse_ops_of(FORMAL.read_text())

    def test_arms_cover_exactly_the_ledger(self) -> None:
        self.assertEqual(
            sorted(self.arms), sorted(f.fid for f in FLOW_LEDGER)
        )

    def test_tail_arm_discards_meta_indices(self) -> None:
        self.assertRegex(FORMAL.read_text(), r"\|\s*_\s*=>\s*\[\]")

    def test_arms_match_flow_ops_with_absorb_compiled(self) -> None:
        for fid, lean_ops in self.arms.items():
            py_ops = FLOW_OPS[fid]
            self.assertEqual(len(lean_ops), len(py_ops), msg=f"#{fid}")
            for lean_op, py_op in zip(lean_ops, py_ops):
                expected = "consume" if py_op == "absorb" else py_op
                self.assertEqual(lean_op, expected, msg=f"#{fid}")

    def test_arms_satisfy_liveness_and_sink_split(self) -> None:
        for fid, ops in self.arms.items():
            self.assertEqual(
                ops.count("emit"), 1, msg=f"#{fid} liveness"
            )
            expected_consume = 0 if fid in SINK_FLOWS else 1
            self.assertEqual(
                ops.count("consume"), expected_consume, msg=f"#{fid}"
            )

    def test_profile_markers_name_the_cycle_and_sink_flows(self) -> None:
        text = FORMAL.read_text()
        fbox = re.search(r"def fbox.*", text).group(0)
        for fid in CYCLE_FLOWS:
            self.assertIn(f"i = {fid}", fbox, msg=f"fbox missing cycle #{fid}")
        fsinkone = re.search(r"def fsinkone.*", text).group(0)
        for fid in SINK_FLOWS:
            self.assertTrue(
                re.search(rf"i = {fid}\b", fsinkone),
                msg=f"fsinkone missing sink #{fid}",
            )
        for fid in (f.fid for f in FLOW_LEDGER if f.fid not in CYCLE_FLOWS):
            self.assertIsNone(
                re.search(rf"i = {fid}\b", fbox),
                msg=f"fbox must not mark #{fid}",
            )

    def test_scalar_replay_from_parsed_lean_table(self) -> None:
        boxes = {fid: 0 for fid in self.arms}
        missed: Counter[int] = Counter()
        refused: Counter[int] = Counter()
        sunk: Counter[int] = Counter()
        accepted: Counter[int] = Counter()
        for tick in range(1, 65):
            for fid, ops in self.arms.items():
                (
                    boxes[fid],
                    missed_delta,
                    refused_delta,
                    sunk_delta,
                    accepted_delta,
                    _consumed,
                ) = step(ops, boxes[fid])
                missed[fid] += missed_delta
                refused[fid] += refused_delta
                sunk[fid] += sunk_delta
                accepted[fid] += accepted_delta
                self.assertEqual(
                    accepted_delta + refused_delta,
                    ops.count("emit"),
                    msg=f"tick {tick}: identity A at #{fid}",
                )
            self.assertTrue(
                all(depth <= 1 for depth in boxes.values()),
                msg=f"tick {tick}: box overflow",
            )
            self.assertEqual(
                {fid: d for fid, d in boxes.items() if d},
                {10: 1, 17: 1},
                msg=f"tick {tick}: steady-state depth profile",
            )
            self.assertEqual(
                {fid: n for fid, n in missed.items() if n},
                {10: 1, 17: 1},
                msg=f"tick {tick}: miss profile",
            )
            self.assertEqual(
                {fid: n for fid, n in refused.items() if n},
                {},
                msg=f"tick {tick}: canonical refusal",
            )
            expected_sunk = {fid: tick for fid in SINK_FLOWS}
            self.assertEqual(
                {fid: n for fid, n in sunk.items() if n},
                expected_sunk,
                msg=f"tick {tick}: sink drain profile",
            )


@unittest.skipIf(not os.path.exists(LEAN_BIN), "lean binary not installed")
class TestLeanCompiles(unittest.TestCase):
    def test_dryrun_lean_compiles_with_no_sorry(self) -> None:
        proc = subprocess.run(
            [LEAN_BIN, str(FORMAL)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        self.assertNotIn("sorry", output)

    def test_master_theorems_present(self) -> None:
        text = FORMAL.read_text()
        for name in (
            "boxes_bounded",
            "boxes_master",
            "misses_master",
            "refused_master",
            "sunk_master",
            "applyOp_attempt",
            "fold_attempt",
            "applyOp_event",
            "fold_event",
            "fold_append",
            "fold_box_bounded",
            "step_contraction",
            "emit_full_box",
            "emit_empty_box",
            "cycle_warm_no_refusal",
            "cycle_burst_refuses_extras",
            "flow_attempts_raw",
            "flow_take_raw",
            "flow_event",
            "emit_liveness",
            "consume_sink_split",
            "step_used",
            "step_consumed",
            "attempts_at_master",
            "balance_at_master",
            "geometry_write_channel_purity",
            "geometry_writes_absorbed",
            "matTick_length",
            "mat_operational_policy",
            "mat_fresh_policy",
            "mat_bounded_policy",
            "mat_operational",
            "mat_fresh",
            "mat_bounded",
        ):
            self.assertRegex(text, rf"\btheorem {name}\b")

    def test_cycle_ordering_is_an_explicit_theorem_hypothesis(self) -> None:
        for name in (
            "cycle_warm_no_refusal",
            "cycle_burst_refuses_extras",
        ):
            head = FORMAL.read_text().split(f"theorem {name}", 1)[1]
            body = head.split(":= by", 1)[0]
            self.assertIn(
                "opsOf i = [Op.consume, Op.emit]",
                body,
                msg=f"{name}: ordering assumption must be stated",
            )

    def test_mat_theorems_carry_explicit_schedule_hypotheses(self) -> None:
        """Cadence audit: the mat theorems quantify over every stream
        obeying the update law — the schedule is a stated hypothesis, not
        a silent assumption baked into the proof."""
        text = FORMAL.read_text()
        for name in (
            "mat_operational_policy",
            "mat_fresh_policy",
            "mat_bounded_policy",
        ):
            body = text.split(f"theorem {name}", 1)[1].split(":= by", 1)[0]
            self.assertIn("run 0 = [0]", body)
            self.assertIn("run (k + 1) = matTick (run k) (k + 1)", body)
        for name in ("mat_operational", "mat_fresh", "mat_bounded"):
            self.assertIn(
                f"{name}_policy matRun rfl (fun _ => rfl)", text
            )

    def test_payload_schema_markers_match_python(self) -> None:
        text = FORMAL.read_text()
        self.assertIn(
            "def d4WriteFids : List Nat := [22]", text
        )
        self.assertIn(
            "def d2WriteFids : List Nat := [18, 24]", text
        )
        self.assertIn("rev.length > 4", text)


if __name__ == "__main__":
    unittest.main()
