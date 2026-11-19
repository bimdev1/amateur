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
    arms: dict[int, tuple[str, ...]] = {}
    for match in re.finditer(r"^\s*\| (\d+) => \[([^\]]*)\]", text, re.M):
        words = tuple(
            _LEAN_TO_WORD[w.strip()]
            for w in match.group(2).split(",")
            if w.strip()
        )
        arms[int(match.group(1))] = words
    return arms


def step(ops: tuple[str, ...], box: int) -> tuple[int, int, int]:
    missed = sunk = 0
    for op in ops:
        if op == "emit":
            box += 1
        elif op == "consume":
            if box > 0:
                box -= 1
            else:
                missed += 1
        elif op == "sink":
            sunk += box
            box = 0
        else:
            raise ValueError(f"unknown op word {op!r}")
    return box, missed, sunk


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
        sunk: Counter[int] = Counter()
        for tick in range(1, 65):
            for fid, ops in self.arms.items():
                boxes[fid], missed_delta, sunk_delta = step(ops, boxes[fid])
                missed[fid] += missed_delta
                sunk[fid] += sunk_delta
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
            "sunk_master",
        ):
            self.assertRegex(text, rf"\btheorem {name}\b")


if __name__ == "__main__":
    unittest.main()
