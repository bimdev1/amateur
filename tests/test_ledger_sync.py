"""Contract: the DFD document and the code ledger never drift.

Parses v0.10-untested.md's flow table and Mermaid edges and asserts both
match amateur.ledger.FLOW_LEDGER exactly, and that every flow has a typed
event carrier.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from amateur import events, ledger

ROOT = Path(__file__).resolve().parents[1]
DFD_PATH = ROOT / "v0.10-untested.md"

TABLE_ROW = re.compile(r"^\| (\d+) \| (.+?) \| (.+?) \| (.+?) \|$")
MERMAID_EDGE = re.compile(r"^    (\w+) -->\|\"(.+?)\"\| (\w+)$")


def parse_table(text: str) -> list[tuple[int, str, str, str]]:
    return [
        (int(m[1]), m[2], m[3], m[4])
        for line in text.splitlines()
        if (m := TABLE_ROW.match(line))
    ]


def parse_mermaid(text: str) -> list[tuple[str, str, str]]:
    return [
        (m[1], m[2], m[3])
        for line in text.splitlines()
        if (m := MERMAID_EDGE.match(line))
    ]


class TestLedgerSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dfd = DFD_PATH.read_text(encoding="utf-8")

    def test_markdown_table_matches_code(self) -> None:
        rows = parse_table(self.dfd)
        expected = [(f.fid, f.source, f.flow, f.target) for f in ledger.FLOW_LEDGER]
        self.assertEqual(rows, expected)

    def test_flow_ids_are_dense(self) -> None:
        fids = [f.fid for f in ledger.FLOW_LEDGER]
        self.assertEqual(fids, list(range(1, len(fids) + 1)))

    def test_mermaid_edges_match_code(self) -> None:
        self.assertEqual(parse_mermaid(self.dfd), list(ledger.edges()))

    def test_every_flow_has_a_carrier(self) -> None:
        for flow in ledger.FLOW_LEDGER:
            carrier = events.carrier_for(flow.fid)
            self.assertEqual(carrier.FLOW_ID, flow.fid)

    def test_carriers_cover_exactly_the_ledger(self) -> None:
        carriers = {
            cls.FLOW_ID
            for cls in vars(events).values()
            if isinstance(cls, type)
            and issubclass(cls, events.FlowEvent)
            and cls is not events.FlowEvent
        }
        self.assertEqual(carriers, {f.fid for f in ledger.FLOW_LEDGER})

    def test_every_node_is_declared(self) -> None:
        for flow in ledger.FLOW_LEDGER:
            self.assertIn(flow.source, ledger.NODE_NAMES.values())
            self.assertIn(flow.target, ledger.NODE_NAMES.values())


if __name__ == "__main__":
    unittest.main()
