"""Design-intent tests: the wiring must not merely be a graph, it must be
*this* graph. Each test here is a clause of the source prose or a rule of
data-flow methodology, checked against the ledger.
"""
from __future__ import annotations

import unittest
from collections import defaultdict
from pathlib import Path

from amateur import events, ledger


class TopologyMixin:
    def edge_view(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        out: dict[str, list[str]] = defaultdict(list)
        inn: dict[str, list[str]] = defaultdict(list)
        for src, flow, dst in ledger.edges():
            out[src].append(flow)
            inn[dst].append(flow)
        return dict(out), dict(inn)


class TestGraphHealth(TopologyMixin, unittest.TestCase):
    def test_no_black_holes_no_miracles(self) -> None:
        out, inn = self.edge_view()
        for node in ledger.NODE_NAMES:
            self.assertTrue(out[node], msg=f"miracle (no output): {node}")
            self.assertTrue(inn[node], msg=f"black hole (no input): {node}")

    def test_flow_names_are_unique(self) -> None:
        labels = [flow for _, flow, _ in ledger.edges()]
        self.assertEqual(len(labels), len(set(labels)))

    def test_every_flow_appears_once(self) -> None:
        fids = [f.fid for f in ledger.FLOW_LEDGER]
        self.assertEqual(len(fids), len(set(fids)))


class TestSeparationOfTimescales(TopologyMixin, unittest.TestCase):
    """'Its stillness is parametric, not dynamical... frozen geometry for
    unfixed unfolding': the long-term geometry may only move on the
    sediment timescale."""

    def test_only_the_ratchet_writes_the_base_core(self) -> None:
        self._written_by("D4")

    def test_only_the_ratchet_writes_the_mat(self) -> None:
        self._written_by("D2")

    def _written_by(self, store: str) -> None:
        writers = {src for src, _, dst in ledger.edges() if dst == store}
        self.assertEqual(writers, {"P6"}, msg=f"unauthorized writer of {store}")


class TestTheCoreCarriesNoTrajectories(TopologyMixin, unittest.TestCase):
    """'The core contains no trajectories... No particular world occupies
    the core': the observation loop feels long-term geometry only through
    routing's bias (#7, #8, #26) and never touches it directly."""

    def test_observer_never_touches_geometry_or_entities(self) -> None:
        touch = {dst for src, _, dst in ledger.edges() if src == "P3"}
        touch |= {src for src, _, dst in ledger.edges() if dst == "P3"}
        self.assertFalse(touch & {"D2", "D4"}, msg=touch)
        self.assertFalse(touch & {"E1", "E2"}, msg=touch)

    def test_blanket_receives_only_crossings_and_resolutions(self) -> None:
        self.assertEqual(
            ledger.inbound(ledger.P1),
            ("Gradients", "Perturbations", "Resolved Trajectory"),
        )


class TestDebrisDiscipline(TopologyMixin, unittest.TestCase):
    """Debris is recycled through digestion and biases routing; it never
    decorates the core and never reaches the entities."""

    def test_debris_pool_serves_only_digestion_and_routing(self) -> None:
        touch = {dst for src, _, dst in ledger.edges() if src == "D3"}
        touch |= {src for src, _, dst in ledger.edges() if dst == "D3"}
        self.assertEqual(touch, {"P5", "P2"})


class TestCarrierSemantics(unittest.TestCase):
    def test_carriers_are_flow_pinned(self) -> None:
        for flow in ledger.FLOW_LEDGER:
            cls = events.carrier_for(flow.fid)
            event = cls(flow.flow)
            self.assertEqual(cls.FLOW_ID, flow.fid)
            self.assertEqual(event.FLOW_ID, flow.fid)

    def test_equality_is_class_sensitive_not_payload_sensitive(self) -> None:
        self.assertNotEqual(events.Gradients(payload=1), events.PredictionError(payload=1))
        self.assertEqual(events.Gradients(payload=1), events.Gradients(payload=1))

    def test_hash_semantics_are_tuple_like(self) -> None:
        a = events.Gradients(payload=("k", 1))
        self.assertEqual(hash(a), hash(events.Gradients(payload=("k", 1))))
        with self.assertRaises(TypeError):
            hash(events.Gradients(payload={"k": 1}))


class TestCrystalLexicon(unittest.TestCase):
    """The resolved #18 disambiguation, pinned so it cannot silently
    reopen: candidate / extracted-rule / vetted are three different terms
    on three different edges, and the ledger wiring matches the lexicon."""

    RESOLVED = "#18, RESOLVED"

    def _doc(self) -> str:
        path = Path(__file__).resolve().parents[1] / "v0.10-untested.md"
        return path.read_text()

    def test_resolution_is_on_record(self) -> None:
        doc = self._doc()
        self.assertIn(self.RESOLVED, doc)
        self.assertIn("Lexicon pinned", doc)
        self.assertIn("Falsifier on record", doc)
        for term in (
            "*Emergent Crystal* = candidate (3.0 -> 5.0)",
            "*Structural Motif* =\n  extracted rule (5.0 -> 6.0)",
            "*Surviving Crystal* = vetted, ratcheted\n  structure (6.0 -> D2)",
        ):
            self.assertIn(term, doc)

    def test_ledger_matches_the_lexicon(self) -> None:
        flow = {f.fid: f for f in ledger.FLOW_LEDGER}
        self.assertEqual(flow[18].source, ledger.P3)
        self.assertEqual(flow[18].target, ledger.P5)
        self.assertEqual(flow[21].source, ledger.P5)
        self.assertEqual(flow[21].target, ledger.P6)
        self.assertEqual(flow[23].source, ledger.P6)
        self.assertEqual(flow[23].target, ledger.D2)


if __name__ == "__main__":
    unittest.main()
