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
        writers = {src for src, _, dst in ledger.edges() if dst == "D4"}
        self.assertEqual(writers, {"P6"}, msg="unauthorized writer of D4")

    def test_the_nursery_has_two_owners(self) -> None:
        """Round 10: D2 is a nursery, not a ratchet-only store.  The loop
        births the candidate (#18) and the ratchet writes the revalidated
        overlay (#24); D4 stays the only ratchet-exclusive store."""
        writers = {
            src for src, _, dst in ledger.edges() if dst == "D2"
        }
        self.assertEqual(writers, {"P3", "P6"}, msg="unauthorized writer of D2")


class TestTheCoreCarriesNoTrajectories(TopologyMixin, unittest.TestCase):
    """'The core contains no trajectories... No particular world occupies
    the core': the observation loop feels long-term geometry only through
    routing's bias (#7, #8, #26) and never touches it directly."""

    def test_observer_births_candidates_but_touches_no_frozen_geometry(self) -> None:
        """Round 10 corrects the old "never touches D2" clause: 3.0 births
        the emergent crystal into the nursery (#18).  What survives is the
        real separation: the loop never touches the frozen base core (D4),
        never reaches an entity, and never hardens the mat itself."""
        touch = {dst for src, _, dst in ledger.edges() if src == "P3"}
        touch |= {src for src, _, dst in ledger.edges() if dst == "P3"}
        self.assertNotIn("D4", touch, msg=touch)
        self.assertFalse(touch & {"E1", "E2"}, msg=touch)
        self.assertIn("D2", touch, msg="the corrected #18 birth is missing")

    def test_blanket_receives_only_crossings_and_resolutions(self) -> None:
        self.assertEqual(
            ledger.inbound(ledger.P1),
            (
                "Gradients",
                "Perturbations",
                "Resolved Trajectory",
                "Dissipated Entropy (read)",
            ),
        )


class TestDebrisDiscipline(TopologyMixin, unittest.TestCase):
    """Debris is recycled through digestion and biases routing; it never
    decorates the core and never reaches the entities."""

    def test_debris_pool_serves_reuse_and_dissipation(self) -> None:
        """Round 10: the corrected ASCII routes D3 to the digester, the
        regulator (#20 scaffolds), the ratchet (#21 motifs), routing (#26
        trails) and back out through the boundary (#27 dissipated entropy).
        The pool never reaches the entities or the frozen base core."""
        touch = {dst for src, _, dst in ledger.edges() if src == "D3"}
        touch |= {src for src, _, dst in ledger.edges() if dst == "D3"}
        self.assertEqual(touch, {"P1", "P2", "P4", "P5", "P6"})
        self.assertFalse(touch & {"E1", "E2", "D4"})


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
    """The round-10 audit record, pinned so the correction cannot silently
    reopen.  The old lexicon (candidate into 5.0, motif out of 5.0,
    survivor `6.0 -> D2`) was built on the drifted transcription; the
    corrected ASCII reading is the spec and the lexicon is explicitly
    marked as being rewritten / not authoritative."""

    def _doc(self) -> str:
        path = Path(__file__).resolve().parents[1] / "v0.10-untested.md"
        return path.read_text()

    def test_audit_record_is_on_file(self) -> None:
        doc = self._doc()
        self.assertIn("Round 10 audit record", doc)
        for corrected in (
            "**3.0 → 5.0**",        # #16
            "**3.0 → D2**",         # #18
            "**D3 → 4.0**",         # #20
            "**D3 → 6.0 (read)**",  # #21
            "**D2 → 6.0 (read)**",  # #23
            "**D2 → 5.0**",         # #25
            "**D3 → 1.0 (read), restored**",  # #27
        ):
            self.assertIn(corrected, doc)

    def test_ledger_matches_the_corrected_reading(self) -> None:
        flow = {f.fid: f for f in ledger.FLOW_LEDGER}
        self.assertEqual(flow[16].source, ledger.P3)
        self.assertEqual(flow[16].target, ledger.P5)
        self.assertEqual(flow[18].source, ledger.P3)
        self.assertEqual(flow[18].target, ledger.D2)
        self.assertEqual(flow[20].source, ledger.D3)
        self.assertEqual(flow[20].target, ledger.P4)
        self.assertEqual(flow[21].source, ledger.D3)
        self.assertEqual(flow[21].target, ledger.P6)
        self.assertEqual(flow[23].source, ledger.D2)
        self.assertEqual(flow[23].target, ledger.P6)
        self.assertEqual(flow[25].source, ledger.D2)
        self.assertEqual(flow[25].target, ledger.P5)
        self.assertEqual(flow[27].source, ledger.D3)
        self.assertEqual(flow[27].target, ledger.P1)
        # 5.0 -> 6.0 no longer exists.
        self.assertFalse(
            [(src, dst) for src, _, dst in ledger.edges()
             if src == "P5" and dst == "P6"]
        )

    def test_old_lexicon_is_recorded_as_superseded(self) -> None:
        doc = self._doc()
        self.assertIn("Superseded readings", doc)
        self.assertNotIn("Lexicon pinned", doc)


if __name__ == "__main__":
    unittest.main()
