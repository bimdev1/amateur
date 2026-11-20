"""Physics invariants, enforced against the ledger graph.

These are the clauses of the source text ("Emptiness") that a wiring can
obey or violate; violations made v0.10's first ASCII draft wrong.
"""
from __future__ import annotations

import unittest

from amateur import ledger

STORE_SHORTS = ("D1", "D2", "D3", "D4")
ENTITY_SHORTS = ("E1", "E2")


class TestThermodynamics(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = ledger.edges()

    def _flows_from(self, node: str) -> list[str]:
        return [flow for src, flow, _ in self.edges if src == node]

    def _flows_to(self, node: str) -> list[str]:
        return [flow for _src, flow, dst in self.edges if dst == node]

    def test_entities_only_touch_the_blanket(self) -> None:
        for src, flow, dst in self.edges:
            if src in ENTITY_SHORTS:
                self.assertEqual(dst, "P1", msg=f"flow '{flow}' bypasses P1")
            if dst in ENTITY_SHORTS:
                self.assertEqual(src, "P1", msg=f"flow '{flow}' bypasses P1")

    def test_perturbation_does_not_instruct(self) -> None:
        self.assertEqual(self._flows_from("E1"), ["Perturbations"])
        self.assertEqual(self._flows_from("E2"), ["Gradients"])

    def test_entropy_leaves_through_the_boundary(self) -> None:
        """RE-SPEC PENDING (round 10): the old "Entropy Exhaust is the only
        degenerate outflow" reading is VOID -- the corrective ASCII restores
        #27.  Provisional reading: two entropy-typed flows exist, the
        unlabeled terminal exhaust #5 (P1 -> E1) and the labeled store read
        #27 (D3 -> P1, the dissipation exit)."""
        entropy_flows = [f for _, f, _ in self.edges if "Entropy" in f]
        self.assertEqual(
            entropy_flows, ["Entropy Exhaust", "Dissipated Entropy (read)"]
        )
        exhaust = next(f for f in ledger.FLOW_LEDGER if f.flow == "Entropy Exhaust")
        self.assertEqual((exhaust.source, exhaust.target), (ledger.P1, ledger.E1))

    def test_residue_edge_is_the_dissipation_exit(self) -> None:
        """RE-SPEC PENDING (round 10): replaces the void
        ``test_no_residue_edge_back_into_transduction``.  The corrected
        graph HAS D3 -> P1; it is a store read and the only residue edge
        back into the transducer, and the transducer still leaves through
        the boundary exhaust, so residue drains rather than circulates."""
        residue = [flow for src, flow, dst in self.edges if (src, dst) == ("D3", "P1")]
        self.assertEqual(len(residue), 1, msg="exactly one D3 -> P1 edge")
        self.assertTrue(residue[0].endswith("(read)"))
        self.assertIn("Entropy Exhaust", self._flows_from("P1"))

    def test_store_outbound_edges_are_reads(self) -> None:
        for src, flow, _dst in self.edges:
            if src in STORE_SHORTS:
                self.assertTrue(flow.endswith("(read)"), msg=f"unlabeled store read: {flow}")

    def test_store_inbound_edges_are_writes(self) -> None:
        for _src, flow, dst in self.edges:
            if dst in STORE_SHORTS:
                self.assertNotIn("(read)", flow, msg=f"read marked as write: {flow}")

    def test_base_core_curvature_is_nearly_fixed(self) -> None:
        self.assertEqual(self._flows_to("D4"), ["Sedimented Invariants"])
        self.assertEqual(self._flows_from("D4"), ["Base Ops (read)"])

    def test_entity_directions_are_asymmetric(self) -> None:
        self.assertEqual(ledger.outbound(ledger.E1), ("Perturbations",))
        self.assertEqual(ledger.inbound(ledger.E1), ("Active Probes", "Entropy Exhaust"))
        self.assertEqual(ledger.outbound(ledger.E2), ("Gradients",))
        self.assertEqual(ledger.inbound(ledger.E2), ("Crystallized Output",))


class TestRoutingBiasOnly(unittest.TestCase):
    def test_router_emits_a_single_flow(self) -> None:
        self.assertEqual(ledger.outbound(ledger.P2), ("Routed Dynamics",))

    def test_density_of_the_mat_reaches_routing(self) -> None:
        self.assertIn("Mat Overlays (read)", ledger.outbound(ledger.D2))
        self.assertIn("Base Ops (read)", ledger.inbound(ledger.P2))
        self.assertIn("Avoidance Trails (read)", ledger.inbound(ledger.P2))


if __name__ == "__main__":
    unittest.main()
