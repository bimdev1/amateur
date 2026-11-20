"""Structural invariants over the corrected 27-flow graph and the tick order.

The user-supplied master invariants split into proof obligations
(formal/DryRun.lean), empirical bookkeeping (tests/test_accounting.py), and
these structural checks.  Round 11 re-spec:

- INV-GEOMETRY has three tiers: D4 is the ratchet-exclusive store (P6/#22
  only); D2 is a nursery with two owners (P3 births #18, P6 writes #24) and
  exact per-flow kinds; D1/D3 are operational stores with dynamic writes.
- INV-CENSUS is a **derived fingerprint**, not a hard-coded cycle list:
  the simple-cycle census is computed from ``FLOW_LEDGER`` at test time;
  the literals 26/24 and 12/11/15/4 are regression expectations only
  (tested-not-proven topology).
- INV-STAGE-ORDER (renamed from INV-CYCLE-DELAY): the delayed set is exactly
  the stage-backward process edges, computed from the ledger + spine.  Store
  reads are injected, so they cannot be delayed; the set is minimal and
  sufficient.
- INV-EXHAUST: sinks are {2, 4, 5}; the dissipation chain is
  D3 -> P1 (#27 read) -> E1 (#5); #27 is destructive and no D3 material is
  re-stored as state afterward (every D3 outbound edge is a read).
"""
from __future__ import annotations

import unittest

from amateur.dryrun import (
    PLACEHOLDER_FLOWS,
    PER_TICK_INJECTIONS,
    SINK_FLOWS,
    SPINE,
    STAGE_BACKWARD_FLOWS,
    STAGE_BACKWARD_FLOWS as DELAYED,
    WRITE_FLOWS,
    DryRun,
)
from amateur.ledger import E1, E2, FLOW_LEDGER, NODE_NAMES, STORES, edges
from amateur.payloads import (
    PAYLOAD_SCHEMA,
    BiasPrimitive,
    DigestedDebrisBody,
    EmergentCrystalBody,
    MatOverlay,
)

SHORT_OF = {name: short for short, name in NODE_NAMES.items()}
STAGE_ORDER = {SHORT_OF[stage.name]: index for index, stage in enumerate(SPINE)}
D1, D2, D3, D4 = (NODE_NAMES[s] for s in ("D1", "D2", "D3", "D4"))
P1 = NODE_NAMES["P1"]


def event_order() -> list[tuple[str, int]]:
    events = [("emit", fid) for fid in PER_TICK_INJECTIONS]
    for stage in SPINE:
        events.extend(("consume", fid) for fid in stage.consumes)
        events.extend(("emit", fid) for fid in stage.emits)
    events.extend(("absorb", fid) for fid in WRITE_FLOWS)
    events.extend(("sink", fid) for fid in SINK_FLOWS)
    return events


def adjacency(graph_edges) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for src, _flow, dst in graph_edges:
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set())
    return adj


def simple_cycles(adj: dict[str, set[str]]) -> set[tuple[str, ...]]:
    """Node-distinct simple directed cycles, canonical rotation only."""
    ordered = {node: sorted(targets) for node, targets in adj.items()}
    cycles: set[tuple[str, ...]] = set()

    def dfs(start: str, node: str, path: tuple[str, ...]) -> None:
        for nxt in ordered[node]:
            if nxt == start:
                cycles.add(path)
            elif nxt > start and nxt not in path:
                dfs(start, nxt, path + (nxt,))

    for start in sorted(adj):
        dfs(start, start, (start,))
    return cycles


class TestLiveness(unittest.TestCase):
    def test_inv_liveness_event_schedule(self) -> None:
        emits = [fid for kind, fid in event_order() if kind == "emit"]
        self.assertEqual(sorted(emits), sorted(f.fid for f in FLOW_LEDGER))

    def test_inv_liveness_empirical_per_tick(self) -> None:
        report = DryRun().run(4)
        for stat in report.history:
            self.assertEqual(
                stat.accepted, {f.fid: 1 for f in FLOW_LEDGER}
            )
            self.assertEqual(stat.attempted, stat.accepted)
            self.assertEqual(stat.refused, {})
            expected_misses = {10: 1, 17: 1}
            if stat is not report.history[0]:
                expected_misses = {}
            self.assertEqual(stat.missed, expected_misses)


class TestCausal(unittest.TestCase):
    def setUp(self) -> None:
        self.arrival: dict[int, int] = {}
        self.pickup: dict[int, int] = {}
        for index, (kind, fid) in enumerate(event_order()):
            if kind == "emit":
                self.arrival.setdefault(fid, index)
            elif kind in ("consume", "absorb", "sink"):
                self.pickup.setdefault(fid, index)

    def test_inv_causal_no_clairvoyance(self) -> None:
        for fid, at in self.pickup.items():
            if fid in STAGE_BACKWARD_FLOWS:
                continue
            self.assertLess(
                self.arrival[fid],
                at,
                msg=f"#{fid} read before it was sent",
            )

    def test_inv_stage_order_delayed_flows(self) -> None:
        for fid in STAGE_BACKWARD_FLOWS:
            self.assertIn(fid, self.arrival, msg=f"#{fid} never emitted")
            self.assertLess(
                self.pickup[fid],
                self.arrival[fid],
                msg=f"#{fid} must be consumed before its same-tick emit",
            )

    def test_inv_causal_stage_consumes_exactly_its_inflows(self) -> None:
        inflows = {
            SHORT_OF[t]: {g.fid for g in FLOW_LEDGER if g.target == t}
            for t in {f.target for f in FLOW_LEDGER}
        }
        for stage in SPINE:
            short = SHORT_OF[stage.name]
            self.assertEqual(
                set(stage.consumes), inflows[short], msg=stage.name
            )
            outflows = {
                g.fid for g in FLOW_LEDGER if SHORT_OF[g.source] == short
            }
            self.assertEqual(set(stage.emits), outflows, msg=stage.name)


class TestStageOrder(unittest.TestCase):
    def test_inv_stage_order_is_exactly_the_stage_backward_edges(self) -> None:
        """Derived from ledger + spine: a process edge is delayed iff its
        source stage is after its target stage (the same-tick read would
        precede the same-tick emit)."""
        derived = {
            flow.fid
            for flow in FLOW_LEDGER
            if flow.source in NODE_NAMES.values()
            and flow.target in NODE_NAMES.values()
            and SHORT_OF[flow.source] in STAGE_ORDER
            and SHORT_OF[flow.target] in STAGE_ORDER
            and STAGE_ORDER[SHORT_OF[flow.source]]
            > STAGE_ORDER[SHORT_OF[flow.target]]
        }
        self.assertEqual(derived, set(STAGE_BACKWARD_FLOWS))
        self.assertEqual(set(STAGE_BACKWARD_FLOWS), {10, 17})

    def test_inv_stage_order_store_reads_are_injected_not_delayed(self) -> None:
        store_reads = {
            flow.fid
            for flow in FLOW_LEDGER
            if "(read)" in flow.flow
        }
        self.assertTrue(store_reads <= set(PER_TICK_INJECTIONS))
        self.assertFalse(store_reads & set(STAGE_BACKWARD_FLOWS))

    def test_inv_stage_order_minimality_and_sufficiency(self) -> None:
        arrivals: dict[int, int] = {}
        pickups: dict[int, int] = {}
        for index, (kind, fid) in enumerate(event_order()):
            if kind == "emit":
                arrivals.setdefault(fid, index)
            elif kind in ("consume", "absorb", "sink"):
                pickups.setdefault(fid, index)
        same_tick = {
            fid for fid in pickups if pickups[fid] < arrivals[fid]
        }
        self.assertEqual(same_tick, set(STAGE_BACKWARD_FLOWS))
        # Every delayed flow is necessary: dropping it would leave a
        # same-tick read before its emit unhandled.
        for fid in set(STAGE_BACKWARD_FLOWS):
            self.assertLess(pickups[fid], arrivals[fid])


class TestBlanket(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ent = {SHORT_OF[n] for n in (E1, E2)}

    def test_inv_blanket_exact_port_pairs(self) -> None:
        pairs = {
            (s, t) for s, _f, t in edges() if s in self.ent or t in self.ent
        }
        self.assertEqual(
            pairs,
            {("E1", "P1"), ("P1", "E1"), ("E2", "P1"), ("P1", "E2")},
        )

    def test_inv_blanket_periphery_flows(self) -> None:
        periphery = [
            f
            for f in FLOW_LEDGER
            if f.source in (E1, E2) or f.target in (E1, E2)
        ]
        self.assertEqual({f.fid for f in periphery}, {1, 2, 3, 4, 5})


class TestGeometry(unittest.TestCase):
    """INV-GEOMETRY three tiers (round 11 ruling G)."""

    def test_tier_1_d4_is_ratchet_exclusive(self) -> None:
        writers = [
            (flow.fid, SHORT_OF[flow.source])
            for flow in FLOW_LEDGER
            if flow.target == D4 and "(read)" not in flow.flow
        ]
        self.assertEqual(writers, [(22, "P6")])
        self.assertEqual(
            PAYLOAD_SCHEMA[22], frozenset({BiasPrimitive})
        )

    def test_tier_2_d2_is_a_two_owner_nursery_with_pinned_kinds(self) -> None:
        writers = {
            flow.fid: SHORT_OF[flow.source]
            for flow in FLOW_LEDGER
            if flow.target == D2 and "(read)" not in flow.flow
        }
        self.assertEqual(writers, {18: "P3", 24: "P6"})
        self.assertEqual(
            PAYLOAD_SCHEMA[18], frozenset({EmergentCrystalBody})
        )
        self.assertEqual(PAYLOAD_SCHEMA[24], frozenset({MatOverlay}))
        reads = {
            flow.fid
            for flow in FLOW_LEDGER
            if flow.source == D2 and "(read)" in flow.flow
        }
        self.assertEqual(reads, {8, 23, 25})

    def test_tier_3_d1_d3_are_operational_stores(self) -> None:
        writers = {
            flow.fid: SHORT_OF[flow.source]
            for flow in FLOW_LEDGER
            if flow.target in (D1, D3)
            and "(read)" not in flow.flow
        }
        self.assertEqual(writers, {13: "P4", 19: "P5"})
        self.assertEqual(
            PAYLOAD_SCHEMA[19], frozenset({DigestedDebrisBody})
        )
        # D1 stays the frozen slice: only #13 writes, and #11/#14/#15 are
        # its (placeholder) reads.
        self.assertIn(13, writers)
        self.assertNotIn(14, writers)


class TestCensus(unittest.TestCase):
    """INV-CENSUS: derived fingerprint; literals are regression only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.adj = adjacency(edges())
        cls.cycles = simple_cycles(cls.adj)
        cls.interior = cls.cycles - {
            cycle for cycle in cls.cycles
            if any(node in ("E1", "E2") for node in cycle)
        }

    def test_inv_census_aggregate_fingerprint(self) -> None:
        self.assertEqual(len(self.cycles), 26)
        self.assertEqual(len(self.interior), 24)
        members = {
            store: sum(store in cycle for cycle in self.interior)
            for store in ("D1", "D2", "D3", "D4")
        }
        self.assertEqual(members, {"D1": 12, "D2": 11, "D3": 15, "D4": 4})

    def test_inv_census_store_shapes(self) -> None:
        lengths = {
            store: [
                len(cycle) for cycle in self.interior if store in cycle
            ]
            for store in ("D1", "D2", "D3", "D4")
        }
        self.assertNotIn(2, lengths["D3"], msg="D3 must have no 2-cycle")
        self.assertGreaterEqual(min(lengths["D4"]), 5)
        self.assertIn(
            ("D2", "P2", "P3"), self.interior,
            msg="the D2 nursery loop #8/#9/#18 is missing",
        )

    def test_inv_census_is_derived_not_listed(self) -> None:
        # A one-edge perturbation changes the fingerprint: the test is
        # reading the ledger, not a stored literal.
        perturbed = [
            edge for edge in edges()
            if not edge[1].startswith("Emergent")
        ]
        self.assertNotEqual(
            len(simple_cycles(adjacency(perturbed))), len(self.cycles)
        )


class TestExhaust(unittest.TestCase):
    def test_inv_exhaust_sinks_are_the_three_boundary_flows(self) -> None:
        self.assertEqual(set(SINK_FLOWS), {2, 4, 5})

    def test_inv_exhaust_dissipation_chain_and_destructive_27(self) -> None:
        flow = {f.fid: f for f in FLOW_LEDGER}
        self.assertEqual((flow[27].source, flow[27].target), (D3, P1))
        self.assertEqual((flow[5].source, flow[5].target), (P1, E1))
        # #27 is consumed at P1 before #5 is emitted from P1.
        arrivals: dict[int, int] = {}
        pickups: dict[int, int] = {}
        for index, (kind, fid) in enumerate(event_order()):
            if kind == "emit":
                arrivals.setdefault(fid, index)
            elif kind in ("consume", "absorb", "sink"):
                pickups.setdefault(fid, index)
        self.assertLess(pickups[27], arrivals[5])

    def test_inv_exhaust_no_d3_material_restored_as_state(self) -> None:
        for src, _flow, _dst in edges():
            if src == D3:
                self.assertEqual(
                    _flow.endswith("(read)"), True,
                    msg=f"D3 outbound write would restore drained material: {_flow}",
                )

    def test_inv_exhaust_27_actually_drains(self) -> None:
        run = DryRun()
        run.run(16)
        balance = run.debris.mass_balance()
        self.assertGreater(balance["drained"], 0)
        self.assertEqual(
            balance["deposits"],
            balance["resident"] + balance["drained"] + balance["evicted"],
        )
        # The drained ids are not resident after their #27 consume.
        resident = {cohort.cohort_id for cohort in run.debris.bodies()}
        self.assertFalse(
            resident & {cohort.cohort_id for cohort in run.debris.drains}
        )


class TestPlaceholders(unittest.TestCase):
    def test_placeholder_set_is_the_reduced_round11_set(self) -> None:
        self.assertEqual(PLACEHOLDER_FLOWS, frozenset({1, 2, 4, 5, 9, 13}))
        self.assertNotIn(16, PLACEHOLDER_FLOWS)


if __name__ == "__main__":
    unittest.main()
