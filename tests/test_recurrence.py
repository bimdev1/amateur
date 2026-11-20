"""Census of the ledger's recurrent structure, computed independently of
the way the dry run exploits it.

Domain results asserted here:

- Over the interior subgraph {P1..P6, D1..D4}: exactly one strongly
  connected component.
- Ports: the only entity-adjacent flows are three lateral outputs
  (#2 Crystallized Output, #4 Active Probes, #5 Entropy Exhaust, all
  from P1) and two fresh inputs (#1 Gradients, #3 Perturbations, both
  into P1) — hence the port condensation is the four-edge dag
  {E1 -> core, E2 -> core, core -> E1, core -> E2}.
- Over the interior subgraph, the node-canonical simple cycles are
  exactly the expected seventeen: the perception loop through P1, the
  whiteboard pair (P4, P5), and per store D, the loops that close
  through it. Raw membership is D1 in seven, D2 in four, D3 in four,
  D4 in three (each store's cycles include its bare D<->P pair except
  D4, and three cycles bridge D1 through P6 into D2/D3/D4); membership
  counts overlap there, which sums to twenty over seventeen distinct
  cycles. The Base Core D4 is the only store with no 2- or 3-hop loop.
- Thermodynamic law at the boundary: every simple cycle of the full
  graph that touches an entity necessarily carries both a fresh input
  (#1 or #3) and a lateral output (#2, #4, or #5). The driving gradient
  itself never returns in a cycle ("Emptiness": it passes through
  organization and leaves degraded).
"""
from __future__ import annotations

import unittest

from amateur.ledger import ENTITIES, FLOW_LEDGER, NODE_NAMES, edges

SHORT_OF = {name: short for short, name in NODE_NAMES.items()}
ENTITY_SHORT = {SHORT_OF[name] for name in ENTITIES}

LEDGER_NODES = frozenset(
    {s for s, _f, t in edges()} | {t for _s, _f, t in edges()}
)

INTERIOR = LEDGER_NODES - ENTITY_SHORT

INTERIOR_FLOWS = tuple(
    (s, name, t) for s, name, t in edges() if s in INTERIOR and t in INTERIOR
)

PORT_EDGES = {
    (s, t) for s, _name, t in edges() if s in ENTITY_SHORT or t in ENTITY_SHORT
}

FRESH_INPUT_FLOWS = (1, 3)
LATERAL_OUTPUT_FLOWS = (2, 4, 5)

EXPECTED_CYCLES = {
    ("D1", "P3", "P4"),
    ("D1", "P3", "P5", "P4"),
    ("D1", "P4"),
    ("D1", "P5", "D3", "P2", "P3", "P4"),
    ("D1", "P5", "P4"),
    ("D1", "P5", "P6", "D2", "P2", "P3", "P4"),
    ("D1", "P5", "P6", "D4", "P2", "P3", "P4"),
    ("D2", "P2", "P3", "P4", "P5", "P6"),
    ("D2", "P2", "P3", "P5", "P6"),
    ("D2", "P6"),
    ("D3", "P2", "P3", "P4", "P5"),
    ("D3", "P2", "P3", "P5"),
    ("D3", "P5"),
    ("D4", "P2", "P3", "P4", "P5", "P6"),
    ("D4", "P2", "P3", "P5", "P6"),
    ("P1", "P2", "P3"),
    ("P4", "P5"),
}

EXPECTED_DAG = {
    ("E1", "CORE"),
    ("E2", "CORE"),
    ("CORE", "E1"),
    ("CORE", "E2"),
}


def adjacency(edge_triples) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for src, _flow, dst in edge_triples:
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set())
    return adj


def tarjan_sccs(adj: dict[str, set[str]]) -> list[frozenset[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[frozenset[str]] = []
    counter = [0]

    def connect(node: str) -> None:
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in adj[node]:
            if nxt not in index:
                connect(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            sccs.append(frozenset(component))

    for node in sorted(adj):
        if node not in index:
            connect(node)
    return sccs


def simplecycles(adj: dict[str, set[str]]) -> set[tuple[str, ...]]:
    ordered = {n: sorted(targets) for n, targets in adj.items()}
    cycles: set[tuple[str, ...]] = set()

    def dfs(start: str, node: str, path: tuple[str, ...]) -> None:
        for nxt in ordered[node]:
            if nxt == start:
                cycles.add(path)
            elif nxt > start and nxt not in path:
                dfs(start, nxt, path + (nxt,))

    for node in sorted(ordered):
        dfs(node, node, (node,))
    return cycles


class TestRecurrentStructure(unittest.TestCase):
    def test_ports_are_exactly_the_five_boundary_flows(self) -> None:
        boundary = [
            f for f in FLOW_LEDGER if f.source in ENTITIES or f.target in ENTITIES
        ]
        self.assertEqual(len(boundary), 5)
        self.assertEqual(
            {(SHORT_OF[f.source], SHORT_OF[f.target]) for f in boundary},
            PORT_EDGES,
        )
        self.assertEqual(len(INTERIOR_FLOWS), len(edges()) - len(boundary))

    def test_interior_subgraph_is_one_scc(self) -> None:
        sccs = tarjan_sccs(adjacency(INTERIOR_FLOWS))
        self.assertEqual(sorted(map(sorted, sccs)), [sorted(INTERIOR)])

    def test_port_condensation_is_the_four_edge_dag(self) -> None:
        collapse = lambda n: "CORE" if n in INTERIOR else n
        self.assertEqual(
            {(collapse(s), collapse(t)) for s, t in PORT_EDGES},
            EXPECTED_DAG,
        )

    def test_interior_cycle_census(self) -> None:
        self.assertEqual(simplecycles(adjacency(INTERIOR_FLOWS)), EXPECTED_CYCLES)

    def test_inv_census_per_store_cycle_counts(self) -> None:
        counts = {
            store: sum(store in cycle for cycle in EXPECTED_CYCLES)
            for store in ("D1", "D2", "D3", "D4")
        }
        self.assertEqual(counts, {"D1": 7, "D2": 4, "D3": 4, "D4": 3})

    def test_inv_census_d4_has_no_short_loops(self) -> None:
        d4_hops = {
            len(cycle) for cycle in EXPECTED_CYCLES if "D4" in cycle
        }
        self.assertEqual(d4_hops & {2, 3}, set(), msg=str(d4_hops))

    def test_no_interior_cycle_touches_an_entity(self) -> None:
        for cycle in simplecycles(adjacency(INTERIOR_FLOWS)):
            self.assertFalse(ENTITY_SHORT & set(cycle), msg=str(cycle))

    def test_every_entity_cycle_carries_throughput(self) -> None:
        adj = adjacency(tuple(edges()))
        fid_by_name = {f.flow: f.fid for f in FLOW_LEDGER}
        pairs: dict[tuple[str, str], set[int]] = {}
        for s, name, t in edges():
            pairs.setdefault((s, t), set()).add(fid_by_name[name])
        for cycle in simplecycles(adj):
            if not (ENTITY_SHORT & set(cycle)):
                continue
            fids = set().union(
                *(pairs[(a, b)] for a, b in zip(cycle, cycle[1:] + cycle[:1]))
            )
            self.assertTrue(
                set(FRESH_INPUT_FLOWS) & fids,
                msg=f"entity cycle without fresh input: {cycle} {fids}",
            )
            self.assertTrue(
                set(LATERAL_OUTPUT_FLOWS) & fids,
                msg=f"entity cycle without lateral output: {cycle} {fids}",
            )

    def test_every_interior_cycle_step_is_a_ledger_edge(self) -> None:
        ledger_pairs = {(s, t) for s, _name, t in INTERIOR_FLOWS}
        adj = adjacency(INTERIOR_FLOWS)
        for cycle in simplecycles(adj):
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                self.assertIn(b, adj[a], msg=f"{a} -> {b}")
                self.assertIn((a, b), ledger_pairs, msg=f"{a} -> {b}")


if __name__ == "__main__":
    unittest.main()
