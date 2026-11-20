"""Invariant-mapped structural tests over the tick's event order.

The user-supplied master invariants (INV-*) split into proof obligations
(formal/DryRun.lean), empirical bookkeeping (tests/test_accounting.py),
and these structural checks over what the plumbing may do in one tick:

- INV-LIVENESS: every ledger flow is emitted exactly once per tick. The
  Lean theorem `emit_liveness` proves the op table's weight side; this
  pins the event schedule itself.
- INV-CAUSAL: an event is never read before it was sent in the same
  tick, except the two declared delayed cycles (#10, #17), whose
  mailbox deliberately carries the prior tick's deformation into the
  next ("recurrence becomes differential"). A stage consumes exactly
  its node's ledger inflow set and emits exactly its outflow set, so
  every prerequisite of a stage is processed by construction.
- INV-BLANKET: the only entity-adjacent edges are the four port pairs
  around P1; every entity crossing goes through 1.0.
- INV-GEOMETRY (round-10 re-spec, provisional): the only ratchet-exclusive
  store is D4; D2 is a nursery with two owners — the loop births the
  candidate (#18, P3) and the ratchet writes the revalidated overlay
  (#24, P6).  D1/D3 stay operational stores with per-tick dynamic writes
  by design — D1 <- P4, D3 <- P5; no other flow writes any store.
"""
from __future__ import annotations

import unittest

from amateur.dryrun import (
    CYCLE_FLOWS,
    DryRun,
    PER_TICK_INJECTIONS,
    SINK_FLOWS,
    SPINE,
    WRITE_FLOWS,
)
from amateur.ledger import E1, E2, FLOW_LEDGER, NODE_NAMES, STORES, edges

SHORT_OF = {name: short for short, name in NODE_NAMES.items()}


def event_order() -> list[tuple[str, int]]:
    events = [("emit", fid) for fid in PER_TICK_INJECTIONS]
    for stage in SPINE:
        events.extend(("consume", fid) for fid in stage.consumes)
        events.extend(("emit", fid) for fid in stage.emits)
    events.extend(("absorb", fid) for fid in WRITE_FLOWS)
    events.extend(("sink", fid) for fid in SINK_FLOWS)
    return events


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
            self.assertEqual(stat.missed, {10: 1, 17: 1} if stat is report.history[0] else {})


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
            if fid in CYCLE_FLOWS:
                continue
            self.assertLess(
                self.arrival[fid],
                at,
                msg=f"#{fid} read before it was sent",
            )

    def test_inv_causal_cycles_are_delayed_not_impossible(self) -> None:
        for fid in CYCLE_FLOWS:
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
    def test_inv_geometry_adjudicated_writer_sets(self) -> None:
        writers = {
            SHORT_OF[store]: {
                src for src, _f, dst in edges() if dst == SHORT_OF[store]
            }
            for store in sorted(STORES)
        }
        self.assertEqual(
            writers,
            {"D1": {"P4"}, "D2": {"P3", "P6"}, "D3": {"P5"}, "D4": {"P6"}},
        )

    def test_inv_geometry_d4_is_the_only_ratchet_exclusive_store(self) -> None:
        """Round 10: D2's ratchet-only reading is withdrawn (the nursery
        is born into by 3.0); D4 stays the sole ratchet-exclusive store."""
        writes = {
            store: [
                (src, name)
                for src, name, dst in edges()
                if dst == store and "(read)" not in name
            ]
            for store in ("D2", "D4")
        }
        self.assertEqual(
            {(src, name) for src, name in writes["D4"]},
            {("P6", "Sedimented Invariants")},
        )
        self.assertEqual(
            {src for src, _name in writes["D2"]}, {"P3", "P6"}
        )
        self.assertGreater(len(writes["D2"]), 1)


if __name__ == "__main__":
    unittest.main()
