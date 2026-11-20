"""Machine-readable copy of the v0.10 flow ledger.

Single source of truth for the wiring. `v0.10-untested.md' is the human
document; tests assert that its markdown table, its Mermaid edges, and this
module agree exactly. Convention: a store's outbound edge is always a read
(`(read)` suffix), a store's inbound edge is always a write or an operation.
"""
from __future__ import annotations

from typing import NamedTuple

E1 = "E1 External Reality"
E2 = "E2 Gardener"
P1 = "1.0 Transduce Markov Blanket"
P2 = "2.0 Route Without Ossification"
P3 = "3.0 Execute Observation Loop"
P4 = "4.0 Allocate & Regulate Whiteboard"
P5 = "5.0 Digest & Collapse Ceiling-Bound Units"
P6 = "6.0 Ratchet & Sediment Grammar"
D1 = "D1 Elastic Board"
D2 = "D2 Multi-Layer Mat"
D3 = "D3 Debris & Scaffolding Pool"
D4 = "D4 Base Core"

NODE_NAMES = {
    "E1": E1,
    "E2": E2,
    "P1": P1,
    "P2": P2,
    "P3": P3,
    "P4": P4,
    "P5": P5,
    "P6": P6,
    "D1": D1,
    "D2": D2,
    "D3": D3,
    "D4": D4,
}
_NAME_TO_SHORT = {name: short for short, name in NODE_NAMES.items()}


class Flow(NamedTuple):
    """One row of the DFD ledger: source node, flow, target node."""

    fid: int
    source: str
    flow: str
    target: str


FLOW_LEDGER: tuple[Flow, ...] = (
    Flow(1, E2, "Gradients", P1),
    Flow(2, P1, "Crystallized Output", E2),
    Flow(3, E1, "Perturbations", P1),
    Flow(4, P1, "Active Probes", E1),
    Flow(5, P1, "Entropy Exhaust", E1),
    Flow(6, P1, "Prediction Error", P2),
    Flow(7, D4, "Base Ops (read)", P2),
    Flow(8, D2, "Mat Overlays (read)", P2),
    Flow(9, P2, "Routed Dynamics", P3),
    Flow(10, P3, "Resolved Trajectory", P1),
    Flow(11, D1, "Active Context (read)", P3),
    Flow(12, P3, "State Deltas & Surprise", P4),
    Flow(13, P4, "Write / Erase / Resize / Evict", D1),
    Flow(14, D1, "Slot State (read)", P4),
    Flow(15, D1, "Evicted Clusters (read)", P5),
    Flow(16, P3, "Cost vs. Error Metrics", P5),
    Flow(17, P5, "Collapse Mandate & Freed Slots", P4),
    Flow(18, P3, "Emergent Crystals", D2),
    Flow(19, P5, "Digested Debris", D3),
    Flow(20, D3, "Consumed Scaffolds (read)", P4),
    Flow(21, D3, "Structural Motifs (read)", P6),
    Flow(22, P6, "Sedimented Invariants", D4),
    Flow(23, D2, "Surviving Crystals (read)", P6),
    Flow(24, P6, "Ratcheted Overlays", D2),
    Flow(25, D2, "Stale Overlays (read)", P5),
    Flow(26, D3, "Avoidance Trails (read)", P2),
    Flow(27, D3, "Dissipated Entropy (read)", P1),
)

ENTITIES = frozenset((E1, E2))
PROCESSES = frozenset((P1, P2, P3, P4, P5, P6))
STORES = frozenset((D1, D2, D3, D4))


def edges() -> tuple[tuple[str, str, str], ...]:
    """Ledger as (source_short, flow, target_short) triples."""
    return tuple(
        (_NAME_TO_SHORT[f.source], f.flow, _NAME_TO_SHORT[f.target])
        for f in FLOW_LEDGER
    )


def outbound(node: str) -> tuple[str, ...]:
    return tuple(f.flow for f in FLOW_LEDGER if f.source == node)


def inbound(node: str) -> tuple[str, ...]:
    return tuple(f.flow for f in FLOW_LEDGER if f.target == node)
