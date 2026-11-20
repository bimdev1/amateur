"""Data stores (D1-D4).

Direction contract, enforced by tests: a store's outbound edge is always a
read, a store's inbound edge is always a write or an operation. D4 is the
implemented exception to dynamical writability — "Its stillness is
parametric, not dynamical" — writable only through sedimentation on the
ratchet timescale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .events import (
    ActiveContext,
    BaseOps,
    BoardOps,
    ConsumedScaffolds,
    DigestedDebris,
    DissipatedEntropy,
    EmergentCrystals,
    EvictedClusters,
    MatOverlays,
    RatchetedOverlays,
    SedimentedInvariants,
    SlotState,
    StateDeltas,
    StructuralMotifs,
    StaleOverlays,
    SurvivingCrystals,
)


class ElasticBoard(ABC):
    """D1 — active context, capacity-bounded working surface."""

    @abstractmethod
    def read_context(self) -> ActiveContext:
        """Flow #11: hand active context to the observation loop."""

    @abstractmethod
    def apply(self, deltas: StateDeltas) -> None:
        """P4 input only: #12 is the regulator's evidence, not a D1 write.

        D1's only write edge is #13 (`operate`); a concrete store must not
        treat this hook as a second mutation path (Lead adjudication §9.10,
        2026-11-20).
        """

    @abstractmethod
    def operate(self, ops: BoardOps) -> None:
        """Flow #13: regulator-issued write / erase / resize / evict."""

    @abstractmethod
    def slot_state(self) -> SlotState:
        """Flow #14: report occupancy to the regulator."""

    @abstractmethod
    def evictable_clusters(self) -> EvictedClusters:
        """Flow #15: hand evicted clusters to the digester."""


class MultiLayerMat(ABC):
    """D2 — the nursery: candidate density field, biases routing."""

    @abstractmethod
    def mat_overlays(self) -> MatOverlays:
        """Flow #8: 'Density modifies the next transition'."""

    @abstractmethod
    def birth_crystals(self, crystals: EmergentCrystals) -> None:
        """Flow #18: the loop's emergent crystal enters the nursery as a
        candidate (adjudicated round 11: `EmergentCrystalBody` is the
        candidate kind, in its own D2 index; it is not a `MatOverlay`
        status and it may carry a shadow-scored patch but never a live
        one)."""

    @abstractmethod
    def lay_overlays(self, overlays: RatchetedOverlays) -> None:
        """Flow #24: ratcheted overlays harden into the mat."""

    @abstractmethod
    def surviving_crystals(self) -> SurvivingCrystals:
        """Flow #23: hand survivors to the ratchet for sediment/write-back."""

    @abstractmethod
    def stale_overlays(self) -> StaleOverlays:
        """Flow #25: hand stale overlays to digestion for review."""


class BaseCore(ABC):
    """D4 — the frozen curvature. Read-only at dynamical timescale."""

    @abstractmethod
    def base_ops(self) -> BaseOps:
        """Flow #7: 'A frozen geometry for unfixed unfolding'."""

    @abstractmethod
    def sediment(self, invariants: SedimentedInvariants) -> None:
        """Flow #22: the sole write path, on the sediment timescale:
        'None of these configurations is intrinsically preserved'."""


class DebrisPool(ABC):
    """D3 — compressed remnants awaiting reuse; not a destination."""

    @abstractmethod
    def deposit(self, debris: DigestedDebris) -> None:
        """Flow #19: digestion's debris is stored, never dissolved in
        place: the gradient does not circulate."""

    @abstractmethod
    def scaffolds(self) -> ConsumedScaffolds:
        """Flow #20: recyclable scaffolds back to the digester."""

    @abstractmethod
    def structural_motifs(self) -> StructuralMotifs:
        """Flow #21: the ratchet mines motifs from the pool (round 10)."""

    @abstractmethod
    def avoidance_trails(self) -> AvoidanceTrails:
        """Flow #26: 'prior deformation alters subsequent deformation' —
        trails enter routing as bias, not as instruction."""

    @abstractmethod
    def dissipated_entropy(self) -> DissipatedEntropy:
        """Flow #27: degraded throughput drains back to the boundary."""
