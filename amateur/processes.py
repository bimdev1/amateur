"""Processes (P1-P6).

Each method cites the ledger flows it owns. Method bodies are abstract:
the scaffold fixes the wiring and the contracts, not the physics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .events import (
    ActiveContext,
    ActiveProbes,
    AvoidanceTrails,
    BaseOps,
    CollapseMandate,
    ConsumedScaffolds,
    CostErrorMetrics,
    CrystallizedOutput,
    DigestedDebris,
    EmergentCrystals,
    EntropyExhaust,
    EvictedClusters,
    Gradients,
    MatOverlays,
    Perturbations,
    PredictionError,
    ResolvedTrajectory,
    RoutedDynamics,
    SlotState,
    StaleOverlays,
    StateDeltas,
    StructuralMotifs,
)


class MarkovBlanketTransducer(ABC):
    """P1 — 1.0 Transduce Markov Blanket.

    Inputs: #1 Gradients (E2), #3 Perturbations (E1).
    Outputs: #2, #4, #5, #6. The boundary crossing point; nothing reaches
    the interior except through here.
    """

    @abstractmethod
    def transduce(
        self, gradients: Sequence[Gradients], perturbations: Sequence[Perturbations]
    ) -> tuple[
        CrystallizedOutput,
        Sequence[ActiveProbes],
        Sequence[EntropyExhaust],
        Sequence[PredictionError],
    ]:
        """Flows #2, #4, #5, #6: output crystallization, probing,
        terminal exhaust, and surprise extraction."""


class Router(ABC):
    """P2 — 2.0 Route Without Ossification.

    Inputs: #6 Prediction Error, #7 Base Ops, #8 Mat Overlays, #26 Avoidance
    Trails. Output: #9. 'No direction carries a name': routing reads bias
    and density only.
    """

    @abstractmethod
    def route(
        self,
        error: Sequence[PredictionError],
        base: BaseOps,
        overlays: MatOverlays,
        trails: AvoidanceTrails,
    ) -> RoutedDynamics:
        """Flow #9: 'Routed Dynamics' — bias-conditioned continuation."""


class Observer(ABC):
    """P3 — 3.0 Execute Observation Loop (Core Engine).

    Inputs: #9 Routed Dynamics, #11 Active Context.
    Outputs: #10 Resolved Trajectory, #12 State Deltas, #18 Emergent
    Crystals.
    """

    @abstractmethod
    def observe(
        self, dynamics: RoutedDynamics, context: ActiveContext
    ) -> tuple[ResolvedTrajectory, StateDeltas, EmergentCrystals]:
        """Flows #10, #12, #18."""


class WhiteboardRegulator(ABC):
    """P4 — 4.0 Allocate & Regulate Whiteboard.

    Inputs: #12 State Deltas, #14 Slot State (read), #17 Collapse Mandate.
    Outputs: #13 write/erase/resize/evict, #16 Cost vs. Error Metrics.
    """

    @abstractmethod
    def regulate(
        self, deltas: StateDeltas, slots: SlotState, mandate: CollapseMandate
    ) -> tuple[BoardOps, CostErrorMetrics]:
        """Flows #13, #16."""


class Digester(ABC):
    """P5 — 5.0 Digest & Collapse Ceiling-Bound Units.

    Inputs: #15 Evicted Clusters, #16 Cost vs. Error Metrics, #18 Emergent
    Crystals, #20 Consumed Scaffolds.
    Outputs: #17 Collapse Mandate, #19 Digested Debris, #21 Structural
    Motifs. 'A contracted region suppresses difference until sufficient
    discrepancy restores variation.'
    """

    @abstractmethod
    def digest(
        self,
        clusters: EvictedClusters,
        metrics: CostErrorMetrics,
        crystals: EmergentCrystals,
        scaffolds: ConsumedScaffolds,
    ) -> tuple[CollapseMandate, DigestedDebris, StructuralMotifs]:
        """Flows #17, #19, #21."""


class Ratchet(ABC):
    """P6 — 6.0 Ratchet & Sediment Grammar.

    Inputs: #21 Structural Motifs, #25 Stale Overlays (read).
    Outputs: #22 Sedimented Invariants, #23 Surviving Crystals, #24
    Ratcheted Overlays. 'A settled region resists displacement.'
    """

    @abstractmethod
    def ratchet(
        self, motifs: StructuralMotifs, stale: StaleOverlays
    ) -> tuple[SedimentedInvariants, SurvivingCrystals, RatchetedOverlays]:
        """Flows #22, #23, #24: only survivors harden."""
