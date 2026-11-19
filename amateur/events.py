"""Typed carriers for the 26 ledger flows.

Payloads stay opaque: v0.10 declares the wiring, not the physics. Every
carrier pins exactly one flow id, so a test can prove that the event
vocabulary and the DFD ledger never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class FlowEvent:
    """One traversal of one ledger edge."""

    payload: Any
    FLOW_ID: ClassVar[int]


_CARRIERS: dict[int, type[FlowEvent]] = {}


def _carrier(name: str, fid: int) -> type[FlowEvent]:
    cls = type(name, (FlowEvent,), {"FLOW_ID": fid, "__module__": __name__})
    cls.__doc__ = f"Flow #{fid} payload carrier."
    carrier = dataclass(frozen=True)(cls)
    _CARRIERS[fid] = carrier
    return carrier


Gradients = _carrier("Gradients", 1)
CrystallizedOutput = _carrier("CrystallizedOutput", 2)
Perturbations = _carrier("Perturbations", 3)
ActiveProbes = _carrier("ActiveProbes", 4)
EntropyExhaust = _carrier("EntropyExhaust", 5)
PredictionError = _carrier("PredictionError", 6)
BaseOps = _carrier("BaseOps", 7)
MatOverlays = _carrier("MatOverlays", 8)
RoutedDynamics = _carrier("RoutedDynamics", 9)
ResolvedTrajectory = _carrier("ResolvedTrajectory", 10)
ActiveContext = _carrier("ActiveContext", 11)
StateDeltas = _carrier("StateDeltas", 12)
BoardOps = _carrier("BoardOps", 13)
SlotState = _carrier("SlotState", 14)
EvictedClusters = _carrier("EvictedClusters", 15)
CostErrorMetrics = _carrier("CostErrorMetrics", 16)
CollapseMandate = _carrier("CollapseMandate", 17)
EmergentCrystals = _carrier("EmergentCrystals", 18)
DigestedDebris = _carrier("DigestedDebris", 19)
ConsumedScaffolds = _carrier("ConsumedScaffolds", 20)
StructuralMotifs = _carrier("StructuralMotifs", 21)
SedimentedInvariants = _carrier("SedimentedInvariants", 22)
SurvivingCrystals = _carrier("SurvivingCrystals", 23)
RatchetedOverlays = _carrier("RatchetedOverlays", 24)
StaleOverlays = _carrier("StaleOverlays", 25)
AvoidanceTrails = _carrier("AvoidanceTrails", 26)


def carrier_for(fid: int) -> type[FlowEvent]:
    """Return the carrier class registered for a ledger flow id."""
    try:
        return _CARRIERS[fid]
    except KeyError:
        raise ValueError(f"no carrier registered for flow #{fid}") from None
