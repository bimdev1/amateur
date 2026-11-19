"""External entities (E1, E2).

Both entities converse with the system only through the Markov blanket
transducer (P1). E1 does not instruct: "Perturbation crosses the boundary.
It does not instruct. It changes local conditions."
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .events import (
    ActiveProbes,
    CrystallizedOutput,
    EntropyExhaust,
    Gradients,
    Perturbations,
)


class ExternalReality(ABC):
    """E1 — source of perturbations, sink for probes and exhaust."""

    @abstractmethod
    def perturb(self) -> Sequence[Perturbations]:
        """Flow #3: emit perturbations across the boundary."""

    @abstractmethod
    def absorb_probes(self, probes: Sequence[ActiveProbes]) -> None:
        """Flow #4: consume the system's active probes."""

    @abstractmethod
    def absorb_exhaust(self, exhaust: Sequence[EntropyExhaust]) -> None:
        """Flow #5: terminal sink. Degraded throughput leaves and never
        circulates back into the system."""


class Gardener(ABC):
    """E2 — injects gradients, receives crystallized output."""

    @abstractmethod
    def gradient(self) -> Gradients:
        """Flow #1: the driving gradient entering the blanket."""

    @abstractmethod
    def receive(self, output: CrystallizedOutput) -> None:
        """Flow #2: 'It gives it curvature' — the core's output to E2."""
