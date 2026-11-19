"""Engine wiring (the loop) — v0.10 scaffold.

The step order follows the ledger's causal spine:

    transduce (#6) -> route (#9) -> observe (#10, #12, #18)
    observe -> regulate (#13, #16) -> digest (#17, #19, #21)
    digest -> ratchet (#22, #23, #24) -> stores

Thermodynamic invariant: the only degenerate throughflow out of the system
is P1's Entropy Exhaust (#5), drained once per tick into E1. No edge
returns it; see tests/test_thermodynamics.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from .entities import ExternalReality, Gardener
from .ledger import Flow, FLOW_LEDGER
from .processes import (
    MarkovBlanketTransducer,
    Observer,
    Router,
    Ratchet,
    WhiteboardRegulator,
    Digester,
)
from .stores import BaseCore, DebrisPool, ElasticBoard, MultiLayerMat


@dataclass(frozen=True)
class Loop:
    """Declarative wiring of the v0.10 architecture.

    Every node of the DFD is a field here: entities E1/E2, processes
    P1-P6, stores D1-D4. A wiring that cannot name its stores is not a
    wiring.
    """

    environment: ExternalReality
    gardener: Gardener
    blanket: MarkovBlanketTransducer
    router: Router
    observer: Observer
    regulator: WhiteboardRegulator
    digester: Digester
    ratchet: Ratchet
    board: ElasticBoard
    mat: MultiLayerMat
    debris: DebrisPool
    base_core: BaseCore

    def blueprint(self) -> tuple[Flow, ...]:
        """The 26-flow wiring this loop is contractually bound to."""
        return FLOW_LEDGER

    def step(self) -> None:
        """Advance one tick of the loop.

        Unimplemented by design (v0.10 scaffold): the DFD declares the
        wiring; dynamics arrive in a later revision, implemented against
        the process and store contracts.
        """
        raise NotImplementedError(
            "v0.10 scaffold: wiring only, no dynamics yet"
        )
