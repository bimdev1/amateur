"""Loop wiring tests: the engine must be constructible, immutable, and
complete — every DFD node is a field, and every field satisfies its
abstract contract. Stubs prove the contracts are implementable.
"""
from __future__ import annotations

import dataclasses
import unittest
from typing import Any

from amateur.engine import Loop
from amateur.entities import ExternalReality, Gardener
from amateur.ledger import FLOW_LEDGER, NODE_NAMES, PROCESSES, STORES
from amateur.processes import (
    MarkovBlanketTransducer,
    Observer,
    Router,
    Ratchet,
    WhiteboardRegulator,
    Digester,
)
from amateur.stores import BaseCore, DebrisPool, ElasticBoard, MultiLayerMat


def _stub(abstract_cls: type) -> Any:
    """Minimal concrete instance of an ABC: None-returning overrides."""
    methods = {
        name: (lambda *args, **kwargs: None)
        for name in abstract_cls.__abstractmethods__
    }
    return type(f"{abstract_cls.__name__}Stub", (abstract_cls,), methods)()


def _wired_loop(**overrides: Any) -> Loop:
    parts: dict[str, Any] = {
        "environment": _stub(ExternalReality),
        "gardener": _stub(Gardener),
        "blanket": _stub(MarkovBlanketTransducer),
        "router": _stub(Router),
        "observer": _stub(Observer),
        "regulator": _stub(WhiteboardRegulator),
        "digester": _stub(Digester),
        "ratchet": _stub(Ratchet),
        "board": _stub(ElasticBoard),
        "mat": _stub(MultiLayerMat),
        "debris": _stub(DebrisPool),
        "base_core": _stub(BaseCore),
    }
    parts.update(overrides)
    return Loop(**parts)


class TestLoopWiring(unittest.TestCase):
    def test_loop_is_constructible_from_contracts(self) -> None:
        loop = _wired_loop()
        self.assertEqual(loop.blueprint(), FLOW_LEDGER)

    def test_every_dfd_node_has_a_field(self) -> None:
        n_processes, n_stores = len(PROCESSES), len(STORES)
        n_entities = len(NODE_NAMES) - n_processes - n_stores
        self.assertEqual(len(dataclasses.fields(Loop)), n_processes + n_stores + n_entities)

    def test_step_is_an_explicit_punt(self) -> None:
        with self.assertRaises(NotImplementedError):
            _wired_loop().step()

    def test_wiring_is_frozen(self) -> None:
        loop = _wired_loop()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            loop.ratchet = _stub(Ratchet)  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
