"""amateur — not agi.

v0.10 scaffold of the loop architecture declared in `v0.10-untested.md`:
entities E1/E2, processes P1-P6, stores D1-D4, and the 27-flow ledger that
binds the diagram, the prose, and the code together.
"""
from __future__ import annotations

from .ledger import FLOW_LEDGER, NODE_NAMES, Flow
from .events import FlowEvent, carrier_for

__version__ = "0.10.0"

__all__ = [
    "FLOW_LEDGER",
    "Flow",
    "FlowEvent",
    "NODE_NAMES",
    "__version__",
    "carrier_for",
]
