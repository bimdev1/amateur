"""Contract tests: documentation and error paths.

- Every abstract method on every component must cite the ledger flow ids
  it owns (`#<n>` in its docstring), and the union of citations must cover
  exactly flows #1-#27. Doc drift becomes a test failure.
- Error paths: unknown flow ids rejected; abstract components refuse
  direct instantiation.
"""
from __future__ import annotations

import re
import unittest

from amateur import entities, events, ledger, processes, stores

FLOW_REF = re.compile(r"#(\d+)")
COMPONENT_MODULES = (entities, stores, processes)


class TestDocCitations(unittest.TestCase):
    def test_every_abstract_method_cites_its_flows(self) -> None:
        for module in COMPONENT_MODULES:
            for name, cls in vars(module).items():
                if not (isinstance(cls, type) and cls.__module__ == module.__name__):
                    continue
                for method_name in cls.__abstractmethods__:
                    method = getattr(cls, method_name)
                    self.assertIsNotNone(
                        method.__doc__,
                        msg=f"{name}.{method_name} lacks a docstring",
                    )
                    self.assertRegex(
                        method.__doc__ or "",
                        FLOW_REF,
                        msg=f"{name}.{method_name} does not cite flow ids",
                    )

    def test_citations_cover_exactly_the_ledger(self) -> None:
        cited: set[int] = set()
        for module in COMPONENT_MODULES:
            for cls in vars(module).values():
                if not (isinstance(cls, type) and cls.__module__ == module.__name__):
                    continue
                for method_name in cls.__abstractmethods__:
                    doc = getattr(cls, method_name).__doc__ or ""
                    cited.update(int(n) for n in FLOW_REF.findall(doc))
        self.assertEqual(cited, {f.fid for f in ledger.FLOW_LEDGER})


class TestErrorPaths(unittest.TestCase):
    def test_unknown_flow_id_is_rejected(self) -> None:
        for bad in (0, 28, -1):
            with self.assertRaises(ValueError):
                events.carrier_for(bad)

    def test_abstract_components_refuse_instantiation(self) -> None:
        for module in COMPONENT_MODULES:
            for cls in vars(module).values():
                if isinstance(cls, type) and cls.__module__ == module.__name__:
                    with self.assertRaises(TypeError, msg=cls.__name__):
                        cls()


if __name__ == "__main__":
    unittest.main()
