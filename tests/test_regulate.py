"""P4 minimal regulator: evidence-only #13/#16 derivation (phase 1).

These tests pin the adjudicated guards that are testable without the
executor (Lead adjudication §9.11):

- refusal atomicity is pinned in ``test_payloads.TestBoardStore``;
- **no-constant audit**: ``nbytes == 8 * len(delta)``, ``cost ==
  sum(slot.nbytes)``, ``body_id``/``slot_id`` are digest-derived, and no
  state-bearing literal (the latent-width byte count) appears in the
  regulator or store source;
- **no #17 consumption**: the signature has no mandate parameter and the
  module imports no mandate type (cross-round obligation is recorded in
  ``amateur/regulate.py``);
- purity: no side effects, deterministic, stateless.

The regulator is Python-tested, not proved.
"""
from __future__ import annotations

import inspect
import unittest

import amateur.regulate as regulate
from amateur.payloads import (
    BoardOpBody,
    BoardStore,
    CostErrorBody,
    SlotStateBody,
    StateDeltaBody,
)
from amateur.regulate import Regulator, content_digest, represented_bytes
from amateur.provenance import Provenance


def _delta(fill: float = 1.0, surprise: float = 0.25) -> StateDeltaBody:
    return StateDeltaBody((fill,) + (0.0,) * 63, surprise)


def _state(
    occupied: int = 2,
    capacity: int = 8,
    nbytes: int = 1024,
    slot_ids: tuple[str, ...] | None = None,
    oldest_slot_id: str | None = None,
) -> SlotStateBody:
    ids = slot_ids if slot_ids is not None else tuple(
        f"s{index}" for index in range(occupied)
    )
    if oldest_slot_id is None and ids:
        oldest_slot_id = ids[0]
    return SlotStateBody(
        capacity=capacity,
        occupied=occupied,
        nbytes=nbytes,
        slot_ids=ids,
        oldest_slot_id=oldest_slot_id,
    )


class TestP4ColdStartAndCapacity(unittest.TestCase):
    def test_cold_start_writes_the_first_slot(self) -> None:
        delta = _delta()
        op, metrics = Regulator().regulate(delta, None, 7)
        self.assertIs(type(op), BoardOpBody)
        self.assertIs(type(metrics), CostErrorBody)
        self.assertEqual(op.op, "write")
        self.assertEqual(op.reason, "cold-start")
        self.assertEqual(op.tick, 7)
        self.assertEqual(op.body_id, content_digest(delta))
        self.assertEqual(op.slot_id, f"ctx-7-{content_digest(delta)[:12]}")
        self.assertEqual(op.nbytes, 8 * len(delta.delta))
        self.assertGreater(op.nbytes, 0)
        self.assertEqual(metrics.cost, 0.0)
        self.assertEqual(metrics.error, delta.surprise)

    def test_free_slot_writes_and_costs_the_current_bytes(self) -> None:
        delta = _delta()
        slots = _state(occupied=2, nbytes=1024)
        op, metrics = Regulator().regulate(delta, slots, 8)
        self.assertEqual(op.op, "write")
        self.assertEqual(op.reason, "free-slot")
        self.assertEqual(op.slot_id, f"ctx-8-{content_digest(delta)[:12]}")
        self.assertEqual(op.body_id, content_digest(delta))
        self.assertEqual(metrics.cost, float(slots.nbytes))
        self.assertEqual(metrics.error, delta.surprise)

    def test_full_board_evicts_the_oldest_slot(self) -> None:
        slots = _state(occupied=8, nbytes=4096)
        op, metrics = Regulator().regulate(_delta(), slots, 9)
        self.assertEqual(op.op, "evict")
        self.assertEqual(op.reason, "capacity-evict")
        self.assertEqual(op.slot_id, "s0")
        self.assertIsNone(op.body_id)
        self.assertEqual(op.nbytes, 0)
        self.assertEqual(metrics.cost, float(slots.nbytes))

    def test_oldest_slot_is_the_eviction_target(self) -> None:
        slots = _state(
            occupied=3,
            capacity=3,
            nbytes=1536,
            slot_ids=("old", "middle", "new"),
            oldest_slot_id="old",
        )
        op, _metrics = Regulator().regulate(_delta(), slots, 5)
        self.assertEqual(op.op, "evict")
        self.assertEqual(op.slot_id, "old")


class TestP4EvidenceDependence(unittest.TestCase):
    def test_delta_content_drives_body_and_slot_identity(self) -> None:
        first = _delta(fill=1.0)
        second = _delta(fill=2.0)
        first_op, _ = Regulator().regulate(first, None, 3)
        second_op, _ = Regulator().regulate(second, None, 3)
        self.assertNotEqual(first_op.body_id, second_op.body_id)
        self.assertNotEqual(first_op.slot_id, second_op.slot_id)
        self.assertEqual(
            first_op.body_id, content_digest(first)
        )
        self.assertEqual(
            second_op.body_id, content_digest(second)
        )

    def test_same_evidence_is_deterministic_across_ticks(self) -> None:
        delta = _delta()
        first, first_metrics = Regulator().regulate(delta, None, 3)
        second, second_metrics = Regulator().regulate(delta, None, 4)
        self.assertEqual(first.body_id, second.body_id)
        self.assertEqual(first.nbytes, second.nbytes)
        self.assertEqual(first_metrics, second_metrics)
        self.assertNotEqual(first.slot_id, second.slot_id)  # tick names it

    def test_error_equals_the_measured_surprise(self) -> None:
        for surprise in (0.0, 0.25, 3.5):
            _op, metrics = Regulator().regulate(
                _delta(surprise=surprise), None, 1
            )
            self.assertEqual(metrics.error, surprise)

    def test_cost_tracks_d1_bytes_and_occupancy_drives_the_branch(self) -> None:
        state = _state(occupied=2, nbytes=1024)
        _op, metrics = Regulator().regulate(_delta(), state, 4)
        self.assertEqual(metrics.cost, 1024.0)
        # Free-slot -> full changes the op, not just the numbers.
        full = _state(occupied=8, nbytes=4096)
        op, metrics = Regulator().regulate(_delta(), full, 4)
        self.assertEqual(op.op, "evict")
        self.assertEqual(metrics.cost, 4096.0)

    def test_content_digest_is_stable_and_hexadecimal(self) -> None:
        delta = _delta()
        digest = content_digest(delta)
        self.assertEqual(digest, content_digest(delta))
        self.assertEqual(len(digest), 32)  # 16-byte blake2b
        int(digest, 16)  # raises if not hex

    def test_represented_bytes_is_computed_from_the_delta(self) -> None:
        delta = _delta()
        self.assertEqual(represented_bytes(delta), 8 * len(delta.delta))
        self.assertGreater(represented_bytes(delta), 0)


class TestP4NoConstantAudit(unittest.TestCase):
    """§9.11b: every state-bearing field is traceable to evidence."""

    def test_nbytes_is_a_computation_not_a_literal(self) -> None:
        source = inspect.getsource(regulate)
        self.assertNotIn("512", source)
        self.assertIn("8 * len(delta.delta)", source)
        delta = _delta()
        op, _metrics = Regulator().regulate(delta, None, 6)
        self.assertEqual(op.nbytes, 8 * len(delta.delta))

    def test_cost_matches_the_store_byte_sum(self) -> None:
        store = BoardStore()
        delta = _delta()
        size = represented_bytes(delta)
        for tick, slot_id in ((1, "a"), (2, "b"), (3, "c")):
            op = BoardOpBody(
                "write", slot_id, content_digest(delta), size, "test", tick
            )
            store.apply(op, Provenance(13, tick))
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.occupied, len(state.slot_ids))
        self.assertEqual(
            state.nbytes, sum(slot.nbytes for slot in store.slots())
        )
        _op, metrics = Regulator().regulate(delta, state, 4)
        self.assertEqual(metrics.cost, float(state.nbytes))
        self.assertEqual(metrics.cost, float(sum(
            slot.nbytes for slot in store.slots()
        )))

    def test_slot_and_body_ids_are_digest_derived(self) -> None:
        delta = _delta()
        digest = content_digest(delta)
        op, _metrics = Regulator().regulate(delta, None, 11)
        self.assertEqual(op.body_id, digest)
        self.assertIn(digest[:12], op.slot_id)
        self.assertNotEqual(op.body_id, "0" * 32)
        self.assertNotEqual(op.slot_id, "ctx")

    def test_no_state_literal_in_the_store(self) -> None:
        source = inspect.getsource(BoardStore)
        self.assertNotIn("512", source)
        self.assertIn("metadata footprint", BoardStore.__doc__ or "")
        self.assertIn("metadata footprint", CostErrorBody.__doc__ or "")


class TestP4PurityAndContract(unittest.TestCase):
    def test_regulator_is_stateless(self) -> None:
        regulator = Regulator()
        self.assertEqual(vars(regulator), {})
        regulator.regulate(_delta(), None, 1)
        regulator.regulate(_delta(fill=2.0), _state(), 2)
        self.assertEqual(vars(regulator), {})

    def test_regulator_does_not_touch_the_store(self) -> None:
        store = BoardStore()
        store.apply(
            BoardOpBody("write", "a", content_digest(_delta()), 512, "t", 1),
            Provenance(13, 1),
        )
        before = store.state_digest()
        state = store.slot_state()
        Regulator().regulate(_delta(), state, 2)
        self.assertEqual(store.state_digest(), before)

    def test_no_mandate_parameter_and_no_mandate_import(self) -> None:
        """§9.6: #17 is not consumed this slice; the obligation is recorded."""
        signature = inspect.signature(Regulator.regulate)
        self.assertEqual(
            tuple(signature.parameters), ("self", "delta", "slots", "tick")
        )
        self.assertFalse(hasattr(regulate, "CollapseMandateBody"))
        self.assertFalse(hasattr(regulate, "PhysicsOrgan"))
        # The module namespace carries only the payload types it consumes.
        self.assertEqual(
            {
                name for name in vars(regulate)
                if name.endswith("Body")
            },
            {"BoardOpBody", "CostErrorBody", "SlotStateBody", "StateDeltaBody"},
        )

    def test_wrong_types_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Regulator().regulate("delta", None, 1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Regulator().regulate(_delta(), "slots", 1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Regulator().regulate(_delta(), None, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Regulator().regulate(_delta(), None, -1)


if __name__ == "__main__":
    unittest.main()
