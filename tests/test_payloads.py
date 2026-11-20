"""Typed payload schema: INV-EMPTINESS by construction, INV-NO-OSSIFY
mechanically.

The whitelist is the invariant. Nothing inspects D4's contents for
trajectories and rejects them; NO type other than `BiasPrimitive` is even
representable on the D4 write channel (#22), and no overlay record other
than `MatOverlay` on the D2 write channels (#23, #24). Type-inexact
subclassing does not sneak past: membership is exact-type.

The mat lifecycle is the formal `matTick` policy from DryRun.lean:
per tick, one birth (appended), one revalidation (head recycled at the
current epoch), one eviction iff the board exceeds MAT_CAPACITY. Seeded
with a single epoch-0 overlay so the #25 stale read pairs with #24 from
tick 1. The Lean file proves no resident epoch falls more than one tick
behind (mat_fresh) and the board never exceeds capacity (mat_bounded);
these tests verify the same laws empirically and the #25 -> #24 pairing.
"""
from __future__ import annotations

import unittest

from amateur.dryrun import WRITE_FLOWS, DryRun
from amateur.ledger import FLOW_LEDGER
from amateur.payloads import (
    D2_FLOWS,
    D2_WRITES,
    D4_FLOWS,
    D4_WRITES,
    MAT_CAPACITY,
    MAT_SEED,
    BiasPrimitive,
    MatBoard,
    MatOverlay,
    PAYLOAD_SCHEMA,
    allowed_payload,
    require_payload,
)

_TURNS = 200


class _Trajectory:
    """A payload type that must be unrepresentable on D4 (and D2)."""


class TestSchema(unittest.TestCase):
    def test_flows_derived_from_the_ledger(self) -> None:
        def store_flows(store: str) -> set[int]:
            return {
                f.fid
                for f in FLOW_LEDGER
                if f.target.startswith(store) or f.source.startswith(store)
            }

        self.assertEqual(D2_FLOWS, store_flows("D2"))
        self.assertEqual(D4_FLOWS, store_flows("D4"))

    def test_write_channels_are_ratchet_owned(self) -> None:
        for fid in D2_WRITES | D4_WRITES:
            flow = next(f for f in FLOW_LEDGER if f.fid == fid)
            self.assertEqual(flow.source, "6.0 Ratchet & Sediment Grammar")
            self.assertIn(fid, WRITE_FLOWS)

    def test_schema_covers_every_flow(self) -> None:
        self.assertEqual(
            set(PAYLOAD_SCHEMA), {f.fid for f in FLOW_LEDGER}
        )

    def test_d2_d4_channels_are_typed(self) -> None:
        for fid in D2_WRITES:
            self.assertEqual(PAYLOAD_SCHEMA[fid], frozenset({MatOverlay}))
        for fid in D2_FLOWS - D2_WRITES:
            self.assertEqual(
                PAYLOAD_SCHEMA[fid], frozenset({MatOverlay, type(None)})
            )
        for fid in D4_WRITES:
            self.assertEqual(PAYLOAD_SCHEMA[fid], frozenset({BiasPrimitive}))
        for fid in D4_FLOWS - D4_WRITES:
            self.assertEqual(
                PAYLOAD_SCHEMA[fid], frozenset({BiasPrimitive, type(None)})
            )
        for fid in set(range(1, 27)) - D2_FLOWS - D4_FLOWS:
            self.assertEqual(PAYLOAD_SCHEMA[fid], frozenset({type(None)}))

    def test_inv_emptiness_trajectory_unrepresentable(self) -> None:
        for fid in D2_WRITES | D4_WRITES:
            self.assertFalse(
                allowed_payload(fid, _Trajectory()),
                msg=f"#{fid} accepted a trajectory-like payload",
            )
            with self.assertRaises(TypeError, msg=f"#{fid}"):
                require_payload(fid, _Trajectory())

    def test_inv_emptiness_subclassing_is_not_a_loophole(self) -> None:
        class Sneaky(MatOverlay):
            pass

        class Smooth(BiasPrimitive):
            pass

        for fid in D4_WRITES:
            self.assertFalse(allowed_payload(fid, Sneaky(1, "x", "active")))
            self.assertFalse(allowed_payload(fid, Smooth("x", 0.5)))

    def test_record_validation(self) -> None:
        with self.assertRaises(ValueError):
            MatOverlay(0, "x", "frozen")
        with self.assertRaises(ValueError):
            MatOverlay(-1, "x", "active")
        with self.assertRaises(ValueError):
            BiasPrimitive("x", 1.5)
        with self.assertRaises(ValueError):
            BiasPrimitive("x", -0.1)


class TestMatBoard(unittest.TestCase):
    def _drive(self, board: MatBoard, ticks: int) -> None:
        for t in range(1, ticks + 1):
            board.birth(MatOverlay(t, f"cry-{t}", "active"))
            stale_id, _e = board.head()
            board.revalidate(MatOverlay(t, stale_id, "active"))
            board.evict_if_needed(t)

    def test_lifecycle_tracks_the_formal_policy(self) -> None:
        board = MatBoard()
        self._drive(board, _TURNS)
        self.assertLessEqual(len(board.residents()), MAT_CAPACITY)
        for _oid, e in board.residents():
            self.assertIn(e, {_TURNS - 1, _TURNS})

    def test_pairing_and_eviction_over_64_ticks(self) -> None:
        board = MatBoard()
        read_ids: dict[int, str] = {}
        for t in range(1, 65):
            read_ids[t] = board.head()[0]
            board.birth(MatOverlay(t, f"cry-{t}", "active"))
            stale_id, _e = board.head()
            self.assertEqual(stale_id, read_ids[t])
            board.revalidate(MatOverlay(t, stale_id, "active"))
            board.evict_if_needed(t)
            self.assertLessEqual(
                len(board.residents()), MAT_CAPACITY, msg=f"tick {t}"
            )
            for _oid, e in board.residents():
                self.assertIn(e, {t - 1, t}, msg=f"tick {t}")
        for oid, epoch, tick in board.evictions:
            self.assertIn(epoch, {tick - 2})


class TestExecutorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dry = DryRun()
        cls.dry.run(_TURNS)
        cls.events = cls.dry.emissions

    def test_inv_emptiness_no_trajectory_ever_reaches_d4(self) -> None:
        for tick, fid, payload in self.events:
            if fid in D4_FLOWS:
                self.assertIsInstance(
                    payload,
                    (BiasPrimitive, type(None)),
                    msg=f"tick {tick} flow #{fid}: {type(payload).__name__}",
                )

    def test_d2_channels_carry_overlays(self) -> None:
        for tick, fid, payload in self.events:
            if fid in D2_FLOWS:
                self.assertIsInstance(
                    payload,
                    (MatOverlay, type(None)),
                    msg=f"tick {tick} flow #{fid}: {type(payload).__name__}",
                )

    def test_stale_read_pairs_with_same_tick_ratchet_write(self) -> None:
        by_tick: dict[int, dict[int, object]] = {}
        for tick, fid, payload in self.events:
            by_tick.setdefault(tick, {})[fid] = payload
        for tick, payloads in by_tick.items():
            self.assertEqual(
                payloads[25].overlay_id,
                payloads[24].overlay_id,
                msg=f"tick {tick}: #25 read is not what #24 ratcheted",
            )
            self.assertEqual(payloads[25].status, "stale")
            self.assertEqual(payloads[24].status, "active")

    def test_epochs_strictly_increase_per_overlay(self) -> None:
        seen: dict[str, int] = {}
        for tick, fid, payload in self.events:
            if fid in D2_WRITES:
                prior = seen.get(payload.overlay_id)
                if prior is not None:
                    self.assertGreater(
                        payload.epoch,
                        prior,
                        msg=f"{payload.overlay_id} epoch frozen at tick {tick}",
                    )
                seen[payload.overlay_id] = payload.epoch

    def test_no_resident_ever_stale_beyond_one_tick(self) -> None:
        run = DryRun()
        for t in range(1, 65):
            run.board.evict_if_needed(t)
            run.tick()
            for _oid, epoch in run.board.residents():
                self.assertIn(epoch, {t - 1, t}, msg=f"tick {t}")

    def test_evicted_overlays_were_previously_read_stale(self) -> None:
        run = DryRun()
        run.run(_TURNS)
        read_ids = {
            payload.overlay_id
            for _t, fid, payload in run.emissions
            if fid == 25 and payload is not None
        }
        self.assertTrue(run.board.evictions, msg="no evictions ever fired")
        for oid, _epoch, _tick in run.board.evictions:
            self.assertIn(oid, read_ids, msg=f"{oid} evicted unread")

    def test_d4_core_accumulates_only_bias_primitives(self) -> None:
        self.assertEqual(len(self.dry.board.d4), _TURNS)
        for bias in self.dry.board.d4:
            self.assertIsInstance(bias, BiasPrimitive)


if __name__ == "__main__":
    unittest.main()
