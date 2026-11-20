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
    BOARD_CAPACITY,
    D2_FLOWS,
    D2_WRITES,
    D4_FLOWS,
    D4_WRITES,
    DEBRIS_CAPACITY,
    DEBRIS_MAX_BYTES,
    MAX_PATCH_BYTES,
    MAX_PATCH_TOTAL_BYTES,
    MAT_CAPACITY,
    MAT_SEED,
    NO_DEBRIS_ID,
    NOOP_OVERLAY_ID,
    REAL_BODIES,
    BiasPrimitive,
    BoardOpBody,
    BoardSlot,
    BoardStore,
    CollapseMandateBody,
    CostErrorBody,
    DebrisPool,
    DigestedDebrisBody,
    EmergentCrystalBody,
    FrameBody,
    LatentRolloutBody,
    MatBoard,
    MatOverlay,
    PAYLOAD_SCHEMA,
    PredictionErrorBody,
    SlotStateBody,
    StateDeltaBody,
    StructuralMotifBody,
    allowed_payload,
    noop_debris,
    noop_mandate,
    noop_motif,
    noop_overlay,
    require_payload,
)
from amateur.provenance import Envelope, Provenance

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
            if fid in REAL_BODIES:
                self.assertEqual(
                    PAYLOAD_SCHEMA[fid],
                    frozenset({REAL_BODIES[fid], type(None)}),
                )
            else:
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
        for tick, fid, envelope in self.events:
            if fid in D4_FLOWS:
                body = envelope.body
                self.assertIsInstance(
                    body,
                    (BiasPrimitive, type(None)),
                    msg=f"tick {tick} flow #{fid}: {type(body).__name__}",
                )

    def test_d2_channels_carry_overlays(self) -> None:
        for tick, fid, envelope in self.events:
            if fid in D2_FLOWS:
                body = envelope.body
                self.assertIsInstance(
                    body,
                    (MatOverlay, type(None)),
                    msg=f"tick {tick} flow #{fid}: {type(body).__name__}",
                )

    def test_stale_read_pairs_with_same_tick_ratchet_write(self) -> None:
        by_tick: dict[int, dict[int, Envelope]] = {}
        for tick, fid, envelope in self.events:
            by_tick.setdefault(tick, {})[fid] = envelope
        for tick, envelopes in by_tick.items():
            self.assertEqual(
                envelopes[25].body.overlay_id,
                envelopes[24].body.overlay_id,
                msg=f"tick {tick}: #25 read is not what #24 ratcheted",
            )
            self.assertEqual(envelopes[25].body.status, "stale")
            self.assertEqual(envelopes[24].body.status, "active")
            # The read predates the same-tick rewrite; it inherits the
            # stored record's provenance, not #24's new birth stamp.
            self.assertLess(envelopes[25].provenance.tick_born, tick)

    def test_epochs_strictly_increase_per_overlay(self) -> None:
        seen: dict[str, int] = {}
        for tick, fid, envelope in self.events:
            if fid in D2_WRITES:
                body = envelope.body
                prior = seen.get(body.overlay_id)
                if prior is not None:
                    self.assertGreater(
                        body.epoch,
                        prior,
                        msg=f"{body.overlay_id} epoch frozen at tick {tick}",
                    )
                seen[body.overlay_id] = body.epoch

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
            envelope.body.overlay_id
            for _t, fid, envelope in run.emissions
            if fid == 25 and envelope.body is not None
        }
        self.assertTrue(run.board.evictions, msg="no evictions ever fired")
        for oid, _epoch, _tick in run.board.evictions:
            self.assertIn(oid, read_ids, msg=f"{oid} evicted unread")

    def test_d4_core_accumulates_only_bias_primitives(self) -> None:
        self.assertEqual(len(self.dry.board.d4), _TURNS)
        for bias in self.dry.board.d4:
            self.assertIsInstance(bias, BiasPrimitive)


class TestEmissionProvenance(unittest.TestCase):
    """A2 at the channel: every emission is an envelope, reads inherit the
    stored record's provenance, #7 hands back the exact sedimented object,
    and writes are born at their emission tick."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dry = DryRun()
        cls.dry.run(32)
        cls.events = cls.dry.emissions

    def test_every_emission_is_an_envelope_with_ordered_provenance(self) -> None:
        for tick, fid, envelope in self.events:
            self.assertIs(
                type(envelope), Envelope, msg=f"tick {tick} flow #{fid}"
            )
            self.assertIs(
                type(envelope.provenance),
                Provenance,
                msg=f"tick {tick} flow #{fid}",
            )
            self.assertIn(fid, {f.fid for f in FLOW_LEDGER})
            self.assertLessEqual(
                envelope.provenance.tick_born, tick, msg=f"#{fid}"
            )

    def test_d4_read_body_is_the_exact_sedimented_object(self) -> None:
        by_fid: dict[int, dict[int, Envelope]] = {}
        for tick, fid, envelope in self.events:
            by_fid.setdefault(fid, {})[tick] = envelope
        for tick in range(2, 33):
            read = by_fid[7][tick]
            prior = by_fid[22][tick - 1]
            self.assertIs(read.body, prior.body, msg=f"tick {tick}")
            self.assertIs(read.body, self.dry.board.d4[tick - 2])
            self.assertEqual(read.provenance, prior.provenance)
            self.assertLess(read.provenance.tick_born, tick)

    def test_reads_inherit_the_producing_write_provenance(self) -> None:
        writes = {
            (envelope.provenance.source_fid, envelope.provenance.tick_born):
                envelope
            for _tick, fid, envelope in self.events
            if fid in D2_WRITES
        }
        for tick, fid, envelope in self.events:
            if fid not in (8, 25) or envelope.body is None:
                continue
            provenance = envelope.provenance
            self.assertLess(provenance.tick_born, tick, msg=f"#{fid}")
            if provenance == Provenance(23, 0):
                continue  # the MAT_SEED record predates every write
            writer = writes[(provenance.source_fid, provenance.tick_born)]
            self.assertEqual(
                writer.body.overlay_id,
                envelope.body.overlay_id,
                msg=f"tick {tick} flow #{fid}",
            )



# ---------------------------------------------------------------------------
# Phase 4-5b: typed physics bodies, D3 pool, D2 patch accounting
# ---------------------------------------------------------------------------
def _digest_body(debris_id: str = "debris-001", count: int = 3) -> DigestedDebrisBody:
    return DigestedDebrisBody(
        debris_id=debris_id,
        centroid=(0.5,) * 64,
        count=count,
        spread=1.0,
        first_tick=1,
        last_tick=2,
        peak_surprise=0.5,
    )


class TestPhysicsBodies(unittest.TestCase):
    def test_bodies_are_frozen_and_hashable(self) -> None:
        bodies = [
            FrameBody(b"\x00" * 3, 1, 1, 0),
            PredictionErrorBody(0.5, (0.0,) * 64),
            LatentRolloutBody(((0.0,) * 64,) * 4),
            StateDeltaBody((0.0,) * 64, 0.5),
            CollapseMandateBody("ceiling-pressure", 8, 8, 7, "debris-001", 3),
            EmergentCrystalBody((0.0,) * 64, 0.5, 3),
            _digest_body(),
            StructuralMotifBody("motif-001", "debris-001", (0.0,) * 64, 4, 1.0, 4, 0.0, 1, 4),
        ]
        for body in bodies:
            self.assertEqual(hash(body), hash(body))
            self.assertIn(type(body), PAYLOAD_SCHEMA[_fid_for_body(body)])

    def test_body_validation(self) -> None:
        with self.assertRaises(ValueError):
            FrameBody(b"\x00" * 5, 1, 1, 0)  # length != h*w*3
        with self.assertRaises(ValueError):
            PredictionErrorBody(float("nan"), (0.0,) * 64)
        with self.assertRaises(ValueError):
            PredictionErrorBody(0.5, (0.0,) * 3)  # wrong latent width
        with self.assertRaises(ValueError):
            MatOverlay(0, "x", "frozen")
        with self.assertRaises(ValueError):
            _digest_body().__class__("id", (0.0,) * 3, 1, 0.0, 0, 0, 0.0)

    def test_no_op_sentinels_are_real_exact_type_bodies(self) -> None:
        self.assertEqual(noop_overlay().overlay_id, NOOP_OVERLAY_ID)
        self.assertEqual(noop_debris().debris_id, NO_DEBRIS_ID)
        self.assertEqual(noop_debris().count, 0)
        self.assertEqual(noop_motif().persistence, 0)
        self.assertEqual(noop_mandate().pressure, 0)
        self.assertIn(type(noop_debris()), PAYLOAD_SCHEMA[19])
        self.assertIn(type(noop_motif()), PAYLOAD_SCHEMA[21])
        self.assertIn(type(noop_mandate()), PAYLOAD_SCHEMA[17])
        self.assertIn(type(noop_overlay()), PAYLOAD_SCHEMA[23])


def _fid_for_body(body: object) -> int:
    for fid, body_type in REAL_BODIES.items():
        if type(body) is body_type:
            return fid
    if type(body) is MatOverlay:
        return 23
    raise AssertionError(f"{type(body).__name__} has no flow")


class TestDebrisPool(unittest.TestCase):
    def test_capacity_and_provenance(self) -> None:
        pool = DebrisPool(capacity=2, max_bytes=DEBRIS_MAX_BYTES)
        provenance = (19, 1)
        pool.deposit(_digest_body("a"), provenance, 1)
        pool.deposit(_digest_body("b"), provenance, 2)
        self.assertEqual(len(pool), 2)
        self.assertLessEqual(pool.byte_size(), DEBRIS_MAX_BYTES)
        evicted = pool.deposit(_digest_body("c"), provenance, 3)
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].body.debris_id, "a")
        self.assertEqual(pool.newest().body.debris_id, "c")
        self.assertEqual(pool.newest().provenance, provenance)

    def test_byte_ceiling_evicts(self) -> None:
        one = len(_digest_body("a").debris_id) + 64 * 8 + 8 * 5
        pool = DebrisPool(capacity=8, max_bytes=one + 8)  # room for one body only
        pool.deposit(_digest_body("a"), (19, 1), 1)
        evicted = pool.deposit(_digest_body("b"), (19, 2), 2)
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].body.debris_id, "a")
        self.assertLessEqual(pool.byte_size(), pool.max_bytes)

    def test_no_op_is_never_deposited(self) -> None:
        pool = DebrisPool()
        with self.assertRaises(ValueError):
            pool.deposit(noop_debris(), (19, 1), 1)
        self.assertEqual(len(pool), 0)

    def test_default_bounds(self) -> None:
        self.assertEqual(DebrisPool().capacity, DEBRIS_CAPACITY)
        self.assertEqual(DebrisPool().max_bytes, DEBRIS_MAX_BYTES)


class _Patch:
    def __init__(self, overlay_id: str, nbytes: int, param_count: int = 12) -> None:
        self.overlay_id = overlay_id
        self.nbytes = nbytes
        self.param_count = param_count


class TestMatBoardPatches(unittest.TestCase):
    def test_attach_and_account(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-1", "active"))
        board.attach_patch("ovr-1", _Patch("ovr-1", 1000))
        self.assertEqual(board.patch_bytes(), 1000)
        self.assertEqual(board.patch_params(), 12)
        self.assertIsNotNone(board.patch("ovr-1"))
        board.assert_patch_bounds(require_complete=True)
        self.assertIn("ovr-1", {oid for oid, _ in board.residents()})

    def test_per_patch_ceiling_rejects_full_patch(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-full", "active"))
        with self.assertRaises(ValueError):
            board.attach_patch("ovr-full", _Patch("ovr-full", 843_680))
        self.assertEqual(board.patch_bytes(), 0)
        self.assertGreater(MAX_PATCH_BYTES, 136_960)
        self.assertLess(MAX_PATCH_BYTES, 843_680)

    def test_attach_requires_a_resident(self) -> None:
        board = MatBoard(seed=False)
        with self.assertRaises(AssertionError):
            board.attach_patch("ghost", _Patch("ghost", 10))

    def test_table_ceiling_evicts_record_and_patch_atomically(self) -> None:
        board = MatBoard(seed=False, max_patch_bytes=1500, max_patch_total_bytes=1500)
        board.birth(MatOverlay(1, "ovr-1", "active"))
        board.attach_patch("ovr-1", _Patch("ovr-1", 1000))
        board.birth(MatOverlay(2, "ovr-2", "active"))
        board.attach_patch("ovr-2", _Patch("ovr-2", 1000))
        board.enforce_patch_ceiling(3)
        self.assertEqual(board.resident_ids(), frozenset({"ovr-2"}))
        self.assertEqual(tuple(oid for oid, _p in board.patches()), ("ovr-2",))
        self.assertIn(("ovr-1", 1, 3), board.byte_evictions)
        board.assert_patch_bounds(require_complete=True)

    def test_capacity_eviction_removes_the_patch(self) -> None:
        board = MatBoard(seed=False)
        for index in range(MAT_CAPACITY + 1):
            board.birth(MatOverlay(index + 1, f"ovr-{index}", "active"))
            board.attach_patch(f"ovr-{index}", _Patch(f"ovr-{index}", 10))
        board.evict_if_needed(9)
        self.assertEqual(len(board.residents()), MAT_CAPACITY)
        self.assertEqual(len(board.patches()), MAT_CAPACITY)
        self.assertNotIn("ovr-0", board.resident_ids())
        self.assertIsNone(board.patch("ovr-0"))
        board.assert_patch_bounds(require_complete=True)

    def test_completeness_assert_catches_unpatched_record(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-1", "active"))
        with self.assertRaises(AssertionError):
            board.assert_patch_bounds(require_complete=True)

    def test_evict_record_is_atomic(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-1", "active"))
        board.attach_patch("ovr-1", _Patch("ovr-1", 10))
        board.evict_record("ovr-1", 5)
        self.assertEqual(board.resident_ids(), frozenset())
        self.assertEqual(board.patches(), ())
        self.assertIn(("ovr-1", 1, 5), board.evictions)

    def test_total_ceiling_constant(self) -> None:
        self.assertEqual(MAX_PATCH_TOTAL_BYTES, 4 * MAX_PATCH_BYTES)


class TestMatBoardGuardRails(unittest.TestCase):
    """D: guard rails must fail without mutating the record/patch table."""

    def test_ghost_revalidation_leaves_state_intact(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-a", "active"))
        board.attach_patch("ovr-a", _Patch("ovr-a", 10))
        with self.assertRaises(RuntimeError):
            board.revalidate(MatOverlay(2, "ghost", "active"))
        self.assertEqual(board.residents(), (("ovr-a", 1),))
        self.assertEqual(tuple(oid for oid, _p in board.patches()), ("ovr-a",))
        self.assertEqual(board.patch_bytes(), 10)

    def test_revalidation_on_empty_board_raises_and_stays_empty(self) -> None:
        board = MatBoard(seed=False)
        with self.assertRaises(RuntimeError):
            board.revalidate(MatOverlay(1, "ovr-a", "active"))
        self.assertEqual(board.residents(), ())
        self.assertEqual(board.patches(), ())

    def test_duplicate_birth_is_rejected(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-a", "active"))
        with self.assertRaises(ValueError):
            board.birth(MatOverlay(2, "ovr-a", "active"))
        self.assertEqual(board.residents(), (("ovr-a", 1),))
        board.assert_patch_bounds()

    def test_evict_record_removes_exactly_one_record_and_patch(self) -> None:
        board = MatBoard(seed=False)
        for index in (1, 2):
            board.birth(MatOverlay(index, f"ovr-{index}", "active"))
            board.attach_patch(f"ovr-{index}", _Patch(f"ovr-{index}", 10))
        self.assertIsNone(board.evict_record("missing", 3))
        self.assertEqual(len(board.residents()), 2)
        board.evict_record("ovr-1", 4)
        self.assertEqual(tuple(oid for oid, _e in board.residents()), ("ovr-2",))
        self.assertEqual(tuple(oid for oid, _p in board.patches()), ("ovr-2",))
        self.assertEqual(board.patch_bytes(), 10)
        self.assertEqual(board.evictions, [("ovr-1", 1, 4)])
        board.assert_patch_bounds(require_complete=True)


class TestBoardBodies(unittest.TestCase):
    """#13/#14 bodies: frozen, exact-type, validated, hashable."""

    def _op(self, **overrides: object) -> BoardOpBody:
        values: dict[str, object] = {
            "op": "write",
            "slot_id": "ctx-1-abc",
            "body_id": "a" * 32,
            "nbytes": 8 * 64,
            "reason": "cold-start",
            "tick": 1,
        }
        values.update(overrides)
        return BoardOpBody(**values)  # type: ignore[arg-type]

    def _state(self, **overrides: object) -> SlotStateBody:
        values: dict[str, object] = {
            "capacity": 8,
            "occupied": 2,
            "nbytes": 1024,
            "slot_ids": ("s0", "s1"),
            "oldest_slot_id": "s0",
        }
        values.update(overrides)
        return SlotStateBody(**values)  # type: ignore[arg-type]

    def test_bodies_are_frozen_hashable_and_exact_typed(self) -> None:
        bodies = [
            self._op(),
            self._state(),
            BoardSlot("s0", "b" * 32, 512, 1),
        ]
        for body in bodies:
            self.assertEqual(hash(body), hash(body))
        self.assertEqual(
            PAYLOAD_SCHEMA[13], frozenset({BoardOpBody, type(None)})
        )
        self.assertEqual(
            PAYLOAD_SCHEMA[14], frozenset({SlotStateBody, type(None)})
        )
        self.assertEqual(
            PAYLOAD_SCHEMA[16], frozenset({CostErrorBody, type(None)})
        )
        self.assertIn(BoardOpBody, PAYLOAD_SCHEMA[13])
        self.assertIn(SlotStateBody, PAYLOAD_SCHEMA[14])

    def test_board_op_validation(self) -> None:
        for bad in (
            {"op": "mint"},
            {"op": "write", "body_id": None},
            {"op": "write", "nbytes": 0},
            {"op": "write", "slot_id": ""},
            {"op": "write", "tick": -1},
            {"op": "write", "reason": ""},
            {"op": "evict", "body_id": "x", "nbytes": 0},
            {"op": "evict", "body_id": None, "nbytes": 8},
            {"op": "resize", "body_id": None, "nbytes": 0},
            {"op": "resize", "body_id": "x", "nbytes": 4},
        ):
            with self.assertRaises((TypeError, ValueError), msg=repr(bad)):
                self._op(**bad)
        with self.assertRaises(TypeError):
            self._op(nbytes=True)
        with self.assertRaises(TypeError):
            self._op(tick=False)

    def test_slot_state_validation(self) -> None:
        for bad in (
            {"occupied": 1},
            {"occupied": 3},
            {"slot_ids": ("s0", "s0")},
            {"oldest_slot_id": "s1"},
            {"oldest_slot_id": None},
            {"capacity": 0},
            {"nbytes": -1},
            {"slot_ids": ["s0", "s1"]},
        ):
            with self.assertRaises((TypeError, ValueError), msg=repr(bad)):
                self._state(**bad)
        empty = self._state(occupied=0, nbytes=0, slot_ids=(), oldest_slot_id=None)
        self.assertEqual(empty.occupied, 0)

    def test_board_slot_validation(self) -> None:
        for bad in (
            ("", "b" * 32, 512, 1),
            ("s0", "", 512, 1),
            ("s0", "b" * 32, 0, 1),
            ("s0", "b" * 32, 512, -1),
        ):
            with self.assertRaises((TypeError, ValueError), msg=repr(bad)):
                BoardSlot(*bad)  # type: ignore[arg-type]


class TestBoardStore(unittest.TestCase):
    """D1 minimal store: atomic #13 commits, single writer, no constants."""

    def _write(
        self, tick: int, slot_id: str = "s0", nbytes: int = 512
    ) -> BoardOpBody:
        return BoardOpBody(
            "write", slot_id, "d" * 32, nbytes, "test-write", tick
        )

    def _evict(self, tick: int, slot_id: str) -> BoardOpBody:
        return BoardOpBody("evict", slot_id, None, 0, "test-evict", tick)

    def _erase(self, tick: int, slot_id: str) -> BoardOpBody:
        return BoardOpBody("erase", slot_id, None, 0, "test-erase", tick)

    def _resize(self, tick: int, capacity: int) -> BoardOpBody:
        return BoardOpBody("resize", "s0", None, capacity, "test-resize", tick)

    def test_initial_state_is_empty(self) -> None:
        store = BoardStore()
        self.assertEqual(store.capacity, BOARD_CAPACITY)
        self.assertEqual(store.slots(), ())
        self.assertEqual(store.evictions(), ())
        self.assertIsNone(store.slot_state())
        self.assertIsNone(store.last_commit)
        self.assertIsNone(store.state_provenance)

    def test_write_commits_slot_and_state_provenance(self) -> None:
        store = BoardStore()
        store.apply(self._write(1, "s0", 512), Provenance(13, 1))
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.capacity, BOARD_CAPACITY)
        self.assertEqual(state.occupied, 1)
        self.assertEqual(state.nbytes, 512)
        self.assertEqual(state.slot_ids, ("s0",))
        self.assertEqual(state.oldest_slot_id, "s0")
        self.assertEqual(store.last_commit, 1)
        self.assertEqual(store.state_provenance, Provenance(13, 1))
        self.assertEqual(store.slots()[0].body_id, "d" * 32)
        self.assertEqual(store.evictions(), ())

    def test_oldest_and_byte_sum_follow_insertion_order(self) -> None:
        store = BoardStore()
        for tick, slot_id in ((1, "a"), (2, "b"), (3, "c")):
            store.apply(self._write(tick, slot_id, 8 * 64), Provenance(13, tick))
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.slot_ids, ("a", "b", "c"))
        self.assertEqual(state.oldest_slot_id, "a")
        self.assertEqual(state.occupied, len(state.slot_ids))
        self.assertEqual(
            state.nbytes, sum(slot.nbytes for slot in store.slots())
        )

    def test_evict_removes_oldest_and_records_eviction(self) -> None:
        store = BoardStore()
        for tick, slot_id in ((1, "a"), (2, "b"), (3, "c")):
            store.apply(self._write(tick, slot_id), Provenance(13, tick))
        store.apply(self._evict(4, "a"), Provenance(13, 4))
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.slot_ids, ("b", "c"))
        self.assertEqual(store.evictions(), (("a", 4),))
        self.assertEqual(store.last_commit, 4)

    def test_erase_removes_without_eviction_record(self) -> None:
        store = BoardStore()
        store.apply(self._write(1, "a"), Provenance(13, 1))
        store.apply(self._erase(2, "a"), Provenance(13, 2))
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.occupied, 0)
        self.assertEqual(store.evictions(), ())
        self.assertEqual(store.last_commit, 2)

    def test_resize_changes_capacity(self) -> None:
        store = BoardStore()
        store.apply(self._write(1, "a"), Provenance(13, 1))
        store.apply(self._resize(2, 10), Provenance(13, 2))
        self.assertEqual(store.capacity, 10)
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.capacity, 10)
        self.assertEqual(store.last_commit, 2)

    def test_forced_refusal_is_atomic(self) -> None:
        """§9.11a: a refused #13 leaves D1 byte-identical, no eviction."""
        store = BoardStore()
        for tick, slot_id in enumerate(("s0", "s1", "s2", "s3", "s4",
                                        "s5", "s6", "s7"), start=1):
            store.apply(self._write(tick, slot_id), Provenance(13, tick))
        before = (
            store.state_digest(),
            store.evictions(),
            store.last_commit,
            store.slots(),
            store.slot_state(),
        )
        with self.assertRaises(ValueError):
            store.apply(self._write(9, "s8"), Provenance(13, 9))
        after = (
            store.state_digest(),
            store.evictions(),
            store.last_commit,
            store.slots(),
            store.slot_state(),
        )
        self.assertEqual(before, after)
        self.assertEqual(store.last_commit, 8)
        self.assertEqual(store.evictions(), ())

    def test_other_refusals_are_atomic(self) -> None:
        store = BoardStore()
        store.apply(self._write(1, "a"), Provenance(13, 1))
        before = store.state_digest()
        for bad in (
            self._write(2, "a"),          # duplicate
            self._evict(2, "missing"),    # unknown evict
            self._erase(2, "missing"),    # unknown erase
            self._resize(2, BOARD_CAPACITY),  # unchanged capacity
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                store.apply(bad, Provenance(13, bad.tick))
            self.assertEqual(store.state_digest(), before, msg=repr(bad))
        # A shrink below occupancy is also refused atomically.
        store.apply(self._write(2, "b"), Provenance(13, 2))
        before = store.state_digest()
        with self.assertRaises(ValueError):
            store.apply(self._resize(3, 1), Provenance(13, 3))
        self.assertEqual(store.state_digest(), before)

    def test_provenance_guard(self) -> None:
        store = BoardStore()
        before = store.state_digest()
        with self.assertRaises(TypeError):
            store.apply(self._write(1), (13, 1))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.apply(self._write(1), Provenance(14, 1))
        with self.assertRaises(ValueError):
            store.apply(self._write(1), Provenance(13, 2))
        with self.assertRaises(TypeError):
            store.apply({"op": "write"}, Provenance(13, 1))  # type: ignore[arg-type]
        self.assertEqual(store.state_digest(), before)
        self.assertIsNone(store.slot_state())

    def test_state_digest_is_deterministic(self) -> None:
        first = BoardStore()
        second = BoardStore()
        for store in (first, second):
            store.apply(self._write(1, "a"), Provenance(13, 1))
            store.apply(self._write(2, "b"), Provenance(13, 2))
        self.assertEqual(first.state_digest(), second.state_digest())
        second.apply(self._evict(3, "a"), Provenance(13, 3))
        self.assertNotEqual(first.state_digest(), second.state_digest())

    def test_single_writer_surface(self) -> None:
        """§9.11c: apply is the only public mutator; #12 must not touch D1."""
        public = {
            name for name in dir(BoardStore) if not name.startswith("_")
        }
        self.assertEqual(
            public,
            {
                "apply",
                "capacity",
                "evictions",
                "last_commit",
                "slot_state",
                "slots",
                "state_digest",
                "state_provenance",
            },
        )
        import inspect

        source = inspect.getsource(BoardStore)
        self.assertNotIn("StateDeltaBody", source)
        self.assertEqual(
            tuple(inspect.signature(BoardStore.apply).parameters),
            ("self", "op", "provenance"),
        )
        from amateur.stores import ElasticBoard

        doc = ElasticBoard.apply.__doc__ or ""
        self.assertIn("#13", doc)
        self.assertNotIn("Flow #12: write", doc)

    def test_read_accessors_do_not_mutate(self) -> None:
        store = BoardStore()
        store.apply(self._write(1, "a"), Provenance(13, 1))
        before = store.state_digest()
        store.slot_state()
        store.slots()
        store.evictions()
        store.last_commit
        store.state_provenance
        store.state_digest()
        self.assertEqual(store.state_digest(), before)

    def test_synthetic_source_still_returns_none_for_13_and_16(self) -> None:
        """§9.4: dropping 13/16 from PLACEHOLDER later does not change the
        synthetic default envelope (verify, don't assume)."""
        from amateur.dryrun import SyntheticSource

        source = SyntheticSource()
        for fid in (13, 16):
            self.assertIsNone(
                source.derive_body(
                    fid, 1, board=MatBoard(), debris=DebrisPool()
                ),
                msg=f"#{fid}",
            )


if __name__ == "__main__":
    unittest.main()
