"""Typed payload schema and store contracts (Round 11, rulings A-G).

The whitelist is the invariant.  Round 11 pins the D2 write kinds **per
flow**: #18 carries only `EmergentCrystalBody` (the nursery candidate kind)
and #24 only `MatOverlay` (the ratcheted overlay kind); `candidate` is not a
`MatOverlay` status.  D2 reads are pinned too (#8/#25 `MatOverlay`, #23
`SurvivingCrystalBody`).  D3 is a cohort-level store with per-edge filters
and the destructive #27 drain.  No modeled store channel accepts `None`; a
"no event" is an exact-type sentinel.

The overlay index is deliberately tested against its unchanged `matTick`
policy; the candidate index is a separate nursery with its own FIFO
capacity.  D1 (`BoardStore`, `BoardOpBody`, `SlotStateBody`) is frozen work
from the #13 slice and keeps its prior behaviour.

The executor-integration classes below run the real `DryRun` and assert
the same claims end-to-end; they are kept here (not only in
`test_dryrun.py`/`test_provenance.py`) because a schema test that never
executes the executor cannot see a wiring regression.
"""
from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from amateur.dryrun import DryRun
from amateur.ledger import FLOW_LEDGER
from amateur.payloads import (
    BOARD_CAPACITY,
    COHORT_KINDS,
    D2_FLOWS,
    D2_WRITES,
    D4_FLOWS,
    D4_WRITES,
    DEBRIS_CAPACITY,
    DEBRIS_MAX_BYTES,
    DEBRIS_TTL,
    MAX_PATCH_BYTES,
    MAX_PATCH_TOTAL_BYTES,
    MAT_CAPACITY,
    MAT_SEED,
    NO_CANDIDATE_ID,
    NO_DEBRIS_ID,
    NO_MOTIF_ID,
    NOOP_OVERLAY_ID,
    NURSERY_CAPACITY,
    PAYLOAD_SCHEMA,
    REAL_BODIES,
    RESIDENCE_N,
    AvoidanceTrailsBody,
    BiasPrimitive,
    BoardOpBody,
    BoardSlot,
    BoardStore,
    CollapseMandateBody,
    CostErrorBody,
    DebrisCeilingExceeded,
    DebrisCohort,
    DebrisPool,
    DigestedDebrisBody,
    DissipatedEntropyBody,
    EmergentCrystalBody,
    FrameBody,
    LatentRolloutBody,
    MatBoard,
    MatOverlay,
    PredictionErrorBody,
    ScaffoldsBody,
    SlotStateBody,
    StateDeltaBody,
    StructuralMotifBody,
    SurvivingCrystalBody,
    allowed_payload,
    noop_bias,
    noop_crystal,
    noop_debris,
    noop_entropy,
    noop_mandate,
    noop_motif,
    noop_overlay,
    noop_scaffolds,
    noop_survivor,
    noop_trails,
    require_payload,
)
from amateur.provenance import Provenance

_TURNS = 200


class _Trajectory:
    """A payload type that must be unrepresentable on geometry channels."""


def _patch(owner_id: str, nbytes: int = 100, params: int = 12) -> object:
    return SimpleNamespace(
        overlay_id=owner_id, nbytes=nbytes, param_count=params
    )


def _candidate(
    candidate_id: str = "cand-1", born_tick: int = 1
) -> EmergentCrystalBody:
    return EmergentCrystalBody(
        candidate_id=candidate_id,
        residual=(0.5,) * 64,
        surprise=0.2,
        born_tick=born_tick,
    )


def _cohort(
    cohort_id: str = "c1",
    kind: str = "motif",
    count: int = 3,
    first_tick: int = 1,
    last_tick: int = 2,
    provenance: object | None = None,
) -> DebrisCohort:
    return DebrisCohort(
        cohort_id=cohort_id,
        kind=kind,
        centroid=(0.5,) * 64,
        count=count,
        spread=1.0,
        first_tick=first_tick,
        last_tick=last_tick,
        peak_surprise=0.5,
        provenance=provenance
        if provenance is not None
        else Provenance(19, last_tick),
    )


def _debris(
    debris_id: str, cohorts: tuple[DebrisCohort, ...]
) -> DigestedDebrisBody:
    return DigestedDebrisBody(debris_id=debris_id, cohorts=cohorts)


def _random_debris_history(seed: int) -> DebrisPool:
    """A seeded deposit/read/drain/promote/evict sequence for both accounts.

    Every cohort id is unique, as the executor's #19 composite guarantees.
    The mix is chosen to reach all four issuance fates -- promoted and still
    resident, resident unpromoted, drained unpromoted, evicted unpromoted --
    plus the promoted-then-evicted slack (`evicted_promoted`), so the
    partition is exercised rather than assumed.
    """
    rng = random.Random(seed)
    pool = DebrisPool(capacity=6)
    counter = 0
    for step in range(40):
        resident_ids = [cohort.cohort_id for cohort in pool.bodies()]
        roll = rng.random()
        if roll < 0.5 or not resident_ids:
            n = rng.randint(1, 3)
            cohorts = tuple(
                _cohort(
                    f"c{counter + i}",
                    rng.choice(COHORT_KINDS),
                    count=rng.randint(0, 5),
                    last_tick=step,
                )
                for i in range(n)
            )
            counter += n
            body = _debris(f"d{step}", cohorts)
            pool.deposit(
                body,
                Provenance(19, step),
                step,
                allow_ceiling_evict=True,
            )
        elif roll < 0.7:
            for kind in COHORT_KINDS:
                pool.read(kind)
        elif roll < 0.85:
            chosen = rng.sample(
                resident_ids, rng.randint(1, min(2, len(resident_ids)))
            )
            pool.drain(tuple(chosen), step)
        else:
            candidate = rng.choice(resident_ids)
            try:
                pool.promote(candidate, step)
            except ValueError:  # INV-PROMOTE-ONCE: already promoted
                pool.read(rng.choice(COHORT_KINDS))
        if roll >= 0.5 and pool.byte_size() > 1 and rng.random() < 0.15:
            pool.max_bytes = max(1, pool.byte_size() // 2)
            pool.evict_to_ceiling(step)
    return pool


def _survivor(
    *,
    candidate_id: str = "cand-1",
    born_tick: int = 1,
    residence_ticks: int = RESIDENCE_N,
    adapted_mean: float = 0.2,
    base_mean: float = 0.4,
    qualified: bool | None = None,
) -> SurvivingCrystalBody:
    ratio = adapted_mean / base_mean
    if qualified is None:
        qualified = ratio < 1.0 and residence_ticks >= RESIDENCE_N
    return SurvivingCrystalBody(
        candidate_id=candidate_id,
        born_tick=born_tick,
        residence_ticks=residence_ticks,
        adapted_mean=adapted_mean,
        base_mean=base_mean,
        ratio=ratio,
        qualified=qualified,
    )


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
        self.assertEqual(D2_FLOWS, frozenset({8, 18, 23, 24, 25}))
        self.assertEqual(D2_WRITES, frozenset({18, 24}))
        self.assertEqual(D4_WRITES, frozenset({22}))

    def test_d2_write_kinds_are_pinned_per_flow(self) -> None:
        """§1.1: #18 is the candidate kind, #24 the overlay kind; no None."""
        self.assertEqual(
            PAYLOAD_SCHEMA[18], frozenset({EmergentCrystalBody})
        )
        self.assertEqual(PAYLOAD_SCHEMA[24], frozenset({MatOverlay}))
        self.assertNotIn(type(None), PAYLOAD_SCHEMA[18])
        self.assertNotIn(type(None), PAYLOAD_SCHEMA[24])
        self.assertEqual(
            next(f for f in FLOW_LEDGER if f.fid == 18).source,
            "3.0 Execute Observation Loop",
        )
        self.assertEqual(
            next(f for f in FLOW_LEDGER if f.fid == 24).source,
            "6.0 Ratchet & Sediment Grammar",
        )

    def test_store_read_channels_are_typed_or_none(self) -> None:
        """Corrected §1.1: reads carry the exact body OR None; writes never None."""
        reads = {
            7: BiasPrimitive,
            8: MatOverlay,
            16: CostErrorBody,
            20: ScaffoldsBody,
            21: StructuralMotifBody,
            23: SurvivingCrystalBody,
            25: MatOverlay,
            26: AvoidanceTrailsBody,
            27: DissipatedEntropyBody,
        }
        for fid, body_type in reads.items():
            self.assertEqual(
                PAYLOAD_SCHEMA[fid],
                frozenset({body_type, type(None)}),
                msg=f"#{fid}",
            )
        writes = {
            18: EmergentCrystalBody,
            19: DigestedDebrisBody,
            22: BiasPrimitive,
            24: MatOverlay,
        }
        for fid, body_type in writes.items():
            self.assertEqual(
                PAYLOAD_SCHEMA[fid], frozenset({body_type}), msg=f"#{fid}"
            )
            self.assertNotIn(type(None), PAYLOAD_SCHEMA[fid], msg=f"#{fid}")

    def test_d3_write_is_exact(self) -> None:
        self.assertEqual(
            PAYLOAD_SCHEMA[19], frozenset({DigestedDebrisBody})
        )
        self.assertNotIn(type(None), PAYLOAD_SCHEMA[19])

    def test_cost_error_is_a_read_with_optional_none(self) -> None:
        self.assertEqual(
            PAYLOAD_SCHEMA[16], frozenset({CostErrorBody, type(None)})
        )

    def test_non_store_placeholders_and_real_bodies(self) -> None:
        for fid in (1, 2, 4, 5, 9, 11, 15):
            self.assertEqual(PAYLOAD_SCHEMA[fid], frozenset({type(None)}))
        for fid in (3, 6, 10, 12, 17):
            self.assertEqual(
                PAYLOAD_SCHEMA[fid],
                frozenset({REAL_BODIES[fid], type(None)}),
                msg=f"#{fid}",
            )
        # D1 is frozen: #13/#14 keep their prior optional body shape.
        self.assertEqual(
            PAYLOAD_SCHEMA[13], frozenset({BoardOpBody, type(None)})
        )
        self.assertEqual(
            PAYLOAD_SCHEMA[14], frozenset({SlotStateBody, type(None)})
        )

    def test_schema_covers_every_flow(self) -> None:
        self.assertEqual(
            set(PAYLOAD_SCHEMA), {f.fid for f in FLOW_LEDGER}
        )

    def test_inv_emptiness_trajectory_unrepresentable(self) -> None:
        for fid in (18, 22, 24):
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

        class CrystalSneak(EmergentCrystalBody):
            pass

        self.assertFalse(
            allowed_payload(24, Sneaky(1, "x", "active"))
        )
        self.assertFalse(allowed_payload(22, Smooth("x", 0.5)))
        self.assertFalse(
            allowed_payload(18, CrystalSneak("c", (0.0,) * 64, 0.1, 1))
        )

    def test_per_flow_pins_reject_the_wrong_kind(self) -> None:
        # A candidate cannot be born through the overlay path ...
        self.assertFalse(allowed_payload(18, MatOverlay(1, "c", "active")))
        with self.assertRaises(TypeError):
            require_payload(18, MatOverlay(1, "c", "active"))
        # ... and an overlay cannot be written through the candidate path.
        self.assertFalse(allowed_payload(24, _candidate()))
        with self.assertRaises(TypeError):
            require_payload(24, _candidate())
        # The survivor read is not the overlay read.
        self.assertFalse(allowed_payload(23, MatOverlay(1, "x", "stale")))
        self.assertFalse(allowed_payload(8, _survivor()))

    def test_stores_enforce_per_flow_pins(self) -> None:
        board = MatBoard(seed=False)
        with self.assertRaises(TypeError):
            board.birth(_candidate("c1", 1))
        with self.assertRaises(TypeError):
            board.nursery_birth(MatOverlay(1, "x", "active"), 1)
        with self.assertRaises(TypeError):
            board.sediment(MatOverlay(1, "x", "active"))

    def test_candidate_status_is_gone(self) -> None:
        with self.assertRaises(ValueError):
            MatOverlay(0, "x", "candidate")
        self.assertEqual(MAT_SEED, ("ovr-0", 0))
        seed = MatOverlay(MAT_SEED[1], MAT_SEED[0], "active")
        self.assertIsNone(seed.motif_lineage)

    def test_record_validation(self) -> None:
        with self.assertRaises(ValueError):
            MatOverlay(0, "x", "frozen")
        with self.assertRaises(ValueError):
            MatOverlay(-1, "x", "active")
        with self.assertRaises(ValueError):
            MatOverlay(0, "x", "active", motif_lineage="")
        with self.assertRaises(ValueError):
            BiasPrimitive("x", 1.5)
        with self.assertRaises(ValueError):
            BiasPrimitive("x", -0.1)
        with self.assertRaises(ValueError):
            BiasPrimitive("x", 0.5, candidate_lineage="")
        self.assertEqual(
            BiasPrimitive("x", 0.5, "cand-1").candidate_lineage, "cand-1"
        )


class TestMatBoard(unittest.TestCase):
    """Overlay index: matTick policy is unchanged (not re-specified)."""

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


class TestMatBoardPatches(unittest.TestCase):
    def test_attach_and_account(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-1", "active"))
        board.attach_patch("ovr-1", _patch("ovr-1", 1000))
        self.assertEqual(board.patch_bytes(), 1000)
        self.assertEqual(board.patch_params(), 12)
        self.assertIsNotNone(board.patch("ovr-1"))
        board.assert_patch_bounds(require_complete=True)
        self.assertIn("ovr-1", {oid for oid, _ in board.residents()})

    def test_per_patch_ceiling_rejects_full_patch(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-full", "active"))
        with self.assertRaises(ValueError):
            board.attach_patch("ovr-full", _patch("ovr-full", 843_680))
        self.assertEqual(board.patch_bytes(), 0)
        self.assertGreater(MAX_PATCH_BYTES, 136_960)
        self.assertLess(MAX_PATCH_BYTES, 843_680)

    def test_attach_requires_a_resident(self) -> None:
        board = MatBoard(seed=False)
        with self.assertRaises(AssertionError):
            board.attach_patch("ghost", _patch("ghost", 10))

    def test_table_ceiling_evicts_record_and_patch_atomically(self) -> None:
        board = MatBoard(
            seed=False, max_patch_bytes=1500, max_patch_total_bytes=1500
        )
        board.birth(MatOverlay(1, "ovr-1", "active"))
        board.attach_patch("ovr-1", _patch("ovr-1", 1000))
        board.birth(MatOverlay(2, "ovr-2", "active"))
        board.attach_patch("ovr-2", _patch("ovr-2", 1000))
        board.enforce_patch_ceiling(3)
        self.assertEqual(board.resident_ids(), frozenset({"ovr-2"}))
        self.assertEqual(tuple(oid for oid, _p in board.patches()), ("ovr-2",))
        self.assertIn(("ovr-1", 1, 3), board.byte_evictions)
        board.assert_patch_bounds(require_complete=True)

    def test_capacity_eviction_removes_the_patch(self) -> None:
        board = MatBoard(seed=False)
        for index in range(MAT_CAPACITY + 1):
            board.birth(MatOverlay(index + 1, f"ovr-{index}", "active"))
            board.attach_patch(f"ovr-{index}", _patch(f"ovr-{index}", 10))
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
        board.attach_patch("ovr-1", _patch("ovr-1", 10))
        board.evict_record("ovr-1", 5)
        self.assertEqual(board.resident_ids(), frozenset())
        self.assertEqual(board.patches(), ())
        self.assertIn(("ovr-1", 1, 5), board.evictions)

    def test_total_ceiling_constant(self) -> None:
        self.assertEqual(MAX_PATCH_TOTAL_BYTES, 4 * MAX_PATCH_BYTES)


class TestMatBoardGuardRails(unittest.TestCase):
    def test_ghost_revalidation_leaves_state_intact(self) -> None:
        board = MatBoard(seed=False)
        board.birth(MatOverlay(1, "ovr-a", "active"))
        board.attach_patch("ovr-a", _patch("ovr-a", 10))
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
            board.attach_patch(f"ovr-{index}", _patch(f"ovr-{index}", 10))
        self.assertIsNone(board.evict_record("missing", 3))
        self.assertEqual(len(board.residents()), 2)
        board.evict_record("ovr-1", 4)
        self.assertEqual(tuple(oid for oid, _e in board.residents()), ("ovr-2",))
        self.assertEqual(tuple(oid for oid, _p in board.patches()), ("ovr-2",))
        self.assertEqual(board.patch_bytes(), 10)
        self.assertEqual(board.evictions, [("ovr-1", 1, 4)])
        board.assert_patch_bounds(require_complete=True)


class TestD2Nursery(unittest.TestCase):
    """Candidate index: FIFO, evidence, survivor selection, promotion."""

    def test_birth_appends_candidate(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        self.assertEqual(board.nursery_ids(), ("c1",))
        self.assertEqual(
            tuple(body.candidate_id for body in board.nursery_records()),
            ("c1",),
        )
        self.assertEqual(board.nursery_evidence("c1"), ())
        self.assertEqual(board.nursery_patch_bytes(), 0)

    def test_duplicate_and_resurrected_ids_are_refused(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        with self.assertRaises(ValueError):
            board.nursery_birth(_candidate("c1", 2), 2)
        board.nursery_promote("c1", 3)
        with self.assertRaises(ValueError):
            board.nursery_birth(_candidate("c1", 4), 4)

    def test_birth_tick_must_match(self) -> None:
        board = MatBoard(seed=False)
        with self.assertRaises(ValueError):
            board.nursery_birth(_candidate("c1", 1), 2)

    def test_capacity_is_fifo(self) -> None:
        board = MatBoard(seed=False)
        for tick in range(1, NURSERY_CAPACITY + 2):
            board.nursery_birth(_candidate(f"c{tick}", tick), tick)
        self.assertEqual(len(board.nursery_ids()), NURSERY_CAPACITY)
        self.assertNotIn("c1", board.nursery_ids())
        self.assertEqual(board.nursery_ids(), ("c2", "c3", "c4", "c5"))

    def test_observe_unknown_raises_and_cannot_create(self) -> None:
        board = MatBoard(seed=False)
        with self.assertRaises(ValueError):
            board.nursery_observe("ghost", 0.2, 0.4, 1)
        self.assertEqual(board.nursery_ids(), ())
        self.assertEqual(board.nursery_records(), ())

    def test_observe_appends_strictly_increasing_evidence(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        board.nursery_observe("c1", 0.2, 0.4, 1)
        board.nursery_observe("c1", 0.3, 0.4, 2)
        self.assertEqual(
            board.nursery_evidence("c1"),
            ((1, 0.2, 0.4), (2, 0.3, 0.4)),
        )
        with self.assertRaises(ValueError):
            board.nursery_observe("c1", 0.2, 0.4, 2)

    def test_survivor_requires_residence_n_and_ratio_below_one(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        for tick in range(1, RESIDENCE_N):  # one short
            board.nursery_observe("c1", 0.2, 0.4, tick)
        self.assertIsNone(board.nursery_survivor(RESIDENCE_N))
        board.nursery_observe("c1", 0.2, 0.4, RESIDENCE_N)
        survivor = board.nursery_survivor(RESIDENCE_N)
        assert survivor is not None
        self.assertEqual(survivor.candidate_id, "c1")
        self.assertEqual(survivor.residence_ticks, RESIDENCE_N)
        self.assertAlmostEqual(survivor.ratio, 0.5)
        self.assertTrue(survivor.qualified)

    def test_harmful_survivor_is_surfaced_for_the_gate_to_refuse(self) -> None:
        """Correction 2: no qualified pre-filter; #22's gate must see it."""
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        for tick in range(1, RESIDENCE_N + 1):
            board.nursery_observe("c1", 0.6, 0.4, tick)  # ratio 1.5
        survivor = board.nursery_survivor(RESIDENCE_N)
        assert survivor is not None
        self.assertEqual(survivor.candidate_id, "c1")
        self.assertFalse(survivor.qualified)
        self.assertGreaterEqual(survivor.ratio, 1.0)

    def test_survivor_tie_break_lowest_ratio_then_oldest(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("old", 1), 1)
        board.nursery_birth(_candidate("new", 2), 2)
        for tick in range(2, RESIDENCE_N + 2):
            board.nursery_observe("old", 0.3, 0.4, tick)  # 0.75
            board.nursery_observe("new", 0.2, 0.4, tick)  # 0.5
        survivor = board.nursery_survivor(RESIDENCE_N + 2)
        assert survivor is not None
        self.assertEqual(survivor.candidate_id, "new")
        # Same ratio: oldest birth wins.
        board2 = MatBoard(seed=False)
        board2.nursery_birth(_candidate("old", 1), 1)
        board2.nursery_birth(_candidate("new", 2), 2)
        for tick in range(2, RESIDENCE_N + 2):
            board2.nursery_observe("old", 0.2, 0.4, tick)
            board2.nursery_observe("new", 0.2, 0.4, tick)
        survivor = board2.nursery_survivor(RESIDENCE_N + 2)
        assert survivor is not None
        self.assertEqual(survivor.candidate_id, "old")

    def test_candidate_patches_are_counted(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        board.attach_candidate_patch("c1", _patch("c1", 1000))
        self.assertEqual(board.nursery_patch("c1").nbytes, 1000)
        self.assertEqual(board.nursery_patch_bytes(), 1000)
        self.assertEqual(board.patch_bytes(), 1000)
        board.assert_patch_bounds(require_complete=True)

    def test_candidate_patch_requires_a_resident_and_matching_id(self) -> None:
        board = MatBoard(seed=False)
        with self.assertRaises(AssertionError):
            board.attach_candidate_patch("ghost", _patch("ghost", 10))
        board.nursery_birth(_candidate("c1", 1), 1)
        with self.assertRaises(ValueError):
            board.attach_candidate_patch("c1", _patch("other", 10))
        with self.assertRaises(ValueError):
            board.attach_candidate_patch("c1", _patch("c1", 843_680))

    def test_promotion_removes_record_and_patch_atomically(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        board.attach_candidate_patch("c1", _patch("c1", 1000))
        body = board.nursery_promote("c1", 2)
        self.assertEqual(body.candidate_id, "c1")
        self.assertEqual(board.nursery_ids(), ())
        self.assertIsNone(board.nursery_patch("c1"))
        self.assertEqual(board.patch_bytes(), 0)
        with self.assertRaises(ValueError):
            board.nursery_promote("c1", 3)

    def test_nursery_evict_record_removes_record_and_patch(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        board.attach_candidate_patch("c1", _patch("c1", 1000))
        board.nursery_evict_record("c1", 2, "organ")
        self.assertEqual(board.nursery_ids(), ())
        self.assertIsNone(board.nursery_patch("c1"))
        self.assertEqual(board.patch_bytes(), 0)
        with self.assertRaises(ValueError):
            board.nursery_evict_record("c1", 3, "organ")

    def test_require_complete_covers_both_indices(self) -> None:
        board = MatBoard(seed=False)
        board.nursery_birth(_candidate("c1", 1), 1)
        with self.assertRaises(AssertionError):
            board.assert_patch_bounds(require_complete=True)
        board.attach_candidate_patch("c1", _patch("c1", 10))
        board.assert_patch_bounds(require_complete=True)

    def test_seed_overlay_is_exempt_from_completeness(self) -> None:
        board = MatBoard()
        board.assert_patch_bounds(require_complete=True)

    def test_candidate_patch_ceiling_evicts_candidate(self) -> None:
        board = MatBoard(
            seed=False, max_patch_bytes=1500, max_patch_total_bytes=1500
        )
        board.nursery_birth(_candidate("c1", 1), 1)
        board.attach_candidate_patch("c1", _patch("c1", 1000))
        board.nursery_birth(_candidate("c2", 2), 2)
        board.attach_candidate_patch("c2", _patch("c2", 1000))
        board.enforce_patch_ceiling(3)
        self.assertIn("c2", board.nursery_ids())
        self.assertNotIn("c1", board.nursery_ids())
        self.assertEqual(board.patch_bytes(), 1000)
        board.assert_patch_bounds(require_complete=True)


class TestBoardBodies(unittest.TestCase):
    """#13/#14 D1 bodies: frozen, exact-type, validated, hashable."""

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
        # D1 is frozen: its optional-body shape is retained.
        self.assertEqual(
            PAYLOAD_SCHEMA[13], frozenset({BoardOpBody, type(None)})
        )
        self.assertEqual(
            PAYLOAD_SCHEMA[14], frozenset({SlotStateBody, type(None)})
        )
        self.assertEqual(
            PAYLOAD_SCHEMA[16], frozenset({CostErrorBody, type(None)})
        )

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
        empty = self._state(
            occupied=0, nbytes=0, slot_ids=(), oldest_slot_id=None
        )
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
    """D1 minimal store: atomic #13 commits, single writer (frozen)."""

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
        return BoardOpBody(
            "resize", "s0", None, capacity, "test-resize", tick
        )

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
            store.apply(
                self._write(tick, slot_id, 8 * 64), Provenance(13, tick)
            )
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
        store = BoardStore()
        for tick, slot_id in enumerate(
            ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"), start=1
        ):
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
            self._write(2, "a"),
            self._evict(2, "missing"),
            self._erase(2, "missing"),
            self._resize(2, BOARD_CAPACITY),
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                store.apply(bad, Provenance(13, bad.tick))
            self.assertEqual(store.state_digest(), before, msg=repr(bad))
        store.apply(self._write(2, "b"), Provenance(13, 2))
        before = store.state_digest()
        with self.assertRaises(ValueError):
            store.apply(self._resize(3, 1), Provenance(13, 3))
        self.assertEqual(store.state_digest(), before)

    def test_stale_and_equal_tick_commits_are_refused_atomically(self) -> None:
        store = BoardStore()
        store.apply(self._write(2, "a"), Provenance(13, 2))
        before = (
            store.state_digest(),
            store.evictions(),
            store.last_commit,
            store.slots(),
            store.slot_state(),
        )
        for tick, op in (
            (1, self._write(1, "b")),
            (2, self._write(2, "b")),
            (1, self._evict(1, "a")),
            (2, self._erase(2, "a")),
        ):
            with self.assertRaises(ValueError, msg=repr(op)):
                store.apply(op, Provenance(13, tick))
            after = (
                store.state_digest(),
                store.evictions(),
                store.last_commit,
                store.slots(),
                store.slot_state(),
            )
            self.assertEqual(before, after, msg=repr(op))
        self.assertEqual(store.last_commit, 2)
        self.assertEqual(
            [slot.slot_id for slot in store.slots()], ["a"]
        )

    def test_oldest_is_born_tick_order_across_evict_then_write(self) -> None:
        store = BoardStore()
        for tick, slot_id in ((1, "a"), (2, "b"), (3, "c")):
            store.apply(self._write(tick, slot_id), Provenance(13, tick))
        self.assertEqual(
            [slot.born_tick for slot in store.slots()], [1, 2, 3]
        )
        store.apply(self._evict(4, "a"), Provenance(13, 4))
        store.apply(self._write(5, "d"), Provenance(13, 5))
        born = [slot.born_tick for slot in store.slots()]
        self.assertEqual(born, sorted(born))
        self.assertEqual(born, [2, 3, 5])
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.slot_ids, ("b", "c", "d"))
        self.assertEqual(state.oldest_slot_id, "b")
        self.assertNotEqual(state.oldest_slot_id, "d")
        store.apply(self._erase(6, "c"), Provenance(13, 6))
        store.apply(self._write(7, "e"), Provenance(13, 7))
        born = [slot.born_tick for slot in store.slots()]
        self.assertEqual(born, sorted(born))
        state = store.slot_state()
        assert state is not None
        self.assertEqual(state.slot_ids, ("b", "d", "e"))
        self.assertEqual(state.oldest_slot_id, "b")
        self.assertEqual(state.occupied, len(state.slot_ids))

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


class TestPhysicsBodies(unittest.TestCase):
    def test_bodies_are_frozen_hashable_and_exact_typed(self) -> None:
        bodies = [
            FrameBody(b"\x00" * 3, 1, 1, 0),
            PredictionErrorBody(0.5, (0.0,) * 64),
            LatentRolloutBody(((0.0,) * 64,) * 4),
            StateDeltaBody((0.0,) * 64, 0.5),
            CostErrorBody(0.5, 0.4),
            CollapseMandateBody("ceiling-pressure", 8, 8, 7, "debris-001", 3),
            _candidate(),
            _debris("debris-001", (_cohort("m1", "motif"),)),
            StructuralMotifBody(
                "motif-001", "debris-001", (0.0,) * 64, 4, 1.0, 4, 1, 4
            ),
            ScaffoldsBody((_cohort("s1", "scaffold"),)),
            AvoidanceTrailsBody((_cohort("t1", "trail"),)),
            DissipatedEntropyBody(("r1",), (1.0,), 3),
            _survivor(),
        ]
        for body in bodies:
            self.assertEqual(hash(body), hash(body))
            self.assertIn(type(body), PAYLOAD_SCHEMA[_fid_for_body(body)])

    def test_body_validation(self) -> None:
        with self.assertRaises(ValueError):
            FrameBody(b"\x00" * 5, 1, 1, 0)
        with self.assertRaises(ValueError):
            PredictionErrorBody(float("nan"), (0.0,) * 64)
        with self.assertRaises(ValueError):
            PredictionErrorBody(0.5, (0.0,) * 3)
        with self.assertRaises(ValueError):
            MatOverlay(0, "x", "frozen")
        with self.assertRaises(ValueError):
            _cohort(kind="mystery")
        with self.assertRaises(ValueError):
            _debris("d", (_cohort("same", "motif"), _cohort("same", "trail")))
        with self.assertRaises(ValueError):
            ScaffoldsBody((_cohort("m1", "motif"),))
        with self.assertRaises(ValueError):
            AvoidanceTrailsBody((_cohort("m1", "motif"),))
        with self.assertRaises(ValueError):
            DissipatedEntropyBody(("a", "b"), (1.0,), 0)
        with self.assertRaises(ValueError):
            _survivor(qualified=True, adapted_mean=0.5, base_mean=0.4)
        with self.assertRaises(ValueError):
            _survivor(qualified=False, adapted_mean=0.2, base_mean=0.4)

    def test_no_op_sentinels_are_real_exact_type_bodies(self) -> None:
        sentinels = {
            7: noop_bias(),
            8: noop_overlay(),
            18: noop_crystal(),
            19: noop_debris(),
            20: noop_scaffolds(),
            21: noop_motif(),
            22: noop_bias(),
            23: noop_survivor(),
            24: noop_overlay(),
            25: noop_overlay(),
            26: noop_trails(),
            27: noop_entropy(),
            17: noop_mandate(),
        }
        for fid, body in sentinels.items():
            self.assertIn(type(body), PAYLOAD_SCHEMA[fid], msg=f"#{fid}")
        self.assertEqual(noop_crystal().candidate_id, NO_CANDIDATE_ID)
        self.assertEqual(noop_survivor().candidate_id, NO_CANDIDATE_ID)
        self.assertEqual(noop_debris().debris_id, NO_DEBRIS_ID)
        self.assertEqual(noop_debris().cohorts, ())
        self.assertEqual(noop_motif().motif_id, NO_MOTIF_ID)
        self.assertTrue(noop_scaffolds().no_event)
        self.assertTrue(noop_trails().no_event)
        self.assertEqual(noop_entropy().cohort_ids, ())
        self.assertEqual(noop_overlay().overlay_id, NOOP_OVERLAY_ID)
        self.assertEqual(noop_bias().confidence, 0.0)


def _fid_for_body(body: object) -> int:
    for fid, allowed in PAYLOAD_SCHEMA.items():
        if type(body) in allowed:
            return fid
    raise AssertionError(f"{type(body).__name__} has no flow")


class TestDebrisCohorts(unittest.TestCase):
    def test_default_cohort_capacity_is_32(self) -> None:
        pool = DebrisPool()
        self.assertEqual(pool.capacity, 32)
        self.assertEqual(DEBRIS_CAPACITY, 32)
        self.assertEqual(pool.max_bytes, DEBRIS_MAX_BYTES)

    def test_deposit_read_filters_and_non_destructive_reads(self) -> None:
        pool = DebrisPool()
        body = _debris(
            "d1",
            (
                _cohort("m1", "motif"),
                _cohort("s1", "scaffold"),
                _cohort("t1", "trail"),
                _cohort("r1", "residual"),
            ),
        )
        pool.deposit(body, Provenance(19, 1), 1)
        self.assertEqual(len(pool), 4)
        self.assertEqual(
            tuple(c.cohort_id for c in pool.read("motif")), ("m1",)
        )
        self.assertEqual(
            tuple(c.cohort_id for c in pool.read("scaffold")), ("s1",)
        )
        self.assertEqual(
            tuple(c.cohort_id for c in pool.read("trail")), ("t1",)
        )
        self.assertEqual(
            tuple(c.cohort_id for c in pool.read("residual")), ("r1",)
        )
        # Read is non-destructive and counted.
        self.assertEqual(pool.read("motif"), pool.read("motif"))
        self.assertEqual(len(pool), 4)
        self.assertEqual(pool.mass_balance()["reads_by_kind"]["motif"], 3)
        with self.assertRaises(ValueError):
            pool.read("mystery")

    def test_capacity_evicts_fifo_and_mass_balance_holds(self) -> None:
        pool = DebrisPool(capacity=2)
        body = _debris(
            "d1",
            (
                _cohort("a", "motif"),
                _cohort("b", "motif"),
                _cohort("c", "motif"),
            ),
        )
        evicted = pool.deposit(body, Provenance(19, 1), 1)
        self.assertEqual(tuple(c.cohort_id for c in evicted), ("a",))
        self.assertEqual(len(pool), 2)
        self.assertEqual(
            pool.mass_balance(),
            {
                **pool.mass_balance(),
                "deposits": 3,
                "resident": 2,
                "drained": 0,
                "evicted": 1,
                "ceiling_evicted": 0,
                "promoted": 0,
            },
        )
        balance = pool.mass_balance()
        self.assertEqual(
            balance["deposits"],
            balance["resident"] + balance["drained"] + balance["evicted"],
        )

    def test_byte_ceiling_raises_atomically(self) -> None:
        pool = DebrisPool()
        body = _debris("d1", (_cohort("a", "motif"),))
        pool.deposit(body, Provenance(19, 1), 1)
        before = (len(pool), pool.byte_size(), pool.mass_balance())
        pool.max_bytes = pool.byte_size()  # zero headroom
        with self.assertRaises(DebrisCeilingExceeded):
            pool.deposit(
                _debris("d2", (_cohort("b", "motif"),)),
                Provenance(19, 2),
                2,
            )
        self.assertEqual(
            (len(pool), pool.byte_size(), pool.mass_balance()), before
        )

    def test_byte_ceiling_backstop_counts_ceiling_evictions(self) -> None:
        pool = DebrisPool()
        first = _debris("d1", (_cohort("a", "motif"),))
        pool.deposit(first, Provenance(19, 1), 1)
        pool.max_bytes = pool.byte_size()
        evicted = pool.deposit(
            _debris("d2", (_cohort("b", "motif"),)),
            Provenance(19, 2),
            2,
            allow_ceiling_evict=True,
        )
        self.assertEqual(len(evicted), 1)
        balance = pool.mass_balance()
        self.assertEqual(balance["ceiling_evicted"], 1)
        self.assertEqual(balance["evicted"], 1)
        self.assertLessEqual(pool.byte_size(), pool.max_bytes)
        self.assertEqual(
            balance["deposits"],
            balance["resident"] + balance["drained"] + balance["evicted"],
        )

    def test_evict_to_ceiling_records_backstop(self) -> None:
        pool = DebrisPool()
        pool.deposit(
            _debris("d1", (_cohort("a", "motif"),)), Provenance(19, 1), 1
        )
        pool.max_bytes = 1  # force the backstop
        evicted = pool.evict_to_ceiling(2)
        self.assertEqual(len(evicted), 1)
        self.assertEqual(pool.mass_balance()["ceiling_evicted"], 1)
        self.assertEqual(len(pool), 0)

    def test_drainable_selects_residual_and_expired_oldest_first(self) -> None:
        pool = DebrisPool()
        pool.deposit(
            _debris(
                "d1",
                (
                    _cohort("r", "residual", last_tick=1),
                    _cohort("old", "scaffold", last_tick=1),
                    _cohort("young", "scaffold", last_tick=20),
                ),
            ),
            Provenance(19, 1),
            1,
        )
        drainable = pool.drainable(20)
        self.assertEqual(
            tuple(c.cohort_id for c in drainable), ("r", "old")
        )
        self.assertEqual(
            tuple(c.cohort_id for c in pool.drainable(1 + DEBRIS_TTL)),
            ("r",),
        )

    def test_drain_is_destructive_and_double_drain_raises(self) -> None:
        pool = DebrisPool()
        pool.deposit(
            _debris("d1", (_cohort("a", "residual"),)),
            Provenance(19, 1),
            1,
        )
        drained = pool.drain(("a",), 2)
        self.assertEqual(tuple(c.cohort_id for c in drained), ("a",))
        self.assertEqual(len(pool), 0)
        self.assertEqual(pool.mass_balance()["drained"], 1)
        with self.assertRaises(ValueError):
            pool.drain(("a",), 3)
        with self.assertRaises(ValueError):
            pool.drain(("ghost",), 3)

    def test_promote_is_once_and_non_destructive(self) -> None:
        pool = DebrisPool()
        pool.deposit(
            _debris("d1", (_cohort("m1", "motif"),)),
            Provenance(19, 1),
            1,
        )
        pool.promote("m1", 2)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool.mass_balance()["promoted"], 1)
        with self.assertRaises(ValueError):
            pool.promote("m1", 3)
        with self.assertRaises(ValueError):
            pool.promote("ghost", 3)

    def test_best_motif_selects_count_then_persistence(self) -> None:
        pool = DebrisPool()
        pool.deposit(
            _debris(
                "d1",
                (
                    _cohort("a", "motif", count=2, first_tick=1, last_tick=2),
                    _cohort("b", "motif", count=4, first_tick=1, last_tick=2),
                    _cohort("c", "motif", count=4, first_tick=1, last_tick=6),
                ),
            ),
            Provenance(19, 1),
            1,
        )
        best = pool.best_motif()
        assert best is not None
        self.assertEqual(best.cohort_id, "c")

    def test_no_op_and_empty_bodies_are_not_deposited(self) -> None:
        pool = DebrisPool()
        with self.assertRaises(ValueError):
            pool.deposit(noop_debris(), Provenance(19, 1), 1)
        with self.assertRaises(ValueError):
            pool.deposit(_debris("empty", ()), Provenance(19, 1), 1)
        self.assertEqual(len(pool), 0)

    def test_mass_balance_identity_after_mixed_sequence(self) -> None:
        pool = DebrisPool(capacity=3)
        pool.deposit(
            _debris(
                "d1",
                (
                    _cohort("a", "motif"),
                    _cohort("b", "scaffold"),
                    _cohort("c", "trail"),
                ),
            ),
            Provenance(19, 1),
            1,
        )
        pool.drain(("c",), 2)
        pool.deposit(
            _debris("d2", (_cohort("d", "residual"),)),
            Provenance(19, 3),
            3,
        )
        balance = pool.mass_balance()
        self.assertEqual(set(balance), {
            "deposits",
            "resident",
            "drained",
            "evicted",
            "ceiling_evicted",
            "promoted",
            "reads_by_kind",
            "issuance",
            "conservation_account",
            "issuance_account",
        })
        self.assertEqual(
            balance["deposits"],
            balance["resident"] + balance["drained"] + balance["evicted"],
        )
        self.assertIn("promoted", balance["issuance"])
        self.assertIn("resident_unpromoted", balance["issuance"])
        self.assertIn("drained_unpromoted", balance["issuance"])
        # Both names, and never one instead of the other: the flat Round-11
        # keys are the conservation view, the issuance account is separate.
        self.assertEqual(
            balance["conservation_account"],
            {
                "deposits": balance["deposits"],
                "resident": balance["resident"],
                "drained": balance["drained"],
                "evicted": balance["evicted"],
                "ceiling_evicted": balance["ceiling_evicted"],
                "identity": "deposits = resident + drained + evicted",
                "holds": True,
            },
        )
        self.assertEqual(balance["issuance_account"], balance["issuance"])
        self.assertTrue(balance["issuance_account"]["holds"])
        self.assertEqual(
            balance["issuance_account"]["issued"], balance["deposits"]
        )

    def test_conservation_and_issuance_accounts_close_on_random_sequences(
        self,
    ) -> None:
        for seed in range(8):
            with self.subTest(seed=seed):
                pool = _random_debris_history(seed)
                conservation = pool.conservation_account()
                issuance = pool.issuance_account()
                self.assertTrue(conservation["holds"], msg=conservation)
                self.assertTrue(issuance["holds"], msg=issuance)
                self.assertEqual(conservation["deposits"], issuance["deposits"])
                self.assertEqual(issuance["issued"], conservation["deposits"])
                # The issuance terms are a *partition* of the deposits.
                self.assertEqual(
                    issuance["promoted"]
                    + issuance["resident_unpromoted"]
                    + issuance["drained_unpromoted"]
                    + issuance["evicted"],
                    conservation["deposits"],
                )
                # The conservation terms are the three cohort fates.
                self.assertEqual(
                    conservation["resident"]
                    + conservation["drained"]
                    + conservation["evicted"],
                    conservation["deposits"],
                )
                self.assertEqual(
                    conservation["evicted"],
                    pool.mass_balance()["evicted"],
                )

    def test_promoted_then_evicted_cohort_is_not_double_counted(self) -> None:
        pool = DebrisPool(capacity=2)
        pool.deposit(
            _debris("d1", (_cohort("a", "motif"), _cohort("b", "motif"))),
            Provenance(19, 1),
            1,
        )
        pool.promote("a", 2)
        pool.deposit(
            _debris("d2", (_cohort("c", "motif"), _cohort("d", "motif"))),
            Provenance(19, 2),
            2,
        )
        issuance = pool.issuance_account()
        # "a" was promoted and then evicted: counted once, under `promoted`.
        self.assertEqual(issuance["promoted"], 1)
        self.assertEqual(issuance["evicted_promoted"], 1)
        self.assertEqual(issuance["evicted"], 1)
        self.assertEqual(issuance["resident_unpromoted"], 2)
        self.assertEqual(issuance["issued"], 4)
        self.assertTrue(issuance["holds"])
        # Taking `_evicted` whole (`_ceiling_evicted` is a subset of it, not
        # an extra term) would count "a" twice and miss the deposits by one.
        self.assertEqual(
            issuance["promoted"]
            + issuance["resident_unpromoted"]
            + issuance["drained_unpromoted"]
            + issuance["evicted"]
            + issuance["evicted_promoted"],
            issuance["deposits"] + 1,
        )

    def test_naive_ruling_formula_overcounts_by_exactly_the_reads(self) -> None:
        """The double count the user ratified as our correction.

        Ruling (F) proposed ``in = out_reads + drained + delta_resident``.
        Reads are non-destructive, so every cohort a read surfaces is added
        to the outflow while never leaving the pool: the naive right-hand
        side exceeds the true inflow by exactly the mass the reads handed
        out.  Neither named account makes that mistake.
        """
        pool = DebrisPool(capacity=8)
        pool.deposit(
            _debris(
                "d1",
                (
                    _cohort("a", "motif"),
                    _cohort("b", "motif"),
                    _cohort("c", "scaffold"),
                ),
            ),
            Provenance(19, 1),
            1,
        )
        start_resident = 0  # the window opens on an empty pool
        read_surfaces = sum(
            len(pool.read(kind)) for kind in ("motif", "motif", "scaffold")
        )
        self.assertEqual(read_surfaces, 5)  # two reads surface two each
        self.assertEqual(
            sum(pool.mass_balance()["reads_by_kind"].values()), 3
        )
        pool.drain(("c",), 2)
        balance = pool.mass_balance()
        delta_resident = balance["resident"] - start_resident
        naive = read_surfaces + balance["drained"] + delta_resident
        self.assertEqual(naive - balance["deposits"], read_surfaces)
        self.assertEqual(naive, 8)  # 3 deposits + 5 double-counted
        # The conservation account excludes reads by construction ...
        self.assertTrue(balance["conservation_account"]["holds"])
        self.assertEqual(balance["conservation_account"]["deposits"], 3)
        self.assertNotIn("reads", balance["conservation_account"]["identity"])
        # ... and the issuance account closes without a reads term too.
        self.assertTrue(balance["issuance_account"]["holds"])
        self.assertEqual(balance["issuance_account"]["issued"], 3)
        self.assertNotIn("read", balance["issuance_account"]["formula"])

    def test_reads_are_not_mass_so_they_cannot_break_either_account(self) -> None:
        pool = DebrisPool(capacity=4)
        pool.deposit(
            _debris(
                "d1",
                (_cohort("r", "residual"), _cohort("m", "motif")),
            ),
            Provenance(19, 1),
            1,
        )
        before = (
            pool.conservation_account(),
            pool.issuance_account(),
            len(pool),
        )
        for _ in range(3):
            for kind in COHORT_KINDS:
                pool.read(kind)
        self.assertEqual(
            (pool.conservation_account(), pool.issuance_account(), len(pool)),
            before,
        )


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
                    body, (BiasPrimitive, type(None)),
                    msg=f"tick {tick} flow #{fid}: {type(body).__name__}",
                )

    def test_d2_channels_carry_pinned_kinds(self) -> None:
        for tick, fid, envelope in self.events:
            if fid in D2_FLOWS:
                body = envelope.body
                self.assertIn(type(body), PAYLOAD_SCHEMA[fid], msg=f"#{fid}")

    def test_d4_core_accumulates_only_bias_primitives(self) -> None:
        self.assertEqual(len(self.dry.board.d4), _TURNS)
        for bias in self.dry.board.d4:
            self.assertIsInstance(bias, BiasPrimitive)


class TestEmissionProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dry = DryRun()
        cls.dry.run(32)
        cls.events = cls.dry.emissions

    def test_every_emission_is_an_envelope_with_ordered_provenance(self) -> None:
        for tick, fid, envelope in self.events:
            self.assertIn(type(envelope.provenance), (Provenance,))
            self.assertLessEqual(
                envelope.provenance.tick_born, tick, msg=f"#{fid}"
            )


if __name__ == "__main__":
    unittest.main()
