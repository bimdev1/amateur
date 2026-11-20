"""Round 11 (wave 3): the LeWM physics run through the ledger executor.

This module is the thin adapter between the physics organ
(:mod:`amateur.physics.organ`) and the ledger executor (:class:`DryRun`)
under the Round-11 contract (``docs/round11-contract.md`` §3.5 and §5).

PRE-REGISTERED GATES (written here, and in the module constants below,
**before** any measurement; §5).  Both are computed from the same
:class:`RunTick` rows / promotion log:

============  =======================================================  ========
Gate          Definition                                               PASS iff
============  =======================================================  ========
ADAPTATION    mean over ticks with ``phase == "flipped"`` **and** at    ``< 1.0``
              least one shadow-scored resident candidate of
              ``shadow adapted / base``
ADMISSION     count of **promoted** candidates (surfaced by ``#23``,    ``== 0``
              lineage-consumed by ``#22``) whose final residence
              ``ratio >= 1.0``
============  =======================================================  ========

Pre-registered, **not** gates (§5(i)-(iii)): the seed-7 residence-evidence
curve reported per resident tick as ``(candidate_id, residence_ticks,
adapted_mean, base_mean, ratio, qualified)``; seed 0 as the rejection
control with the expectation **zero promotions**; the live-overlay
behaviour (#24 -> stale -> evict, D4 retained contents byte-identical,
encoder bit-identical) reported as measured state.

Registered scenario (§3.5): ``warmup=8``, gravity flip at frame 8,
``post_flip=20``, flip back at frame 28, ``recovery=8`` -> 36 ticks.  The
numbers are re-measured this round; no round 7-9a measurement is
re-asserted.

The single-stream regime cannot reach the ``#21`` -> ``#24`` live-overlay
path: ``MOTIF_MIN_PERSISTENCE = 10`` was derived on a multi-stream bank, and
one stream never accumulates that much support (measured: longest support
streak 2, zero motif cohorts).  Per the Lead's binding ruling the constant
is **not tuned**: the registered trace reports ``motif cohorts = 0`` and
zero live-overlay events, and the full lifecycle is demonstrated by the
organ-level 16-stream probe
(:meth:`amateur.physics.organ.PhysicsOrgan.nursery_scenario`), clearly
labelled POST-HOC PROBE / NOT A GATE.

Three honesty constraints are encoded structurally:

1. The candidate's patch is **substance, never steering**: it is fitted and
   shadow-scored by the organ; the live model is asserted bit-identical on
   every shadow call, and only a ``#22`` promotion puts it into D4
   provenance (never into the live predictor).
2. The D2 patch table is in memory only.  A resident record without a patch
   is an assertion, never a silent unpatched run; restart persistence is
   out of scope this round.
3. Every gate verdict is computed from the trace independently in the tests
   and cross-checked against the summary, so the trace stays the single
   source of truth.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dryrun import PLACEHOLDER_FLOWS, DryRun
from ..ledger import FLOW_LEDGER
from ..payloads import (
    EmergentCrystalBody,
    NON_READ_FLOWS,
    NOOP_OVERLAY_ID,
    NO_MOTIF_ID,
    BiasPrimitive,
    CollapseMandateBody,
    CostErrorBody,
    DebrisCohort,
    DigestedDebrisBody,
    FrameBody,
    LatentRolloutBody,
    MatOverlay,
    PredictionErrorBody,
    StateDeltaBody,
    noop_bias,
    noop_crystal,
    noop_debris,
    noop_mandate,
    noop_overlay,
)
from ..provenance import Envelope, Provenance
from .organ import (
    COHORT_KINDS,
    DEFAULT_FIT_WINDOW,
    DigestResult,
    Observation,
    OverlayRecord,
    PhysicsOrgan,
    StructuralMotif,
)

__all__ = [
    "ADAPTATION_GATE",
    "ADMISSION_GATE",
    "CONTROL_SEED",
    "FLIP_AT",
    "FLIP_BACK_AT",
    "POST_FLIP",
    "RECOVERY",
    "REGISTERED_SEED",
    "TOTAL_TICKS",
    "WARMUP",
    "AcceptanceResult",
    "PhysicsRun",
    "PhysicsSource",
    "RunTick",
    "coverage_ledger",
]

# ---------------------------------------------------------------------------
# PRE-REGISTERED CONSTANTS (contract §5; fixed before measuring)
# ---------------------------------------------------------------------------
#: ADAPTATION gate threshold: PASS iff the flipped-phase mean shadow ratio
#: is strictly below this value.
ADAPTATION_GATE: float = 1.0
#: ADMISSION gate threshold: PASS iff exactly this many promoted candidates
#: have a final residence ratio >= 1.0.
ADMISSION_GATE: int = 0
#: The registered trace's seed.
REGISTERED_SEED: int = 7
#: The rejection-control seed (pre-registered expectation: zero promotions).
CONTROL_SEED: int = 0
#: Registered scenario shape (frames 1-based).
WARMUP: int = 8
FLIP_AT: int = WARMUP
POST_FLIP: int = 20
FLIP_BACK_AT: int = WARMUP + POST_FLIP
RECOVERY: int = 8
TOTAL_TICKS: int = WARMUP + POST_FLIP + RECOVERY

#: Non-read flows minus the fixed placeholders: the flows a physics source
#: must derive.  Derived from the executor's own ledger constants, never
#: hard-coded (contract §2.7).
DERIVED_FLOWS: frozenset[int] = frozenset(NON_READ_FLOWS) - PLACEHOLDER_FLOWS
#: Store reads: executor-derived, no source body (contract §2.1).
STORE_READ_FLOWS: frozenset[int] = frozenset(
    flow.fid for flow in FLOW_LEDGER
) - frozenset(NON_READ_FLOWS)


def coverage_ledger() -> dict[str, Any]:
    """The physics coverage account, **derived** from the ledger (§2.7).

    ``derived`` is the source's dispatch set, ``placeholders`` is the
    executor's fixed placeholder set intersected with the non-read flows,
    and ``store_reads`` is the complement of the non-read flows.  The
    counts are computed, not asserted constants: the physics source must
    cover exactly ``derived`` and the tests check the arithmetic against
    this account.
    """
    non_read = frozenset(NON_READ_FLOWS)
    placeholders = frozenset(non_read & PLACEHOLDER_FLOWS)
    derived = frozenset(non_read - PLACEHOLDER_FLOWS)
    return {
        "non_read": tuple(sorted(non_read)),
        "derived": tuple(sorted(derived)),
        "placeholders": tuple(sorted(placeholders)),
        "store_reads": tuple(sorted(STORE_READ_FLOWS)),
        "n_non_read": len(non_read),
        "n_derived": len(derived),
        "n_placeholders": len(placeholders),
        "n_store_reads": len(STORE_READ_FLOWS),
    }


def _vector(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


def _matrix(values: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(_vector(row) for row in np.asarray(values))


class PhysicsSource:
    """Content-only source hook over a :class:`PhysicsOrgan` (Q2).

    Round 11 dispatch (contract §2.1/§3.5): ``#3`` frame, ``#6`` prediction
    error, ``#10`` rollout, ``#12`` state delta, ``#16`` cost/error,
    ``#17`` collapse mandate, ``#18`` candidate birth, ``#19`` tagged
    composite, ``#22`` the ``#23``-derived ratchet gate, ``#24`` the
    ``#21``-motif-driven live overlay.  ``#21`` and ``#23`` are executor
    store reads and are *not* in this dispatch; asking for any fid outside
    it raises (no synthetic fallback).

    The derivation is a fixed method map; ``coverage`` is plain data read by
    tests only and no behaviour branches on it.  The source never touches
    the schedule, the stage order, or the channel semantics.
    """

    mode = "physics"

    def __init__(self, organ: PhysicsOrgan) -> None:
        self.organ = organ
        account = coverage_ledger()
        self.coverage: dict[int, str] = {
            fid: (
                "placeholder" if fid in account["placeholders"] else "derived"
            )
            for fid in sorted(account["non_read"])
        }
        self.read_coverage: dict[int, str] = {
            fid: "store-read/executor-derived" for fid in sorted(STORE_READ_FLOWS)
        }
        #: Fixed derivation dispatch (asserted == the ledger-derived set).
        self._derivers = {
            3: self._frame,
            6: self._prediction_error,
            10: self._rollout,
            12: self._state_delta,
            16: self._cost_error,
            17: self._mandate,
            18: self._candidate,
            19: self._debris,
            22: self._gate,
            24: self._revalidation_token,
        }
        if frozenset(self._derivers) != frozenset(account["derived"]):
            raise AssertionError(
                "the physics source dispatch does not cover the ledger's "
                f"derived flows: {sorted(set(self._derivers) ^ set(account['derived']))}"
            )
        # Tick context.
        self._frame_body: FrameBody | None = None
        self._observation: Observation | None = None
        self._digest: DigestResult | None = None
        self._delta: tuple[float, ...] = ()
        self._surprise: float = float("nan")
        self._born: Any = None
        # Per-tick outcomes, kept for inspection/debugging.  The runner reads
        # the committed bodies back from the executor's derived envelopes, so
        # these are a second view of the same tick, never a second source.
        self.last_gate: BiasPrimitive | None = None
        self.last_survivor: Any = None
        self.last_adoption: OverlayRecord | None = None
        self.last_motif_id: str | None = None

    # ------------------------------------------------------------- context
    def begin_tick(
        self,
        frame: np.ndarray,
        observation: Observation,
        digest: DigestResult,
        *,
        candidate: Any = None,
    ) -> None:
        """Bind this tick's real context; called before ``DryRun.tick``."""
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be (H, W, 3)")
        self._frame_body = FrameBody(
            frame=np.ascontiguousarray(frame).tobytes(),
            height=int(frame.shape[0]),
            width=int(frame.shape[1]),
            tick=int(observation.tick),
        )
        self._observation = observation
        self._digest = digest
        self._born = candidate
        self._delta = _vector(observation.latent_delta)
        self._surprise = float(observation.surprise)
        self.last_gate = None
        self.last_survivor = None
        self.last_adoption = None
        self.last_motif_id = None

    def _require_context(self) -> tuple[Observation, DigestResult]:
        if (
            self._observation is None
            or self._digest is None
            or self._frame_body is None
        ):
            raise AssertionError(
                "PhysicsSource.derive_body called before begin_tick(); "
                "no derived body can be supplied without the organ context"
            )
        return self._observation, self._digest

    # ----------------------------------------------------------- derivations
    def derive_body(
        self, fid: int, tick: int, *, board: Any, debris: Any
    ) -> object:
        deriver = self._derivers.get(fid)
        if deriver is None:
            raise AssertionError(
                f"physics source has no derivation for flow #{fid}; "
                "it must never fall back to a synthetic body"
            )
        self._require_context()
        return deriver(int(tick), board, debris)

    def _frame(self, tick: int, board: Any, debris: Any) -> FrameBody:
        assert self._frame_body is not None
        return self._frame_body

    def _prediction_error(
        self, tick: int, board: Any, debris: Any
    ) -> PredictionErrorBody:
        observation, _ = self._require_context()
        return PredictionErrorBody(
            surprise=self._surprise,
            residual=_vector(observation.residual),
        )

    def _rollout(
        self, tick: int, board: Any, debris: Any
    ) -> LatentRolloutBody:
        observation, _ = self._require_context()
        return LatentRolloutBody(latents=_matrix(observation.rollout))

    def _state_delta(
        self, tick: int, board: Any, debris: Any
    ) -> StateDeltaBody:
        return StateDeltaBody(delta=self._delta, surprise=self._surprise)

    def _cost_error(
        self, tick: int, board: Any, debris: Any
    ) -> CostErrorBody:
        """#16 (contract §2.6): same values #12 carries, same tick.

        ``cost`` is the L2 norm of the **#12 delta vector itself** (the same
        tuple, not a re-read), so ``#16.cost == ||#12.delta||_2`` and
        ``#16.error == #12.surprise`` hold by construction; the executor
        test asserts them from the derived envelopes.
        """
        cost = float(np.linalg.norm(np.asarray(self._delta, dtype=np.float64)))
        return CostErrorBody(cost=cost, error=self._surprise)

    def _mandate(
        self, tick: int, board: Any, debris: Any
    ) -> CollapseMandateBody:
        _, digest = self._require_context()
        mandate = digest.mandate
        if mandate is None:
            return noop_mandate()
        return CollapseMandateBody(
            reason=mandate.reason,
            pressure=int(mandate.pressure),
            capacity=int(mandate.capacity),
            active_bodies=int(mandate.active_bodies),
            freed_body=mandate.freed_body,
            tick=int(mandate.tick),
        )

    def _candidate(
        self, tick: int, board: Any, debris: Any
    ) -> object:
        """#18: the nursery birth, or the exact-type no-op sentinel.

        The candidate is fitted and registered org-side by the runner's
        trigger *before* the derivation pass (``birth_candidate``), so this
        derivation only transports the birth into D2 -- it never decides
        one.  ``born_tick`` is the emission tick by construction.
        """
        candidate = self._born
        if candidate is None:
            return noop_crystal()
        observation, _ = self._require_context()
        return EmergentCrystalBody(
            candidate_id=str(candidate.candidate_id),
            residual=_vector(observation.residual),
            surprise=self._surprise,
            born_tick=int(tick),
        )

    def _debris(
        self, tick: int, board: Any, debris: Any
    ) -> DigestedDebrisBody:
        """#19: pack the tick's tagged digest cohorts, or the no-op sentinel.

        Every cohort is provenance-stamped with the depositing ``#19``
        write's own provenance ``(19, tick)`` (contract §1.7/§2.1); a tick
        with no cohorts emits ``noop_debris()``.
        """
        _, digest = self._require_context()
        if not digest.cohorts:
            return noop_debris()
        provenance = Provenance(19, int(tick))
        cohorts = tuple(
            DebrisCohort(
                cohort_id=str(item.cohort_id),
                kind=str(item.kind),
                centroid=_vector(item.centroid),
                count=int(item.count),
                spread=float(item.spread),
                first_tick=int(item.first_tick),
                last_tick=int(item.last_tick),
                peak_surprise=float(item.peak_surprise),
                provenance=provenance,
            )
            for item in digest.cohorts
        )
        return DigestedDebrisBody(
            debris_id=f"debris-{int(tick):04d}", cohorts=cohorts
        )

    def _gate(
        self, tick: int, board: Any, debris: Any
    ) -> BiasPrimitive:
        """#22: derived from the **same** ``#23`` survivor store read.

        ``board.nursery_survivor(tick)`` is called here exactly as the
        executor's ``#23`` derivation calls it, and the survivor is
        surfaced **without** a ``qualified`` pre-filter (contract §1.9/§3.4).
        A qualified survivor (`ratio < 1.0` over at least ``RESIDENCE_N``
        shadow ticks) yields a positive sediment carrying that candidate's
        lineage; an unqualified (harmful) survivor, or no survivor at all,
        yields the Q4 zero-confidence sentinel and no promotion.
        """
        survivor = board.nursery_survivor(int(tick))
        self.last_survivor = survivor
        if survivor is None:
            body = noop_bias()
            self.last_gate = body
            return body
        ratio = float(survivor.ratio)
        if survivor.qualified:
            confidence = float(min(max(1.0 - ratio, 0.0), 1.0))
            body = BiasPrimitive(
                "ratchet-gate-passed",
                confidence,
                str(survivor.candidate_id),
            )
        else:
            body = noop_bias()
        self.last_gate = body
        return body

    def _revalidation_token(
        self, tick: int, board: Any, debris: Any
    ) -> MatOverlay:
        """#24: the live overlay, fitted/revalidated from the ``#21`` motif.

        The motif is read from D3 with the same call the executor's ``#21``
        derivation uses (``board.debris.best_motif()``) and normalised to
        the organ's :class:`StructuralMotif` shape with the same field
        mapping, so the body's ``motif_lineage`` equals the motif id the
        executor consumed this tick (its lineage gate asserts exactly that).
        A tick with no motif, or an adoption the ratchet refuses (cooldown,
        another active overlay, capacity), emits ``noop_overlay()``.
        """
        motif_cohort = debris.best_motif()
        if motif_cohort is None:
            return noop_overlay()
        motif = StructuralMotif(
            motif_id=str(motif_cohort.cohort_id),
            debris_id=str(motif_cohort.cohort_id),
            centroid=np.asarray(motif_cohort.centroid, dtype=np.float64),
            count=int(motif_cohort.count),
            spread=float(motif_cohort.spread),
            persistence=max(
                0, int(motif_cohort.last_tick) - int(motif_cohort.first_tick)
            ),
            drift=0.0,
            first_tick=int(motif_cohort.first_tick),
            last_tick=int(motif_cohort.last_tick),
        )
        self.last_motif_id = motif.motif_id
        record = self.organ.adopt_motif(motif, int(tick))
        self.last_adoption = record
        if record is None:
            return noop_overlay()
        epoch, overlay_id, status = record.token()
        return MatOverlay(epoch, overlay_id, status, motif_lineage=motif.motif_id)

    # ------------------------------------------------------- shadow evidence
    def shadow_evidence(self, tick: int) -> dict[str, tuple[float, float]]:
        """§2.5: this tick's shadow scores, one-way and read-only.

        The store's candidate index receives the ``#18`` birth at the end of
        the birth tick, but the executor's annotation hook runs *before* the
        derivation pass -- so evidence is offered only for candidates the
        store already holds (``born_tick < tick``).  The runner already
        excludes the in-sample birth-tick measurement (``exclude_born_at``),
        so the two evidence sets are identical; the ``born_tick`` filter is
        kept as a hard guard (it must never drop a candidate, or the organ
        and store would diverge).
        """
        scores = self.organ.shadow_scores(
            int(tick), stream=0, exclude_born_at=int(tick)
        )
        if not scores:
            return {}
        offered: dict[str, tuple[float, float]] = {}
        for candidate_id, pair in scores.items():
            record = self.organ.candidate_record(candidate_id)
            if record is None or int(record.born_tick) >= int(tick):
                raise AssertionError(
                    "shadow evidence offered for a candidate the store does "
                    f"not hold yet ({candidate_id!r} born {getattr(record, 'born_tick', None)} "
                    f"at tick {tick}): the in-sample birth-tick sample must "
                    "never leave the organ"
                )
            offered[candidate_id] = pair
        return offered

    # ------------------------------------------------------------ commits
    def on_commit(
        self,
        fid: int,
        envelope: Envelope,
        *,
        board: Any,
        debris: Any,
    ) -> None:
        """One-way post-commit notification (Q2-3, Q1-A).

        ``#18`` attaches the candidate's **substance** patch to the store's
        candidate index; ``#24`` attaches the live patch; a positive ``#22``
        sediments *after* the executor promoted the candidate, so the organ
        mirrors that promotion here (the ``#23`` lineage consumption).
        A positive ``#22`` also runs the one-way prune that keeps both
        indices, both patch tables and both lifecycle owners aligned (the
        runner prunes again at the end of every tick).
        """
        tick = int(envelope.provenance.tick_born)
        body = envelope.body
        if fid == 18 and type(body) is EmergentCrystalBody:
            candidate_id = str(body.candidate_id)
            if candidate_id != noop_crystal().candidate_id:
                patch = self.organ.candidate_patch(str(candidate_id))
                if patch is None:
                    raise AssertionError(
                        f"candidate {candidate_id!r} has no shadow patch; "
                        "refusing to run with an unowned candidate substance"
                    )
                board.attach_candidate_patch(str(candidate_id), patch)
        elif fid == 24 and type(body) is MatOverlay:
            if body.overlay_id != NOOP_OVERLAY_ID:
                patch = self.organ.overlay_table.adapter(body.overlay_id)
                if patch is None:
                    raise AssertionError(
                        f"D2 record {body.overlay_id!r} has no predictor "
                        "patch; refusing to run unpatched (Q1-A)"
                    )
                board.attach_patch(body.overlay_id, patch)
        elif fid == 22:
            # The store has already promoted (contract §2.3: promotion before
            # the sediment), so the organ mirror runs first -- a prune here
            # would see the store's absence and close the candidate before
            # the promotion could be recorded.
            if type(body) is BiasPrimitive and body.confidence > 0.0:
                lineage = body.candidate_lineage
                if lineage is None:
                    raise AssertionError(
                        "a positive #22 sediment must carry candidate_lineage"
                    )
                self.organ.promote_candidate(str(lineage), tick)
            self.prune(board, tick=tick)

    def prune(self, board: Any, *, tick: int) -> None:
        """Align organ and store on both indices, then assert the bounds.

        Overlay index: the organ owns the lifecycle, so a store record the
        organ has dropped is evicted (Q1-A).  Candidate index: the store's
        ``#18`` birth / ``#22`` promotion are authoritative, so an organ
        candidate the store no longer holds is pruned org-side, and a store
        candidate the organ never held (a refused ``#18``) is evicted.
        """
        live = self.organ.live_overlay_ids()
        for overlay_id in sorted(board.resident_ids() - live):
            board.evict_record(overlay_id, tick)
        self.organ.prune_overlays(board.resident_ids())
        organ_candidates = set(self.organ.candidate_ids())
        store_candidates = set(board.nursery_ids())
        for candidate_id in sorted(store_candidates - organ_candidates):
            board.nursery_evict_record(candidate_id, tick, "organ")
        self.organ.prune_candidates(frozenset(board.nursery_ids()))
        board.assert_patch_bounds(require_complete=True)


# ---------------------------------------------------------------------------
# Single-stream executor run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunTick:
    """One executor tick of the single-stream run (measured state).

    Contract §3.5 fields plus the survivor fields the Lead's ruling requires
    (so the report can show whether a harmful survivor occurred and was
    refused) and the four residence-curve fields §5(i) needs per resident
    tick.  Every field is measured; nothing is inferred.

    With more than one resident candidate the curve fields describe the
    **newest** resident (the registered run births at most one, so the choice
    is not exercised there); ``nursery_ids`` always carries every resident, so
    no candidate is hidden by the choice.  Absent evidence is ``None``, never
    a fabricated mean.
    """

    tick: int
    frame_tick: int
    phase: str
    action: int
    surprise: float
    base_surprise: float
    overlay_id: str | None
    adopted: str | None
    staled: str | None
    evicted: tuple[str, ...]
    residents: tuple[str, ...]
    patch_bytes: int
    d4_size: int
    # --- candidate / residence evidence (§3.5 + §5(i)) --------------------
    candidate_id: str | None
    candidate_residence_ticks: int
    candidate_adapted_mean: float | None
    candidate_base_mean: float | None
    candidate_ratio: float | None
    candidate_qualified: bool
    shadow_ratio: float | None
    shadow_n: int
    # --- the #23 survivor read and the #22 gate ---------------------------
    survivor_id: str | None
    survivor_ratio: float | None
    survivor_qualified: bool | None
    gate: str
    promoted_id: str | None
    # --- the #21 motif read and the #19/#27 D3 account --------------------
    motif_id: str | None
    cohort_kinds: tuple[str, ...]
    drained_ids: tuple[str, ...]
    d3_mass: dict[str, Any]
    d3_bytes: int
    nursery_ids: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceResult:
    """The single-stream scenario record; every number is measured."""

    ticks: tuple[RunTick, ...]
    summary: dict[str, Any]
    #: Per-tick D4 snapshots (immutable tuples), for the retained-contents
    #: byte-identity assertion.
    d4_trace: tuple[tuple[BiasPrimitive, ...], ...] = ()

    def rows(self, phase: str) -> tuple[RunTick, ...]:
        return tuple(row for row in self.ticks if row.phase == phase)


class PhysicsRun:
    """Drive one physical stream through the ledger executor.

    ``PhysicsRun`` owns E1 (through the organ), the frozen-encoder model,
    the candidate nursery (organ-side) and the executor.  The per-tick
    order is contract §3.5:

    1. step E1 and observe,
    2. digest (P5, tagged cohorts),
    3. ``birth_candidate`` when the loop trigger fires (``#18`` birth is
       immediate -- the round-9a validation delay is gone),
    4. shadow-score every resident candidate (never steering),
    5. resolve the ratchet's staleness/eviction decision,
    6. bind the source context and run one ``DryRun.tick()`` (derivation ->
       arrivals -> write absorption: ``#23`` survivor read, ``#22`` gate,
       ``#21`` motif read, ``#24`` live overlay, ``#19`` deposit, ``#27``
       drain),
    7. prune both indices and record the row.

    The loop trigger is the round-9a ``_should_propose`` shape without the
    delay: no fit attempted yet, no active live overlay, enough history for
    the registered fit window, and this tick's surprise above the measured
    spike threshold.  Birth is immediate at the trigger tick.
    """

    def __init__(
        self,
        organ: PhysicsOrgan | None = None,
        *,
        seed: int = REGISTERED_SEED,
        fit_window: int = DEFAULT_FIT_WINDOW,
        birth_on_spike: bool = True,
    ) -> None:
        self.organ = organ if organ is not None else PhysicsOrgan(
            store_frames=True, fit_streams={0}
        )
        # A run is a single-stream experiment: the fit and the shadow probe
        # are pinned to stream 0 so a supplied organ cannot leak bank data.
        self.organ.fit_streams = {0}
        self.organ.store_frames = True  # the source needs the #3 frame body
        # A runner always starts from a fresh P6: a supplied organ may carry
        # candidate records, evidence, promotions or overlay state from a
        # previous experiment, and reusing them would make the run
        # unrepeatable.
        self.organ.reset_p6_state()
        self.source = PhysicsSource(self.organ)
        self.dry = DryRun(source=self.source, seed_board=False)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.fit_window = int(fit_window)
        self.birth_on_spike = bool(birth_on_spike)
        self.ticks: list[RunTick] = []
        self.d4_trace: list[tuple[BiasPrimitive, ...]] = []
        #: The candidate births, in order (the run births at most one).
        self.births: list[tuple[str, int]] = []
        self.started = False
        self._fit_attempted = False
        self._flip_at: int | None = None
        self._flip_back_at: int | None = None

    # -------------------------------------------------------------- helpers
    @property
    def board(self) -> Any:
        return self.dry.board

    def reset(self, seed: int | None = None) -> Observation:
        """Reset E1 and the run state (fresh D2/D3/D4/patches/nursery)."""
        self.seed = self.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._flip_at = None
        self._flip_back_at = None
        # Clear the nursery, evidence, promotions, overlay records/patches,
        # cooldown and fit history, then rebuild the source/executor so the
        # stores start clean.
        self.organ.reset_p6_state()
        self.source = PhysicsSource(self.organ)
        self.dry = DryRun(source=self.source, seed_board=False)
        observation = self.organ.reset(seed=self.seed)
        self.ticks.clear()
        self.births.clear()
        self.d4_trace.clear()
        self.d4_trace.append(tuple(self.board.d4))
        self.started = True
        self._fit_attempted = False
        return observation

    def flip_gravity(self) -> int | dict[int, int]:
        return self.organ.flip_gravity(stream=0)

    def _should_birth(self, observation: Observation) -> bool:
        """The loop trigger: the ``_should_propose`` shape, birth immediate."""
        if not self.birth_on_spike or self._fit_attempted:
            return False
        if self.organ.overlay_table.active():
            return False
        if int(observation.tick) < self.fit_window:
            return False
        surprise = float(observation.surprise)
        return bool(
            np.isfinite(surprise)
            and surprise > float(self.organ.thresholds.spike_threshold)
        )

    def _phase(self, frame_tick: int) -> str:
        if self._flip_at is not None and frame_tick >= self._flip_at:
            if self._flip_back_at is not None and frame_tick >= self._flip_back_at:
                return "recovered"
            return "flipped"
        return "normal"

    def tick(
        self,
        action: int | None = None,
        *,
        injections: dict[int, int] | None = None,
    ) -> RunTick:
        """One executor tick: E1 -> organ -> source -> DryRun.

        ``injections`` is forwarded to :meth:`DryRun.tick` for the
        adversarial-multiplicity (reject-with-charge) tests; the canonical
        scenario never passes it.
        """
        if not self.started:
            self.reset()
        if action is None:
            action = int(
                self.rng.integers(0, self.organ.base_model.config.action_dim)
            )
        # 1-2. step and digest.
        observation = self.organ.step(int(action), stream=0)
        if observation.frame is None:  # pragma: no cover - run forces frames
            raise AssertionError(
                "PhysicsRun needs observation.frame (store_frames=True)"
            )
        digest = self.organ.digest(observation)
        # 3. #18: immediate birth at the trigger tick (no validation delay).
        born = None
        if self._should_birth(observation):
            self._fit_attempted = True
            born = self.organ.birth_candidate(
                int(observation.tick), fit_window=self.fit_window, stream=0
            )
            if born is not None:
                self.births.append((str(born.candidate_id), int(observation.tick)))
        # 4. shadow-score every resident candidate on this tick's transition.
        #    Out-of-sample only: a candidate born this tick was fitted on the
        #    last ``fit_window`` transitions *ending at this tick*, so its
        #    birth-tick transition is in-sample and cannot be evidence (the
        #    store cannot even carry it: its annotation runs before #18
        #    commits).  Excluding it keeps the organ's evidence set identical
        #    to the store's, so the ADAPTATION gate reads the same samples the
        #    #22 gate does.
        scores = self.organ.shadow_scores(
            int(observation.tick),
            stream=0,
            exclude_born_at=(
                None if born is None else int(born.born_tick)
            ),
        )
        if scores:
            adapted = float(np.mean([pair[0] for pair in scores.values()]))
            base = float(np.mean([pair[1] for pair in scores.values()]))
            shadow_ratio: float | None = (
                adapted / base if base > 0.0 else float("inf")
            )
        else:
            shadow_ratio = None
        # The residence curve is captured *before* the executor tick: a
        # promotion inside it removes the candidate, and the curve must
        # report the evidence the tick actually measured.
        residents = self.organ.nursery.resident()
        tracked = born if born is not None else (
            residents[-1] if residents else None
        )
        if tracked is None:
            residue = (None, 0, None, None, None, False)
        elif int(tracked.residence_ticks) == 0:
            # The birth tick itself is not shadow-scored (in-sample), so the
            # curve row exists with no evidence yet -- ``None``, never a
            # fabricated mean (and never NaN, so rows compare exactly).
            residue = (str(tracked.candidate_id), 0, None, None, None, False)
        else:
            residue = (
                str(tracked.candidate_id),
                int(tracked.residence_ticks),
                float(tracked.adapted_mean),
                float(tracked.base_mean),
                float(tracked.ratio),
                bool(tracked.qualified(self.organ.thresholds.residence_n)),
            )
        # 5. ratchet: staleness / eviction of the live overlay (never births).
        stales_before = len(self.organ.overlay_table.stales)
        self.organ.ratchet(digest)
        staled = None
        if len(self.organ.overlay_table.stales) > stales_before:
            staled = self.organ.overlay_table.stales[-1].overlay_id
        # 6. bind the source context and run the executor tick.
        promotions_before = len(self.organ.promotions)
        evictions_before = len(self.board.evictions)
        self.source.begin_tick(
            observation.frame, observation, digest, candidate=born
        )
        self.dry.tick(injections=injections)
        # 7. prune both indices and read back the tick's committed state.
        self.source.prune(self.board, tick=self.dry.tick_no)
        evicted = tuple(
            record[0] for record in self.board.evictions[evictions_before:]
        )
        survivor = self.dry.derived[23].body
        gate = self.dry.derived[22].body
        overlay_body = self.dry.derived[24].body
        motif_body = self.dry.derived[21].body
        entropy_body = self.dry.derived[27].body
        debris_body = self.dry.derived[19].body
        promoted_id = None
        if len(self.organ.promotions) > promotions_before:
            promoted_id = self.organ.promotions[-1].candidate_id
        row = RunTick(
            tick=self.dry.tick_no,
            frame_tick=int(observation.tick),
            phase=self._phase(int(observation.tick)),
            action=int(action),
            surprise=float(observation.surprise),
            base_surprise=float(observation.base_surprise),
            overlay_id=observation.overlay_id,
            adopted=(
                None
                if overlay_body is None
                or overlay_body.overlay_id == NOOP_OVERLAY_ID
                else overlay_body.overlay_id
            ),
            staled=staled,
            evicted=evicted,
            residents=tuple(
                resident_id for resident_id, _epoch in self.board.residents()
            ),
            patch_bytes=self.board.patch_bytes(),
            d4_size=len(self.board.d4),
            candidate_id=residue[0],
            candidate_residence_ticks=residue[1],
            candidate_adapted_mean=residue[2],
            candidate_base_mean=residue[3],
            candidate_ratio=residue[4],
            candidate_qualified=residue[5],
            shadow_ratio=shadow_ratio,
            shadow_n=len(scores),
            survivor_id=(
                None if survivor is None else str(survivor.candidate_id)
            ),
            survivor_ratio=(
                None if survivor is None else float(survivor.ratio)
            ),
            survivor_qualified=(
                None if survivor is None else bool(survivor.qualified)
            ),
            gate=str(gate.bias_primitive),
            promoted_id=promoted_id,
            motif_id=(
                None
                if motif_body is None
                or motif_body.motif_id == NO_MOTIF_ID
                else str(motif_body.motif_id)
            ),
            cohort_kinds=(
                ()
                if debris_body is None or not debris_body.cohorts
                else tuple(sorted({str(c.kind) for c in debris_body.cohorts}))
            ),
            drained_ids=(
                ()
                if entropy_body is None
                else tuple(str(cid) for cid in entropy_body.cohort_ids)
            ),
            d3_mass=self.dry.debris.mass_balance(),
            d3_bytes=self.dry.debris.byte_size(),
            nursery_ids=tuple(self.board.nursery_ids()),
        )
        self.ticks.append(row)
        self.d4_trace.append(tuple(self.board.d4))
        return row

    def run(
        self,
        ticks: int,
        *,
        flip_at: int | None = None,
        flip_back_at: int | None = None,
    ) -> tuple[RunTick, ...]:
        """Run ``ticks`` executor ticks with optional gravity changes."""
        if ticks < 1:
            raise ValueError("ticks must be >= 1")
        self.reset()
        return self._drive(ticks, flip_at=flip_at, flip_back_at=flip_back_at)

    def _drive(
        self,
        ticks: int,
        *,
        flip_at: int | None = None,
        flip_back_at: int | None = None,
    ) -> tuple[RunTick, ...]:
        """The tick loop without the reset (``run`` / ``scenario`` share it)."""
        if ticks < 1:
            raise ValueError("ticks must be >= 1")
        self._flip_at = flip_at
        self._flip_back_at = flip_back_at
        rows: list[RunTick] = []
        for index in range(1, ticks + 1):
            if flip_at is not None and index == flip_at:
                self.flip_gravity()
            if flip_back_at is not None and index == flip_back_at:
                self.flip_gravity()
            rows.append(self.tick())
        self._flip_at = None
        self._flip_back_at = None
        return tuple(rows)

    # ---------------------------------------------------------------- gates
    def adaptation(self) -> dict[str, Any]:
        """ADAPTATION (pre-registered §5): flipped-phase shadow ratio."""
        rows = [
            row
            for row in self.ticks
            if row.phase == "flipped" and row.shadow_ratio is not None
        ]
        n = len(rows)
        if n == 0:
            return {
                "value": float("nan"),
                "n": 0,
                "threshold": ADAPTATION_GATE,
                "passed": False,
                "reason": "no shadow-scored flipped-phase rows",
            }
        value = float(np.mean([row.shadow_ratio for row in rows]))
        return {
            "value": value,
            "n": n,
            "threshold": ADAPTATION_GATE,
            "passed": bool(value < ADAPTATION_GATE),
            "reason": "measured",
        }

    def admission(self) -> dict[str, Any]:
        """ADMISSION (pre-registered §5): promoted candidates with ratio >= 1."""
        promotions = self.organ.promotions
        harmful = [
            promotion
            for promotion in promotions
            if not (np.isfinite(promotion.ratio) and promotion.ratio < 1.0)
        ]
        return {
            "value": len(harmful),
            "n": len(promotions),
            "threshold": ADMISSION_GATE,
            "passed": bool(len(harmful) == ADMISSION_GATE),
            "harmful": tuple(
                {
                    "candidate_id": promotion.candidate_id,
                    "ratio": float(promotion.ratio),
                    "promoted_tick": int(promotion.promoted_tick),
                }
                for promotion in harmful
            ),
            "promotions": tuple(
                {
                    "candidate_id": promotion.candidate_id,
                    "born_tick": int(promotion.born_tick),
                    "promoted_tick": int(promotion.promoted_tick),
                    "residence_ticks": int(promotion.residence_ticks),
                    "adapted_mean": float(promotion.adapted_mean),
                    "base_mean": float(promotion.base_mean),
                    "ratio": float(promotion.ratio),
                    "qualified": bool(promotion.qualified),
                }
                for promotion in promotions
            ),
        }

    def scenario(
        self,
        *,
        seed: int = REGISTERED_SEED,
        warmup: int = WARMUP,
        post_flip: int = POST_FLIP,
        recovery: int = RECOVERY,
    ) -> AcceptanceResult:
        """The registered scenario: normal -> flip -> flip back -> recover.

        ``warmup=8`` normal frames, gravity flipped at frame 8,
        ``post_flip=20`` frames, flipped back at frame 28, ``recovery=8``
        frames -- 36 executor ticks.  The gates are computed from the rows
        (``adaptation``/``admission``) and the summary cross-checks them.
        """
        self.reset(seed=seed)
        before = tuple(self.board.d4)
        total = int(warmup) + int(post_flip) + int(recovery)
        rows = self._drive(
            total,
            flip_at=int(warmup),
            flip_back_at=int(warmup) + int(post_flip),
        )
        after = tuple(self.board.d4)
        return self._result(rows, before=before, after=after, seed=seed)

    def _result(
        self,
        rows: tuple[RunTick, ...],
        *,
        before: tuple[BiasPrimitive, ...],
        after: tuple[BiasPrimitive, ...],
        seed: int,
    ) -> AcceptanceResult:
        adaptation = self.adaptation()
        admission = self.admission()
        normal = [row for row in rows if row.phase == "normal"]
        flipped = [row for row in rows if row.phase == "flipped"]
        shadow_rows = [row for row in rows if row.shadow_ratio is not None]
        survivor_rows = [row for row in rows if row.survivor_id is not None]
        harmful_survivors = [
            row
            for row in survivor_rows
            if row.survivor_qualified is False
        ]
        resident_rows = [
            row for row in rows if row.candidate_id is not None
        ]
        curve = tuple(
            {
                "tick": row.tick,
                "phase": row.phase,
                "candidate_id": row.candidate_id,
                "residence_ticks": row.candidate_residence_ticks,
                "adapted_mean": row.candidate_adapted_mean,
                "base_mean": row.candidate_base_mean,
                "ratio": row.candidate_ratio,
                "qualified": row.candidate_qualified,
            }
            for row in resident_rows
        )
        d4_before = pickle.dumps(before, protocol=4)
        d4_after = pickle.dumps(after, protocol=4)
        retained_ok = all(
            pickle.dumps(previous, protocol=4)
            == pickle.dumps(nxt[: len(previous)], protocol=4)
            for previous, nxt in zip(self.d4_trace, self.d4_trace[1:])
        )
        cohort_counts: dict[str, int] = {kind: 0 for kind in COHORT_KINDS}
        for row in rows:
            for kind in row.cohort_kinds:
                cohort_counts[kind] += 1
        summary: dict[str, Any] = {
            "scenario": {
                "seed": int(seed),
                "warmup": WARMUP,
                "flip_at": FLIP_AT,
                "post_flip": POST_FLIP,
                "flip_back_at": FLIP_BACK_AT,
                "recovery": RECOVERY,
                "ticks": len(rows),
                "fit_window": self.fit_window,
            },
            "gates": {
                "adaptation": adaptation,
                "admission": admission,
            },
            # --- measured state, not gates (§5(ii)-(iii)) -----------------
            "control_expectation": (
                "pre-registered: zero promotions (seed 0)"
                if int(seed) == CONTROL_SEED
                else "registered trace (seed 7)"
            ),
            "normal_mean": float(
                np.mean([row.base_surprise for row in normal])
            ) if normal else float("nan"),
            "flipped_base_mean": float(
                np.mean([row.base_surprise for row in flipped])
            ) if flipped else float("nan"),
            "spike_peak": float(
                max((row.base_surprise for row in flipped), default=float("nan"))
            ),
            "residence_curve": curve,
            "resident_ticks": len(resident_rows),
            "shadow_ticks": len(shadow_rows),
            "survivor_ticks": len(survivor_rows),
            "harmful_survivor_ticks": len(harmful_survivors),
            "harmful_survivors": tuple(
                {
                    "tick": row.tick,
                    "candidate_id": row.survivor_id,
                    "ratio": row.survivor_ratio,
                }
                for row in harmful_survivors
            ),
            "candidate_fit": {
                "attempted": bool(self.organ.fit_history),
                "record": (
                    None
                    if not self.organ.fit_history
                    else {
                        "rank": self.organ.fit_history[-1].rank,
                        "param_count": self.organ.fit_history[-1].param_count,
                        "nbytes": self.organ.fit_history[-1].nbytes,
                        "loss_before": self.organ.fit_history[-1].loss_before,
                        "loss_after": self.organ.fit_history[-1].loss_after,
                        "budget_n": self.organ.fit_history[-1].budget_n,
                        "budget_capacity": (
                            self.organ.fit_history[-1].budget_capacity
                        ),
                        "reason": self.organ.fit_history[-1].reason,
                    }
                ),
                "refusals": tuple(self.organ.refusals),
            },
            "lifecycle": {
                "candidate_birth": (
                    self.births[0][1] if self.births else None
                ),
                "candidate_births": tuple(self.births),
                "first_shadow": next(
                    (row.tick for row in rows if row.shadow_ratio is not None),
                    None,
                ),
                "first_survivor": next(
                    (row.tick for row in rows if row.survivor_id is not None),
                    None,
                ),
                "promotions": tuple(
                    (promotion.candidate_id, promotion.promoted_tick)
                    for promotion in self.organ.promotions
                ),
                "overlay_birth": next(
                    (row.tick for row in rows if row.adopted is not None), None
                ),
                "overlay_stale": next(
                    (row.tick for row in rows if row.staled is not None), None
                ),
                "overlay_evict": next(
                    (row.tick for row in rows if row.evicted), None
                ),
                "motif_reads": sum(1 for row in rows if row.motif_id is not None),
                "nursery_empty_at_end": not self.organ.candidate_ids(),
                "d2_empty_at_end": not self.board.residents(),
            },
            "cohort_counts": cohort_counts,
            "d3": {
                "mass": dict(rows[-1].d3_mass) if rows else {},
                "bytes": rows[-1].d3_bytes if rows else 0,
                "drain_ticks": sum(1 for row in rows if row.drained_ids),
                "drained": sum(len(row.drained_ids) for row in rows),
                "deposits": rows[-1].d3_mass.get("deposits", 0) if rows else 0,
                "ceiling_evictions": len(self.dry.ceiling_evictions),
            },
            "d4": {
                "before_bytes": len(d4_before),
                "after_bytes": len(d4_after),
                "byte_identical": d4_before == d4_after,
                "retained_byte_identical": bool(retained_ok),
                "sediments": tuple(
                    {
                        "bias_primitive": body.bias_primitive,
                        "confidence": float(body.confidence),
                        "candidate_lineage": body.candidate_lineage,
                        "tick": tick,
                    }
                    for tick, body in self._d4_sediment_ticks()
                ),
            },
            "encoder_bit_identical": bool(
                self.organ.encoder_bit_identical()
            ),
            "patches_end": tuple(
                overlay_id for overlay_id, _patch in self.board.patches()
            ),
            "patch_bytes_end": self.board.patch_bytes(),
            "nursery_patch_bytes_end": self.board.nursery_patch_bytes(),
            "max_patch_bytes_seen": max(
                (row.patch_bytes for row in rows), default=0
            ),
            "zero_confidence_writes": self.dry.zero_confidence_absorbed.get(22, 0),
            "accepted_22": self.dry.accepted.get(22, 0),
            "consumed_22": self.dry.consumed.get(22, 0),
            "refused_22": self.dry.refused.get(22, 0),
            "coverage": coverage_ledger(),
        }
        return AcceptanceResult(ticks=rows, summary=summary, d4_trace=tuple(self.d4_trace))

    def _d4_sediment_ticks(self) -> tuple[tuple[int, BiasPrimitive], ...]:
        """``(tick, body)`` for every committed #22 sediment, from emissions."""
        found: list[tuple[int, BiasPrimitive]] = []
        for tick, fid, envelope in self.dry.emissions:
            if fid == 22 and type(envelope.body) is BiasPrimitive:
                if envelope.body.confidence > 0.0:
                    found.append((int(tick), envelope.body))
        return tuple(found)

    #: Backwards-compatible alias: Round 9a called the scenario ``acceptance``.
    def acceptance(self, **kwargs: Any) -> AcceptanceResult:
        return self.scenario(**kwargs)

