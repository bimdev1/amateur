"""Phase 4-5b: executor wiring for the physics organ (Q1-Q4, all option A).

This module is the thin adapter between the standalone physics organ
(:mod:`amateur.physics.organ`) and the ledger executor (:class:`DryRun`):

* :class:`PhysicsSource` is the **content-only payload source** (Q2).  It
  turns the organ's per-tick observation/digest/ratchet state into the typed
  bodies of the real flows, attaches the predictor patch to D2 through the
  one-way post-commit hook (Q1-A), and prunes the organ/store in step.  It
  never touches the schedule, stage order, or channel semantics.
* :class:`PhysicsRun` drives a **single** physical stream through the
  executor: E1 step -> observation -> digest -> ratchet -> proposal ->
  validation -> decision -> derivation -> arrivals -> write absorption.
  The acceptance scenario is the phase-5 event (normal -> flip -> spike ->
  digest -> proposal at 16 -> held-out validation on 17-18 -> birth or
  rejection at 18 -> flip back -> stale -> evict) measured state-by-state.

Three honesty constraints are encoded structurally:

1. The executor acceptance fits from a **bounded same-stream replay**
   (the pre-registered default: the last 16 transitions of this stream, no
   cross-stream bank data, no post-hoc variant selection), and the Round 9a
   **capacity bound** caps the stored patch at the rank the evidence admits
   (rank 26 / 5,056 params / 20,224 B on this trace).  A single-stream fit
   is not guaranteed to improve forward surprise; the candidate must pass a
   held-out validation window before it is born, and the summary reports the
   measured ratios (all-active and active-flipped) as measured, never a
   fabricated drop.
2. The D2 patch table is in memory only.  A resident record without a patch
   is an assertion, never a silent unpatched run; restart persistence is
   explicitly out of scope this round.
3. Every pre-adoption decision is public: ``organ.decisions`` records both
   outcomes with the measured means, and ``organ.rejections`` is the
   test-readable rejection log.  A rejected candidate never reaches D2,
   D4 or a prediction.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..dryrun import DryRun, PLACEHOLDER_FLOWS, PayloadSource
from ..payloads import (
    MAX_PATCH_BYTES,
    MAX_PATCH_TOTAL_BYTES,
    NON_READ_FLOWS,
    NO_DEBRIS_ID,
    NOOP_OVERLAY_ID,
    BiasPrimitive,
    CollapseMandateBody,
    DigestedDebrisBody,
    EmergentCrystalBody,
    FrameBody,
    LatentRolloutBody,
    MatOverlay,
    PredictionErrorBody,
    StateDeltaBody,
    StructuralMotifBody,
    noop_debris,
    noop_mandate,
    noop_motif,
    noop_overlay,
)
from ..provenance import Envelope
from .organ import (
    DigestResult,
    Observation,
    OverlayRecord,
    PhysicsOrgan,
    RatchetResult,
    StructuralMotif,
)

__all__ = [
    "AcceptanceResult",
    "PhysicsRun",
    "PhysicsSource",
    "RunTick",
]

#: The pre-registered single-stream replay window (Lead fork (a)).
DEFAULT_FIT_WINDOW = 16
#: Ticks after the first spike before the single-stream adoption may fire.
DEFAULT_ADOPTION_AFTER_SPIKE = 4


def _vector(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


class PhysicsSource:
    """Content-only source hook over a :class:`PhysicsOrgan` (Q2).

    The coverage ledger is plain data (``fid -> derived | synthetic |
    placeholder``) read by tests; the derivation dispatch is a fixed method
    map, never a branch on the ledger.  ``derive_body`` raises
    ``AssertionError`` for any fid outside the dispatch, including the
    placeholder fids -- the executor's fixed ``PLACEHOLDER_FLOWS`` dispatch
    is what keeps those out of this call.
    """

    mode = "physics"

    def __init__(self, organ: PhysicsOrgan) -> None:
        self.organ = organ
        self.coverage: dict[int, str] = {
            fid: ("placeholder" if fid in PLACEHOLDER_FLOWS else "derived")
            for fid in sorted(NON_READ_FLOWS)
        }
        #: Fixed derivation dispatch; not consulted through ``coverage``.
        self._derivers = {
            3: self._frame,
            6: self._prediction_error,
            10: self._rollout,
            12: self._state_delta,
            17: self._mandate,
            18: self._crystal,
            19: self._debris,
            21: self._motif,
            22: self._gate,
            23: self._birth_token,
            24: self._revalidation_token,
        }
        self._frame_body: FrameBody | None = None
        self._observation: Observation | None = None
        self._digest: DigestResult | None = None
        self._ratchet: RatchetResult | None = None
        self._pending_births: list[OverlayRecord] = []
        self._pending_revalidations: list[OverlayRecord] = []

    # ------------------------------------------------------------- context
    def begin_tick(
        self,
        frame: np.ndarray,
        observation: Observation,
        digest: DigestResult,
        ratchet: RatchetResult,
        *,
        adoption: OverlayRecord | None = None,
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
        self._ratchet = ratchet
        live = self.organ.live_overlay_ids()
        for record in ratchet.births:
            if record.overlay_id in live:
                self._pending_births.append(record)
        for record in ratchet.revalidations:
            if record.overlay_id in live:
                self._pending_revalidations.append(record)
        if adoption is not None and adoption.overlay_id in live:
            self._pending_births.append(adoption)
        # Drop events for overlays that were evicted before they could be
        # committed (no organ-side retention, Q1-A).
        self._pending_births = [
            record for record in self._pending_births if record.overlay_id in live
        ]
        self._pending_revalidations = [
            record
            for record in self._pending_revalidations
            if record.overlay_id in live
        ]

    def _require_context(self) -> tuple[Observation, DigestResult]:
        if self._observation is None or self._digest is None or self._frame_body is None:
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
        return deriver(tick, board)

    def _frame(self, tick: int, board: Any) -> FrameBody:
        assert self._frame_body is not None
        return self._frame_body

    def _prediction_error(self, tick: int, board: Any) -> PredictionErrorBody:
        observation, _ = self._require_context()
        return PredictionErrorBody(
            surprise=float(observation.surprise),
            residual=_vector(observation.residual),
        )

    def _rollout(self, tick: int, board: Any) -> LatentRolloutBody:
        observation, _ = self._require_context()
        return LatentRolloutBody(
            latents=tuple(_vector(row) for row in np.asarray(observation.rollout))
        )

    def _state_delta(self, tick: int, board: Any) -> StateDeltaBody:
        observation, _ = self._require_context()
        return StateDeltaBody(
            delta=_vector(observation.latent_delta),
            surprise=float(observation.surprise),
        )

    def _mandate(self, tick: int, board: Any) -> CollapseMandateBody:
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

    def _crystal(self, tick: int, board: Any) -> EmergentCrystalBody:
        observation, _ = self._require_context()
        return EmergentCrystalBody(
            residual=_vector(observation.residual),
            surprise=float(observation.surprise),
            tick=int(observation.tick),
        )

    def _debris(self, tick: int, board: Any) -> DigestedDebrisBody:
        _, digest = self._require_context()
        body = digest.created
        if body is None and digest.assigned_debris_id is not None:
            for candidate in digest.pool:
                if candidate.debris_id == digest.assigned_debris_id:
                    body = candidate
                    break
        if body is None:
            return noop_debris()
        return DigestedDebrisBody(
            debris_id=body.debris_id,
            centroid=_vector(body.centroid),
            count=int(body.count),
            spread=float(body.spread),
            first_tick=int(body.first_tick),
            last_tick=int(body.last_tick),
            peak_surprise=float(body.peak_surprise),
        )

    def _motif(self, tick: int, board: Any) -> StructuralMotifBody:
        _, digest = self._require_context()
        motif = digest.motif
        if motif is None:
            return noop_motif()
        return StructuralMotifBody(
            motif_id=motif.motif_id,
            debris_id=motif.debris_id,
            centroid=_vector(motif.centroid),
            count=int(motif.count),
            spread=float(motif.spread),
            persistence=int(motif.persistence),
            drift=float(motif.drift),
            first_tick=int(motif.first_tick),
            last_tick=int(motif.last_tick),
        )

    def _gate(self, tick: int, board: Any) -> BiasPrimitive:
        """The real #22 ratchet gate's derived output (Q4).

        The organ's adapted-vs-base evidence decides: an active overlay whose
        cumulative adapted surprise is strictly below base reports
        ``ratchet-gate-passed`` with ``clamp(1 - adapted/base, 0, 1)``, and
        the store sediments that body into D4.  No verdict, no active
        overlay, non-finite values, ``base <= 0`` or ``adapted >= base``
        report the zero-confidence ``ratchet-gate-blocked`` sentinel, which
        the executor absorbs as a consumed no-op.  Round 9a: the registered
        trace really passes on 11 active ticks (measured), so both branches
        occur without any test double.
        """
        passed, confidence = self.organ.ratchet_gate_passed()
        if passed:
            return BiasPrimitive("ratchet-gate-passed", float(confidence))
        return BiasPrimitive("ratchet-gate-blocked", 0.0)

    def _birth_token(self, tick: int, board: Any) -> MatOverlay:
        """The #23 birth token, with the attached patch pre-validated.

        Q1 atomicity: ``_absorb_writes`` commits the birth before the
        one-way ``on_commit`` attaches the patch, so a patch that violates
        the outer byte ceilings must be refused here, in the derivation
        pass -- before any arrival, commit, or charge.  The Round 9a
        evidence bound is applied earlier, at fit time: only a
        capacity-admissible low-rank patch can reach this hook.  The outer
        ``MatBoard`` byte ceilings remain storage bounds and still fail
        closed here (a synthetic oversized patch is refused before any
        commit).
        """
        self._pending_births = [
            record
            for record in self._pending_births
            if record.overlay_id in self.organ.live_overlay_ids()
        ]
        if not self._pending_births:
            return noop_overlay()
        record = self._pending_births[0]
        patch = self.organ.overlay_table.adapter(record.overlay_id)
        try:
            if patch is None:
                raise AssertionError(
                    f"overlay {record.overlay_id!r} has no predictor patch; "
                    "refusing to derive a birth that cannot be attached "
                    "(Q1-A)"
                )
            resident = record.overlay_id in board.resident_ids()
            nbytes = int(getattr(patch, "nbytes"))
            if not resident and nbytes > MAX_PATCH_BYTES:
                raise AssertionError(
                    f"overlay {record.overlay_id!r} patch is {nbytes} bytes, "
                    f"over the {MAX_PATCH_BYTES}-byte per-patch ceiling; the "
                    "#23 birth is refused in the derivation pass (Q1 atomicity)"
                )
            projected = board.patch_bytes() + (0 if resident else nbytes)
            if projected > MAX_PATCH_TOTAL_BYTES:
                raise AssertionError(
                    f"attaching overlay {record.overlay_id!r} would make the "
                    f"patch table {projected} bytes, over the "
                    f"{MAX_PATCH_TOTAL_BYTES}-byte ceiling; the #23 birth is "
                    "refused in the derivation pass (Q1 atomicity)"
                )
        except AssertionError:
            # The overlay was born organ-side before this tick; drop it so a
            # refused birth leaves no organ-side retention either.
            self.organ.prune_overlays(frozenset())
            raise
        self._pending_births.pop(0)
        return MatOverlay(*record.token())

    def _revalidation_token(self, tick: int, board: Any) -> MatOverlay:
        self._pending_revalidations = [
            record
            for record in self._pending_revalidations
            if record.overlay_id in self.organ.live_overlay_ids()
        ]
        if not self._pending_revalidations:
            return noop_overlay()
        record = self._pending_revalidations.pop(0)
        return MatOverlay(*record.token())

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

        On #23/#24 the store has already committed the record; the source
        attaches the matching predictor patch (or asserts if the organ has
        none).  On the post-eviction notifications (#19/#22) it prunes the
        organ/store to the resident set so no overlay-side retention
        survives a D2 eviction.
        """
        if fid in (23, 24) and type(envelope.body) is MatOverlay and \
                envelope.body.overlay_id != NOOP_OVERLAY_ID:
            overlay_id = envelope.body.overlay_id
            patch = self.organ.overlay_table.adapter(overlay_id)
            if patch is None:
                raise AssertionError(
                    f"D2 record {overlay_id!r} has no predictor patch; "
                    "refusing to run unpatched (Q1-A)"
                )
            board.attach_patch(overlay_id, patch)
        if fid in (19, 22):
            self.prune(board, tick=int(envelope.provenance.tick_born))

    def prune(self, board: Any, *, tick: int) -> None:
        """Align organ and store: keep only D2-resident overlays (Q1-A)."""
        live = self.organ.live_overlay_ids()
        for overlay_id in sorted(board.resident_ids() - live):
            board.evict_record(overlay_id, tick)
        self.organ.prune_overlays(board.resident_ids())
        board.assert_patch_bounds(require_complete=True)


# ---------------------------------------------------------------------------
# Single-stream executor run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunTick:
    """One executor tick of the single-stream run (measured state).

    Round 9a adds the pre-adoption validation trace: ``proposal_tick`` /
    ``proposal_rank`` / ``proposal_param_count`` / ``proposal_capacity``
    describe the candidate, ``validation_n`` / ``candidate_mean`` /
    ``base_mean`` its held-out evidence (``nan`` before the first validation
    tick), and ``decision`` / ``rejection`` the resolved outcome (``None``
    while the window is open).
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
    proposal_tick: int | None
    proposal_rank: int
    proposal_param_count: int
    proposal_capacity: int
    validation_n: int
    validation_ticks: tuple[int, ...]
    candidate_mean: float | None
    base_mean: float | None
    decision: str | None
    rejection: str | None


@dataclass(frozen=True)
class AcceptanceResult:
    """The single-stream acceptance record; every number is measured."""

    ticks: tuple[RunTick, ...]
    summary: dict[str, Any]

    def rows(self, phase: str) -> tuple[RunTick, ...]:
        return tuple(row for row in self.ticks if row.phase == phase)


class PhysicsRun:
    """Drive one physical stream through the ledger executor.

    ``PhysicsRun`` owns E1 (through the organ), the frozen-encoder model,
    and the executor.  Every tick: step the world, observe/digest/ratchet,
    optionally adopt the pre-registered single-stream overlay, bind the
    source context, run one ``DryRun.tick()`` (derivation -> arrivals ->
    write absorption), prune the organ/store, and record the state.
    """

    def __init__(
        self,
        organ: PhysicsOrgan | None = None,
        *,
        seed: int = 0,
        single_stream_adoption: bool = True,
        fit_window: int = DEFAULT_FIT_WINDOW,
        adoption_after_spike: int = DEFAULT_ADOPTION_AFTER_SPIKE,
    ) -> None:
        self.organ = organ if organ is not None else PhysicsOrgan(
            store_frames=True, fit_streams={0}
        )
        if self.organ.fit_streams is None:
            self.organ.fit_streams = {0}
        self.organ.store_frames = True  # the source needs the #3 frame body
        # A runner always starts from a fresh P6: a supplied organ may carry
        # overlay records, pending decisions or gate evidence from a previous
        # experiment, and reusing them would make the acceptance unrepeatable.
        self.organ.reset_p6_state()
        self.source = PhysicsSource(self.organ)
        self.dry = DryRun(source=self.source, seed_board=False)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.single_stream_adoption = bool(single_stream_adoption)
        self.fit_window = int(fit_window)
        self.adoption_after_spike = int(adoption_after_spike)
        self.ticks: list[RunTick] = []
        self.started = False
        self._spike_at: int | None = None
        self._fit_attempted = False
        self._flip_at: int | None = None
        self._flip_back_at: int | None = None

    # -------------------------------------------------------------- helpers
    @property
    def board(self) -> Any:
        return self.dry.board

    def reset(self, seed: int | None = None) -> Observation:
        """Reset E1 and the executor run state (fresh D2/D3/D4/patches/P6)."""
        self.seed = self.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._flip_at = None
        self._flip_back_at = None
        # Clear P6 (pending decisions, cooldown, stale run, gate evidence,
        # overlay records/patches, fit history), then rebuild the
        # source/executor so the store and patch table start clean.
        self.organ.reset_p6_state()
        self.source = PhysicsSource(self.organ)
        self.dry = DryRun(source=self.source, seed_board=False)
        observation = self.organ.reset(seed=self.seed)
        self.ticks.clear()
        self.started = True
        self._spike_at = None
        self._fit_attempted = False
        return observation

    def flip_gravity(self) -> int | dict[int, int]:
        return self.organ.flip_gravity()

    def _should_propose(self, observation: Observation) -> bool:
        """Round 9a trigger: propose at the same point that used to adopt.

        Proposal, not adoption: the candidate is fitted at this tick and
        held for the pre-adoption validation window (Round 9a's binding
        condition).  Trigger gating remains a 9b concern and is untouched.
        """
        if not self.single_stream_adoption or self._fit_attempted:
            return False
        if self.organ.overlay_table.active():
            return False
        if np.isfinite(observation.surprise) and \
                observation.surprise > self.organ.thresholds.spike_threshold:
            if self._spike_at is None:
                self._spike_at = int(observation.tick)
        if self._spike_at is None:
            return False
        if observation.tick - self._spike_at < self.adoption_after_spike:
            return False
        if observation.tick < self.fit_window:
            return False
        return True

    def tick(self, action: int | None = None) -> RunTick:
        """One executor tick: E1 -> organ -> source -> DryRun."""
        if not self.started:
            self.reset()
        if action is None:
            action = int(self.rng.integers(0, self.organ.base_model.config.action_dim))
        observation = self.organ.step(int(action))
        if observation.frame is None:  # pragma: no cover - run forces store_frames
            raise AssertionError("PhysicsRun needs observation.frame (store_frames=True)")
        digest = self.organ.digest(observation)
        stales_before = len(self.organ.overlay_table.stales)
        ratchet = self.organ.ratchet(digest)
        rejections_before = len(self.organ.rejections)
        if self._should_propose(observation):
            self._fit_attempted = True
            self.organ.propose_overlay_from_buffer(
                int(observation.tick), fit_window=self.fit_window
            )
        # The decision is taken in the same executor tick that completes the
        # validation window (frame ticks 17-18 for the proposal at 16), and
        # after the ratchet has finalised the previous tick, so a just-born
        # overlay cannot be revalidated before its #23 has even committed.
        adoption: OverlayRecord | None = self.organ.resolve_proposal(
            int(observation.tick)
        )
        state = self.organ.proposal_state()
        evictions_before = len(self.board.evictions)
        self.source.begin_tick(
            observation.frame,
            observation,
            digest,
            ratchet,
            adoption=adoption,
        )
        self.dry.tick()
        self.source.prune(self.board, tick=self.dry.tick_no)
        evicted = tuple(
            record[0] for record in self.board.evictions[evictions_before:]
        )
        staled = None
        if len(self.organ.overlay_table.stales) > stales_before:
            staled = self.organ.overlay_table.stales[-1].overlay_id
        if adoption is not None:
            decision = "birth"
        elif len(self.organ.rejections) > rejections_before:
            decision = "interference"
        else:
            decision = None
        row = RunTick(
            tick=self.dry.tick_no,
            frame_tick=int(observation.tick),
            phase=self._phase(int(observation.tick)),
            action=int(action),
            surprise=float(observation.surprise),
            base_surprise=float(observation.base_surprise),
            overlay_id=observation.overlay_id,
            adopted=None if adoption is None else adoption.overlay_id,
            staled=staled,
            evicted=evicted,
            residents=tuple(rid for rid, _epoch in self.board.residents()),
            patch_bytes=self.board.patch_bytes(),
            d4_size=len(self.board.d4),
            proposal_tick=state["proposal_tick"],
            proposal_rank=int(state["rank"]),
            proposal_param_count=int(state["param_count"]),
            proposal_capacity=int(state["capacity"]),
            validation_n=int(state["validation_n"]),
            validation_ticks=tuple(state["validation_ticks"]),
            candidate_mean=(
                None if state["candidate_mean"] is None else float(state["candidate_mean"])
            ),
            base_mean=(
                None if state["base_mean"] is None else float(state["base_mean"])
            ),
            decision=decision,
            rejection=None if decision != "interference" else "interference",
        )
        self.ticks.append(row)
        return row

    def _phase(self, frame_tick: int) -> str:
        if self._flip_at is not None and frame_tick >= self._flip_at:
            if self._flip_back_at is not None and frame_tick >= self._flip_back_at:
                return "recovered"
            return "flipped"
        return "normal"

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
        """The tick loop without the reset (``run`` / ``acceptance`` share it)."""
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

    # ---------------------------------------------------------- acceptance
    def acceptance(
        self,
        *,
        seed: int = 7,
        warmup: int = 8,
        post_flip: int = 20,
        recovery: int = 8,
    ) -> AcceptanceResult:
        """Single-stream phase-5 acceptance: the measured state trace.

        Normal (``warmup``) -> flip -> ``post_flip`` -> flip back ->
        ``recovery``.  Round 9a's policy is used unchanged: the candidate is
        proposed at the same trigger point, validated on the next two
        held-out ticks, and born only if the validation passes.  The summary
        reports the measured adapted/base ratios exactly as measured, both
        over every active tick (the pre-registered definition) and over the
        active *flipped* ticks (the failure window proper).
        """
        self.reset()
        d4_before = pickle.dumps(self.board.d4, protocol=4)
        encoder_before = [
            parameter.detach().clone()
            for parameter in self.organ.base_model.encoder_parameters()
        ]
        rows = self._drive(
            warmup + post_flip + recovery,
            flip_at=warmup,
            flip_back_at=warmup + post_flip,
        )
        d4_after = pickle.dumps(self.board.d4, protocol=4)
        encoder_after = [
            parameter.detach().clone()
            for parameter in self.organ.base_model.encoder_parameters()
        ]
        encoder_ok = all(
            torch.equal(before, after)
            for before, after in zip(encoder_before, encoder_after)
        )
        normal = [row for row in rows if row.phase == "normal"]
        flipped = [row for row in rows if row.phase == "flipped"]
        active = [row for row in rows if row.overlay_id is not None]
        normal_mean = float(np.mean([row.base_surprise for row in normal])) if normal else float("nan")
        spike = max((row.base_surprise for row in flipped), default=float("nan"))
        adopted = next((row for row in rows if row.adopted is not None), None)
        staled = next((row for row in rows if row.staled is not None), None)
        evicted = next((row for row in rows if row.evicted), None)
        adoption_frame = None if adopted is None else adopted.frame_tick
        pre_adoption = [
            row
            for row in flipped
            if adoption_frame is None or row.frame_tick < adoption_frame
        ]
        pre_adoption_base = (
            float(np.mean([row.base_surprise for row in pre_adoption]))
            if pre_adoption
            else float("nan")
        )
        stale_record = (
            None
            if staled is None
            else self.organ.overlay_table.stales[-1]
        )
        eviction_record = (
            None if evicted is None else self.board.evictions[-1]
        )
        if active:
            adapted_mean = float(np.mean([row.surprise for row in active]))
            base_mean = float(np.mean([row.base_surprise for row in active]))
            forward_ratio = adapted_mean / base_mean if base_mean else float("nan")
        else:
            adapted_mean = float("nan")
            base_mean = float("nan")
            forward_ratio = float("nan")
        # Secondary view: the active ticks that fall inside the *flipped*
        # window proper.  The pre-registered `forward_ratio` above spans every
        # active tick (including the flip-back ticks on which a passed patch
        # goes stale); this one isolates the failure window the user's
        # criterion names.  Both are reported, neither replaces the other.
        active_flipped = [row for row in active if row.phase == "flipped"]
        if active_flipped:
            flip_adapted_mean = float(np.mean([row.surprise for row in active_flipped]))
            flip_base_mean = float(np.mean([row.base_surprise for row in active_flipped]))
            forward_ratio_flip_only = (
                flip_adapted_mean / flip_base_mean if flip_base_mean else float("nan")
            )
        else:
            flip_adapted_mean = float("nan")
            flip_base_mean = float("nan")
            forward_ratio_flip_only = float("nan")
        fit = self.organ.fit_history[-1] if self.organ.fit_history else None
        decision = self.organ.decisions[-1] if self.organ.decisions else None
        proposal = self.organ.proposals[-1] if self.organ.proposals else None
        validation_ticks = () if decision is None else decision.validation_ticks
        validation_candidate = (
            None if decision is None else float(decision.candidate_mean)
        )
        validation_base = (
            None if decision is None else float(decision.base_mean)
        )
        validation_ratio = (
            validation_candidate / validation_base
            if validation_candidate is not None
            and validation_base is not None
            and np.isfinite(validation_candidate)
            and np.isfinite(validation_base)
            and validation_base
            else None
        )
        summary = {
            "streams": 1,
            "normal_mean": normal_mean,
            "spike_peak": float(spike),
            "spike_over_normal": (
                float(spike) / normal_mean if normal_mean else float("nan")
            ),
            "proposal_tick": None if proposal is None else proposal.proposal_tick,
            "proposal_rank": None if proposal is None else proposal.candidate.rank,
            "proposal_param_count": (
                None if proposal is None else proposal.candidate.param_count
            ),
            "proposal_capacity": (
                None if proposal is None else proposal.budget.max_params
            ),
            "proposal_budget_n": (
                None if proposal is None else proposal.budget.n_transitions
            ),
            "validation_ticks": validation_ticks,
            "validation_n": 0 if decision is None else decision.n,
            "validation_candidate_mean": validation_candidate,
            "validation_base_mean": validation_base,
            "validation_ratio": validation_ratio,
            "decision": None if decision is None else decision.outcome,
            "rejection_tick": (
                None
                if decision is None or decision.outcome == "birth"
                else decision.tick
            ),
            "rejection_reason": (
                None
                if decision is None or decision.outcome == "birth"
                else decision.outcome
            ),
            "adoption_tick": adoption_frame,
            "stale_tick": (
                None if stale_record is None else stale_record.stale_tick
            ),
            "stale_detected_frame": None if staled is None else staled.frame_tick,
            "eviction_tick": (
                None if eviction_record is None else eviction_record[2]
            ),
            "eviction_frame": None if evicted is None else evicted.frame_tick,
            "active_ticks": len(active),
            "pre_adoption_flipped_base_mean": pre_adoption_base,
            "adapted_mean": adapted_mean,
            "base_mean": base_mean,
            "forward_ratio": forward_ratio,
            "active_flip_ticks": len(active_flipped),
            "active_flip_adapted_mean": flip_adapted_mean,
            "active_flip_base_mean": flip_base_mean,
            "forward_ratio_flip_only": forward_ratio_flip_only,
            "d4_sediments": len(self.board.d4),
            "direction": (
                "drop" if np.isfinite(forward_ratio) and forward_ratio < 1.0
                else ("no-drop" if np.isfinite(forward_ratio) else "no-overlay")
            ),
            "d4_bytes_before": len(d4_before),
            "d4_bytes_after": len(d4_after),
            "d4_byte_identical": d4_before == d4_after,
            "residents_end": tuple(rid for rid, _epoch in self.board.residents()),
            "patches_end": tuple(overlay_id for overlay_id, _patch in self.board.patches()),
            "patch_bytes_end": self.board.patch_bytes(),
            "max_patch_bytes_seen": max(
                (row.patch_bytes for row in rows), default=0
            ),
            "debris_bodies": len(self.dry.debris),
            "debris_bytes": self.dry.debris.byte_size(),
            "zero_confidence_writes": self.dry.zero_confidence_absorbed.get(22, 0),
            "accepted_22": self.dry.accepted.get(22, 0),
            "consumed_22": self.dry.consumed.get(22, 0),
            "refused_22": self.dry.refused.get(22, 0),
            "sink_depth_22": len(self.dry.mailboxes[22]) if hasattr(self.dry, "mailboxes") else 0,
            "encoder_bit_identical": bool(encoder_ok),
            "fit_patch": None if fit is None else fit.patch,
            "fit_param_count": None if fit is None else fit.param_count,
            "fit_nbytes": None if fit is None else fit.nbytes,
            "fit_loss_before": None if fit is None else fit.loss_before,
            "fit_loss_after": None if fit is None else fit.loss_after,
            "fit_rank": None if fit is None else fit.rank,
            "fit_budget_n": None if fit is None else fit.budget_n,
            "fit_budget_capacity": None if fit is None else fit.budget_capacity,
            "fit_reason": None if fit is None else fit.reason,
            "refusals": tuple(self.organ.refusals),
            "rejections": tuple(
                {
                    "tick": record.tick,
                    "reason": record.reason,
                    "candidate_mean": record.candidate_mean,
                    "base_mean": record.base_mean,
                    "n": record.n,
                }
                for record in self.organ.rejections
            ),
        }
        return AcceptanceResult(ticks=rows, summary=summary)
