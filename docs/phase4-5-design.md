# Phase 4–5 integration design: physics organ + executor wiring

Status: **phase 4–5a and 4–5b built and measured** (task-6 is the executor
wiring; §14 is its integration record). This memo is the decision record for
the physics-epoch P5/P6 work: the organ API, the derived operating point, the
payload-type plan, the D2/D3/D4 accounting rules, the content-only source
hook, and the end-to-end acceptance trace. The user adjudicated the four open
forks as **option A** (Q1–Q4, §7); the Lead then fixed the sub-forks (patch
scope, byte ceilings, replay policy, gate naming, placeholders) — §11.

Code and artifacts:

| Item | Path | Status |
| --- | --- | --- |
| Physics organ (observation, digest, overlay) | `amateur/physics/organ.py` | built, 19 tests |
| Organ tests | `tests/test_physics_organ.py` | 19 tests, OK (~15 s) |
| Phase-3 model + stats | `.work/forge/checkpoint.pt`, `.work/forge/stats.json` | measured artifact |
| Probe scripts / raw traces | `.work/organ/` | scratch only, not committed |

The organ is deliberately **not** wired into the executor in phase 4–5a.
The wiring plan (§8) is written against the phase-2 `DryRun` in the working
tree (`amateur/dryrun.py`, `_derive` at the current lines 245–303,
`_absorb_writes` at 350–384), which the Lead's task-6 owns.

---

## 1. What the organ is

`PhysicsOrgan` owns E1 (a `PhysicsEnv` per stream) and the phase-3
micro-LeWM (CNN encoder + GRU predictor, 521,000 parameters, encoder frozen).
It exposes P5/P6 as pure, testable state machines:

| Method | Input | Output |
| --- | --- | --- |
| `reset(seed, gravity_sign, stream=0)` | E1 reset | `Observation` (reset frame; no prediction) |
| `step(action, stream=0)` | E1 step | `Observation` |
| `observe(frame, action, stream=0, ingest=True)` | external frame | `Observation` |
| `digest(observation)` | `Observation` | `DigestResult` (P5) |
| `ratchet(digest)` | `DigestResult` | `RatchetResult` (P6, finalised tick) |
| `flush()` | — | finalise pending ticks |
| `flip_gravity(stream=0)` | — | new sign |

`Observation` carries the real bodies the executor needs: `latent`,
`latent_delta` (#12 state delta), `surprise` (mean squared latent error,
#6/#12), `residual` (the signed 64-d error vector, #18 candidate),
`rollout` (4-step resolved latent trajectory, #10), `window` (3-frame causal
latent window, the longest context the phase-3 predictor was measured with),
`overlay_id`, and `base_surprise` (the unpatched predictor's surprise, used
for staleness). Online surprise is asserted equal to the batched
`MicroLeWM.predict` reference (`tests/test_physics_organ.py::TestObservation`).

`RatchetResult` carries the D2-shaped triples
`records: tuple[(epoch, overlay_id, status)]`, plus birth/revalidation/
stale/eviction lists and `d4_writes`. The adapter itself lives in the
`OverlayTable` side map keyed by `overlay_id`; it is never part of a payload.

## 2. Threshold derivation (measured, not synthetic)

Source: `.work/forge/stats.json` (phase 3; 384 held-out transitions per
regime, 16 episodes, flip at step 8). Reproduced this session to the artifact
values: normal mean `0.04938631` (artifact `0.04938632`), post-flip mean
`0.28597303` (artifact `0.28597304`).

| Constant | Derivation | Value |
| --- | --- | --- |
| `debris_threshold` | measured normal q99 | **0.4130417** |
| `spike_threshold` | measured normal q99 (same tail) | **0.4130417** |
| `merge_radius` | `sqrt(latent_dim * post_flip_q95)` = `sqrt(64 * 0.7105834)` | **6.7436888** |
| `normal_baseline` | measured normal mean | **0.0493863** |
| `post_flip_q95` (reference) | measured | **0.7105834** |
| `immediate4_mean` (reference) | measured | **0.3467064** |

The merge radius uses the identity `surprise = ||e||² / D` (`D = 64`), so a
residual at the measured post-flip q95 has norm `sqrt(64 * 0.7106) = 6.7437`.
The measured residual geometry supports this:

| Measurement (post-flip window, 16 episodes) | Value |
| --- | --- |
| residual norm, mean | 3.9322 |
| residual norm, q95 | 6.7436 |
| RMS distance of residuals to their centroid | 3.9190 |
| distance-to-centroid, q95 | 6.7898 |
| flipped centroid norm | 0.4715 (the residuals point in varied directions) |
| normal residual norm, q95 / q99 | 3.3197 / 5.1393 |
| normal-anomaly (q99) count, whole episode | 4 / 384 = 1.04 % |
| normal-anomaly count, phase-matched window (transitions 8–23) | 0 / 256 |
| flipped residuals above q99, phase-matched window | **57 / 256 = 22.27 %** |
| distance between normal-anomaly centroid and flipped centroid | 3.0514 |

So one cluster of radius 6.7437 covers the flipped regime: the measured
q95 spread is 6.7898, i.e. 99.3 % of the derived radius. The q99 gate has a
measured 1.04 % whole-episode false-positive rate on normal data and admits
22.27 % of phase-matched post-flip residuals.

**Design constants** (not from the phase-3 statistics; listed so they can be
challenged rather than hidden): D3 capacity 8 bodies; D2/overlay capacity 4
(mirrors `MAT_CAPACITY`); motif minimum support 4 residuals (the phase-3
"first 4 post-flip" window was the smallest measured separating window,
10.03× phase-matched normal); motif persistence 10 ticks (**probe-derived**:
forward-generalising overlays appeared at ≥ 10 post-flip ticks of banked
support, see §6); one isolated missed tick is tolerated
(`MOTIF_STREAK_GAP = 2`); stale after 2 consecutive ticks where the patch
stops improving; eviction 3 ticks after staleness; 4-tick cooldown after a
stale overlay before a new birth.

## 3. P5 digest: bounded pool, collapse mandate

`ResidualDigester` is an online nearest-centroid clusterer:

1. A residual enters only when `surprise > debris_threshold` (the measured
   normal q99). Everything else is resolved and never becomes debris.
2. It joins the nearest centroid when the Euclidean distance is
   ≤ `merge_radius`; the centroid is a running mean and the spread an
   online RMS.
3. An unmatched residual with a full pool triggers one collapse: merge the
   closest body pair when the pair is within `2 * merge_radius`, otherwise
   evict the least-supported body (count tie-break, then newest last-tick);
   then admit the new body. A `CollapseMandate` is emitted naming the freed
   body — the #17 "Collapse Mandate & Freed Slots" body.
4. After admission, centroid pairs closer than `merge_radius / 2` are merged
   (compaction), so the pool cannot accumulate near-duplicates.
5. A body becomes a `StructuralMotif` when `count ≥ 4`, its support streak
   is ≥ 10 ticks, `spread ≤ merge_radius`, and the centroid drift from the
   streak origin is ≤ `merge_radius`. A support gap of more than one tick
   resets the streak and re-origins the stability window.

Measured mechanics (test `test_pool_capacity_and_ceiling_mandate`, 13
deliberately distinct clusters built from a real flipped residual):
pool caps at 8, 5 mandates, 5 evictions, 0 merges; every reported pool has
`spread ≥ 0` and centroid shape `(64,)`. Measured separation (test
`test_q99_gate_separates_flipped_from_normal`): normal 0/256, flipped
57/256 in the phase-matched window; whole-episode normal 4/384.

## 4. P6 overlay: patch, lifecycle, and accounting

`PredictorOverlay` is a predictor-only additive patch:

| Field | Meaning |
| --- | --- |
| `deltas: dict[str, Tensor]` | parameter-name → additive delta, **predictor only** |
| `param_count`, `nbytes` | measured: head patch 34,240 params / 136,960 bytes |
| `changes_gru` | true iff the patch touches GRU parameters |
| `fit_streams`, `fit_transitions`, `loss_before/after`, `patch` | provenance of the fit |

Fitting is a short AdamW fit on a **deep copy** of the base model over the
organ's real cached latents/actions, restricted to the fit streams and the
transitions from the motif's streak origin to the decision tick. The base
encoder is never in the optimizer; `encoder_bit_identical()` compares every
base-encoder parameter with a construction-time `torch.equal` snapshot after
every fit (tested).

`OverlayTable` (capacity 4) enforces the lifecycle:

| Transition | Effect | Measured API |
| --- | --- | --- |
| birth | record `(epoch=1, id, "active")`, attach adapter | `RatchetResult.births` |
| revalidate | `epoch += 1`, `last_validated_tick = tick`, status active | `RatchetResult.revalidations` |
| stale | status `"stale"`, `resolve()` returns `None` (stops affecting prediction) | `RatchetResult.stales` |
| evict | record **and** adapter removed atomically | `RatchetResult.evictions` |

The acceptance run's overlay reached epoch 11 (10 revalidations) before it
went stale, and the table was empty after eviction. When an overlay is
applied, `working_model` is a fresh deep copy of the base model plus the
patch; when it is removed, the working model is a fresh copy of the base.
A head patch does not touch the GRU, so the online hidden state remains
valid and no replay is needed; a full-patch overlay replays each stream's
causal cache to reseed the hidden state.

## 5. Overlay representation options (adjudicated Q1 = A)

Three shapes were considered for how the patch reaches the predictor:

| Option | Where the adapter lives | Consequences |
| --- | --- | --- |
| **A. Store-held patch, token payload** (chosen) | D2 store holds `overlay_id → patch`; #23/#24/#25 payloads stay `MatOverlay(epoch, id, status)` | payload schema/whitelist/Lean untouched; record and patch must evict together (atomic); capacity accounting must count patch params/bytes, not only records; a restart/kill loses the in-memory patch map unless it is rebuilt |
| B. Payload-carried patch | each #8/#23/#24/#25 envelope carries the adapter | 137 KB per head token × every tick/read; provenance carrier cost; revalidation re-sends the same bytes; the D2 schema and Lean mirror change |
| C. Executor-owned side table, D2 untouched | same as A but the table is not part of D2 | record/patch atomicity is not enforceable by the store; a D2 eviction can orphan a patch unless a second GC path exists |

**Decision (Q1 A):** D2 holds the patch keyed by `overlay_id`; the payload
stays `MatOverlay`. Binding conditions: (1) record and patch evict atomically
in the same tick, with no orphan patches and no organ-side retention after
eviction; (2) D2 ceiling accounting covers substance — capacity counts patch
params/bytes, not only records (Python-asserted, marked *tested-not-proven*
because the Lean bound covers records only); (3) the patch body is derived
from digest output with provenance, predictor-side only, encoder
bit-identical.

Measured patch sizes for the accounting decision:

| Patch scope | Parameters | Bytes (fp32) | Held-out ratio | `changes_gru` |
| --- | --- | --- | --- | --- |
| head + norm (default) | 34,240 | 136,960 | **0.690×** | no |
| full predictor | 210,920 | 843,680 | **0.668×** | yes |

The head patch is the recommended default: 6.2× smaller, applies/clears
exactly without hidden replay, and trades away only 2.2 percentage points
of the measured held-out improvement. The `MAX_PATCH_BYTES` ceiling in task-6
must be at least the head size; a full patch fits under a 1 MB ceiling but
consumes 4× the head budget.

## 6. Measured overlay evidence (honest, including the null result)

**Held-out episode fit (the strong result).** Fit the head patch on episodes
0–15 of phase-3 seed 31 (transitions 8–23), evaluate on 16 unseen seed-77
episodes, same transition window:

| Metric | Value |
| --- | --- |
| base post-flip surprise | 0.2616 |
| adapted post-flip surprise | **0.1805** |
| ratio | **0.690×** (31.0 % reduction) |
| in-fit loss | 0.2860 → 0.1075 |
| fit corpus | 16 streams, 256 transitions |
| head patch | 34,240 params / 136,960 bytes |
| encoder bit-identical | yes (`torch.equal` on every parameter) |

Per-transition adapted/base ratios over transitions 8–23: 0.99, 0.74, 0.58,
0.63, 0.59, 0.72, 0.93, 0.87, 0.78, 0.79, 0.62, 0.47, 0.64, 0.64, 0.61,
0.88. No transition is worse than 0.99× for the head patch. The full patch
(0.668× overall) has two transitions worse than baseline (1.31× and 1.02×),
another reason to prefer the head patch.

**Single-stream fit (the null result).** Fit the head patch on one episode's
post-flip window (transitions 9–24), evaluate the following 16 transitions:

| Metric | Value |
| --- | --- |
| in-fit 9–24 | 0.3145 → 0.0124 (memorised) |
| forward 25–40, base | 0.2728 |
| forward 25–40, adapted | **0.4287** |
| ratio | **1.571× (worse)** |

Earlier probe windows gave 0.99–1.18×. **A short predictor-only fit on a
single stream does not transfer forward.** The executor is a single stream;
this is a first-class honesty constraint for task-6: the end-to-end
acceptance must report the measured single-stream result, and if it is not a
drop, say so. The minimum viable adaptation is to fit from the organ's
recent real buffer once it contains enough diverse transitions (the bank
result required 16 streams × 10+ post-flip ticks, i.e. a replay corpus, not
a single episode's window). Do not fabricate a single-stream drop.

## 7. Binding adjudications (Q1–Q4, all option A)

Recorded verbatim from the Lead's adjudication message; they constrain
task-6 (executor wiring).

**Q1 (A) — D2 holds the patch keyed by `overlay_id`; payload stays
`MatOverlay`.** (1) record and patch evict atomically, same tick — no orphan
patches, no organ-side retention after eviction; (2) D2 ceiling accounting
covers substance: capacity bound counts patch params/bytes, not only records
— Python-asserted, marked *tested-not-proven* (Lean bounds records only);
(3) patch body derived from digest output with provenance, predictor-side
only, encoder bit-identical.

**Q2 (A) — content-only source hook.** (1) No silent mint fallback: if the
source cannot supply a derived body, fail loudly (assert/raise). Synthetic
mode is explicit (constructor sentinel/flag) for plumbing tests only; a
coverage ledger records mode per flow. (2) The source cannot influence
schedule, stage order, or channel semantics — ops table and Lean untouched.
(3) `_absorb_writes` notifications are one-way, post-commit; the organ may
apply committed store state internally; no new events, no back-channel
writes. Source failures are assertions, not charges.

**Q3 (A) — end-to-end through the executor with the fixed cadence.**
Assertions are state-based (overlay installed → surprise drops → stale
flagged → evicted → base-core contents unchanged), never emission counts.
Document the cadence quarantine: uniform cadence is plumbing; lifecycle
semantics are organ/store-owned until the trigger-gating round.

**Q4 (A) — #22 zero-confidence sentinel.** The #22 sentinel is the real
ratchet gate's derived output (confidence 0 because the gate did not pass),
carrying provenance; zero-confidence writes are absorbed as no-ops (event
consumed, identity `B: 1 = 1 + 0 + 0`, state delta empty, nothing retained);
D4 retained contents must be byte-identical across the flip scenario
(assert on retained contents, not on a field check).

### Q2 compatibility check against the existing `_derive` read rules

No conflict, but three points must be implemented exactly:

1. The phase-2 rule is that **store reads inherit the stored record's
   provenance** and are built by the executor (#7 D4, #8/#25 D2; #20/#26 D3
   once populated). Q2 agrees: the source supplies only non-read bodies, so
   the read rules are untouched. The `PayloadSource` contract must be called
   for non-read flows only, after the executor has placed the read bodies.
2. **#22 is a write** (non-read), so it moves to the source; under Q4 it is
   the zero-confidence sentinel, not the current synthetic rotating
   `BiasPrimitive`. `_absorb_writes` must treat confidence 0 as a consumed
   no-op, so D4's retained contents stay byte-identical.
3. **#23/#24 are writes** and move to the source as `MatOverlay` tokens; the
   patch reaches D2 through a one-way post-commit notification or a board
   patch table (Q1). This is not a read-rule change, but it does mean
   `_derive` no longer constructs `cry-{tick}` or the head-revalidation
   token itself.

One ambiguity to confirm with the Lead: the current `_derive` builds a body
for **every** flow (placeholders included), while Q2's "fail loudly if the
source cannot supply a body" applies to the physics source. The plumbing
`SyntheticSource` sentinel must therefore still cover every non-read flow
explicitly (a coverage ledger), otherwise default plumbing tests would raise.

## 8. Payload-type plan (real flows)

All new bodies are frozen, explicitly typed, and separate from `Envelope`
provenance. Exact fields (the task-6 schema):

| # | Flow (source → target) | Proposed body | Fields | Producer / gate |
| --- | --- | --- | --- | --- |
| 3 | E1 → P1 perturbation | frames body (existing/planned) | frame array (uint8 96×96×3) | E1 boundary; task-6 |
| 6 | P1 → P2 prediction error | `PredictionError` | `surprise: float`, `residual: tuple[float, ...]` (64) | organ `Observation`; every tick |
| 10 | P3 → P1 resolved trajectory | `LatentRollout` | `latents: tuple[tuple[float, ...], ...]` (4×64) | organ `Observation.rollout`; one-tick-delayed cycle |
| 12 | P3 → P4 state deltas & surprise | `StateDelta` | `delta: tuple[float, ...]` (64), `surprise: float` | organ `Observation`; every tick |
| 16 | P4 → P5 cost vs error | `CostError` | `cost: float`, `error: float` | P4 not yet implemented; **stays placeholder** (see honesty note 11) |
| 17 | P5 → P4 collapse mandate | `CollapseMandate` | `reason: str`, `pressure: int`, `capacity: int`, `active_bodies: int`, `freed_body: str \| None`, `tick: int` | organ `DigestResult.mandate`; trigger-gated |
| 18 | P3 → P5 emergent crystal | `EmergentCrystal` | `residual: tuple[float, ...]` (64), `surprise: float`, `tick: int` | organ `Observation`; every ingested tick |
| 19 | P5 → D3 digested debris | `DigestedDebris` | `debris_id: str`, `centroid: tuple[float, ...]`, `count: int`, `spread: float`, `first_tick: int`, `last_tick: int`, `peak_surprise: float` | organ `DigestResult.created/assigned`; D3 capacity 8 |
| 21 | P5 → P6 structural motif | `StructuralMotif` | `motif_id: str`, `debris_id: str`, `centroid: tuple[float, ...]`, `count: int`, `spread: float`, `persistence: int`, `drift: float`, `first_tick: int`, `last_tick: int` | organ `DigestResult.motif`; trigger-gated |
| 2 | P1 → E2 crystallized output | `CrystallizedOutput` (decision) | `tick: int`, `overlay_id: str \| None`, `motif_id: str \| None` | proposed; only on birth/revalidation; **decision needed** |
| 22 | P6 → D4 sedimented invariant | `BiasPrimitive` (unchanged) | `bias_primitive: str`, `confidence: float` | Q4: confidence-0 sentinel when the gate fails; confidence-0 absorbed as no-op |
| 23 | P6 → D2 surviving crystal | `MatOverlay` (unchanged) | `(epoch, overlay_id, status)` | organ birth record |
| 24 | P6 → D2 ratcheted overlay | `MatOverlay` (unchanged) | `(epoch, overlay_id, status)` | organ revalidation record |
| 25 | D2 → P6 stale overlays (read) | `MatOverlay \| None` (unchanged) | as stored | executor-derived read |
| 7 | D4 → P2 base ops (read) | `BiasPrimitive \| None` (unchanged) | as stored | executor-derived read |
| 8 | D2 → P2 mat overlays (read) | `MatOverlay \| None` (unchanged) | as stored | executor-derived read |

Explicitly deferred (still placeholder, and P5 does **not** consume them in
this slice): #11 D1 context read, #13 D1 write, #14 D1 slot read, #15 D1
evicted clusters, #16 P4 metrics, #20 D3 consumed scaffolds, #26 D3
avoidance trails. The organ-derived P5 path uses #18 (and #19 for the D3
pool side) only; P5's *declared* inputs #15/#16/#20 remain synthetic, which
is a gate limitation, not a solved claim (honesty note 11).

D3 (`DebrisPool`) in task-6: capacity 8 bodies **plus** a byte ceiling
(the exact ceiling to be chosen from the measured body size; a 64-d float64
centroid is 512 B, so 8 bodies ≤ ~4 KB plus bookkeeping — the ceiling is
therefore not the binding constraint, but it must still be asserted).

## 9. Executor wiring plan (`DryRun._derive`, current phase-2 lines)

The phase-2 integration in the working tree is the base. Line numbers are
from the current file and will move; use the `put(...)` call as the anchor.

| Current site | Phase-4/5 body | Notes |
| --- | --- | --- |
| `put(1, None, ...)`, `put(3, None, ...)` (lines 266–267) | unchanged (exogenous) | E1 boundary |
| `put(8, MatOverlay(epoch, head_id, "stale"), head_provenance)` (278) | unchanged read | executor-derived |
| `put(25, MatOverlay(epoch, head_id, "stale"), head_provenance)` (279) | unchanged read | feeds revalidation |
| `put(23, MatOverlay(tick, f"cry-{tick}", "active"), Provenance(23, tick))` (280) | **`MatOverlay(*record.token())` from the organ's birth event**; provenance born at the emitting flow | no minting; trigger-gated |
| `put(24, MatOverlay(tick, head_id, "active"), Provenance(24, tick))` (282) | **`MatOverlay(*record.token())` from the revalidation event** | epoch advances |
| `put(7, latest / None, provenance)` (286–290) | unchanged read | D4 empty until a real sediment exists |
| `put(22, BiasPrimitive(...), Provenance(22, tick))` (291–293) | **`BiasPrimitive("ratchet-gate-passed", conf)` or `BiasPrimitive("ratchet-gate-blocked", 0.0)`** as the real evidence gate returns it, with the ratchet gate's provenance | Q4: confidence-0 is absorbed as a no-op; a passing body sediments D4 |
| `put(11,14,15,20,26, None, ...)` (296–297) | #20/#26 become D3 reads **only when the D3 pool is real**; #11/#14/#15 stay placeholder (D1 unimplemented) | honest gate |
| `put(2,4,5,6,9,10,12,13,16,17,18,19,21, None, ...)` (300–301) | #6/#10/#12/#18/#19/#21/#17 from the organ (table §8); #2 decision; #4/#5 remain E1 exhaust placeholders; #9/#13/#16 stay placeholder | source hook |

The `PayloadSource` protocol (Q2):

```python
class PayloadSource(Protocol):
    mode: str  # "physics" | "synthetic"
    coverage: Mapping[int, str]  # per-flow mode, for the coverage ledger
    def derive_body(self, fid: int, tick: int, *, board: MatBoard,
                    debris: DebrisPool) -> object: ...
    def on_commit(self, fid: int, envelope: Envelope, *, board: MatBoard,
                  debris: DebrisPool) -> None: ...
```

* `SyntheticSource` is the explicit plumbing sentinel (constructor flag); it
  preserves today's placeholder behaviour and records every flow as
  `synthetic` in the coverage ledger.
* `PhysicsSource(organ)` derives the §8 bodies; a missing derivation **raises**
  (assertion, not a channel charge).
* `derive_body` is called for non-read flows only; the executor keeps the
  store reads (#7/#8/#25, and #20/#26 from D3 once populated).
* `on_commit` is called once per committed write, post-commit, one-way; the
  organ may update its internal state (e.g. mark an overlay committed), but
  it cannot emit events or write back into the channel/budget.
* `_absorb_writes` attaches `#23/#24` patches to D2 (via `on_commit` or a
  board patch table), deposits `#19` into D3, and treats confidence-0 `#22`
  as a consumed no-op (consume the event; no `board.sediment`, no D4 state
  change, empty state delta). That keeps the identity `B: 1 = 1 + 0 + 0`.

Tick ordering (Q3): the organ must observe/digest/ratchet **before**
`_derive`, because `_derive` only reads already-finalised records; the organ
finalises tick `T-1` when the first digest of tick `T` arrives, which is
exactly the one-tick-lag the D2 policy already has ("no overlay persists more
than one tick without revalidation"). `flush()` at end of run finalises the
last tick. `_arrive` and the ops table are untouched; the source cannot
change the schedule.

Cadence quarantine (Q3, recorded): the uniform per-tick emission of
#22/#23/#24 is **plumbing**; in the physics epoch the lifecycle is
organ/store-owned and trigger-gated (a birth/revalidation only when the
digest says so). The acceptance assertions must be state-based
(installed → drop → stale → evict → D4 bytes unchanged), never emission
counts. The Lean `matTick`/`matRun` mirror stays records-only; the patch
param/byte accounting is Python-asserted and marked *tested-not-proven*.

D4 accounting (Q4): `#22` confidence-0 writes are absorbed as no-ops, so
`board.d4` retains exactly its pre-run contents. The acceptance asserts the
retained contents are byte-identical (serialize `board.d4` before/after and
compare bytes), not merely that a field is zero.

## 10. Acceptance record (phase 4–5a, measured)

Probe: 32 paired streams (24 fit + 8 held-out), three-phase rollout
(normal → flipped at action 8 → flipped back at action 28), horizon 36,
phase-3 seed 31; held-out streams are observed but never ingested. This is a
**shaped bank probe**, not the executor's single stream (see honesty note 1).

| Tick | Phase | Mean surprise | Held-out surprise | Overlay |
| --- | --- | --- | --- | --- |
| 1–8 | normal | 0.2465 → 0.0197 (cold start) | 0.2277 → 0.0212 | none |
| 9 | flipped | 0.1313 | 0.1313 | none |
| 10–13 | flipped | 0.4211, 0.3845, 0.3981, 0.4362 | 0.3280, 0.4269, 0.4540, 0.4568 | none |
| 14–19 | flipped | 0.3726 → 0.2498 | 0.4268 → 0.2242 | none |
| 20–28 | flipped | 0.1879 → 0.1935 | 0.2395 → 0.2155 | **ovr-01 active** |
| 29–30 | recovered | 0.1675, 0.2784 | 0.1431, 0.2154 | ovr-01 → stale at 30 |
| 31–36 | recovered | 0.2383 → 0.0678 | 0.1505 → 0.0408 | none (evicted at 33) |

Lifecycle events and measured summary:

| Event / metric | Measured |
| --- | --- |
| birth | tick 19, `(1, "ovr-01", "active")` |
| fit corpus | 24 streams, 240 transitions; in-fit loss 0.327 → 0.1135 |
| patch | head, 34,240 params / **136,960 bytes**, `changes_gru = False` |
| revalidations | epoch 11 by tick 30 (10 revalidations) |
| stale | tick 30, `(11, "ovr-01", "stale")` |
| evict | tick 33 (3 ticks after stale) |
| spike (base, post-flip mean) | 0.28738 vs phase-matched normal 0.03317 → **8.66×** |
| peak per-tick mean | 0.45679 → **13.8×** normal |
| adapted post-adoption mean (held-out, transitions 19–27) | **0.17232** |
| base post-adoption mean (same ticks) | 0.25170 |
| adapted / base | **0.6846×** (31.5 % reduction) |
| adapted / phase-matched normal | 5.08× (toward, not at, baseline) |
| encoder bit-identical | yes |
| D4 writes | **0** (retained contents empty) |
| trace wall time | 9.01 s (concurrent host load) |

Interpretation: the spike, digest/motif, adoption, recovery, staleness,
eviction, and D4-untouched states are all measured in one deterministic run.
The recovery is **toward** baseline (0.68× base, still 5.1× the normal
baseline), not a return to normal.

## 11. Adjudicated decisions (Q1–Q4 and the Lead's sub-forks)

Q1–Q4 are all **option A**; §7 records the binding conditions.  The Lead
fixed the sub-forks in task-6:

1. **Recovery policy (fork a).** Bounded replay of the **same stream** is
   allowed; the pre-registered default is the last 16 same-stream
   transitions, no cross-stream bank data in the executor acceptance, and
   no post-hoc selection of a fit variant.  If the default does not produce
   a drop, the acceptance records the measured numbers honestly and the
   drop test is an explicit recorded failure (not weakened).
2. **Persistence (fork b).** Out of scope this round.  The patch table is
   in-memory; the restart gap is documented.  A D2 record with no patch is
   an assertion, never a silent unpatched run.
3. **Patch scope (fork c).** HEAD patch only.  `MAX_PATCH_BYTES = 262_144`
   (admits the 136,960 B head patch, rejects the 843,680 B full patch) and
   `MAX_PATCH_TOTAL_BYTES = 1_048_576` (4 × per-patch bound); both are
   asserted.
4. **#22 gate expression (fork d).** Body
   `BiasPrimitive("ratchet-gate-passed", confidence)` when the real
   evidence gate passes, `BiasPrimitive("ratchet-gate-blocked", 0.0)`
   otherwise, provenance `(22, emit_tick)`; the body is the gate's derived
   output (the source reads the organ's stored evidence), never a literal
   in `_derive`.  The gate passes only on cumulative adapted<base evidence
   over at least `stale_patience` applied ticks; zero-confidence writes are
   absorbed as consumed no-ops.
5. **#2 (fork e).** Kept as a placeholder (`None`) this round.
6. **#15/#16/#20 (fork f).** Placeholders acceptable for this gate;
   recorded PARTIAL.  The physics digest is real over #18; #20 is not read
   from D3.
7. **D3 byte ceiling (fork g).** Production default 65,536 B; the eviction
   mechanism is tested with a tighter pool ceiling (test-only), not by
   inflating bodies.

## 12. Honesty notes (numbered, measured vs assumed)

1. **The acceptance scenario is a shaped bank probe, not the executor flow.**
   24 fit + 8 held-out paired streams from the phase-3 seeds; the digest
   ingests only fit streams. It is real physics (the env's `flip_gravity`),
   but it is not one executor stream. The executor acceptance must be
   separate and may not reproduce these numbers.
2. **The single-stream overlay does not transfer forward.** Measured on the
   executor's pre-registered default (last 16 same-stream transitions): 1.261×
   worse over the 4 active ticks; other probe windows measured 1.05–1.57×.
   The bank recovery required 16 episodes × 10+ post-flip ticks. Do not claim
   single-stream recovery from a short fit; §14 records the exact numbers.
3. **The recovery is toward, not at, baseline**: 0.1723 adapted vs 0.2517
   base vs 0.0332 phase-matched normal; still 5.08× normal.
4. **Threshold derivation is from phase 3, but not every constant is.** The
   q99 gate, merge radius, and normal baseline are derived from
   `.work/forge/stats.json`. Motif support 4, persistence 10 (probe-derived),
   the one-tick gap tolerance, stale patience 2, eviction age 3, cooldown 4,
   D3 capacity 8, and D2 capacity 4 are design choices, stated here.
5. **The head-vs-full comparison is measured on one fit/eval split.** Head
   0.690×, full 0.668× on seed-31 fit → seed-77 eval; no cross-validation.
   The full patch is worse on two individual transitions (1.31×, 1.02×).
6. **The digest separation test feeds real residuals** from the frozen
   checkpoint over the phase-matched window (normal 0/256, flipped 57/256;
   whole-episode normal 4/384). The capacity test uses a real flipped
   residual pushed along orthogonal directions to force distinct clusters;
   that is a mechanics test, not a claim about observed cluster diversity.
7. **Staleness is detected by the patch's own usefulness**, not by a
   phase-3 statistic: mean patched surprise > mean base-predictor surprise
   for two consecutive ticks. It fired at tick 30, after two recovered ticks;
   there is a one-to-two-tick detection lag.
8. **The overlay was revalidated 10 times but not refitted.** Revalidation
   advanced the epoch only; the patch is frozen at its birth fit. Refitting
   on new digest evidence is the natural next step and is not measured here.
9. **D4 zero writes now holds in the executor too (task-6).** The physics
   source derives the real Q4 gate's output every tick; in the registered
   acceptance the cumulative evidence is harmful (0.26568 vs 0.14746, then
   0.34720 vs 0.23619), so the gate emits `ratchet-gate-blocked` with
   confidence 0 on all 36 ticks; `_absorb_writes` consumes each as a no-op,
   so the executor's retained D4 contents are byte-identical across the
   flip scenario.  The synthetic plumbing source still sediments its
   synthetic bodies (dry-run only).
10. **The phase-2 read rules are unchanged.** Store reads (#7/#8/#25) stay
    executor-derived; the source hook supplies non-read flows only. There is
    no conflict, but the plumbing `SyntheticSource` must keep an explicit
    per-flow coverage ledger so "fail loudly" does not break dry-run tests.
11. **P5's declared inputs #15/#16/#20 are still placeholders.** The digest
    is real over #18 only; P4's cost/error and D1's slot/eviction state are
    not implemented. This is a partial gate, recorded.
12. **The patch map is in-memory.** No persistence, no restart test. Q1's
    atomic eviction is asserted in-process; the restart story is open
    (fork 2).
13. **Timings are contended-host wall time.** Acceptance probe 9.01 s;
    focused suite 19 tests ~14.8 s; full suite 238 tests OK in 55.2 s with
    `LEAN_BIN=$PWD/.work/lean4/bin/lean` while sibling edits were present.
    These are not single-tenant benchmarks.
14. **The checkpoint's eval means reproduce the artifact** (normal
    0.04938631 vs 0.04938632; post-flip 0.28597303 vs 0.28597304) but are
    float-order approximate, as recorded in the phase-3 doc.
15. **No tolerance was loosened to make a test pass.** The held-out test
    asserts `< 0.85` against a measured 0.690; the acceptance asserts
    `< 0.85` against a measured 0.6846; the separation asserts `≥ 0.15`
    against a measured 0.2227.

## 13. Reproduction

```sh
# focused organ tests (~15 s, 8 threads)
python3 -m unittest tests.test_physics_organ -v

# focused executor-wiring tests (~3 s; one recorded expected failure)
python3 -m unittest tests.test_physics_run -v

# full suite (Lean included); baseline command
LEAN_BIN=$PWD/.work/lean4/bin/lean python3 -m unittest discover -s tests -t .

# the probe numbers in this memo (scratch, not committed)
python3 .work/organ/final_measurements.py
```

The phase-3 artifact is regenerated with
`python3 -m amateur.physics.train --out-dir .work/forge --seed 0`
(CPU-only, ~4 min). The organ requires the artifact for all
measurement-bearing tests; if it is absent those tests skip with an explicit
reason rather than silently passing.

---

## 14. Phase 4–5b integration record (executor wiring)

Status: **built and measured**.  `amateur/payloads.py` gained the typed
body schema, the D3 `DebrisPool`, and the D2 patch table/accounting;
`amateur/dryrun.py` gained the content-only `PayloadSource` hook with the
explicit `SyntheticSource` sentinel and the sentinel-aware derivation and
write absorption; `amateur/physics/run.py` adds `PhysicsSource` and
`PhysicsRun`; `tests/test_physics_run.py` pins the contract (32 tests, one
recorded failure); `tests/test_payloads.py` was updated for the schema and
the new accounting.  `formal/DryRun.lean` is untouched and
`python3 tools/gen_lean.py` is byte-identical (checked with `diff -q`).

### 14.1 Payload types (final)

| # | Flow | Body class (frozen, hashable) | Fields |
| --- | --- | --- | --- |
| 3 | E1 → P1 | `FrameBody` | `frame: bytes`, `height`, `width`, `tick` |
| 6 | P1 → P2 | `PredictionErrorBody` | `surprise`, `residual: tuple[64]` |
| 10 | P3 → P1 | `LatentRolloutBody` | `latents: tuple[tuple[64], ...]` (4 rows) |
| 12 | P3 → P4 | `StateDeltaBody` | `delta: tuple[64]`, `surprise` |
| 16 | P4 → P5 | `CostErrorBody` | `cost`, `error` (placeholder producer) |
| 17 | P5 → P4 | `CollapseMandateBody` | `reason`, `pressure`, `capacity`, `active_bodies`, `freed_body`, `tick` |
| 18 | P3 → P5 | `EmergentCrystalBody` | `residual: tuple[64]`, `surprise`, `tick` |
| 19 | P5 → D3 | `DigestedDebrisBody` | `debris_id`, `centroid: tuple[64]`, `count`, `spread`, `first_tick`, `last_tick`, `peak_surprise` |
| 21 | P5 → P6 | `StructuralMotifBody` | `motif_id`, `debris_id`, `centroid: tuple[64]`, `count`, `spread`, `persistence`, `drift`, `first_tick`, `last_tick` |
| 22 | P6 → D4 | `BiasPrimitive` (unchanged) | `bias_primitive`, `confidence` |
| 23/24 | P6 → D2 | `MatOverlay` (unchanged) | `(epoch, overlay_id, status)` |
| 7/8/25 | store reads | unchanged (`BiasPrimitive|None`, `MatOverlay|None`) | executor-derived |

`None` remains allowed on every non-store channel.  Fixed-cadence no-ops
are exact-type bodies, not `None`: `noop_overlay()` (`ovr-noop`, stale,
epoch 0), `noop_debris()` (`debris-none`, count 0), `noop_motif()`
(`motif-none`, persistence 0), `noop_mandate()` (`none`, pressure 0).  The
executor/store absorbs each as a no-op; the physics source never emits a
`None` for a flow in its `derived` ledger.

### 14.2 Content-only source hook (Q2)

`PayloadSource` protocol: `mode`, `coverage` (plain data), `derive_body(fid,
tick, *, board, debris)`, `on_commit(fid, envelope, *, board, debris)`.

* `_derive` keeps the store reads executor-derived (#7 D4, #8/#25 D2,
  #11/#14/#15/#20/#26 as placeholders per Lead decision f) and delegates the
  fixed `SOURCE_FLOWS` set to the source.  `PLACEHOLDER_FLOWS` is a fixed
  executor constant (`{1,2,4,5,9,13,16}`), never read from `coverage`.
* `SyntheticSource` is the explicit plumbing sentinel (`mode="synthetic"`,
  every non-read flow `synthetic`); `DryRun()` defaults to it, preserving
  the historical synthetic dry run.  `PhysicsSource` raises
  `AssertionError` for any fid outside its fixed derivation dispatch,
  including placeholders — there is no fallback.
* Coverage ledger (`PhysicsSource`, non-read flows only):
  `derived = {3,6,10,12,17,18,19,21,22,23,24}`,
  `placeholder = {1,2,4,5,9,13,16}`;
  `SyntheticSource`: `synthetic =` every non-read flow.
* `on_commit` is one-way and post-commit: on #23/#24 it attaches the patch
  (`board.attach_patch`); on #19/#22 it prunes the organ/store to the
  resident set.  It creates no events and cannot touch the schedule, stage
  order, or channel semantics.
* Q1 atomicity is enforced **before** the commit: `_birth_token` computes
  the patch that `on_commit` will attach and asserts the per-patch and
  projected-total byte bounds in the derivation pass.  A birth whose patch
  cannot be attached is an `AssertionError` in `_derive`, so no record is
  committed and no charge is taken; a refused birth also prunes the
  organ-side overlay that was born for it, so neither side retains state.
* A source failure is an assertion in `_derive` before any `_arrive`; the
  test `test_failure_raises_before_any_arrival` proves attempted/accepted/
  refused/missed/consumed all stay 0.

### 14.3 D2 patch and byte accounting (Q1-A)

| Item | Value |
| --- | --- |
| patch location | `MatBoard._patches: overlay_id -> patch` (in-memory) |
| payload | `MatOverlay(epoch, overlay_id, status)`, unchanged |
| head patch (used) | 34,240 params / **136,960 B** |
| full patch (rejected) | 210,920 params / **843,680 B** |
| `MAX_PATCH_BYTES` | **262,144** (256 KiB, per patch) |
| `MAX_PATCH_TOTAL_BYTES` | **1,048,576** (4 × per-patch) |
| accounting | `patch_bytes()`, `patch_params()`, `assert_patch_bounds(require_complete=True)` |
| eviction | `evict_if_needed` / `enforce_patch_ceiling` / `evict_record` remove record and patch together |

The byte bound is Python-asserted and marked *tested-not-proven*: the Lean
`matTick`/`matRun` mirror bounds records only.  In physics mode the board is
constructed unseeded (`seed_board=False`) so every resident is an
organ-owned overlay with a patch; a resident without a patch asserts.

Atomicity (A): the pending #23 patch is validated in the derivation pass
(`_birth_token`) against `MAX_PATCH_BYTES` and the projected table total
before the executor commits anything.  The supported `patch="full"` mode
produces a 843,680 B patch, so its birth is refused with an
`AssertionError` before any arrival: the failing tick leaves
`residents() == ()`, `patches() == ()`, `patch_bytes() == 0`,
`board.d4 == []`, no organ adapters/records, and attempted/accepted/
refused/consumed/missed unchanged.  The head patch (136,960 B) still
commits and attaches.

### 14.4 D3 debris pool

`DebrisPool(capacity=8, max_bytes=65_536)`; `deposit` stores a real
`DigestedDebrisBody` with the #19 envelope's provenance and evicts the
oldest entries until both bounds hold; `noop_debris()` (id `debris-none`) is
rejected.  The acceptance run deposited **7 bodies / 3,934 bytes**.  D3
reads (#20/#26) remain placeholders per Lead decision f, so the pool is
written but not consumed this round.

### 14.5 Q4 #22 gate and D4

`PhysicsSource._gate` calls `organ.ratchet_gate_passed()`; the gate is real
and evidence-based, not a constant.  The organ records one adapted-vs-base
surprise measurement per finalized tick **in which the active overlay was
actually applied to the prediction** (the adoption tick's observation
predates the patch and carries no verdict).  The production recorder
(`_accrue_gate_evidence`) publishes cumulative means only after
`stale_patience` (= 2) applied ticks, the same patience the staleness
decision uses, so one transient tick cannot sediment an invariant.

The gate returns `(True, clamp(1 - adapted/base, 0, 1))` only when a
verdict exists, an overlay is active, and the cumulative adapted mean is
strictly below the base mean.  No verdict, no active overlay, non-finite
values, `base <= 0`, or `adapted >= base` all return `(False, 0.0)`, and the
source emits `BiasPrimitive("ratchet-gate-blocked", 0.0)` with provenance
`(22, tick)`.  `_absorb_writes` consumes a confidence-0 #22 without state
mutation; the measured per-tick identity is
`accepted 36 = consumed 36 + sunk 0 + depth 0`.

Registered acceptance (measured, no test double): the first applied tick
(17) is transiently beneficial (0.10421 vs 0.13162), the second (18) is
harmful (0.42714 vs 0.16329).  At the first publishable point the
cumulative evidence is **(0.26568 adapted, 0.14746 base): harmful**, and
after the third applied tick it is **(0.34720, 0.23619): harmful**; the
overlay is then marked stale and deactivated.  The gate therefore stays
blocked on every tick: **36 confidence-0 no-ops, D4 remains empty**, and
`ratchet_gate_passed() == (False, 0.0)` after the run with no evidence
retained.  The passing branch is exercised only by injected evidence
(`_record_gate_evidence(0.2, 0.4)` + one real tick → `ratchet-gate-passed`
with confidence 0.5, D4 sediments the identical body); this slice never
produces a successful adaptation, and that is disclosed, not glossed
(honesty note 26).

Retention is proven **non-vacuously** by
`TestZeroConfidenceGate.test_retained_invariant_survives_the_scenario`:
tick 1 is seeded with a documented gate test-double so a real
`BiasPrimitive` passes through the normal commit/sediment path (at tick 1
there is no active overlay, so the real gate cannot pass); the remaining 35
ticks of the registered scenario (normal → flip at 8 → flip back at 28 →
recover) are driven directly with `tick()`/`flip_gravity()`.  The retained
serialized contents are **121 B before and 121 B after, byte identical**,
with `len(board.d4) == 1` and the retained body is the *same object* that
was committed (`assertIs`), not a reconstruction.  All 35 blocked ticks
are absorbed as claimed no-ops (`zero_confidence_absorbed[22] == 35`;
`accepted 36 = consumed 36 + 0 + 0`, `refused 0`, mailbox depth 0 every
tick), so no second sediment occurs and no event is skipped.

The registered acceptance cannot make this check itself: it calls `reset()`,
which rebuilds the board, so its before/after comparison is empty-vs-empty.
That test is kept and renamed `test_empty_d4_stays_empty_byte_identical`
(5 B empty pickle either side) — it now states only what it measures.

### 14.6 Single-stream executor acceptance (measured)

Trace: 1 stream, seed 7, warmup 8, flip, 20 post-flip ticks, flip back, 8
recovery ticks (36 executor ticks).  Every number below is measured.

| Item | Measured |
| --- | --- |
| normal-phase mean surprise | 0.1012 |
| spike peak | 1.9496 = **19.27×** normal |
| pre-adoption flipped base mean | 0.4366 |
| overlay installed | frame tick **16** |
| patch | head, 34,240 params / **136,960 B** |
| fit window | last 16 same-stream transitions (frames 0–16; transitions 0–15) |
| fit corpus | 1 stream, 16 transitions |
| in-sample fit loss | 0.2764 → 0.0086 |
| active ticks / post-adoption window | **4** |
| adapted mean (post-adoption) | **0.3264** |
| base mean (same ticks) | **0.2588** |
| forward ratio | **1.2614 (no drop)** |
| stale flagged | finalized tick **19**, detected at frame **20** |
| eviction | executor tick **23** |
| patch/record at end | residents `()`, patches `()`, patch bytes 0 |
| D4 | serialized contents byte-identical; 0 retained writes; 36 confidence-0 no-ops |
| D3 | 7 bodies / 3,934 B (≤ 8 / 65,536) |
| encoder | bit-identical (`torch.equal` after the fit) |

**Honest outcome:** the pre-registered single-stream adaptation does **not**
produce a forward surprise drop; it is 1.261× worse over the four active
ticks, and the organ flags it stale after two harmful ticks and evicts it.
`test_single_stream_surprise_drops` is an explicit `expectedFailure` (the
assertion is kept strict, not weakened); the suite is green with that one
recorded expected failure.  The positive measured result remains the
bank/held-out fit: 0.690× on unseen episodes (§6).

Variants actually run (reported, none selected): pre-registered default
(spike+4, last 16, post 20) 1.2614×; spike+12, last 16, post 28 → 1.0969×;
spike+16 (window entirely post-spike), post 28 → 1.4120×; earlier phase-4-5a
single-stream probes 1.05×–1.57×.  All are no-drop.

### 14.7 Cadence quarantine (Q3)

The uniform per-tick derivation/emission is plumbing and is unchanged
(`flow_ops` and the Lean op table are byte-identical).  In physics mode the
D2/D3/D4 **lifecycle** is organ/store-owned: births and revalidations are
triggered by the digest/ratchet, and the fixed cadence is filled by the
exact-type sentinels above; the mat record transitions are still the
`matTick` policy, with the byte ceiling and organ-driven eviction as extra
Python-side store accounting.  The three-phase acceptance asserts **state**,
never emission counts.

### 14.8 Tests pinning the new invariants

| Invariant | Test |
| --- | --- |
| default source is synthetic; PhysicsRun's source is physics | `test_physics_run.TestSourceSelection` |
| coverage covers every non-read flow | `test_physics_run.TestCoverageLedger.test_ledger_covers_every_non_read_flow` |
| `derived` ⇒ non-None exact-type body | `...test_derived_claim_holds_for_every_derived_fid` |
| `placeholder` ⇒ None in the executor envelope | `...test_placeholder_envelopes_are_none` |
| absent fid asserts, no fallback | `...test_absent_fid_asserts_and_never_falls_back` |
| source failure is not a charge | `test_physics_run.TestSourceFailureIsNotACharge` |
| #22 gate derived output / no-op / identity B | `test_physics_run.TestZeroConfidenceGate` (9 tests) |
| gate blocked: no evidence, no active overlay, harmful evidence | `...test_gate_blocked_without_evidence`, `...test_gate_blocked_with_evidence_but_no_active_overlay`, `...test_gate_blocked_on_harmful_evidence` |
| gate passing branch (injected evidence, conf 0.5, D4 sediment) | `...test_gate_passes_on_published_beneficial_evidence` |
| acceptance gate blocked by measured harm (no double) | `...test_acceptance_gate_blocked_by_measured_harm` |
| D4 false-positive guard only (empty-vs-empty) | `...test_empty_d4_stays_empty_byte_identical` |
| D4 non-vacuous retention (121 B, one body, same object, 35 no-ops) | `...test_retained_invariant_survives_the_scenario` |
| patch attach/eviction, no orphans, byte ceilings | `test_physics_run.TestPatchAccounting` (5) |
| full-patch birth refused pre-commit, clean state; head still attaches | `test_physics_run.TestFullPatchModeFailsClosed` (2) |
| P6 reset: two accepted traces identical | `test_physics_run.TestResetRepeatability` (2) |
| MatBoard guard rails: ghost revalidate, duplicate birth, exact evict | `test_payloads.TestMatBoardGuardRails` (4) |
| record+patch atomic eviction (table ceiling, capacity, organ) | `test_payloads.TestMatBoardPatches` (6) |
| D3 capacity/byte-ceiling eviction | `test_payloads.TestDebrisPool` (4) |
| state lifecycle adopt → stale → evict | `test_physics_run.TestSingleStreamAcceptance.test_state_lifecycle_adopt_stale_evict` |
| single-stream drop (recorded failure) | `...test_single_stream_surprise_drops` (`expectedFailure`) |
| encoder bit-identity | `...test_encoder_bit_identical_after_fit` |

### 14.9 Honesty notes for the wiring record

16. **The executor acceptance does not reproduce the bank improvement.** It
    is 1.261× worse on the pre-registered single-stream replay window; the
    drop test is a recorded expected failure.  No variant was selected to
    make it pass; all measured variants are listed in §14.6.
17. **The acceptance trace's "stale" and "eviction" are the organ's
    lifecycle events, detected at the executor one tick later than the
    organ's finalized tick** (stale record tick 19, detected frame 20;
    eviction at executor tick 23).  This is the D2 one-tick rule, not a
    fabricated schedule.
18. **The board is unseeded in physics mode** (`seed_board=False`) so the
    organ's first birth becomes the D2 head and revalidation pairing is
    meaningful.  The seeded dry-run board is unchanged; this is a
    physics-mode initialization, not a change to `matTick` transitions.
19. **The fixed-cadence sentinels are real bodies, not `None`.**  They keep
    every channel live and are absorbed as no-ops; a future trigger-gating
    round may remove them, which would be a different schedule and a
    different Lean law (the cadence quarantine).
20. **`on_commit` mutates the store** (attach patch / evict non-resident
    records) only through the one-way post-commit path; it cannot create
    events.  This is the Q1-A mechanism for atomic eviction and no
    organ-side retention, not a back-channel.
21. **Patch persistence is out of scope.**  The table is in-memory; a D2
    record without a patch asserts.  A restart that reuses D2 state would
    fail loudly rather than run unpatched.
22. **#15/#16/#20 remain placeholders** (Lead decision f); the executor
    derives them as `None`.  P5's declared inputs are therefore only
    partially real; the digest is real over #18 and the D3 write #19.  The
    gate is PARTIAL, recorded.
23. **The registered acceptance's D4 before/after check is vacuous by
    construction.**  `acceptance()` calls `reset()`, which rebuilds the
    board, so its "retained contents" snapshot is `pickle.dumps([])` on both
    sides (5 B).  That test is kept but renamed
    `test_empty_d4_stays_empty_byte_identical` so it claims only that no
    stray writes appear; the Q4 retention claim rests on
    `test_retained_invariant_survives_the_scenario`, which seeds one real
    sediment at tick 1 with a **documented gate test-double** (no active
    overlay exists that early, so the real gate cannot pass there), drives
    the same 36-tick scenario without `reset()`, and measures **121 B
    retained before and after, byte identical, one body, same object, 35
    absorbed no-ops** (measured, not inferred).
24. **Timings:** full suite with `LEAN_BIN=$PWD/.work/lean4/bin/lean
    python3 -m unittest discover -s tests -t .` = **289 tests, OK, 42.1 s,
    1 expected failure** (the honest single-stream drop gate; the new
    full-patch, reset and gate-evidence tests add scenario runs);
    `tools/gen_lean.py` output is byte-identical to `formal/DryRun.lean`;
    the Lean artifacts are untouched.
25. **The supported `patch="full"` mode cannot attach, and it now fails
    closed.**  A full predictor patch is 843,680 B > the 262,144 B
    per-patch ceiling; `_birth_token` asserts the per-patch and projected
    total bounds in the derivation pass, before the executor commits the
    #23 birth.  Measured on the failing tick: `residents() == ()`,
    `patches() == ()`, `patch_bytes() == 0`, `board.d4 == []`, no organ
    records/adapters, and attempted/accepted/refused/consumed/missed
    unchanged (390/390/0/343/2 before and after the refused tick).  The
    head patch still commits and attaches (34,240 params / 136,960 B).
    The refusal also prunes the organ-side overlay born for the refused
    birth, so the organ keeps no retention.  The Lean mirror is unaffected
    (records only).
26. **The #22 gate is real evidence but only the injected branch was
    observed passing.**  The gate publishes cumulative adapted/base means
    after `stale_patience` (= 2) applied ticks, matching the organ's own
    staleness patience, so a single transient tick cannot sediment an
    invariant.  In the registered acceptance tick 17 is transiently
    beneficial (0.10421 vs 0.13162) but the first publishable cumulative
    evidence is harmful (0.26568 vs 0.14746), then (0.34720 vs 0.23619),
    and the overlay goes stale; the gate is blocked on all 36 ticks.  The
    passing branch is exercised by `_record_gate_evidence(0.2, 0.4)` + one
    real tick (`ratchet-gate-passed`, confidence 0.5, D4 sediments the
    identical body) and by nothing else in this slice: no adaptation was
    successful, and this disclosure is binding for the gate record.
27. **P6 reset is explicit and the registered trace is repeatable.**  A
    second `acceptance()` on the same runner used to inherit pending
    decisions, cooldown, stale run and the overlay table (adoption 16, stale
    None, eviction None, ratio 1.167411, 20 active ticks, resident
    `ovr-02`).  `PhysicsOrgan.reset_p6_state()` now clears all P6 decision
    state, the overlay table (records, adapters, event lists), gate
    evidence and fit history, and restores the working model; both
    `PhysicsRun.__init__` and `PhysicsRun.reset` call it.  Measured: two
    consecutive `acceptance()` calls give identical traces and summaries
    (adoption 16, stale 19, eviction 23, 4 active ticks, forward ratio
    **1.2613833722904746**, residents/patches empty, 36 no-ops); two
    consecutive `run(36, flip_at=8, flip_back_at=28)` calls are
    row-identical.
28. **MatBoard guard rails fail without mutating state (D).**  `revalidate`
    validates the head pairing before popping (a ghost revalidation raises
    `RuntimeError` and leaves the resident and its patch intact); `birth`
    rejects a duplicate `overlay_id` with `ValueError` (no two records can
    share a patch); `evict_record` removes exactly one record and its patch
    and is a no-op for a missing id.  Pinned by
    `test_payloads.TestMatBoardGuardRails` (4 tests).
