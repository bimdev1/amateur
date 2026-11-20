# Testing methodology for the v0.10 loops architecture

The test suite is layered: each layer states what the previous layer cannot.
Run everything with:

```sh
python3 -m unittest discover -s tests -t .
```

## Layer 1 — Structural graph laws (`test_ledger_sync.py`, `test_thermodynamics.py`, `test_recurrence.py`)

The ledger (`amateur/ledger.py`) is the single source of truth; these tests
prove the *wiring* has the shape the design claims, independent of any
execution:

- ledger / Mermaid / markdown table agreement;
- per-node discipline: entities only touch P1, stores only emit `(read)`
  flows, writes never marked reads, P3 quarantined from D2/D4/E1/E2;
- graph-theoretic facts computed by independent algorithms (Tarjan SCC,
  canonical simple-cycle census):
  - the interior {P1..P6, D1..D4} is one strongly connected component;
  - the port condensation is exactly {E1 -> core, E2 -> core,
    core -> E1, core -> E2};
  - the census of interior simple cycles is exactly the seventeen
    hard-coded in `test_recurrence.py` (each verified edge-by-edge against
    flow ids — every store closes a loop, and D4 Base Core is the only
    store whose shortest loop is a five-node arc through P2 and P3);
  - boundary law: every cycle touching an entity carries fresh input
    (#1/#3) and lateral output (#2/#4/#5); the driving gradient itself
    never circulates.

## Typed payload schema (INV-EMPTINESS / INV-NO-OSSIFY by construction)

The two parametric-geometry stores carry typed payloads (`amateur/payloads.py`),
so the philosophy is enforced by the type system, not by observation:

- **D2 Multi-Layer Mat** carries `MatOverlay(epoch: int, overlay_id: str,
  status: {"active", "stale"})` — enough structure to verify the eviction
  pairing (#25 read feeds the #24 revalidation) and to prove INV-NO-OSSIFY:
  no overlay persists more than one tick without re-validation, and the
  board never exceeds `MAT_CAPACITY` (4) residents.
- **D4 Base Core** carries `BiasPrimitive(bias_primitive: str,
  confidence: float)` and *only* that. Trajectory, state-delta, and context
  types are structurally absent from the schema, so INV-EMPTINESS holds by
  construction: a trajectory cannot be written to the core because no
  trajectory type is nameable there.

The schema is an exact-type whitelist (`allowed_payload` / `require_payload`):
per flow id, only one payload type (or `None` outside geometry channels) is
accepted, and subclassing cannot sneak past it (`Sneaky(MatOverlay)` is
rejected). Flow ids derive from the ledger itself: `D2_FLOWS = {8, 18, 23,
24, 25}`, `D4_FLOWS = {7, 22}`, with the D2 writes `D2_WRITES = {18, 24}`
and `D4_WRITES = {22}` (round 10: #18 is a P3 write, so D2 is no longer
ratchet-exclusive).

**Provenance envelope (round 8, ADJUDICATED — option A2).** Every accepted
traversal carries `Envelope(Provenance(source_fid, tick_born), body)`
(`amateur/provenance.py`). The payload schema above is *not* reopened: it
stays the exact-type **body** whitelist, and the envelope is the transport
gate beside it, so `Sneaky(MatOverlay)` and a subclassed envelope are both
rejected. `source_fid` names the producing **flow** (a write names its
emitting edge; a read names the historical write that bore the stored
record; an exogenous input names #1/#3). `tick_born` is the tick the value
came into existence (0 = the `MAT_SEED` overlay). The per-channel rule is:
a store read with a non-`None` body must have `tick_born < emit_tick`
(reads precede the same tick's writes in stage order); every other
traversal requires `tick_born == emit_tick`. Bodies and provenance are
built by `DryRun._derive()` at the start of each tick from live node/store
state; `_arrive` only transports (the old `_payload_for` minter is gone).
`emissions` therefore carries envelopes. The full per-flow coverage table
is below. The honest status is two-tier: carrying and ordering are closed;
the physics source now derives real bodies for the flows it owns (Q4: #22
is the ratchet gate's output, #18/#24 are the organ's records), but the
default `DryRun` keeps `SyntheticSource` and the derivation half remains
**partial** overall — #2/#4/#5/#9/#13/#16 and all D1/D3 reads are still
placeholders.

The mat lifecycle is one closed policy — mirrored byte-for-byte by the Lean
`matTick`/`matRun` recursion and by the executor's `MatBoard`: seed one
overlay (`"ovr-0"`, epoch 0); each tick *t* append a birth (**#18**, epoch
*t*), revalidate the head at epoch *t* (its stale snapshot was just read by
#25 at the same stage, and the survivors were read by #23), then evict the
head iff the board exceeds capacity 4.
Consequences proved for all ticks in Lean: `mat_operational` (closed form
`replicate 2 (t+2) ++ replicate 2 (t+3)`), `mat_fresh` (every resident epoch
is *t* or *t−1* — nothing ossifies), `mat_bounded` (≤ 4 residents).

`test_payloads.py` covers the schema, the subclass/trajectory rejection, the
lifecycle, and 200-tick executor integration (no trajectory ever reaches D4;
#25→#24 same-tick pairing on `overlay_id`). `test_provenance.py` covers the
carrier, the exact-type interaction, the per-channel birth rule, executor
read continuity (#7 identity, #8/#23/#25 inherited provenance, seed `(23, 0)`),
and the structural anti-mint evidence (explicitly labelled evidence, not a
proof of derivation).

### Provenance coverage by flow (round 11, corrected 27-flow graph)

`synthetic` = no body or a body with no state dependency; `derived` = the
body is the stored record or a function of live node/store state at
emission.  The Round-11 schema pins **one exact body type per flow** (writes
have no `None`; reads allow `None` when the store has nothing to hand over,
so the A2 ordering rule is untouched — design §15.11 / contract §1.1).
`source_fid` is the producing-flow attribution carried in the envelope.

| # | Flow | Pinned body | Physics mode | Dry-run synthetic |
| --- | --- | --- | --- | --- |
| 1 | Gradients (E2→P1) | — | placeholder | synthetic |
| 2 | Crystallized Output (P1→E2) | — | placeholder (sink) | synthetic |
| 3 | Perturbations (E1→P1) | `FrameBody` | **derived** (frame carrier) | `None` |
| 4 | Active Probes (P1→E1) | — | placeholder (sink) | synthetic |
| 5 | Entropy Exhaust (P1→E1) | — | placeholder (sink) | synthetic |
| 6 | Prediction Error (P1→P2) | `PredictionErrorBody` | **derived** | synthetic |
| 7 | Base Ops (D4→P2) | `BiasPrimitive` \| `None` | **derived** (stored, identity) | derived |
| 8 | Mat Overlays (D2→P2) | `MatOverlay` \| `None` | **derived** (overlay head) | derived |
| 9 | Routed Dynamics (P2→P3) | — | placeholder | synthetic |
| 10 | Resolved Trajectory (P3→P1) | `LatentRolloutBody` | **derived** | synthetic |
| 11 | Active Context (D1→P3) | `None` | placeholder (D1 frozen) | placeholder |
| 12 | State Deltas & Surprise (P3→P4) | `StateDeltaBody` | **derived** | synthetic |
| 13 | Board Ops (P4→D1) | `BoardOpBody` \| `None` | **not yet derived** — transport only: `BoardStore` and `Regulator` exist and are tested, but `DryRun` owns no D1 instance, `#14` is hardcoded `None`, `#13` is in `PLACEHOLDER_FLOWS`, and `_absorb_writes` has no `#13` handler (round-12a audit §2.4) | placeholder |
| 14 | Slot State (D1→P4) | `SlotStateBody` \| `None` | placeholder (D1 frozen) | placeholder |
| 15 | Evicted Clusters (D1→P5) | `None` | placeholder (D1 frozen) | placeholder |
| 16 | Cost vs. Error (P3→P5) | `CostErrorBody` \| `None` | **derived** (new: `error = #12.surprise`, `cost = ‖#12.delta‖₂`) | synthetic |
| 17 | Collapse Mandate (P5→P4) | `CollapseMandateBody` | **derived** | synthetic |
| 18 | Emergent Crystals (P3→D2) | `EmergentCrystalBody` (**only**) | **derived** (nursery birth) | synthetic no-op |
| 19 | Digested Debris (P5→D3) | `DigestedDebrisBody` (**only**) | **derived** (tagged cohorts) | synthetic no-op |
| 20 | Consumed Scaffolds (D3→P4) | `ScaffoldsBody` \| `None` | **derived** (scaffold cohorts) | None |
| 21 | Structural Motifs (D3→P6) | `StructuralMotifBody` \| `None` | **derived** (motif cohort) | None |
| 22 | Sedimented Invariants (P6→D4) | `BiasPrimitive` (**only**) | **derived** (from the #23 survivor) | synthetic |
| 23 | Surviving Crystals (D2→P6) | `SurvivingCrystalBody` \| `None` | **derived** (nursery survivor + evidence) | None |
| 24 | Ratcheted Overlays (P6→D2) | `MatOverlay` (**only**) | **derived** (from the #21 motif) | synthetic |
| 25 | Stale Overlays (D2→P5) | `MatOverlay` \| `None` | **derived** (overlay head, stale label) | derived |
| 26 | Avoidance Trails (D3→P2) | `AvoidanceTrailsBody` \| `None` | **derived** (trail cohorts) | None |
| 27 | Dissipated Entropy (D3→P1) | `DissipatedEntropyBody` \| `None` | **derived**, **destructive** | None |

Summary (physics mode): **10/16 non-read flows derived**
({3, 6, 10, 12, 16, 17, 18, 19, 22, 24}), **6/16 placeholders**
({1, 2, 4, 5, 9, 13}), and **11/11 store reads executor-derived**
({7, 8, 11, 14, 15, 20, 21, 23, 25, 26, 27}).  D1 reads stay `None`
because no `BoardStore` is owned by the executor.  The round-12a audit
reclassifies the six placeholders: **#1 is the only true node dependency**
(E2 has no implementation); `#2/#4/#5/#9` are body-type/transport work with
live producers; `#13` is pure wiring between two already-tested nodes
(`BoardStore`, `Regulator`) — the old "D1 frozen, so #13 blocked" comment is
circular (`docs/round12a-audit.md` §3).  No placeholder was derived in 12a:
every derivation needs a body type, and body types are a schema change,
which 12a forbids.  The default synthetic dry run
keeps its historical plumbing behaviour (every non-read flow `None` except
the #18/#19/#22/#24 no-op/rotating bodies).

### Cadence audit: universal vs policy-relative (user-adjudicated)

A proof about the minter is not a proof about the board. Classification:

- **Policy-relative with the schedule lifted as an explicit theorem
  hypothesis:** `mat_operational_policy`, `mat_fresh_policy`,
  `mat_bounded_policy` quantify over *every* epoch stream obeying the mat
  update law (`run 0 = [0]`, `run (k+1) = matTick (run k) (k+1)`). The
  dry-run minter (`matRun`) is one such stream, proven — so `mat_operational`,
  `mat_fresh`, `mat_bounded` are corollaries. A physics-epoch schedule is a
  different law, never a silent assumption.
- **Schedule-pinned (dry-run cadence, plumbing-only):** the mailbox
  recurrence theorems (`boxes_master`, `misses_master`, `sunk_master`,
  `emit_liveness`, `consume_sink_split`, `balance_at_master`) quantify
  over the generated op table, which is the dry-run's per-tick schedule
  pinned to the ledger by lockstep tests. When the sedimentation flows
  (#22/#23/#24) become trigger-gated, these are re-derived against the
  new table — the generator makes that a regeneration, not a rewrite.
- **Cadence-free (enduring):** `geometry_write_channel_purity`,
  `geometry_writes_absorbed`, `matTick_length`, and the word-generic
  lemmas (`applyOp_attempt`, `fold_attempt`, `applyOp_take`,
  `fold_take`, `applyOp_event`, `fold_event`, `fold_append`,
  `fold_box_bounded`, `step_contraction`, `emit_full_box`,
  `emit_empty_box`, `flow_attempts_raw`, `flow_take_raw`,
  `flow_event`). The cycle check and the master-profile identities are
  deliberately *not* here: `cycle_warm_no_refusal` and
  `cycle_burst_refuses_extras` quantify under the explicit ordering
  hypothesis `opsOf i = [Op.consume, Op.emit]`, and
  `attempts_at_master` / `balance_at_master` are pinned to the master
  profile.

## Master invariant matrix (INV-*)

How each user-supplied master invariant is currently verified:

| INV-id | Status | Where | Epoch |
| --- | --- | --- | --- |
| INV-BOUNDS | proven intra-tick (adjudicated round 7) + fuzzed | Lean `fold_box_bounded`/`step_contraction` (depth <= 1 after every event prefix), `boxes_bounded` (tick boundaries); Python `prefix_depth` tests, the executor `max_depth` event-boundary monitor, and the fuzz sweep | enduring |
| INV-LIVENESS | proven + empirical + structural **in its current per-flow, per-tick form** | Lean `emit_liveness`; `test_invariants.TestLiveness` | plumbing-only; enduring form: liveness scoped to the always-on spine, the organ-owned writes (#18/#19/#22/#24) trigger-gated in the physics epoch |
| INV-STAGE-ORDER (round 11; renamed from INV-CYCLE-DELAY) | **resolved** — the law is derived from the ledger, not from a cycle count | Lean `boxes_master`, `cycle_warm_no_refusal` (ordering hypothesis explicit), `cycle_burst_refuses_extras` (the delayed flows' warm-up profile, schedule-relative); `test_invariants.TestStageOrder` (4 tests). The delayed set is exactly {#10, #17} = the **stage-backward process edges**, computed at test time from `FLOW_LEDGER` + `SPINE` (P3→P1, P5→P4); every other process→process edge is stage-forward; all 11 store reads are injected at tick start. Tested: set equality, read injection, minimality (removing either delay makes a consumer precede its producer) and sufficiency. `dryrun.STAGE_BACKWARD_FLOWS` is read off the op word (`ops[0] == "consume"`), never off a census; `CYCLE_FLOWS` survives only as the Lean generator's alias | plumbing-only (schedule-relative) for the profile; the *law* is structural and enduring |
| INV-CONSERVE | two exact identities per flow per tick, proven word-generically + empirical | Lean `fold_attempt`/`fold_event` (any op sequence), `attempts_at_master`/`balance_at_master`; `test_accounting` identities A/B; overflow preserves both verbatim under reject-with-charge | enduring |
| INV-EXHAUST | **resolved** (round 11) — #27 is a real flow and the one destructive read | `test_invariants.TestExhaust` (4), `test_thermodynamics` (dissipation-exit test). Terminal sinks are exactly {#2, #4, #5}; the dissipation chain is `D3 --#27--> P1 --#5--> E1`; #27 drains residual + expired cohorts **destructively** at P1's consume (refused copies never drain), and no D3 material is re-stored as state after #27 — only resident cohorts can be re-read (#20/#21/#26). Sink rates from Lean `sunk_master` | plumbing-only for rates; the chain and the destructive read are design |
| INV-CAUSAL | structural | `test_invariants.TestCausal` | enduring |
| INV-GEOMETRY | **resolved** (round 11) — three tiers + **per-flow** kind pins | `test_invariants.TestGeometry` (3), `test_payloads`/`test_design_intent` pins; Lean `geometry_write_channel_purity`, `geometry_writes_absorbed`, `d2_write_split`, `candidate_recorded_from_birth`. Tiers: **D4 parametric** ({P6 #22} only, `BiasPrimitive`); **D2 nursery** — a separate candidate index written only by #18 (`EmergentCrystalBody`) and the overlay index written only by #24 (`MatOverlay`), two indices in `MatBoard`, not a tagged stream; **D1/D3 operational** (D1←P4 #13, D3←P5 #19). A per-store whitelist is not used: `PAYLOAD_SCHEMA[fid]` pins one exact type per write flow, so a candidate cannot be born through the overlay path or vice versa. Purity is unchanged and `mat_*_policy` is untouched (same overlay-index law) | structural/type-level (enduring); the candidate index adds its own Lean lifecycle block |
| INV-NO-OSSIFY | typed + proven (schedule explicit) + empirical | Lean `mat_fresh_policy`/`mat_fresh`, `mat_bounded_policy`/`mat_bounded`; `test_payloads` lifecycle/integration | enduring (statement); current proof policy-relative to the mat law, hypothesis explicit |
| INV-EMPTINESS | typed by construction + proven | `MatOverlay`/`BiasPrimitive` exact-type whitelist; Lean `geometry_write_channel_purity`; `test_payloads` trajectory-rejection | enduring |
| INV-PROVENANCE (round 8; round 11) | carrying + ordering closed, structural anti-mint checked; derivation **10/16 non-read flows** | A2 `Envelope(Provenance(source_fid, tick_born), body)`; `require_channel_provenance`; `test_provenance` (executor continuity for every store read including #20/#21/#23/#26/#27); `test_payloads.TestEmissionProvenance`. Round 11: #16 is now derived at P3 (no D1 dependency) and #16's identity is pinned (`error == #12.surprise`, `cost == ||#12.delta||₂`); placeholders are exactly {1,2,4,5,9,13}; all 11 store reads are executor-derived | enduring (statement); #13/#16 provenance slice still partially blocked (#13 stays blocked; D1 reads are frozen placeholders) |
| INV-PATCH (round 8; round 9a) | Python-asserted, **tested-not-proven** (Lean bounds records only) | `MatBoard` patch table keyed `overlay_id` (Q1-A): record and patch evict atomically on the capacity, byte-ceiling and organ paths; `MAX_PATCH_BYTES=262,144` per patch, `MAX_PATCH_TOTAL_BYTES=1,048,576` per table. Round 9a applies the evidence-derived bound **first**: `K ≤ 5·N·64` (`amateur/physics/capacity.py`), low-rank `head.2` patch `K(r)=192r+64`, 20,224 B at N=16 (rank 26); dense head+norm (34,240) and full (210,920) are inadmissible; `test_physics_capacity`, `test_physics_run.TestPatchAccounting` | physics epoch; substance bound not proved, patch map is in-memory (persistence absent) |
| INV-SOURCE (round 8) | structural + tested (no silent fallback) | content-only `PayloadSource` (Q2-A); fixed `PLACEHOLDER_FLOWS` dispatch that never reads the ledger; `PhysicsSource.derive_body` raises before any arrival; a source failure leaves every charge counter at 0; `test_physics_run.TestSourceSelection`/`TestCoverageLedger`/`TestSourceFailureIsNotACharge` | enduring (contract); physics producer partial |
| INV-ATOMIC (round 8; round 9a) | tested (fail-closed) | record+patch atomicity on every path: capacity, byte ceiling, organ-driven evict; round 9a refuses `patch="full"` at the capacity gate **before any fit/overlay**, and the outer byte-ceiling path is exercised with a synthetic oversized patch (asserted in the derivation pass before any arrival, no state change); `MatBoard` guard rails (validate-before-pop revalidation, duplicate-birth rejection, exactly-one `evict_record`); `test_payloads.TestMatBoardGuardRails`/`TestMatBoardPatches`, `test_physics_run.TestFullPatchModeFailsClosed` | enduring (Python); Lean bounds records only |
| INV-CAPACITY (round 9a) | derivation + tests (arithmetic; not proved in Lean) | `K ≤ γ·N·D` with γ=5.0, N = real evidence transitions used, D=64 (`amateur/physics/capacity.py`); last-layer low-rank patch `K(r)=192r+64`, `r_max=min(floor((γND−D)/192),64)`, `nbytes=4K`; N=4 → rank 6 / K=1,216 / 4,864 B; N=16 → rank 26 / K=5,056 / 20,224 B; N≥64 → rank 64 / K=12,352 / 49,408 B; dense 34,240 and full 210,920 refused at every N; outer byte ceilings are storage bounds only; `test_physics_capacity` (11) | physics epoch; γ=5.0 is pre-registered design slack, not measured; not proved in Lean |
| INV-VALIDATION (round 9a) | tested (state-based; no-peek asserted) | pre-adoption proposal at frame 16; paired candidate-vs-base validation on held-out ticks 17–18 of the same stream; birth only if `candidate_mean < base_mean` strictly, decision at 18; rejection record reason `interference` with both means; no birth/activation on reject; `test_physics_organ.TestPreAdoptionValidation`, `test_physics_run.TestPreAdoptionValidation` | physics epoch; `validation_window=2` == `stale_patience` (documented); one candidate at a time; trigger-gating (9b) not in scope; **scope: executor proposal path only** — the organ's legacy ratchet birth path (`_decide_tick → _adopt_or_revalidate → _fit_overlay → birth`) remains unvalidated, and the 0.8226 bank ratio is a fit-quality diagnostic, not validation evidence (design §15.3, notes 40–41) |
| INV-ADAPTATION / ADMISSION | **current form: round 12a** (ratio-of-means, both tick sets, both numbers); 9a/11 numbers historical | ADAPTATION = `Σ adapted / Σ base` over `phase == "flipped"` ∧ shadow-scored **out-of-sample** rows < 1.0 (the gate); `secondary_mean_of_ratios` and `with_in_sample` (birth sample added back) are reported, labelled, never gated; the three tick sets (adaptation / with-in-sample / residence) are emitted explicitly. ADMISSION = count of promoted candidates with residence ratio ≥ 1.0 == 0. Round-11 primary under this convention: 0.9992248015938371 (n=8, the same set as the residence curve — declared), secondary 0.9555247158775397; registered 12a trace: 1.013377976345238 FAIL. Pinned in `test_physics_run` | physics epoch; admission *policy* still deferred |
| INV-MDL-GAIN (round 12a) | derivation + tests (arithmetic); **user-ratified constant** | `κ·H·dbar > 32·K`, strict: `κ = 934.7982054975386` bits per (latent²·tick) = `d/(2σ²ln2)`, `d=64`, `σ² = 0.04938631785656374` (phase-3 in-distribution mean surprise — the model's own code variance), `H=20` = POST_FLIP, cost `32K` (uniform 32-bit convention). `dbar` = mean per-tick `base−adapted` over **out-of-sample** evidence; `in_sample` is never read. Refusal classes `harmful` (`dbar<=0`) vs `uneconomic`. Antecedent: the round-11 t27 candidate (K=5,056) buys 3.547 bits against 161,792 → refused (`uneconomic`, 4.56e4× short). `tests/test_physics_organ` floor arithmetic + `test_physics_nulls`; `docs/round12a-prereg.md` §3 (ratified 2026-11-20 before the runs) | physics epoch; the exchange constant is a declared modelling choice, not a theorem |
| INV-TRIGGER-EVIDENCE (round 12a) | tested (AST + measured) | `N_MIN = 2` replaces `tick >= fit_window`: the birth trigger is `tick >= N_MIN` ∧ `surprise > spike_threshold`, and the fit window / capacity bound use the actual `N = min(fit_window, tick)`. `N_MIN=2` = one transition to fit + one out-of-sample transition to test (rank-1 is admissible at `N=2` under `K <= γND`). Measured: seed 7 births t12 (K=3,712), seed 0 t9 (K=2,752). `test_physics_run` (AST assertion: `_should_birth` names `N_MIN`, not `fit_window`) | physics epoch; the old 16 was arbitrary |
| INV-MOTIF-DERIVED (round 12a) | derivation (exact DP) + tests | `MOTIF_MIN_COUNT=2`, `MOTIF_MIN_PERSISTENCE=2` derived by criterion M: the null (q99 = 1 %/tick) false-declaration probability over the 36-tick registered run must stay ≤ 1 %. Exact DP over `(count, streak, elapsed)`; pre-registered (looser intervening reading) `P_false(2,2)=0.978403 %`; implemented distance rule `0.671758 %` (stricter; the pre-registered number stands as the conservative bound — `docs/round12a-prereg.md` §5 amendment note). Unchanged from the bank corpus? No: the bank pair (4,10) is withdrawn for the single-stream regime; the derivation is from the regime's null ingest process, not from a seed's observed support pattern. `tests/test_physics_organ` pins both DP readings and the selection | physics epoch; the derivation is statistical, not a measurement of adaptation |
| INV-RESET (round 8; round 9a) | tested | `PhysicsOrgan.reset_p6_state()` clears decisions/pending/cooldown/gate evidence/overlay table/fit history; `PhysicsRun.reset()` calls it; two consecutive `acceptance()` runs are identical (round 9a: proposal 16, birth 18, stale 31, evict 35, all-active 1.0461437156029878, flip-only 0.9665724706513396, 11 D4 sediments, clean end state); `test_physics_run.TestResetRepeatability` | enduring (harness contract); needed for the registered trace to be repeatable |
| INV-ZERO-CONF (round 8; round 11) | tested by serialized-bytes equality + identity B | #22 is a **derived** gate output (Q4-A), now derived from the #23 residence evidence: `qualified` survivor → `BiasPrimitive(confidence=clamp(1-ratio,0,1), candidate_lineage=candidate_id)`; anything else → the zero-confidence sentinel, absorbed as a consumed no-op with no promotion and no D4 change. **D4 criterion adjudicated (2026-11-20): the retained-contents reading is correct and met; the round-8 "no writes at all" reading is rejected** (design §15.10) | physics epoch; the passing branch is real, not injected; D4 reading closed |
| INV-MASS (round 11; named in 12a) | tested; asserted every tick | Two **distinctly named** accounts: `conservation_account()` `deposits = resident + drained + evicted` (canonical; reads are not a term) and `issuance_account()` `deposits = promoted + resident_unpromoted + drained_unpromoted + evicted`, where the eviction term is *unpromoted* evictions and the slack is reported as `evicted_promoted` (a promoted-then-evicted cohort must not be counted twice). `mass_balance()` composes both and keeps the round-11 flat keys. Random-sequence tests in `test_payloads`; per-tick assertion in `test_dryrun`; measured 1000-tick synthetic `4000 = 32 + 999 + 2969`; naive ruling form over-counts by exactly the read surfaces (pinned). **Flagged (12a audit):** `DebrisPool.promote` has no executor caller, so the issuance `promoted` term is structurally 0 in every run — the D3 promotion path is unwired; the nursery (#22) promotion is the live one | tested-not-proven (Python) |
| INV-PROMOTE-ONCE (round 11) | tested (Python) + proved for the candidate index | D3: `DebrisPool.promote` refuses a second promotion of the same cohort. D2: Lean `nursery_promote_resident` / `nursery_promote_once` / `nursery_promote_twice` (an id promoted out is never re-admitted, so it cannot be promoted twice), and `nursery_birth` refuses any id ever seen | enduring (statement) |
| INV-LINEAGE (round 11) | tested, physics mode only, fail-closed | #24 `motif_lineage` must equal this tick's consumed #21 motif; #22 positive sediment requires `candidate_lineage` = the #23 survivor consumed this tick, still resident, and **qualified**; the candidate is promoted before the sediment. A violation is an `AssertionError` (wiring defect), never a charge. The #23 survivor read carries `qualified` as a flag, so a harmful candidate is surfaced and refused rather than pre-filtered away | physics epoch |
| INV-NURSERY (round 11) | typed + tested + proved (index law) | D2 candidate index: `NURSERY_CAPACITY=4` FIFO birth, ids never resurrected, `residence_ticks = shadow observations`, `RESIDENCE_N=4` pre-registered, candidate patch counted in D2 bytes (`patch_bytes` covers both indices), promotion/eviction remove record+patch atomically; Lean nursery block proves boundedness, birth membership, one-tick residence, distinctness and promotion-once | enduring (law); the constants are design pins |
| INV-BLANKET | structural + layer-4 | `test_invariants.TestBlanket`, `test_design_intent`, `test_thermodynamics` | enduring |
| INV-THROUGHPUT | covered | liveness + `test_every_entity_cycle_carries_throughput` | plumbing-only in current form (inherits LIVENESS's cadence); enduring form follows the spine-scoped liveness |
| INV-CENSUS | **resolved as a derived fingerprint** (round 11) | `test_invariants.TestCensus` (3): the census is **computed from `FLOW_LEDGER` at test time** (node-distinct simple cycles, canonical rotation, parallel edges collapsed) — 26 total / 24 interior / D1=12, D2=11, D3=15, D4=4, D3 has no 2-cycle, D4 minimum 5, D2 nursery loop D2→P2→P3→D2. The literals are **regression expectations, not the source of truth**: `test_inv_census_is_derived_not_listed` perturbs the ledger and shows the fingerprint moves. Evidence for structural richness / no degenerate store, never for dynamics. **Tested-not-proven**; no Lean change | evidenced (tested-not-proven) |
| D4 sediment cadence (#22 one per tick), #18 birth cadence | factual in the dry run | `test_payloads` lifecycle | plumbing-only (harness artifact; 6.0 is an interference-checked gate, not a metronome) |
| #18 edge | **VOID — superseded by the round-10 spec correction** | `v0.10-untested.md` round-10 audit record + superseded readings; `test_design_intent.TestCrystalLexicon` now pins the corrected wiring (3.0→D2) and the absence of 5.0→6.0; the crystal lexicon is explicitly non-authoritative until adjudicated | closed as a reading, **not** as a payload decision (see the provisional choices above) |

Lockstep chain (nothing is trusted twice):

```
v0.10-untested.md  ->(sync tests)->  amateur/ledger.py
ledger.py          ->(derived)    ->  amateur/dryrun.flow_ops()
flow_ops()         ->(generator)  ->  formal/DryRun.lean opsOf arms
opsOf arms         ->(parser, table parity test) -> flow_ops()
```

`test_lean_lockstep.py` enforces every arrow: ledger sync, table parity
(with the deliberate `absorb -> consume` compilation), profile markers, an
independent scalar replay of the *parsed Lean arms* over 64 ticks, and —
when a lean binary is present (`LEAN_BIN`, default
`/tmp/lean4/lean-4.21.0-linux/bin/lean`) — an actual compile with no
`sorry`.

Regenerate the Lean file after touching the ledger or the spine:

```sh
python3 tools/gen_lean.py > formal/DryRun.lean
```

## Layer 3 — Empirical bookkeeping (`test_dryrun.py`, `test_accounting.py`,
`test_invariants.py`, `test_payloads.py`, `test_fuzz.py`)

The executor (`DryRun`) is checked against the adjudicated channel
semantics over real runs: first-arrival-wins acceptance, reject-with-charge
on full mailboxes, liveness (all 27 flows accepted exactly once per tick
under the canonical schedule, nothing refused), and the two exact
conservation identities per flow per tick — `attempted = accepted +
refused` (A) and `accepted = consumed + sunk + delta depth` (B). Refused
copies never enter (B); a miss is a deficit charge on the expectation side
and removes nothing. Intra-tick INV-BOUNDS is asserted empirically via
`prefix_depth` (depth <= 1 after every prefix of a flow's event word) and
the executor's `max_depth` monitor (sampled at every arrival and
consumption boundary). Stationarity: per-tick stats identical from tick 2;
tick 1 differs only in the two cycle warm-up misses and the matching
consumption deficit. A final assertion replays `step_flow` for 32 ticks
and confirms the pure scalar model tracks the executor tick for tick.

## Property-based fuzz (`amateur/fuzz.py`, `test_fuzz.py`)

Randomized adversarial injection on the entity channels (#1, #3) is run
once in `test_fuzz.py`'s `setUpClass` and asserted seed by seed:

- distribution: seeds 0–11, 1000 ticks each, multiplicities k drawn
  uniformly in [1, 8] per injected flow per tick (peak multiplicity
  recorded as `peak_k`; every seed reaches 8 on both channels);
- per tick of every seed, for all 27 flows: intra-tick INV-BOUNDS (`max_depth` <= 1
  at every event boundary, mailbox depth <= 1 at every tick boundary);
  identity A and identity B exactly; schedule-shift `attempt == 1`
  outside the plan; scalar agreement with canonical `step_flow`;
- round-10 results (corrected graph): per seed attempted 33,821–34,139,
  accepted = 27,000 exactly (= 27 flows x 1000 ticks), refused
  6,821–7,139, missed = 2 exactly (the #10/#17 cycle warm-ups); over the
  whole 12-seed sweep attempted = 407,794, accepted = 324,000, refused =
  83,794, missed = 24. (Round-7's 26-flow figures — accepted 26,000 per
  seed, attempted 395,794 total — are historical on the old graph.) Adversarial pressure registers in every seed (refused > 0) while
  the depth bound never breaks — the charge stays scalar, the flood
  never relocates into state.
- **Scope pin: multiplicities k >= 1 only.** Zero-injection (starvation)
  semantics are deferred to the physics epoch; an idle tick is a
  legitimate state, not a liveness violation. The distribution above is
  a property of the harness, not part of the spec.

## Layer 4 — Design intent (`test_design_intent.py`, `test_contracts.py`, `test_engine.py`)

Prose-level invariants from the "Emptiness" text and the ABC contracts:
timescale separation, trajectory-free core, debris discipline, carrier
semantics, node vocabulary. These bind the *intended* behavior, not just
the implemented behavior.

## Physics-epoch acceptance gates (pinned now)

- **Payload provenance (round 8, ADJUDICATED — A2 envelope; gate PARTIAL):**
  every accepted payload carries `(source_fid, tick_born)` and is built by
  the derivation pass, not at the emit boundary — no minting at emit time.
  Carrying, ordering, and the structural anti-mint check are closed
  (`test_provenance.py`, `test_payloads.TestEmissionProvenance`). In
  physics mode the source derives real bodies for #3/#6/#10/#12/#17/#18/
  #19/#24 and the organ's gate/record (#22), so the original #22/#23
  objection is closed; the gate stays **PARTIAL** because #2/#4/#5/#9/#13/
  #16 are still placeholders and D3 is written but not read (#20/#21/#26/
  #27 are placeholder reads).
- **Phase 4–5b executor wiring (round 8, ADJUDICATED Q1–Q4 = A; gate
  PARTIAL):** the content-only source is separated from the executor
  (reads stay executor-derived; `PLACEHOLDER_FLOWS` is a fixed executor
  constant, never read from the coverage ledger); a missing derivation
  raises before any arrival, so a source failure is never a channel charge;
  D2 holds the predictor patch keyed by `overlay_id` with atomic
  record+patch eviction and byte accounting
  (`MAX_PATCH_BYTES`/`MAX_PATCH_TOTAL_BYTES`); the encoder is bit-identical
  after fitting (`torch.equal`); #22 is the gate's **derived** output —
  blocked here because the measured cumulative adapted/base evidence is
  harmful (0.2657 vs 0.1475), with the pass condition, patience and
  injected-evidence caveat recorded below; a confidence-0 #22 is a consumed
  no-op and D4's retained contents are byte-identical: the acceptance run's
  empty D4 stays empty (5 B, a vacuous guard), and the separate non-vacuous
  retention probe seeds one real sediment through a passing gate and shows
  121 B retained byte-identically across the full flip scenario (35 further
  confidence-0 no-ops, same object, no second sediment). The supported
  `patch="full"` mode fails **closed**: the 843,680 B patch is refused in
  the derivation pass before any commit, with every charge counter unchanged
  and no record, patch, or organ adapter left behind.
  Open: the byte bound is Python-asserted (Lean bounds records only), the
  patch map is in-memory, and the fixed cadence uses exact-type no-op
  sentinels — cadence is **plumbing**, lifecycle semantics are
  organ/store-owned until the trigger-gating round.
- **Single-stream overlay recovery (round 8, MEASURED — FAILED, recorded):**
  the pre-registered single-stream adaptation (last 16 same-stream
  transitions, one attempt) does **not** produce a forward surprise drop:
  adapted 0.3264 vs base 0.2588 over the 4 active ticks = **1.2614× (26 %
  worse)**; the organ flags it stale after two harmful ticks and evicts it.
  `test_physics_run.TestSingleStreamAcceptance.test_single_stream_surprise_drops`
  is an explicit `expectedFailure` — the assertion is kept strict, not
  weakened into "no material harm". The 0.690× bank/held-out fit reported
  here belongs to the dense patch and is **withdrawn** in round 9a, which
  re-measures it at 0.8226 under the admissible low-rank patch
  ([phase4-5-design.md](phase4-5-design.md) §15.7).
- **Round 9a — evidence-bounded capacity + pre-adoption validation
  (MEASURED; acceptance wording OPEN):** the patch is now a
  capacity-bounded low-rank `head.2` delta, `K ≤ 5·N·64` with N the real
  evidence transitions, stored as factors (N=16 → rank 26 / 5,056 params /
  20,224 B); the dense head+norm (34,240) and full (210,920) patches are
  inadmissible at every N. The candidate is fitted at frame 16, validated
  on paired held-out ticks 17–18 (0.133868 vs 0.147455, 0.9079), and born
  at 18 only on a strict win; a rejection is a public `interference`
  record with no birth/activation. Seed 7 passes and is later
  staleness-evicted (stale 31, evict 35, end state empty); seed 0 is the
  measured rejection trace (0.190908 vs 0.133875, 1.4260; no
  D2/D4/activation). **Open:** over all active ticks the ratio is 1.0461
  (> 1.0, so `test_single_stream_surprise_drops` stays a strict
  `expectedFailure`); over the active *flipped* ticks — the literal
  "active flip window" — it is 0.9666 (< 1.0). Both are recorded; the
  registered metric was not switched after seeing the result.
  `test_physics_capacity` (11) + the updated organ/run suites.
- **Overflow semantics (round 7, ADJUDICATED — reject-with-charge):**
  each mailbox has capacity 1. Acceptance is first-arrival-wins with
  ledger order as tie-break — deterministic, stateless, pointwise. A
  refused arrival increments a scalar excess charge and the event object
  is discarded at the channel: no refusal logs, no charged-payload
  retention, no replay. Charges are O(1) per channel, exactly like
  sinks. Misses (deficit: expected, absent) and refusals (excess:
  arrived, refused) are distinct charge kinds with separate counters,
  never summed. The identities `attempted = accepted + refused` (A) and
  `accepted = consumed + delta-sunk + delta-depth` (B) hold per flow per
  tick and are preserved verbatim; per-flow step contraction and the
  intra-tick depth bound (depth <= 1 at every event boundary, not merely
  at tick snapshots) are proved in Lean (`fold_box_bounded`,
  `step_contraction`, `emit_full_box`, `emit_empty_box`).
- **Starvation deferral (fuzz scope is not the spec):** fuzz
  multiplicities are k >= 1 only; zero-injection (starvation) semantics
  are deferred to the physics epoch together with trigger-gating —
  "without throughput, the geometry remains without unfolding" means an
  idle tick is a legitimate state, not a liveness violation, and the
  enduring form of INV-LIVENESS is re-scoped then.
- **Refusal coupling (design intent, physics epoch):** refusal rate at
  the blanket is the world arriving faster than we can digest. Sustained
  excess must drive collapse pressure at 4.0/5.0, and dissipated refusal
  ultimately exits as entropy exhaust. Today the charges are pure
  accounting with no feedback into dynamics; this coupling is recorded
  now so that it is built, not rediscovered.

## Round 8 physics-mode acceptance record (measured)

*(round 8 record; superseded for the live policy by the Round 9a record
   below — kept as the measured history of the dense-patch attempt.)*

Protocol: one executor stream (`PhysicsRun`, seed 7), 8 warmup ticks
(normal rows 1–7 are the measured normal phase; the gravity flip is
applied before tick 8), gravity flip, 20 post-flip ticks, flip back, 8
recovery ticks. The organ
observes, digests and ratchets **before** `_derive`; the overlay is fitted
once from the organ's same-stream buffer when a spike passes the q99 gate
(last 16 transitions, one attempt, no post-hoc selection).

| Item | Measured |
| --- | --- |
| normal-phase mean surprise | 0.1012 |
| spike peak | 1.9496 (19.27× normal) |
| pre-adoption flipped base mean | 0.4366 |
| overlay installed | frame tick 16 (head patch 34,240 params / 136,960 B) |
| active ticks | 4 |
| adapted vs base, same ticks | 0.3264 vs 0.2588 → **1.2614×, no drop** |
| stale / evict | finalized tick 19 / executor tick 23 |
| end state | D2 residents and patches `()`, patch bytes 0 |
| D4 (acceptance run) | empty before and after (5 B); 0 retained writes; 36 confidence-0 no-ops |
| D4 retention (separate probe) | 1 pre-seeded real sediment, 121 B before and after, same object, 35 further no-ops |
| #22 gate | derived from cumulative evidence, **blocked**: 0.2657 adapted vs 0.1475 base; 36 no-ops; D4 empty |
| D3 | 7 bodies / 3,934 B of capacity 8 / 65,536 B |
| encoder | bit-identical after the fit (`torch.equal`) |
| suite / Lean | 289 tests, OK, 1 expected failure; `gen_lean` byte-identical |

Honest reading: the no-drop is a **genuine null result**, not a wiring
failure — the lifecycle fires end-to-end and four pre-registered/observed
single-stream fits (1.05–1.57×, plus the 1.2614× default) are all no-drop.
The bank/held-out fit (0.690× on unseen episodes) is a different protocol
and is not evidence for the executor stream. Two disclosures on the gate:
production publishes cumulative adapted/base evidence only after
`stale_patience`=2 applied ticks (one transient beneficial tick at frame 17
would have passed a strict per-tick recorder at tick 18, sedimenting one
D4 body — the patience rule prevents that), and the *passing* branch is
exercised only by injected evidence, never by a real beneficial adaptation
in this slice. Open limitations: #2, D1/D3 reads, patch persistence, and
trigger-gating; full list in
[phase4-5-design.md](phase4-5-design.md) §14.9.

## Round 9a physics-mode acceptance record (measured)

Protocol: one executor stream (`PhysicsRun`, seed 7), 8 warmup ticks
(normal rows 1–7 are the measured normal phase; the gravity flip is
applied before tick 8), gravity flip, 20 post-flip ticks, flip back, 8
recovery ticks. The
capacity-bounded candidate is fitted at frame 16 (16 real same-stream
transitions), validated on frames 17–18, and born at the decision tick 18
only if the paired held-out candidate mean is strictly below the base mean.
No cross-stream bank data is used in the executor; no post-hoc fit-variant
or seed selection was made for the registered trace.

| Item | Measured |
| --- | --- |
| capacity bound | `K ≤ 5·N·64`; N=16 → 5,120 capacity, rank 26, 5,056 params, 20,224 B |
| dense patches | head+norm 34,240 and full 210,920: inadmissible at every N |
| proposal / validation | frame 16 / frames 17–18 (paired candidate vs base) |
| candidate vs base, validation | 0.13386843726038933 vs 0.1474553719162941 → **0.9078573097790045** |
| decision / birth | **birth** at frame 18 (strict pass) |
| active ticks | 14 (frames 19–32) |
| adapted / base, active **flipped** ticks (19–27) — **ADAPTATION** gate | 0.25469933084115787 / 0.2635077436765035 = **0.9665724706513396** → **PASS** |
| adapted / base, **all** active ticks (19–32) — **ADMISSION** gate | 0.3444968502626556 / 0.3293016486402069 = **1.0461437156029878** → **FAIL** |
| adapted / base, active recovered ticks (28–32) | 0.5061323852213514 / 0.44773067757487295 = **1.1304393702991506** |
| stale / evict | finalized tick 31 / executor tick 35 (record+patch together) |
| end state | D2 residents and patches `()`, patch bytes 0 |
| D4 | 11 real gate sediments (frames 21–31), 25 confidence-0 no-ops; retained prefix byte-identical and object-identical (no test double) |
| encoder | bit-identical after the fit (`torch.equal`) |
| rejection trace (seed 0) | validations 17–18: 0.19090842455625534 vs 0.13387499749660492 = **1.4260200046771003**; `interference` at 18; no birth/activation; D2 empty, patch bytes 0; D4 byte-identical |
| reassessed bank fit | 0.8225682511949177 (old 0.690× quarantined: dense-patch artifact) |
| suite / Lean | 313 tests, OK (1 expected failure); `gen_lean` byte-identical |

Honest reading: the validation gate passes on the registered trace and the
organ's stale/evict rule removes the patch on the flip-back harm — the
"stale/evict on a failure trace" requirement is met by the real trace. The
closeout ruling (2026-11-20) pre-registers two gates over this one trace:
**ADAPTATION PASS** (0.9666 over the active flipped ticks) and
**ADMISSION FAIL** (1.0461 over all active ticks) — the failure is the
round's primary finding (the 2-tick validation admitted a patch whose
bounded lifetime harm outweighed its benefit; the stale rule, not
validation, protected the trace). The strict `expectedFailure` stays on
ADMISSION until admission policy is set in Round 9c; selecting another seed
after seeing the result would be seed-shopping and was not done (13 of the
first 31 seeds reject, 18 birth; the rejection test uses seed 0, the first
natural seed, not a re-registered acceptance). The D4 reading is adjudicated
(retained-contents, met). Full derivation, per-tick trace, gate definitions
and honesty notes in [phase4-5-design.md](phase4-5-design.md) §15.10 (and
§15.4–§15.9).

*Correction provenance (round-9a audit): this file is Lead-owned.  It was
edited by the implementer in commit `e6d6051` contrary to the task contract
that said "Do NOT edit `docs/TESTING.md` (Lead owns)"; the Lead reviewed that
change and adopts it.  The audit corrections on this page — the
`TestZeroConfidenceGate` 8-test count, the INV-VALIDATION scope caveat and
the "round 8 record" label — were applied by the implementer under explicit
Lead delegation (shared-board task 8, docs-only); the round-9a closeout
edits here — two pre-registered gates, adjudicated D4 reading, 313-test
count — were applied under the same explicit delegation (shared-board
task 9, docs + one test only).*

## Round 10 spec-correction record (2026-11-20)

An independent audit re-read the original ASCII and found that the ledger
shipped in rounds 8–9 had been transcribed from a *transcription*, not from
the ASCII itself: seven edges on the lower half of the drawing were
misread. The user accepted the ASCII reading; the corrected 27-flow ledger
is the spec. `v0.10-untested.md` (mermaid + table + audit record),
`amateur/ledger.py` and everything generated from them were corrected
together. **Scope was plumbing only** — no physics-organ rewiring, no
admission-policy change, no #13/#16 provenance work.

What the correction changes: #16 is 3.0→5.0 (not 4.0→5.0); #18 is 3.0→D2
(not 3.0→5.0); #20 is D3→4.0; #21 is D3→6.0 and a **read**; #23 is D2→6.0
and a **read**; #25 is D2→5.0; #27 (D3→1.0, read) is restored. `5.0→6.0`
no longer exists. D2's write set is `{18, 24}`; D3 gains its exit.

What did *not* change (checked, not assumed): the per-flow op words of the
existing 26 flows, the master profile (depths 1 on #10/#17, 0 elsewhere),
the sink profile `{2, 4, 5}`, and the cycle set `{10, 17}`. Only one Lean
arm is new (`| 27 => [Op.emit, Op.consume]`); the rest of the Lean diff is
the fid-range extension (1..27) and the payload markers. That is a
measurement, not a hope: `boxes_master` / `misses_master` / `sunk_master` /
`refused_master` are unchanged theorem-by-theorem.

Corrected census (computed from code, independent of the dry run):
**24** interior simple cycles; memberships D1=12, D2=11, D3=15, D4=4;
D4 has no 2- or 3-hop loop. The delay set is **{#10, #17}**, justified by
stage inversion (P3→P1, P5→P4) plus store-read injection — *not* by cycles
(the old cycle-based INV-CYCLE-DELAY argument is withdrawn).

Measured on the corrected graph: dry run 1000 ticks — liveness 27/27 per
tick, `attempted == accepted` on every tick, refusals 0, sinks {2: 1000,
4: 1000, 5: 1000}, misses exactly `{10: 1, 17: 1}` (warm-up), in transit
`{10: 1, 17: 1}`, `max_depth` <= 1 on all 27 channels. Fuzz: seeds 0–11,
1000 ticks, k in [1, 8] — attempted 407,794, accepted 324,000, refused
83,794, missed 24 (2 per seed); no depth or identity violation. Suite:
**355 tests, OK (1 expected failure)**. `gen_lean` output byte-identical to
`formal/DryRun.lean`; Lean compiles (2.1 s).

**Provisional decisions — ALL RESOLVED in Round 11** (see the Round-11
closeout below and `docs/round11-record.md`): #18 is now the
`EmergentCrystalBody` nursery kind (not a `MatOverlay` status); #23 reads
the nursery survivor with its residence evidence; #20/#21/#26/#27 carry
real tagged D3 cohort bodies; #16 is derived at P3. The "candidate carries
a predictor patch" tension was adjudicated (ruling E): the candidate does
carry the patch **as substance**, shadow-scored every resident tick and
never applied to behaviour, with live patches arriving only via #24.

**Historical measurements.** Every physics-epoch measurement recorded
above (rounds 7–9a: the 0.9079 validation ratio, ADAPTATION 0.9666 /
ADMISSION 1.0461, the D4 sediment count, the patch byte figures, the
rejection trace) was taken on the **old 26-flow graph**. They are kept as
historical records; they are not re-measured and not re-asserted against
the corrected ledger. The round-9b trigger-gating work was stashed and
superseded by this correction.

## What each layer catches that the others cannot

- Layer 1 catches wiring regressions that no run would exhibit soon (or
  ever, from a cold start).
- Layer 2 catches semantic drift between the documented op table and both
  implementations, and proves limits (`forall t`) that sampling cannot.
- Layer 3 catches executor-only state (counters, drains, evictions,
  payload emissions) that the pointwise model does not cover.
- Layer 4 catches code that satisfies the ledger but betrays the text.

## Round 11 closeout (rulings A–G)

Measured on the corrected 27-flow graph, after the Round-11 implementation:

| Check | Result |
| --- | --- |
| suite | **431 tests, OK, 0 skipped, 0 expected failures** (50–63 s) |
| `python3 tools/gen_lean.py \| diff - formal/DryRun.lean` | empty (byte-identical) |
| `lean formal/DryRun.lean` | exit 0, **76 theorems**, zero `sorry`, zero `axiom`, 2.3–3.4 s |
| op table vs Round-10 HEAD | **byte-identical** (no mailbox theorem restated) |
| 1000-tick canonical dry run | liveness 27/27, accepted 27,000, refused 0, sinks `{2:1000, 4:1000, 5:1000}`, misses `{10:1, 17:1}`, `max_depth ≤ 1`, in transit `{10:1, 17:1}` |
| D3 mass balance (1000-tick synthetic) | `4000 = 32 + 999 + 2969`, ceiling evictions 0 |
| registered physics run (seed 7, 36 ticks) | ADAPTATION **0.9555247158775397** (n=8) PASS; ADMISSION **0 harmful / 1 promotion** PASS; #23 surfaced-and-refused a harmful survivor on 4 real ticks |
| control (seed 0) | 0 promotions, but **vacuously** (no candidate forms under the registered trigger) — reported as finding |
| reachability | motif cohorts **0** on both single-stream seeds (`MOTIF_MIN_PERSISTENCE=10`, `MOTIF_MIN_COUNT=4` unreachable); #21→#24 demonstrated by a bank probe and an executor wiring test only — reported as finding |

The Round-11 findings (motif reachability, vacuous control, knife-edge
promotion, structural ADMISSION, out-of-sample evidence convention,
mass-balance formula, drift placeholder) and the Round-12 adjudication list
are in `docs/round11-record.md` §3–§4. The pre-registered gate text is
`docs/phase4-5-design.md` §15.11 (written before the run).

**Historical. ** The Round-10 figures above (355 tests, the Round-10 dry-run
and fuzz numbers) remain the record of that round; the 27-flow op table is
unchanged, so the mailbox profile above reproduces them exactly, and they
are restated here only as a reproduction check, not as new evidence.

## Round 12a closeout (conventions, gain floor, controls)

Measured on this round's tree; full record `docs/round12a-record.md`.
Pre-registration `docs/round12a-prereg.md` (constants frozen and, for the
MDL exchange, user-ratified **before** the runs).

| Check | Result |
| --- | --- |
| suite | **477 tests, OK** (129 s), 0 skipped, 0 expected failures |
| `python3 tools/gen_lean.py \| diff - formal/DryRun.lean` | empty (byte-identical); Lean untouched by 12a |
| ADAPTATION convention | ratio-of-means primary; mean-of-ratios and with-in-sample reported and labelled; three tick sets emitted |
| Round-11 ADAPTATION recomputed | primary **0.9992248015938371** (n=8; adaptation set == residence set, declared), secondary 0.955525, with-in 0.899061 |
| seed 7 registered (birth t12, K=3,712) | ADAPTATION primary **1.013378 FAIL** (n=15); with-in 0.892135; residence ratio-of-means 1.060769; **0 promotions even pre-floor**; ADMISSION 0/0 vacuous |
| gain floor (κ=934.798, H=20, ratified) | refuses every candidate: Round-11 t27 by **4.56e4×**, seed-0 t14 by **524×**, organ probe by **2,154×** |
| retroactive falsifier | Round-11 t27 refused: gain 3.547 bits vs 161,792 bits (`uneconomic`) |
| trigger (N_MIN=2) | seed 7 births **t12** (was t19 — the reversal of the Round-11 verdict is the round's primary finding); seed 0 births **t9**, K=2,752 |
| motif thresholds (derived 2/2, criterion M) | seed 7: 0 motif cohorts; seed 0: live half fires (2 motif ids, `ovr-02`/`ovr-03`, stale/evict) |
| null suite (seed 7, pre-registered) | 0 promotions for zero/random/shuffled with **and** without the floor; zero exactly neutral (ratio 1.0, dbar 0.0); `no_flip` vacuous (no spike) |
| post-hoc seed-0 null probe | at the shared t14 decision point (n=5): real **0.983203 promoted pre-floor**, zero 1.000000, random 1.005290, shuffled 1.050450 — no null ever qualified at any tick; the separator is the qualification test, not the floor; stop condition did not fire (post-hoc, not a gate) |
| housekeeping | two named mass accounts; dead `#21` `drift` payload field deleted; coverage audit `docs/round12a-audit.md` (#1 blocked, #2/#4/#5/#9 not-yet-derived, #13 transport-only) |

**Open convention issue recorded, not resolved.** ADAPTATION's tick set
depends on the floor it measures (a refused candidate keeps being scored):
seed 0 is n=18 / 1.0321 with the floor on and n=5 / 0.9832 ablated. For 12b
the metric needs a fixed evidence window or a per-candidate form.

**Flags.** `DebrisPool.promote` has no executor caller (`INV-PROMOTE-ONCE`'s
D3 half is tested but never exercised); `INV-ADAPTATION`/`ADMISSION` numbers
above are trace-level and not invariant to the gate. Both are in the 12b
input list.
