# Round 11 record — rulings A–G, then the LeWM circulation

**Status:** implementation + measurement record for Round 11. Every number
here was measured on this worktree; none is inherited from rounds 7–9a.
The pre-registered gate text lives in `docs/phase4-5-design.md` §15.11 and
was written before the seed-7 run; this file appends the measurements and
the findings.

**Scope executed:** rulings A–G (user adjudication 2026-11-20), #16 derived
(this round's newly unblocked provenance item), then the corrected
lifecycle run end-to-end on the flippable 2D env with the CPU micro-LeWM:
seed 7 registered, seed 0 as the control. No 9b trigger-gating was
recovered; #13 stays blocked; reject-with-charge, theorem discipline, the
cadence quarantine, the provenance gate, Q4 zero-confidence and the frozen
encoder are all untouched.

---

## 1. A–G as implemented

| Ruling | What landed | Evidence |
| --- | --- | --- |
| **A** distinct kind, pinned **per flow** | `EmergentCrystalBody` is the D2 candidate kind (reachable; it was only unreachable because the whitelist was wrong). `MatBoard` has two indices — overlay (policy untouched, `mat_*_policy` not restated) and candidate (new). `PAYLOAD_SCHEMA[18]={EmergentCrystalBody}`, `[24]={MatOverlay}`, each write channel exactly one type and no `None`. Lean: `PayloadKind.crystal`, `d2Allowed=[overlay,crystal]`, `d2_write_split`, `candidate_recorded_from_birth`, plus a nursery lifecycle block (`nurseryTick_length`, `nursery_bounded`, `nursery_birth_member`, `nursery_fresh`, `nursery_runs_distinct`, `nursery_promote_resident`, `nursery_promote_once`, `nursery_promote_twice`) | `tests/test_payloads.py` (80), `tests/test_invariants.py::TestGeometry`, Lean 1,145 lines / 76 theorems, compiles in 2.3–3.4 s, zero `sorry`/`axiom` |
| **B** real flow, degraded body, destructive drain | `DissipatedEntropyBody(cohort_ids, magnitudes, tick)` — magnitudes and lineage only, no motif/scaffold/patch/trajectory by type. #27 is the one destructive read: the cohorts named by the read are removed from D3 at P1's consume, in the same tick; a refused copy never drains. D3 mass balance asserted every tick. Byte ceiling is drain-first (drain via #27 then deposit), with a recorded eviction backstop | `tests/test_dryrun.py`, `tests/test_invariants.py::TestExhaust`, `tests/test_payloads.py::TestDebrisPool*`; measured registered run `12 = 4 + 8 + 0` |
| **C** #23 as a precondition for #22; #24 from #21 | Physics mode: a positive #22 requires `candidate_lineage` = the candidate surfaced by *this tick's consumed* #23, still resident and `qualified`; the candidate is promoted before the sediment. A #24 write requires `motif_lineage` = this tick's consumed #21 motif. Violations are `AssertionError` (wiring defect), never a charge. #21 has a real consumer now | `tests/test_dryrun.py` (lineage gates), `tests/test_provenance.py`, `tests/test_physics_run.py::TestLiveOverlayLifecycle` |
| **D** lexicon; survivor is a **read** | Three kinds / three stores: `EmergentCrystalBody` (nursery), `MatOverlay` (ratcheted), `BiasPrimitive` (D4). No `surviving` status and no third D2 status. `#2 Crystallized Output` is the blanket's output to E2, not a D2 crystal. "Surviving Crystal" is the #23 read selection, carrying candidate identity + residence evidence | `tests/test_design_intent.py::TestRulingDLexicon`, `tests/test_invariants.py::TestGeometry` |
| **E** candidate carries the patch, shadow-scored, never live | At birth the candidate gets a capacity-bounded patch as **substance** (accounted in D2 bytes). Every resident tick it is scored on a shadow copy of the predictor against base; encoder and live predictor are asserted bit-identical after each scoring call. Live patches arrive only via #24. #8 reads the overlay index only | `tests/test_physics_organ.py::TestShadowScoring` (6), `TestCandidateFit` (5), `test_scratch` evidence in `.work/round11/` |
| **F** tagged composite D3, per-edge filters, promote-once | `DigestedDebrisBody(debris_id, cohorts)` with `DebrisCohort(kind ∈ {motif, scaffold, trail, residual}, …, provenance)`. Filters pinned: #20→scaffold, #21→motif, #26→trail, #27→residual + expired. Reads non-destructive (except #27); `DebrisPool.promote` refuses a second promotion of the same cohort | `tests/test_payloads.py` (cohort filters, drain, promote-once, mass balance), `tests/test_dryrun.py` |
| **G** census fingerprint; rename | Census computed from `FLOW_LEDGER` at test time; 26/24 and 12/11/15/4 are regression expectations only, labelled tested-not-proven; a ledger perturbation test proves it is derived, not listed. `INV-CYCLE-DELAY` → `INV-STAGE-ORDER`: delayed set = exactly the stage-backward process edges, store reads injected, minimality + sufficiency tested; `dryrun.STAGE_BACKWARD_FLOWS` | `tests/test_invariants.py::TestCensus` (3), `TestStageOrder` (4) |
| **#16** | Derived at P3 this round (no D1 dependency): `CostErrorBody(cost=‖#12.delta‖₂, error=#12.surprise)`, identity-tested | `tests/test_physics_run.py` (#16 identity) |

### 1.1 Schema pins (the shape the ruling asked for)

| Channel | Flow | Pinned body |
| --- | --- | --- |
| D2 write | #18 | `EmergentCrystalBody` **only** |
| D2 write | #24 | `MatOverlay` **only** |
| D4 write | #22 | `BiasPrimitive` **only** |
| D3 write | #19 | `DigestedDebrisBody` **only** |
| D2 read | #8 / #25 | `MatOverlay` \| `None` |
| D2 read | #23 | `SurvivingCrystalBody` \| `None` |
| D3 read | #20 / #21 / #26 / #27 | `ScaffoldsBody` / `StructuralMotifBody` / `AvoidanceTrailsBody` / `DissipatedEntropyBody` \| `None` |
| D4 read | #7 | `BiasPrimitive` \| `None` |
| D1 | #11/#13/#14/#15 | frozen (`None` / `BoardOpBody` / `SlotStateBody`) |

Reads allow `None` so the A2 provenance rule survives unchanged (a read
with a body must carry the stored record's provenance with
`tick_born < emit_tick`); a sentinel body on a read would have forced a
sentinel-provenance exception, i.e. a weakening of the anti-mint gate that
the ruling did not ask for. The "untypeable" requirement is enforced where
it matters: the **write** channels.

---

## 2. The physics round

* **Setup:** own flippable 2D env, CPU micro-LeWM (encoder frozen,
  bit-identical), capacity bound `K ≤ γND` (γ = 5.0) in force,
  reject-with-charge unchanged, provenance gate on every derived body,
  `RESIDENCE_N = 4`.
* **Scenario (pre-registered):** warmup 8, gravity flip at 8, 20 flipped
  ticks, flip back at 28, 8 recovery ticks — 36 executor ticks.
* **Seeds:** 7 registered, 0 control (pre-registered expectation: zero
  promotions).

### 2.1 Pre-registered gates — measured

| Gate | Definition | Seed 7 measured | Verdict |
| --- | --- | --- | --- |
| **ADAPTATION** | mean over shadow-scored ∩ `phase=="flipped"` rows of `shadow adapted/base` | **0.9555247158775397**, n = 8 | **PASS** (`< 1.0`) |
| **ADMISSION** | count of promoted candidates whose final residence `ratio ≥ 1.0` | **0 harmful / 1 promotion** | **PASS** (`== 0`) |

ADMISSION was **non-vacuously exercised**: the harmful survivor was
surfaced by #23 on four real ticks (ratios 1.009547, 1.023553, 1.019522,
1.002973), each with `qualified = False`; the source emitted the
zero-confidence #22, D4 did not change and `promoted_id` stayed `None`.
Only the t27 survivor sedimented.

### 2.2 Seed-7 residence-evidence curve (per resident tick)

| t | residence | adapted mean | base mean | ratio (cumulative) | shadow (tick) | qualified | event |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 0 | — | — | — | — | n | #18 birth (rank 26, 5,056 params, 20,224 B, N=16) |
| 20 | 1 | 0.386725 | 0.326463 | 1.184592 | 1.184592 | n | first shadow tick |
| 21 | 2 | 0.209407 | 0.188007 | 1.113826 | 0.647588 | n | |
| 22 | 3 | 0.202779 | 0.201347 | 1.007115 | 0.831149 | n | |
| 23 | 4 | 0.207068 | 0.205110 | 1.009547 | 1.016336 | n | first #23 surface → **refused** |
| 24 | 5 | 0.213401 | 0.208490 | 1.023553 | 1.075312 | n | refused |
| 25 | 6 | 0.249575 | 0.244796 | 1.019522 | 1.009666 | n | refused |
| 26 | 7 | 0.263071 | 0.262291 | 1.002973 | 0.936787 | n | refused |
| 27 | 8 | 0.244551 | 0.244741 | **0.999225** | 0.942766 | **Y** | **#22 promotion**, confidence 0.000775 |

### 2.3 Lifecycle event ticks

| Trace | #18 | first shadow | #23 | #22 | #24 | #25 stale | evict | motifs | D4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Registered seed 7** | 19 | 20 | 23 | 27 | — | — | — | 0 | 1 |
| **Control seed 0** | — | — | — | — | — | — | — | 0 | 0 |
| Bank probe seed 7 *(post-hoc, not a gate)* | 11 | 12 | 15 | 15 | 28 | 31 | 34 | 4 | — |
| Bank probe seed 0 *(post-hoc, not a gate)* | 10 | 11 | 14 | 14 | 24 | 31 | 34 | 1 | — |

The registered run exercises: #18 birth → shadow scoring → #23 survivor →
gate refusal (×4) → #22 sediment → #19 tagged deposit → #27 destructive
drain → #5 exit. It does **not** exercise #21→#24: see finding F1.

### 2.4 D3 / D4 / end state (seed 7)

* D3: `deposits 12 = resident 4 + drained 8 + evicted 0`; cohort classes
  `trail 6, residual 6, motif 0, scaffold 0`; 2,280 B; 0 ceiling
  evictions.
* D4: 1 sediment (lineage `cand-01`, t27); 35 zero-confidence #22 no-ops;
  `accepted 36 = consumed 36 + 0 + 0` (identity B); retained contents
  byte-identical (append-only, asserted per tick and by object identity).
* End state: D2 records `[]`, patches `[]`, patch bytes 0 (max 20,224 B),
  nursery empty; encoder bit-identical.

---

## 3. Findings

**F1 — the live-overlay half of the lifecycle is unreachable on a single
stream at the phase-3 motif threshold.** With `MOTIF_MIN_PERSISTENCE = 10`
and `MOTIF_MIN_COUNT = 4` untouched, the registered seed-7 run ingests on
6 ticks `{12, 19, 25, 28, 29, 30}` with a longest support streak of **1**
and largest body count **2**; seed 0 ingests on 5 ticks with streak 2 and
count 3. Motif cohorts produced: **0 on both**. Hence #21 always reads
`None`, #24 never emits, and #25-stale/evict do not fire in the registered
trace. This is a **regime mismatch**, not a bug: those thresholds were
derived from the 16-stream bank (16 observations/tick), and one stream
cannot supply 10 supports. The round does **not** tune them (that would be
the 9a failure class); the full lifecycle is demonstrated instead by (i) a
post-hoc 16-stream bank probe and (ii) an executor wiring test that
deposits one valid motif cohort and runs the production #21→#24→#25→evict
path (stale t32, evict t35, clean end state). **Round-12 decision
required:** derive single-stream motif persistence *and* count from the
single-stream ingest distribution, or rule that motif-gated adoption is a
bank-only mechanism.

**F2 — the seed-0 control is met vacuously.** Under the registered trigger
(9a's shape: spike above threshold, `tick ≥ fit_window = 16`, no active
live overlay) seed 0 spikes at frames 9, 10, 11, 12, 14 — all before the
fit window — so **no candidate is ever born**: 0 promotions, no D4 change,
ADAPTATION has n = 0 (no shadow-scored flipped rows) and is therefore not
measurable. The pre-registered "zero promotions" expectation is satisfied,
but it was not *tested* on this seed. A post-hoc probe that removes the
fit-window guard produces a seed-0 candidate at t20 whose final residence
ratio is **1.186745** (harmful, refused) — consistent with the old
rejection's verdict, but that is post-hoc. **Round-12 decision:** design
the control so it exercises the refusal branch (e.g. a scenario whose
spikes land after the fit window, or a control seed chosen for that
property).

**F3 — the promoted candidate is a knife-edge pass.** The seed-7 candidate
was harmful (cumulative ratio > 1) on seven of its eight shadow ticks and
crossed below 1 only at t27 (0.999225, confidence 0.000775). The gate
behaved exactly as ruled — refuse while harmful, promote on a cumulative
pass — but the promoted patch is effectively **neutral**, not beneficial.
Two aggregation conventions are in play and they disagree materially:
ADAPTATION is the **mean of per-tick ratios** (0.9555) per the
pre-registered wording, while `qualified` uses the **ratio of means**
(0.9992). Neither flips the verdict here; the inconsistency should be
pinned in Round 12.

**F4 — ADMISSION is structural under the ruled design.** Because the #22
body is derived from the same survivor body the gate acts on, a harmful
promotion can now only happen through a wiring defect; ADMISSION is a
wiring invariant, not an empirical policy test. Its empirical content
moved to "was a harmful survivor surfaced and refused" (yes, four ticks).
**Round-12 suggestion:** measure a *post-promotion, out-of-sample* rate
for the sedimented candidate to test the policy rather than the wire.

**F5 — the out-of-sample evidence convention moves the ADAPTATION
verdict.** The birth tick's transition is in-sample (the fit saw it) and
cannot be carried by the store (annotation runs before the #18 commit), so
shadow evidence is defined out-of-sample, which also keeps the organ's and
the store's evidence identical (asserted equal on every row). Under the
literal alternative (counting the in-sample birth transition) the seed-7
ADAPTATION would be **1.0077 → FAIL**. The convention was introduced for
the structural reason above, not for the verdict; the Lead ratified it on
those grounds. **The user can overrule it, and the consequence is the
FAIL.**

**F6 — a Wave-2 regression found in Lead review.** `DryRun._derive` had
been made to force `#3` to `None`, which silently dropped the `FrameBody`
carrier and made the physics source's #3 dispatch dead code (a Round-10
behaviour regression). Fixed in review (one line); the source is asked for
#3 again, the synthetic source still returns `None` for it, so plumbing is
unchanged and the physics frame carrier is restored.

**F7 — ruling (F)'s mass-balance formula is not a conservation law.**
`in#19 = out#20/#21/#26 reads + drained#27 + Δresident` double counts while
reads are non-destructive. The implemented and asserted identity is
`deposits = resident + drained + evicted`; the ruling's intended account
is reported separately as an issuance account (promoted /
resident_unpromoted / drained_unpromoted) plus `reads_by_kind`. Both are
in `mass_balance()`; neither is silently substituted for the other.

**F8 — `StructuralMotifBody.drift` is a placeholder.** The D3 cohort
record carries no drift field, so the #21 read body reports `drift = 0.0`.
Flagged rather than presented as derived; it should be carried by the
cohort in Round 12 if the ratchet reads it.

---

## 4. Round-12 adjudication list

1. Single-stream motif persistence **and** count (F1) — derive from the
   single-stream ingest distribution, or rule the adoption bank-only.
2. Control design so the rejection branch is exercised on the registered
   configuration (F2).
3. Pin one aggregation convention for ADAPTATION vs `qualified` (F3).
4. Decide whether ADMISSION should stay a wiring invariant or become a
   post-promotion out-of-sample measure (F4).
5. Ratify or overrule the out-of-sample shadow convention (F5).
6. Carry drift on the D3 cohort or retire the field (F8).
7. #13 remains blocked; #16 is now derived. The #13/#16 provenance memo's
   coverage arithmetic (14/26, 17/26) remains stale and must be re-derived
   when #13 unblocks.

---

## 5. Honesty notes

**Proven (Lean, regenerated from the ledger; 76 theorems, zero `sorry`, zero
`axiom`, 2.3–3.4 s):** per-flow liveness, both conservation identities at
the master profile, boundedness, the master depth/sink/miss profile, the
per-flow D2 write pins (`d2_write_split`, `candidate_recorded_from_birth`),
geometry payload purity, the untouched `mat_*_policy` theorems, and the new
candidate-index lifecycle (boundedness, birth membership, one-tick
residence, distinctness, promotion-once).

**Tested, not proven:** the census fingerprint (derived at test time, with
a perturbation test proving derivation), INV-STAGE-ORDER minimality and
sufficiency, the lineage gates, the D3 mass balance and promote-once, the
residence-evidence arithmetic, the two gates on the registered trace, the
coverage arithmetic (10/16 non-read derived, 6/16 placeholders, 11/11
reads), and the encoder/live-predictor bit-identity assertions.

**Assumed:** store reads see pre-tick committed state; entity inputs are
injected at tick start; `RESIDENCE_N = 4` and the shadow out-of-sample
convention are pre-registered design choices; the seed-7 `fit_window = 16`
trigger shape is 9a's, kept deliberately; the bank probe is post-hoc and is
not evidence for the registered trace.

**Historical (not re-asserted, measured on the old 26-flow topology):**
every round 7–9a number — the 0.9079 validation, ADAPTATION 0.9666 /
ADMISSION 1.0461, sediment counts, patch bytes, rejection trace. The 9a
seed-0 rejection has no successor mechanism on the registered Round-11
configuration because no candidate is born there (F2), and the post-hoc
probe that suggests the same verdict is labelled as such.

**Measured this round, first time:** the 16-stream bank probe live-overlay
numbers (seed 7 promotion ratio 0.966339 at residence 4, live overlay born
t28, stale t31, evict t34, live ratio 1.416059 over its steered ticks, none
of which is flipped; seed 0 live ratio 0.919256 overall / 0.843852 on
flipped ticks) — post-hoc probe, not gates.
