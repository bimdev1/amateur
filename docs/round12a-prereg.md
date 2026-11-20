# Round 12a — pre-registration (written before any 12a measurement)

> Status: **binding constants frozen here, before the runs they govern.**
> Scope: physics-side, metric-side, test-side only. No change to the
> 27-flow ops table, the stage schedule, the store schemas, or Lean.
> The 12b design is pre-registered in `docs/phase4-5-design.md` §17 and is
> **not run** this round.

The three soft greens from Round 11 (out-of-sample convention, aggregation,
and the absent gain condition) are all **conventions or missing conditions**,
not data shortages. This document freezes the rulings that replace them.
Numbers are computed here from constants that already exist in the tree
(the phase-3 measured statistics, the capacity law, the ledger); no 12a
trace is read.

---

## 1. Out-of-sample shadow convention — ratified, re-grounded

The birth transition is **excluded** from the candidate's evidence. The
binding reason is **train/test separation, not storage**: the birth
transition is one of the transitions the patch was fitted on, so scoring on
it is scoring on training data. The Round-11 implementation note ("the
store's annotation runs before `#18` commits") is an implementation fact; it
is **not** the reason and must not become the reason if the store is later
changed.

Binding reporting rule: **both numbers are reported, always and labelled**:

| label | definition |
| --- | --- |
| `out_of_sample` (primary) | ratio-of-means over flipped ticks scored **after** the birth tick |
| `with_in_sample` (secondary) | the same statistic on the tick set that **adds** the birth-tick sample |

The with-in-sample sample is computed by the organ at birth on the shadow
copy and stored on the record as `in_sample` — never in `evidence`, never
seen by `#23`/`#22`, never used by the gain floor.

## 2. Aggregation — ratio-of-means primary, everywhere

**Primary (gate and metric):**

    R = sum_t adapted_surprise(t) / sum_t base_surprise(t)

over the declared tick set. This is the error-mass ratio; it weights a tick
by its error magnitude instead of amplifying near-zero denominators, and it
is the statistic `CandidateRecord.qualified` already uses
(`adapted_mean / base_mean` is the same quantity). Gate and metric are one
convention.

**Secondary (diagnostic only, always labelled):** `mean_t(adapted/base)`.

**Tick sets, declared explicitly in every summary** (never inferred):

* `ADAPTATION_SET` — ticks with `phase == "flipped"`, shadow-scored, and not
  the candidate's birth tick.
* `WITH_IN_SAMPLE_SET` — `ADAPTATION_SET ∪ {birth tick}` (only when the
  birth tick is itself in the flipped phase; otherwise the sets coincide
  and the summary says so).
* `RESIDENCE_SET` — every tick with at least one out-of-sample evidence
  sample (the residence-evidence curve). On the registered 36-tick
  scenario this set is a subset of the flipped phase, but that is a
  measured fact of the trace, not an assumption.

Both `R` values and both tick-set lists are emitted in the run summary.

## 3. MDL gain floor on promotion (NEW, binding)

### 3.1 The condition

A candidate is admissible for promotion iff

    kappa * H * dbar  >  32 * K

where `dbar` is the candidate's measured mean per-tick error-mass reduction
over its **out-of-sample** evidence, `K` its stored parameter count,
`H` the amortisation horizon, and `kappa` the exchange constant derived
below. Equivalently: `dbar > 32K/(kappa*H)`. Strict inequality; an exactly
zero gain is refused.

### 3.2 Units convention (explicit)

The organ's `surprise` is a **mean squared latent prediction error**
(`Observation.surprise == mean(residual**2)`, latent-dimension squared per
dimension) — a *squared* distance, not a distance. `dbar` is therefore in
latent²-per-dimension per tick.

Conversion to bits: the model's code length for a latent residual `r` under
its own Gaussian predictive model with per-component variance `sigma²` is

    L = (d/2) log2(2 pi sigma²) + ||r||² / (2 sigma² ln 2)   bits

The patch changes the mean, not the model's declared variance, so only the
second term changes. With `s = ||r||²/d`:

    dL = (d / (2 sigma² ln 2)) * ds          =>   kappa = d / (2 sigma² ln 2)

### 3.3 Derivations of the constants

* `d = 64` — the phase-3 latent width (`Phase3Stats.latent_dim`).
* `sigma² = normal_mean = 0.04938631785656374` — the phase-3 measured
  **in-distribution** mean surprise, i.e. the model's own calibrated
  residual variance (`Phase3Stats.from_stats`; recorded fallback constant
  identical). This is the variance the model's code actually assumes; a
  model trained on normal data does not widen its likelihood because the
  dynamics changed. Round 11's `post_flip_mean = 0.28597303529750207` is
  reported as a sensitivity variant.
* `kappa = 64 / (2 * 0.04938631785656374 * ln 2) = 934.7982054975386`
  bits per (latent² per dim · tick). At the in-distribution operating
  point one tick of residual magnitude costs `d/(2 ln 2) = 46.2` bits.
  Sensitivity variant: `sigma² = post_flip_mean` gives
  `kappa = 161.43564465936234`.
* `H = 20` — the registered flipped-episode length (`POST_FLIP`), which is
  the only lifetime the registered scenario gives evidence for. Amortising
  beyond the episode is an assumption the evidence does not support.
  Sensitivity over `H` is reported in §3.5; the verdict does not depend on it.
* `32K` — the description cost at the tree-wide uniform 32-bit encoding
  (`amateur.physics.capacity`, same convention as the capacity law); `K` is
  `PredictorOverlay.param_count`.

### 3.4 Worked example (historical numbers, not a 12a measurement)

Seed-7 promotion candidate, Round 11: `K = 5,056` (rank 26), out-of-sample
`R = 0.999225` over `n = 8`, base error scale `b`:

    dbar = b * (1 - R)

with `b = 0.28597303529750207` (the recorded phase-3 post-flip mean, used
here only as the order of magnitude; the 12a run recomputes `b` from its
own trace):

    dbar      = 2.2163e-4
    gain(20)  = 934.798 * 20 * 2.2163e-4 = 4.14 bits
    cost      = 32 * 5056               = 161,792 bits
    verdict   = REFUSED (cost is 39,000x the claimed gain)

Break-even horizon at that gain: `H* = 161,792 / (934.798 * 2.2163e-4)`
= **780,930 ticks** (21,700 registered runs). At `H = 20` the largest
admissible patch is `K < 0.13` parameters — i.e. none.

### 3.5 Pre-registered expectations

* **The current evidence regime cannot justify patches of this size; every
  candidate in the registered, control, and null runs is expected to be
  refused.** That is the finding, not a regression.
* The falsifier: applied retroactively to the seed-7 t27 candidate the
  condition must **refuse** it. If it admits it, it is not a gain floor and
  the round reports that instead.
* Refusal classes are reported separately: `harmful` (`dbar <= 0`) and
  `uneconomic` (`0 < gain <= cost`).
* Sensitivity table: gain at `H in {4, 8, 16, 20, 36, 100, 1000, 10^4,
  10^5}` and at both `sigma²` variants, on the trace's own numbers.

**Adjudication status: RATIFIED (user, 2026-11-20, before the 12a
measurement runs).** `kappa = 934.7982054975386`, `H = 20`, cost `32K`,
strict inequality — exactly as pre-registered; the ratified form is the
Gaussian-NLL exchange at the model's in-distribution variance, amortised
over the registered episode. No constant was changed after ratification;
a changed constant would mean a re-run, not a re-reading.

## 4. Trigger — evidence sufficiency replaces the absolute tick guard

**Old:** `tick >= fit_window (16)` — arbitrary; it is why seed 0 never
birthed (spikes at frames 9–14).

**New:** `N >= N_MIN`, where `N` is the number of real transitions the
fit will actually consume (`N = tick`, capped by `fit_window=16`), and the
capacity is derived from **that** `N` (`K <= gamma*N*D`, gamma=5, D=64,
unchanged). `N_MIN = 2`: the smallest evidence with at least one transition
to fit and at least one out-of-sample transition to test — one transition
has no held-out sample, and the capacity bound already admits a rank-1
patch at `N = 2` (`K(1)=256 <= 640`).

Pre-registered consequences to verify:

* seed 0 births at its first spike (frame 9) with a **small** candidate
  (`N = 9` => `K <= 2,880`, rank 14, `K = 2,752` vs 5,056 at N=16) and it
  is **refused** — the rejection-before-birth branch exercised on the
  registered configuration.
* **seed 7's registered birth moves from t19 to t12** (its first spike;
  the old guard skipped t12 because it was below the magic 16). That is
  the intended consequence of removing the arbitrary guard, not a
  regression: the candidate is born at the onset of the perturbation with
  `N = 12` (`K <= 3,840`, rank 19, `K = 3,712`) and its residence evidence
  then spans the flipped phase and the recovery phase. The Round-11 trace
  (birth t19, K=5056) is **not** the 12a registered trace; it is
  reproduced separately, by a clearly labelled historical diagnostic
  (`trigger="round11"`, i.e. the old `tick >= fit_window` guard, floor
  ablated), for the §3.5 retroactive falsifier.
* the end state of the registered run is no longer "nursery empty": with
  the floor refusing every candidate, the (bounded, non-steering)
  candidate remains resident at tick 36.  Reported as measured state.

## 5. Motif thresholds — derived per regime, not tuned

The bank constants (`MOTIF_MIN_COUNT = 4`, `MOTIF_MIN_PERSISTENCE = 10`)
were calibrated on a multi-stream bank and are unreachable on one stream.
Replacing them with a declared criterion:

**Criterion M.** The thresholds must keep the **null false-declaration
probability over the registered run at or below the same 1 % that already
defines the debris/spike threshold** (`debris_threshold` is normal q99).
Null process: an independent 1 %/tick ingest (by construction of q99), over
the registered `T = 36` ticks, with the implemented streak rule
(consecutive supports may be at most `MOTIF_STREAK_GAP = 2` ticks apart).
`(c, p)` = the smallest pair in `(c+p, -p)` order satisfying
`P_false(c,p) <= 0.01` and `c >= 2` (a single observation is not a repeated
structure).

Exact enumeration of the null process (dynamic programming over
`(count, streak, gap)`, all 36 ticks; the body receives every ingest,
which upper-bounds the true false rate):

| c | p | P_false |
| --- | --- | --- |
| 1 | 1 | 30.359 % |
| 2 | 1 | 5.035 % |
| 2 | 2 | **0.978 %** |
| 3 | 1 | 0.558 % |
| 3 | 2 | 0.146 % |

**Derived: `MOTIF_MIN_COUNT = 2`, `MOTIF_MIN_PERSISTENCE = 2`**
(P_false = 0.978 % <= 1 %). This is a *statistical* derivation from the
normal-regime ingest process; it is not fitted to any seed's observed
support pattern.

**Amendment note (implementation review, before the 12a run outputs; no
constant or threshold changed).** The table above was computed with the
looser reading of `MOTIF_STREAK_GAP` ("intervening": a support continues a
streak iff at most `gap` *missed ticks* lie between, i.e. distance
`<= gap+1`). The implemented `_Body.absorb` uses the literal distance test
(`t - last_support <= gap`), which is **stricter**: the implemented
`P_false(2,2) = 0.6718 %`. The pre-registered `0.978 %` is therefore a
conservative **upper bound**; criterion M holds under both readings and the
derived pair is `(2,2)` either way. The pre-registered number stands as the
reported bound — it is not rewritten to the smaller measured value.

Pre-registered expectations:
* the registered seed 7 (longest streak 1) still produces **no motif** and
  therefore no live-overlay event — the single-flip trace is unchanged;
* seed 0 (streak 2, count 3) **does** reach the motif path; it is the
  control seed, and this is reported as a measured consequence of the
  derivation, not as a success;
* the thresholds are flagged as sitting at the 1 % boundary (0.978 %), so
  any firing on the registered trace would be weak evidence; the live half
  is still demonstrated end-to-end only by the 16-stream probe and by the
  12b repeated-flip scenario.

## 6. Noise floor — null controls (pre-registered)

Four null candidates, each run through the **identical** pipeline (same
trigger, organ, executor, evidence, gate):

| null | construction | expectation |
| --- | --- | --- |
| `zero` | patch factors and bias exactly zero; neutral by construction | 0 promotions |
| `random` | random u, v, bias at the rank the capacity admits at N | 0 promotions |
| `shuffled` | the standard fit with targets permuted inside the fit window | 0 promotions |
| `no_flip` | the registered environment with no flip at all | 0 promotions (report if vacuous) |

Pre-registered: **if any null promotes, the gate is crossing noise and the
seed-7 pass has no content; report that and stop.** Secondary diagnostic
(clearly labelled, not a gate): the same four nulls with the gain floor
ablated, to show whether the *pre-floor* gate would have admitted noise. If
a null promotes there while the real candidate also promotes, the old pass
was noise.

## 7. Housekeeping rulings

* **Mass balance:** the conservation identity (`deposits = resident +
  drained + evicted`, reads non-destructive) is canonical; the ruling-F
  formula is kept as a separately named **issuance account**. Both names,
  never one instead of the other.
* **`StructuralMotifBody.drift = 0.0`:** the field is deleted from the
  payload surface (a constant zero on a measurement surface is the Round-8
  gate-stub class). The organ's `StructuralMotif.drift` stays — it is
  derived (`||centroid - streak_first_centroid||`) and is used by the motif
  gate. Deleting the payload field touches no store schema and no Lean
  table (the Lean mirror carries the op table, not payload fields).
* **Coverage audit:** the six placeholders `{1,2,4,5,9,13}` are classified
  as *blocked on an unimplemented node* vs *not yet derived*, with the
  evidence for each. This round is an audit, not a schema expansion:
  deriving a placeholder requires a body type, and body types are a
  schema change, which 12a forbids.

## 8. Round 12b — pre-registered in the design doc, NOT run

`docs/phase4-5-design.md` §17 carries the 12b pre-registration, written
before any 12a number exists: repeated flips with dwell (persistence
reachable), `RESIDENCE_N in {4,8,16,32}`, a seed sweep, null controls in
every sweep, ratio-of-means primary with both tick sets declared, and a
pre-registered decision rule for "signal vs random walk".
