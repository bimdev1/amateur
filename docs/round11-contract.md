# Round 11 contract — rulings A–G, then LeWM physics

**Status:** Lead-owned implementation contract. Binding on this round's
writers. It transcribes the user's rulings A–G (2026-11-20) into exact
schema and interface decisions so the schema, executor, organ, and Lean
work can proceed on disjoint files without re-litigating the design.

**Baseline:** HEAD `27898c0`, suite 355 tests OK (1 expected failure),
Lean byte-identical, tree clean.

**Scope discipline:** Round 11 implements A–G and then runs the physics.
It does **not** recover 9b trigger-gating, does not start #13, does not
change reject-with-charge, does not relax the theorem discipline, does not
unfreeze the encoder, and does not re-assert any round 7–9a measurement.

**One deliberate deviation, flagged here and in the final report:** ruling
(F) says the D3 reads (#20/#21/#26) are non-destructive *and* asks for a
mass balance of the form `in#19 = out#20/#21/#26 reads + drained#27 +
Δresident`. Those two clauses cannot both be a conservation law: a
non-destructive read removes no mass, so adding it to the out side double
counts. This contract implements the conservation-correct identity
(`deposits = resident + drained + evicted`) and reports the ruling's
intended identity as a *cohort issuance* account (which cohort was issued
to which reader, and which was promoted). Both are asserted; neither is
silently substituted for the other.

---

## 0. Ruling → work item

| Ruling | Work item | Primary file(s) |
| --- | --- | --- |
| A | Per-flow kind pins; `EmergentCrystalBody` promoted to the D2 candidate kind; `MatBoard` split into **candidate index** + **overlay index**; candidate-index lifecycle theorems in Lean | `payloads.py`, `tools/gen_lean.py`, `formal/DryRun.lean` |
| B | `DissipatedEntropyBody`; #27 destructive drain; D3 mass balance + byte ceiling that forces drainage | `payloads.py`, `dryrun.py`, `physics/run.py` |
| C | #23 is a precondition for #22; #24 requires #21 lineage; #21 gains a consumer | `dryrun.py`, `physics/run.py` |
| D | Lexicon adopted; "Surviving Crystal" is a read-time selection, no third status; three kinds / three stores; #2 is not a D2 crystal | `docs/TESTING.md`, tests |
| E | Candidate carries the patch as **substance**, shadow-scored every resident tick, never steering; live patches only via #24; #8 reads the overlay index only | `physics/organ.py`, `physics/run.py` |
| F | `#19` tagged composite with four provenance-stamped cohort classes; per-edge read filters; INV-PROMOTE-ONCE; #27 drains residual + expired | `payloads.py`, `dryrun.py` |
| G | Census = derived fingerprint (26/24, 12/11/15/4); `INV-CYCLE-DELAY` → `INV-STAGE-ORDER` | `tests/test_invariants.py`, `docs/TESTING.md`, `dryrun.py` |

Plus: derive #16 this round (source is P3, no D1 dependency); #13 stays
blocked.

---

## 1. Payload schema (`amateur/payloads.py`)

### 1.1 D2 write kinds are pinned **per flow**, never per store

```python
# Write channels: exactly one exact type each, no None. This is the ruling's
# point — a promotion bug becomes untypeable, not merely rejected.
PAYLOAD_SCHEMA[18] == frozenset({EmergentCrystalBody})   # nursery birth
PAYLOAD_SCHEMA[24] == frozenset({MatOverlay})            # ratcheted overlay
PAYLOAD_SCHEMA[22] == frozenset({BiasPrimitive})         # D4 sediment
PAYLOAD_SCHEMA[19] == frozenset({DigestedDebrisBody})    # D3 deposit
# Read channels: the exact pinned body type OR None ("the store had nothing
# to hand over this tick"). None keeps the A2 provenance rule intact: a read
# with a body must carry the stored record's provenance with tick_born <
# emit_tick; a read with None carries self-origin (fid, tick). A sentinel
# body on a read would force a sentinel-provenance exception to that gate,
# i.e. it would weaken the anti-mint rule to buy type purity that the
# ruling does not ask for (the ruling is about write kinds).
PAYLOAD_SCHEMA[8]  == frozenset({MatOverlay, None})            # active ratcheted only
PAYLOAD_SCHEMA[23] == frozenset({SurvivingCrystalBody, None})  # survivor + evidence
PAYLOAD_SCHEMA[25] == frozenset({MatOverlay, None})            # stale hand-off
PAYLOAD_SCHEMA[20] == frozenset({ScaffoldsBody, None})         # scaffold cohorts
PAYLOAD_SCHEMA[21] == frozenset({StructuralMotifBody, None})   # motif cohort
PAYLOAD_SCHEMA[26] == frozenset({AvoidanceTrailsBody, None})   # trail cohorts
PAYLOAD_SCHEMA[27] == frozenset({DissipatedEntropyBody, None}) # residual + expired
PAYLOAD_SCHEMA[7]  == frozenset({BiasPrimitive, None})         # D4 read
PAYLOAD_SCHEMA[16] == frozenset({CostErrorBody, None})         # derived this round
PAYLOAD_SCHEMA[11] == PAYLOAD_SCHEMA[13] == ... == frozenset({None})  # D1 frozen
```

The no-op sentinels of §1.6 therefore exist for **write** channels (fixed
cadence with no event) and for tests; a read emits `None` when empty.


### 1.2 D2 candidate record (`EmergentCrystalBody`, rewriting the old #18 type)

```python
@dataclass(frozen=True)
class EmergentCrystalBody:
    candidate_id: str          # non-empty; the nursery identity
    residual: tuple[float, ...]  # PAYLOAD_LATENT_DIM
    surprise: float
    born_tick: int             # == the #18 emission tick
```

This is the *only* type #18 may carry and the only type stored in the
candidate index. It is no longer unreachable (`REAL_BODIES[18]` is
reachable by construction now).

### 1.3 The #23 survivor read body

```python
@dataclass(frozen=True)
class SurvivingCrystalBody:
    candidate_id: str
    born_tick: int
    residence_ticks: int       # shadow-scored ticks since birth
    adapted_mean: float        # mean shadow adapted surprise
    base_mean: float           # mean shadow base surprise
    ratio: float               # adapted_mean / base_mean (finite, base > 0)
    qualified: bool            # ratio < 1.0 and residence_ticks >= RESIDENCE_N
```

`qualified` is the *selection* ("survives"), computed at read time. There
is no stored `surviving` kind and no third status (D).

### 1.4 D2 ratcheted record: `MatOverlay` gains lineage

```python
@dataclass(frozen=True)
class MatOverlay:
    epoch: int
    overlay_id: str
    status: str                  # "active" | "stale"
    motif_lineage: str | None = None   # #24: required non-None; the #21 motif id
```

The `candidate` status is **removed** (A: candidates are a distinct kind in
their own index). Seed record `MAT_SEED = ("ovr-0", 0)` has
`motif_lineage=None`.

### 1.5 D4 body: `BiasPrimitive` gains lineage

```python
@dataclass(frozen=True)
class BiasPrimitive:
    bias_primitive: str
    confidence: float            # [0, 1]
    candidate_lineage: str | None = None   # #22: required non-None = #23 candidate id
```

Trajectory/delta/context remain unrepresentable: D4's whitelist is
unchanged in kind, only the lineage string is added.

### 1.6 No-op sentinels (exact-type, content-free) — **write** channels and tests

Reads signal "nothing there" with `None` (§1.1); write channels need a real
body to keep the fixed cadence, so they use these:

| Flow | Sentinel |
| --- | --- |
| #18 | `noop_crystal()` → `candidate_id == NO_CANDIDATE_ID` |
| #24 | `noop_overlay()` → `overlay_id == NOOP_OVERLAY_ID`, `motif_lineage=None` |
| #19 | `noop_debris()` → `debris_id == NO_DEBRIS_ID`, `cohorts == ()` |
| #22 | zero-confidence `BiasPrimitive("ratchet-gate-blocked", 0.0, None)` (Q4 unchanged) |

`noop_survivor`/`noop_motif`/`noop_scaffolds`/`noop_trails`/`noop_entropy`
remain defined for read-channel tests and for any consumer that prefers a
typed no-event over `None`; the executor emits `None` on reads (the A2
provenance rule then applies unchanged).

### 1.7 D3: tagged composite (`DigestedDebrisBody`) + cohort model (F)

```python
COHORT_KINDS = ("motif", "scaffold", "trail", "residual")

@dataclass(frozen=True)
class DebrisCohort:
    cohort_id: str
    kind: str                  # one of COHORT_KINDS
    centroid: tuple[float, ...]  # PAYLOAD_LATENT_DIM
    count: int
    spread: float
    first_tick: int
    last_tick: int
    peak_surprise: float
    provenance: object         # the exact Provenance of the depositing #19

@dataclass(frozen=True)
class DigestedDebrisBody:      # #19 P5 -> D3
    debris_id: str
    cohorts: tuple[DebrisCohort, ...]

@dataclass(frozen=True)
class ScaffoldsBody:           # #20 D3 -> P4: scaffold cohorts only
    cohorts: tuple[DebrisCohort, ...]
    no_event: bool = False

@dataclass(frozen=True)
class StructuralMotifBody:     # #21 D3 -> P6: the motif cohort
    motif_id: str
    debris_id: str
    centroid: tuple[float, ...]
    count: int
    spread: float
    persistence: int
    drift: float
    first_tick: int
    last_tick: int

@dataclass(frozen=True)
class AvoidanceTrailsBody:     # #26 D3 -> P2: trail cohorts only
    cohorts: tuple[DebrisCohort, ...]
    no_event: bool = False

@dataclass(frozen=True)
class DissipatedEntropyBody:   # #27 D3 -> P1: residual + expired, destructive
    cohort_ids: tuple[str, ...]
    magnitudes: tuple[float, ...]   # same length as cohort_ids
    tick: int
```

`DebrisCohort.provenance` is validated structurally (it must expose
`source_fid: int` and `tick_born: int`) to avoid the
`payloads -> provenance -> payloads` import cycle; the executor's
`_derive` passes real `Provenance` objects.

`REAL_BODIES` gains 20/23/26/27 and keeps 16. `_schema()` maps every store
channel to its pinned exact type with no `None`.

### 1.8 D3 pool (`DebrisPool`) — cohorts, filters, drain, mass balance

Storage is **cohort-level**: `deposit(body, provenance, tick)` splits the
body into cohort entries (`DebrisCohort` objects), each retaining the
depositing #19's provenance.

```python
DEBRIS_CAPACITY = 32         # cohort-level: 32 cohorts ~= the old 8 bodies x 4 kinds
DEBRIS_MAX_BYTES = 65_536    # hard byte ceiling
DEBRIS_TTL = 12              # ticks; older cohorts are "expired"
NURSERY_CAPACITY = 4         # D2 candidate index
RESIDENCE_N = 4              # minimum shadow-scored ticks before #23 may surface
```

API:

- `read(kind: str) -> tuple[DebrisCohort, ...]` — non-destructive,
  oldest-first; `kind ∈ COHORT_KINDS`. Never mutates.
- `best_motif() -> DebrisCohort | None` — highest `(count, persistence)`
  motif cohort, oldest-first tie-break.
- `drainable(now: int) -> tuple[DebrisCohort, ...]` — every `residual`
  cohort plus every cohort with `now - last_tick > DEBRIS_TTL`
  ("expired"), oldest-first. This is #27's selection.
- `drain(cohort_ids, tick) -> tuple[DebrisCohort, ...]` — **destructive**
  (#27 only): removes exactly the named cohorts, records them in
  `drains`, refuses unknown ids loudly. Idempotent by construction: a
  second drain of the same id is a loud failure (the read is destructive,
  so it cannot happen twice).
- `promote(cohort_id, tick) -> None` — marks a cohort as promoted for its
  consumer; **INV-PROMOTE-ONCE**: promoting the same id twice raises.
  Non-destructive.
- `deposit(...)` — appends cohorts; evicts FIFO only for **capacity**
  overflow. On byte-ceiling overflow the pool raises
  `DebrisCeilingExceeded` unless `allow_ceiling_evict=True` (the executor
  always tries drainage first; the eviction is the recorded backstop).
- `mass_balance() -> dict` — `{"deposits", "resident", "drained",
  "evicted", "ceiling_evicted", "promoted", "reads_by_kind"}` and the
  asserted identity `deposits == resident + drained + evicted`; the
  issuance account `promoted + resident_unpromoted + drained_unpromoted`
  is reported separately (§0).
- `over_ceiling() -> bool`, `byte_size()`, `bodies()`, `__len__`.

### 1.9 `MatBoard`: two indices (A, D, E)

The **overlay index** keeps the existing record policy **verbatim**
(`mat_*_policy` applies unchanged): `_residents` deque, seed
`MAT_SEED`, `birth` (append), `revalidate` (pop head, append at new
epoch — the #25 → #24 pairing), `evict_if_needed` (oldest-first past
`MAT_CAPACITY`), `head`, `residents`, `resident_ids`, `evict_record`,
`_patches`, `attach_patch`, `patch`, `patch_bytes`, `patch_params`,
`enforce_patch_ceiling`, `assert_patch_bounds` — all unchanged in
semantics. The only change: `birth`/`revalidate` still store
`MatOverlay`, and `attach_patch` still requires the overlay to be
resident.

The **candidate index** is new and separate:

```python
_nursery: deque[tuple[str, int]]        # (candidate_id, born_tick), FIFO
_candidate_patches: dict[str, object]   # (E): candidate substance
_evidence: dict[str, list[tuple[int, float, float]]]  # tick, adapted, base
_nursery_evictions: list[tuple[str, int, int, str]]   # id, born, tick, cause
```

API (all mutations raise loudly on illegal input; no silent no-ops):

- `nursery_birth(body: EmergentCrystalBody, tick: int) -> None` — append;
  duplicate id raises; capacity overflow evicts oldest (cause
  `"capacity"`).
- `nursery_observe(candidate_id, adapted, base, tick) -> None` — appends
  one shadow measurement to a **resident** candidate; unknown id raises;
  never creates or removes records (this is bookkeeping, not a flow).
- `nursery_records() -> tuple[tuple[EmergentCrystalBody, ...]]` — the
  resident records oldest-first (rebuilt from `_nursery` + stored bodies).
- `nursery_evidence(candidate_id) -> tuple[tuple[int,float,float], ...]`.
- `nursery_survivor(now: int) -> SurvivingCrystalBody | None` — the
  selection, and it must **not** pre-filter on `qualified`: among residents
  with `residence_ticks >= RESIDENCE_N`, the lowest ratio (tie-break oldest
  birth); `qualified = (ratio < 1.0)` is carried as a flag, not applied as a
  filter. `None` iff no resident has enough residence. Rationale: if #23
  only ever surfaced qualified candidates, #22's gate and the ADMISSION
  metric would be vacuously true — the round's whole question is whether the
  *gate* refuses a surfaced harmful candidate, so the harmful candidate must
  be observable as a surfaced (and refused) survivor in the trace.
- `nursery_promote(candidate_id, tick) -> EmergentCrystalBody` — removes
  the candidate **and its patch atomically** on #22 lineage consumption;
  records the promotion; refuses an unknown or already-promoted id.
- `nursery_evict_record(candidate_id, tick, cause) -> ...` — the
  organ-owned prune path (Q1-A), removes record + patch together.
- `nursery_ids()`, `nursery_patch(id)`, `nursery_patch_bytes()`.
- `patch_bytes()` **includes** candidate patches (E: D2 byte accounting
  covers candidate substance); `MAT_PATCH_TOTAL_BYTES` therefore bounds
  overlays + candidates together. `assert_patch_bounds(require_complete)`
  checks: overlay patches ⊆ overlay residents; candidate patches ⊆
  nursery residents; per-patch and total byte bounds. In physics mode
  `require_complete` requires a patch for every physics-owned resident of
  either index; the seed overlay is exempt (its patch is not required —
  keep the current physics behaviour, which seeds `seed_board=False`).

Per-flow pin enforcement lives in one place: `require_payload(fid, body)`
already checks `type(body) in PAYLOAD_SCHEMA[fid]`; the executor calls it
on the derived body **and** the store calls it before committing (so a
candidate cannot be born through the overlay path or vice versa).

---

## 2. Executor contract (`amateur/dryrun.py`)

### 2.1 Derivation pass (`_derive`), per flow

| fid | Kind | Body derivation (executor unless stated) | Provenance |
| --- | --- | --- | --- |
| 1 | input | injected placeholder `None` | `(1, tick)` |
| 3 | input (source) | source body — physics: `FrameBody`; synthetic: `None` | `(3, tick)` |
| 2, 4, 5 | sink | placeholder `None` | `(fid, tick)` |
| 6, 10, 12, 17, 18, 22, 24 | derived | `source.derive_body` | `(fid, tick)` |
| 9, 13 | placeholder | `None` | `(fid, tick)` |
| 16 | **derived (new)** | `source.derive_body` → `CostErrorBody` | `(16, tick)` |
| 19 | derived | `source.derive_body` → tagged composite | `(19, tick)` |
| 7 | D4 read | latest stored `BiasPrimitive` (identity, not a copy) or `None` | stored provenance / `(7, tick)` when `None` |
| 8 | D2 overlay read | head overlay, **ratcheted/active only**, or `None` | stored provenance |
| 25 | D2 stale read | head overlay labelled `"stale"` (hand-off), or `None` | stored provenance |
| 23 | D2 nursery read | `board.nursery_survivor(tick)` (no `qualified` pre-filter) or `None` | the surviving record's `#18` provenance |
| 20 | D3 read | `ScaffoldsBody(board.debris.read("scaffold"))` or `None` when empty | newest contributing `#19` provenance |
| 21 | D3 read | `StructuralMotifBody` from `board.debris.best_motif()` or `None` | that cohort's `#19` provenance |
| 26 | D3 read | `AvoidanceTrailsBody(board.debris.read("trail"))` or `None` when empty | newest contributing `#19` provenance |
| 27 | D3 read | `DissipatedEntropyBody(board.debris.drainable(tick))` or `None` when nothing is drainable | `(27, tick)` + per-cohort lineage in the body |
| 11, 14, 15 | D1 reads | `None` placeholders (D1 has no executor writer; #13 blocked) | `(fid, tick)` |

`PER_TICK_INJECTIONS` is unchanged (all 11 store reads + #1/#3).

### 2.2 #27 is the one destructive read (B)

The drain happens **at P1's consume of #27**, in the same tick as the
read: after `_consume(27)` succeeds, the executor calls
`board.debris.drain(body.cohort_ids, tick)`. If the #27 mailbox is empty
(miss — impossible under the canonical schedule, possible under
adversarial multiplicity) no drain occurs. Refused copies never drain.
The executor asserts `drained == body.cohort_ids` and that each drained
cohort's `provenance` is a stored `#19` provenance.

### 2.3 Absorb pass (`_absorb_writes`), per write

- `#18` → `board.nursery_birth(body, tick)`; the source's `on_commit`
  attaches the candidate patch (§3.2). Noop sentinel: nothing.
- `#24` → if `overlay_id` is a resident overlay id, `revalidate`, else
  `birth`; store the record provenance; `on_commit` attaches the live
  patch. Noop: nothing.
- `#19` → `board.debris.deposit(body, provenance, tick)`; ceiling handling
  per §2.4.
- `#22` → **lineage gate (C)**: when `mode == "physics"`,
  `body.candidate_lineage` must be non-`None`, must equal a candidate id
  whose `#23` read was *consumed this tick*, and that candidate must still
  be resident; then `nursery_promote(lineage, tick)` **before** the
  sediment. A confidence-0 sentinel is the Q4 no-op: no promotion, no D4
  change, identity B holds (`nursery` untouched).
- `#24` **lineage gate (C)**: when `mode == "physics"`,
  `body.motif_lineage` must be non-`None` and must equal the motif id of
  the `#21` read consumed this tick. A noop overlay has no lineage and
  skips the gate.
- Both gates raise `AssertionError` (wiring defect), never a charge.

### 2.4 D3 byte ceiling forces drainage (B)

After the #19 deposit, the executor drains via #27 **in the same tick** if
`over_ceiling()`: `drainable(tick)` ids, then `drain`. If the pool is
still over the ceiling with nothing drainable, it calls
`deposit(..., allow_ceiling_evict=True)` semantics through
`evict_to_ceiling(tick)` (FIFO oldest, counted as `ceiling_evicted`), and
records the fact in `report.ceiling_evictions` so the report can state it.
The mass balance is asserted after every tick.

### 2.5 Shadow evidence annotation (E)

Before `_derive`, the executor pulls
`source.shadow_evidence(tick)` (a read-only mapping
`candidate_id -> (adapted, base)`) if the source exposes it, and calls
`board.nursery_observe(...)` for each id. The source never receives the
board back from this call; it is one-way, and an unknown candidate id is a
loud failure (an organ/store divergence, i.e. a bug).

### 2.6 `#16` derived (P3 source, no D1 dependency)

`CostErrorBody(cost, error)` where at P3:

- `error` = this tick's prediction error/surprise (the same value #12
  carries), and
- `cost` = the L2 norm of this tick's latent state delta (the same vector
  #12 carries).

Test identity: `#16.error == #12.surprise` and
`#16.cost == ||#12.delta||₂`, exactly, same tick. This makes #16 a real
derived body whose falsifier is a constant dressed as evidence.

### 2.7 Coverage ledger

`PLACEHOLDER_FLOWS = frozenset({1, 2, 4, 5, 9, 13})` (16 is removed).
`SOURCE_FLOWS = NON_READ_FLOWS - PLACEHOLDER_FLOWS` = 10 flows. Physics
coverage is asserted from the ledger, not hard-coded: 10/16 non-read
flows derived, 11/11 store reads executor-derived, 6/16 placeholders.

### 2.8 Renames (G)

`dryrun.CYCLE_FLOWS` → `STAGE_BACKWARD_FLOWS` (same value `(10, 17)`),
justified by the stage-backward edge test, not by a cycle census. Update
`tools/gen_lean.py` imports and every test/docstring reference. Lean
theorem names are left alone (they prove the warm-up profile of the
delayed flows, which is still true); the invariant id changes, not the
theorems.

---

## 3. Organ contract (`amateur/physics/organ.py`, `run.py`)

### 3.1 Candidate table (E)

- `CandidateTable`: `candidate_id -> CandidateRecord(overlay: PredictorOverlay, born_tick, evidence: list[(tick, adapted, base)])`, capacity `NURSERY_CAPACITY`, FIFO eviction.
- `birth_candidate(residual, surprise, tick) -> CandidateRecord`: fits the
  shadow patch **from the same evidence window as the old proposal**
  (fit on the last `fit_window` real transitions ending at `tick`), under
  the unchanged capacity bound (`K ≤ γND`, `gamma=5.0`) and the same
  `FitRecord`/refusal reasons. **No live application.**
- `shadow_scores(tick) -> dict[str, (adapted, base)]`: for each resident
  candidate, apply its patch to a **shadow copy** of the predictor and
  score the current transition against the base predictor. Never touches
  `working_model` (asserted by encoder and predictor bit-identity checks).
- `retire_candidate(id, tick)`, `promote_candidate(id, tick)` (called from
  the executor's #22 lineage consumption via the store, mirrored
  organ-side in `on_commit`), `prune_candidates(resident_ids)`.

### 3.2 Patch ownership

The organ owns the operative patch; the store owns the accounting mirror
(existing Q1-A pattern). `on_commit(18)` → `board.attach_candidate_patch`
with the organ's patch object; `on_commit(24)` → `board.attach_patch`
(live). `prune` aligns both indices and both patch tables.

### 3.3 Motif path (#21 → #24)

- The digester now emits **tagged cohorts** in its `DigestResult`
  (`motif`, `scaffold`, `trail`, `residual`) instead of one debris body;
  the organ/source packs one `DigestedDebrisBody` per tick for #19.
- P6 reads the motif cohort via #21 (store read, executor-derived from
  D3). The organ's ratchet decision consumes `#21`'s motif: it fits/births
  the overlay from that motif cohort (same `_fit_overlay` machinery,
  capacity bound unchanged) and emits #24 with
  `motif_lineage = motif_id`.
- Staleness/eviction/cooldown behaviour is unchanged; evicted overlays
  stop affecting prediction, record+patch removed atomically.

### 3.4 #22 gate from residence evidence (C, E)

`_gate` is derived from the **#23 survivor body** (executor store read) —
i.e. it reads candidate evidence, not live-overlay evidence:
pass iff a survivor body was consumed this tick and `qualified`; confidence
= `clamp(1 - ratio, 0, 1)`; `candidate_lineage = candidate_id`. Otherwise
the zero-confidence sentinel. The old `_accrue_gate_evidence` path stays
for the live-overlay D4 demonstration? **No** — per (C) nothing reaches D4
that did not reside in D2 and get surfaced by #23. The #22 body is always
derived from the #23 survivor; the live overlay can only be sedimented
*through* a candidate that carries its shadow evidence. Remove the
live-overlay gate path; the D4 sediment now demonstrates the nursery path.

### 3.5 Physics run (`run.py`)

- `PhysicsRun.tick` order: step env → observe/digest → `birth_candidate`
  when the loop trigger fires (#18 emits) → shadow-score residents →
  ratchet decision from #21 → bind source context → `DryRun.tick()` →
  prune → record row.
- `RunTick` gains: `candidate_id`, `candidate_residence_ticks`,
  `shadow_ratio`, `survivor_id`, `promoted_id`, `motif_id`,
  `drained_ids`, `d3_mass` (dict), `d3_bytes`, `nursery_ids`.
- Seed 7 = registered trace; seed 0 = rejection control. Same scenario:
  `warmup=8`, flip at 8, `post_flip=20`, flip back at 28, `recovery=8`
  (36 ticks) — the shape 9a used; the numbers are re-measured, never
  re-asserted.

---

## 4. Lean contract (`tools/gen_lean.py` → `formal/DryRun.lean`)

Generated only; never hand-edited. `python3 tools/gen_lean.py | diff -`
must be empty after regeneration.

1. **Unchanged:** the 27-flow op table, all mailbox theorems
   (`boxes_*`, `misses_*`, `sunk_*`, `attempts_at_master`,
   `balance_at_master`, word-generic lemmas), and the `mat_*_policy`
   theorems (the overlay index policy is unchanged — do **not** restate
   them).
2. **Payload model:** `PayloadKind` gains `crystal`; `d2Allowed :=
   [overlay, crystal]`; the purity theorem still excludes
   traj/delta/ctx. `d2WriteFids = [18, 24]` and a new
   `candidateWriteFids = [18]`, `overlayWriteFids = [24]` pin the
   per-flow split, with a theorem that they are disjoint, cover the D2
   writes, and each is a `Nat` list of absorbed flows.
3. **New candidate-index lifecycle theorems** (the ruling's "its own
   lifecycle theorems"), modeled as a pure list recurrence over candidate
   ids:
   - `nurseryTick` = FIFO birth (append) then drop-oldest past
     `NURSERY_CAPACITY`. `nurseryRun 0 = []`.
   - `nurseryTick_length` / `nursery_bounded`: `length ≤ 4` always.
   - `nursery_fresh`: the tick's id is in the next state.
   - `nursery_runs_distinct`: ids in `nurseryRun` are pairwise distinct
     (fresh births cannot resurrect a promoted/evicted id).
   - `nursery_promote_resident`: `promote l id` changes `l` only when
     `id ∈ l` (no promotion of a non-resident).
   - `nursery_promote_once`: with distinct ids, an id promoted out at
     tick `t` is not in any later `nurseryRun` state, so it cannot be
     promoted twice.
   - `candidate_recorded_from_birth`: the body type pin — `#18` is the
     only birth flow and it writes the candidate kind (a `Nat`-level
     statement over `d2WriteFids`, complementing the Python exact-type
     test).
4. Keep `lean` compiling from core Lean 4 only; no `sorry`, no `axiom`.

---

## 5. Pre-registered Round-11 gates (must be written before measuring)

Pin these in `docs/phase4-5-design.md` §15.11 **before** the seed-7 run,
then measure. Both are computed from the same `RunTick` rows.

| Gate | Definition | PASS condition |
| --- | --- | --- |
| **ADAPTATION** (shadow) | mean over ticks with `phase == "flipped"` **and** a shadow-scored resident candidate of `shadow adapted / base` | `< 1.0` |
| **ADMISSION** (residence) | count of **promoted** candidates (surfaced by #23, lineage-consumed by #22) whose final residence `ratio >= 1.0` | **== 0** |

Also pre-registered, not gates: (i) the residence-evidence curve for seed 7
is reported per resident tick as
`(candidate_id, residence_ticks, adapted_mean, base_mean, ratio,
qualified)`; (ii) seed 0 is the rejection control with the pre-registered
expectation **zero promotions**; (iii) live-overlay behaviour (#24 →
stale → evict → D4 retained contents byte-identical, encoder
bit-identical) is reported as measured state, not as a gate.

Falsifier for the hypothesis: any promoted candidate with `ratio >= 1.0`
means the gate is not reading its evidence — report it as a wiring defect,
do not tune the threshold.

`RESIDENCE_N = 4` is pinned in code and stated as a pre-registered design
constant (like `gamma = 5.0`), not a measurement.

---

## 6. File ownership (disjoint write scopes)

| Wave | Owner | Files |
| --- | --- | --- |
| 1 | `provenance-scout` | `amateur/payloads.py`, `tests/test_payloads.py` |
| 1 | `physics-integrator` | `amateur/physics/organ.py`, `tests/test_physics_organ.py` |
| 1 | `lewm-engine` | `tools/gen_lean.py`, `formal/DryRun.lean`, `tests/test_lean_lockstep.py` |
| 2 | `provenance-scout` | `amateur/dryrun.py`, `amateur/events.py`, `tests/test_dryrun.py`, `tests/test_provenance.py`, `tests/test_invariants.py`, `tests/test_thermodynamics.py`, `tests/test_design_intent.py`, `tests/test_recurrence.py`, `tests/test_accounting.py`, `tests/test_fuzz.py`, `tests/test_ledger_sync.py` |
| 3 | `physics-integrator` | `amateur/physics/run.py`, `tests/test_physics_run.py` |
| 4 | Lead | `docs/TESTING.md`, `docs/phase4-5-design.md`, `docs/round11-record.md`, `v0.10-untested.md` |

Nobody edits another owner's files. If a cross-file change is needed,
report it to the Lead; do not make it.

---

## 7. Verification checklist (Lead runs at the end)

1. `LEAN_BIN=$PWD/.work/lean4/bin/lean python3 -m unittest discover -s tests -t .` — green, count reported.
2. `python3 tools/gen_lean.py | diff -u formal/DryRun.lean -` — empty.
3. `lean formal/DryRun.lean` — compiles, zero `sorry`, zero `axiom`.
4. 1000-tick canonical dry run: 27/27 liveness, depth ≤ 1, identities
   A/B, sinks `{2,4,5}`, misses only `{10,17}` warm-up, `in transit
   {10:1, 17:1}`.
5. Seed-7 and seed-0 physics runs: gates + residence curve + lifecycle
   trace reported with numbers.
6. `git diff` review: no scope leakage, no re-asserted round 7–9a number,
   no 9b recovery.
