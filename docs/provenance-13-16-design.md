# Provenance #13/#16 — D1/P4 prerequisite slice

**Status:** design memo only. No code, test, Lean, executor, or doc change
was made for this task except this file. Implementation waits for Lead
review and the user's decisions on §8.

**Round:** 9b parallel track (user ruling 2026-11-20): #13/#16 provenance
needs a D1/P4 prerequisite slice designed *before* any derivation.

**Target:** make #13 (`P4 -> D1`, Board Ops) and #16 (`P4 -> P5`,
Cost vs. Error) **genuinely derived**, i.e. their bodies are functions of
live D1/P4 evidence rather than constants, with the A2 envelope carrying
`(source_fid, tick_born)` under the existing per-channel rules. The gate
stays **PARTIAL** until this slice lands; nothing here claims 26/26.

---

## 0. Reading trail (current state, not inferred)

| Source | What it establishes for this design |
| --- | --- |
| `docs/TESTING.md` §"Typed payload schema" + §"Provenance coverage by flow" | A2 envelope; exact-type body whitelist; per-channel birth rule; coverage table; `derived = {7,8,24,25}` in the synthetic dry run; physics mode lists #3/#6/#10/#12/#17/#18/#19/#21/#22/#23/#24 as real bodies, and **#13/#16 as placeholders** |
| `docs/TESTING.md` INV matrix, `INV-PROVENANCE` row | carrying + ordering closed; derivation partial; `#2/#4/#5/#9/#13/#16 stay placeholders` |
| `docs/TESTING.md` `INV-SOURCE` / `INV-GEOMETRY` rows | Q2-A content-only source, fixed dispatch, no fallback; D1 is an *operational* store (two-tier geometry reading), not parametric geometry |
| `docs/phase4-5-design.md` §7 (Q1–Q4) | Q2-A source rules; Q3 fixed cadence / state-based acceptance; Q4 #22; Q1 D2 patch (unrelated to D1) |
| `docs/phase4-5-design.md` §8 payload plan | #16 `CostErrorBody` is named but deferred; #13 is explicitly deferred; the organ-derived P5 path uses #18/#19 only |
| `docs/phase4-5-design.md` §9 (executor wiring plan) | current `_derive` `put(...)` sites; #13/#16 listed as placeholder rows |
| `docs/phase4-5-design.md` §12 note 11 and §14.9 note 22 | "P5's declared inputs #15/#16/#20 are still placeholders"; "P4's cost/error and D1's slot/eviction state are not implemented" |
| `docs/phase4-5-design.md` §15.9 note 37/41 | suite and Lean baseline; validation scope caveat (not directly #13/#16) |
| `amateur/payloads.py` | `PAYLOAD_SCHEMA`; `REAL_BODIES` (contains #16 `CostErrorBody`, not #13/#14); D2 `MatBoard`, D3 `DebrisPool`; no D1 store |
| `amateur/dryrun.py` | `PLACEHOLDER_FLOWS = {1,2,4,5,9,13,16}`; store reads hardcoded (`#11/#14/#15/#20/#26` -> `None`); `_derive` per-fid source loop; `_absorb_writes` handles #19/#22/#23/#24 only; no D1 |
| `amateur/provenance.py` | `Provenance`, `Envelope`, `require_channel_provenance`: store read with non-None body requires `tick_born < emit_tick`; everything else `==`; `source_fid` names the producing flow |
| `amateur/physics/run.py` | `PayloadSource` hooks (`derive_body(fid, tick, *, board, debris)`, `on_commit`); `PhysicsSource._derivers` fixed map; coverage ledger is plain data; no #13/#16 producer |
| `formal/DryRun.lean` | **no provenance lemmas exist.** The Lean surface is op words, scalar channel state, `PayloadKind` purity (`geometry_write_channel_purity`), write-fid anchoring, and the mat laws. D1 is not modelled. `python3 tools/gen_lean.py` is byte-identical to the checked-in file (verified for this memo) |
| Current measured suite | `LEAN_BIN=$PWD/.work/lean4/bin/lean python3 -m unittest discover -s tests -t .` = **313 tests, OK (1 expected failure)**, 52.4 s |

**Coverage arithmetic correction (for the Lead's doc update).**
`docs/TESTING.md` says physics mode moves "ten more flows" to derived and
"so **11/26** derived". The code gives **14/26**: 11 non-read flows
(`3,6,10,12,17,18,19,21,22,23,24`) plus 3 implemented store reads
(`#7`, `#8`, `#25`). The "11" is the non-read count, not the total. This
memo uses 14/26 as the current baseline; the doc row should be corrected
when the Lead next touches it (§8 Q8).

---

## 1. What #13 and #16 are, and why they are placeholders

### 1.1 The two flows

| # | Ledger edge | Flow | Declared body today | Schema today | Coverage today | Producer today |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | P4 4.0 Allocate & Regulate Whiteboard -> D1 Elastic Board | Write / Erase / Resize / Evict | none | `{None}` | synthetic | **none** — P4 is an abstract `WhiteboardRegulator`; D1 is an abstract `ElasticBoard` |
| 16 | P4 -> P5 5.0 Digest & Collapse | Cost vs. Error Metrics | `CostErrorBody(cost, error)` exists in `REAL_BODIES` | `{CostErrorBody, None}` | **placeholder** (`PLACEHOLDER_FLOWS`) | **none** — the class is anticipatory; no P4, no D1 evidence to compute it |

`#14` (`D1 -> P4`, Slot State read) is the third flow in the same closed
circuit and is also a placeholder today (`{None}`). It is the required
evidence path for #13/#16, not an optional extra (§2.4, §4).

### 1.2 Why they are placeholders (not "forgotten")

1. **No D1 state exists.** `DryRun` owns D2 (`MatBoard`), D3 (`DebrisPool`)
   and D4 (a list); there is no D1 object with slots, occupancy, or
   eviction state. `_derive` hardcodes `put(11,14,15,20,26, None, ...)`.
   A `#13` body with no slot state to act on, or a `#16` cost with no
   occupancy, would be a constant dressed as derivation — exactly the
   gate's forbidden minting.
2. **No P4 policy exists.** `amateur/processes.py::WhiteboardRegulator` is
   an ABC; `regulate(deltas, slots, mandate)` has no implementation and
   no executor instantiation. #13/#16 are its two outputs, so they cannot
   be produced before it.
3. **The source hook has no evidence for them.** The Q2-A source owns
   non-read bodies, but `derive_body` currently receives only the D2
   board and D3 debris. It has no D1 slot view and no read-only view of
   the already-derived #12/#14 bodies. Deriving #13/#16 from organ
   internals would be a hidden back-channel (§6).
4. **Lead decision f is still in force.** `PLACEHOLDER_FLOWS` includes
   `13,16`; the coverage ledger marks them `placeholder`; the executor
   never asks the source for them. This is deliberate gate honesty, not
   an omission.

### 1.3 Contract anchors this slice must not break

- **INV-EMPTINESS/INV-GEOMETRY:** D1 is operational, not parametric
  geometry; only D2/D4 are geometry whitelists. Adding a D1 body must not
  touch `PAYLOAD_SCHEMA` for D2/D4, `d2WriteFids`, `d4WriteFids`, or the
  Lean purity/anchoring theorems.
- **INV-PROVENANCE:** exact-type body whitelist; `Envelope` transport;
  `source_fid` = producing flow; writes `tick_born == emit_tick`, reads
  with non-`None` bodies `tick_born < emit_tick`; #14 is a store read.
- **INV-SOURCE (Q2-A):** fixed dispatch, no silent fallback, source
  failure is an assertion before any arrival, source cannot influence
  schedule/stage order/channel semantics, `on_commit` is one-way and
  post-commit.
- **Q3 cadence:** #13/#16 are emitted on the existing fixed cadence. This
  slice does not trigger-gate anything; the policy is state-based but the
  schedule is unchanged.
- **Q1/Q4:** D2 patch table and #22 gate are untouched. D1 slots are not
  predictor patches.
- **No D3 read:** #20/#26 stay placeholders. The design must not consume
  D3 to compute #13/#16.

---

## 2. Minimal D1/P4 prerequisite slice

The slice adds exactly three things: a **slot store** (D1), a **pure
regulator** (P4), and enough **executor/evidence plumbing** to derive
#13/#14/#16 from each other. Everything else stays deferred.

### 2.1 D1: what it must store and read

Proposed additions to `amateur/payloads.py` (pure dataclasses, no torch;
same module as `MatBoard`/`DebrisPool`):

```text
BOARD_CAPACITY = 8                 # design choice, not a measurement

@dataclass(frozen=True)
class BoardSlot:                   # one resident slot (D1 state, not a payload)
    slot_id: str
    body_id: str                   # content digest of the written context
    nbytes: int
    born_tick: int

@dataclass(frozen=True)
class BoardOpBody:                 # #13 write body
    op: str                        # "write" | "erase" | "resize" | "evict"
    slot_id: str
    body_id: str | None            # write only
    nbytes: int                    # write: context bytes; resize: new capacity
    reason: str                    # evidence branch, see §2.2
    tick: int

@dataclass(frozen=True)
class SlotStateBody:               # #14 read body
    capacity: int
    occupied: int
    nbytes: int
    slot_ids: tuple[str, ...]      # oldest -> newest
    oldest_slot_id: str | None
```

**`BoardStore` (D1)** — an in-memory store with no persistence (matching
D2/D3/D4 this epoch):

| Surface | Contract |
| --- | --- |
| `__init__(capacity=BOARD_CAPACITY)` | empty; `_state_provenance = None` |
| `apply(op: BoardOpBody, provenance: Provenance)` | requires exact `BoardOpBody`, `provenance == Provenance(13, op.tick)`; applies the op atomically; updates `state_provenance = provenance` on success; raises on overflow/duplicate/missing slot; **the only D1 mutation path** |
| `slot_state() -> SlotStateBody \| None` | `None` until the first committed #13; otherwise a snapshot (no tick field, so the body is pure state) |
| `state_provenance -> Provenance \| None` | the provenance of the last committed #13; used by the #14 read |
| `evictions() -> tuple[tuple[str, int], ...]` | eviction state for the future #15; **maintained, not emitted** this slice |
| `reset()` | clear slots, evictions, provenance (for `PhysicsRun.reset`) |

Operation semantics (smallest honest set):
- `write`: require a free capacity slot; duplicate `slot_id` raises
  `ValueError`; store `BoardSlot(slot_id, body_id, nbytes, tick)`.
- `evict`: remove exactly one slot; append `(slot_id, tick)` to evictions;
  missing id raises (`ValueError`), no silent no-op.
- `erase`: remove the slot without an eviction record (the distinction
  matters for #15 later); not exercised by the minimal policy (§2.2).
- `resize`: change capacity; `new_capacity < occupied` raises; not
  exercised by the minimal policy.
- **No silent eviction on write overflow.** The policy must evict first;
  a capacity violation is a loud store error, so a policy bug cannot be
  absorbed as state.

**Why metadata + digest, not the latent content.** The task's prerequisite
is slot identity and eviction state. Storing the full 64-float context
would make `#11` (D1 content read) nearly free but would force a content
schema and a P3 consumer decision in this slice. The minimal slice stores
`body_id = digest(delta)` plus `nbytes`; the digest is content-addressed
and testable (same delta -> same id, different delta -> different id).
This is a recorded limitation: **#11 is still not a content read**; it
stays a placeholder (#8 Q2).

### 2.2 P4: what it must expose (cost/error)

Proposed `amateur/regulate.py` (new, pure, no torch, no organ access):

```text
class Regulator:
    def regulate(
        self,
        delta: StateDeltaBody,          # #12 evidence
        slots: SlotStateBody | None,    # #14 evidence
        tick: int,
    ) -> tuple[BoardOpBody, CostErrorBody]
```

The policy is deterministic and evidence-only. No constants are returned;
the branch and the body contents are functions of the inputs:

| Precondition (from evidence) | `#13` op | `reason` |
| --- | --- | --- |
| `slots is None` (cold start, no committed D1 state) | `write` a new slot | `"cold-start"` |
| `slots.occupied < slots.capacity` | `write` a new slot | `"free-slot"` |
| `slots.occupied >= slots.capacity` | `evict` `slots.oldest_slot_id` | `"capacity-evict"` |

Body construction:
- `slot_id = f"ctx-{tick}-{digest[:12]}"` — unique, traceable, and carries
  the content digest.
- `body_id = digest` where `digest = blake2b(struct.pack(">…d", *delta.delta))`
  (big-endian float64 serialization, deterministic across runs/platforms).
- `nbytes = len(delta.delta) * 8` — the serialized evidence width, not a
  constant.
- `tick = tick` (the write's birth tick; legitimate for a write).
- `#16 = CostErrorBody(cost=float(slots.nbytes if slots else 0),
  error=float(delta.surprise))` — cost is allocated D1 bytes, error is the
  measured #12 surprise. Both are evidence; neither is a design constant.

`#17` (`CollapseMandateBody`) is **not consumed in this slice**: mandates
are no-op sentinels until the 9b trigger-gating round, and consuming them
here would create an ordering dependency on a flow the 9b round owns. This
is recorded as an unused declared input, the same honesty posture as P5's
unused #15/#16/#20. If the user wants mandate-driven eviction now, that is
a separate decision (§8 Q6).

`resize` and `erase` are schema-legal but not exercised. A future policy
may use them; no body/test pretends they are derived today.

### 2.3 Schema changes (exact-type preserved)

| Change | Detail | Test consequence |
| --- | --- | --- |
| `REAL_BODIES[13] = BoardOpBody` | schema `{BoardOpBody, None}` (the existing `REAL_BODIES` convention, like #3/#16) | `test_payloads.TestSchema.test_d2_d4_channels_are_typed` follows `REAL_BODIES` automatically |
| `REAL_BODIES[14] = SlotStateBody` | schema `{SlotStateBody, None}` for the D1 read | same generic loop; #14 is a store read, not source-derived |
| #16 unchanged | already `{CostErrorBody, None}` | none |
| no D2/D4 change | `PAYLOAD_SCHEMA` for #7/#8/#22/#23/#24/#25 untouched; `D2_WRITES/D4_WRITES` unchanged | `test_payloads` geometry assertions unchanged |

`None` remains allowed on #13/#14 in the **synthetic** plumbing source,
matching #3/#6; physics mode must return exact bodies and the coverage
test asserts `derived => non-None`. If the Lead wants #13 to be a strict
write-only channel (`{BoardOpBody}`), the cost is a synthetic no-op body
and a default dry-run envelope change; it does not change the derivation
(§8 Q4).

### 2.4 Executor and source plumbing (smallest contract change)

**#14 becomes a real D1 store read.** In `DryRun._derive`, replace the
hardcoded #14 `None` with:

```python
slots = self.d1.slot_state()
if slots is None:
    put(14, None, Provenance(14, tick))          # empty read, == rule
else:
    put(14, slots, self.d1.state_provenance)     # (13, last_commit_tick), strict <
```

`#11` and `#15` remain `None` (no content read, no eviction consumer
this slice). `#20/#26` remain `None` (D3 untouched).

**The source gets a read-only evidence view.** Extend the Q2-A hook
keyword-only:

```python
def derive_body(self, fid, tick, *, board, debris,
                evidence: Mapping[int, Envelope]) -> object: ...
```

`_derive` passes an immutable snapshot of the envelopes already built for
this tick (`MappingProxyType(dict(envelopes))`), with a fixed consequence:
for the sorted non-read loop, `12` (StateDeltaBody) precedes `13`/`16`;
`14` (SlotStateBody) is already present. #13/#16 derive **only** from
`evidence[12].body` and `evidence[14]`; absent evidence is an
`AssertionError` (no fallback). The source never receives the D1 store
handle itself, so it cannot bypass the #14 flow or mutate D1.

The fixed derivation dependency is:

```text
#12 (source) --\
                +--> P4 Regulator --> #13 (source) --> D1 apply (executor)
#14 (executor) -/                   --> #16 (source)
```

**D1 mutation is executor-side, post-commit.** In `_absorb_writes`, after
the write mailboxes are drained, apply the committed #13 body:

```python
op = envelopes.get(13)
if op is not None and type(op.body) is BoardOpBody:
    self.d1.apply(op.body, op.provenance)     # updates state_provenance
```

Synthetic mode leaves #13/#16 `None` (schema allows it), so the default
`DryRun` keeps its historical envelopes and `#14` stays empty — no
default-behaviour drift. `on_commit(13)` is not added: D1 is a store
effect, not a source-side notification (Q2-3 remains one-way).

**Coverage ledger.** Remove `13` and `16` from `PLACEHOLDER_FLOWS`
(becomes `{1,2,4,5,9}`), add the two derivers to `PhysicsSource._derivers`
and `#14` to the executor reads. `PhysicsSource.coverage` then marks #13
and #16 `derived` automatically (it is `placeholder` iff in
`PLACEHOLDER_FLOWS`); #14 is an executor read and appears in the docs
coverage table as derived, not in the source ledger (which covers
non-read flows only). This is the entire coverage change.

### 2.5 Contract-preservation check

| Contract | How this slice preserves it |
| --- | --- |
| no silent fallback (Q2-A) | `PhysicsSource` raises if #12/#14 evidence is absent; no `None` fallback in physics mode; fixed `_derivers` map; synthetic `None` is explicit sentinel mode |
| exact-type whitelists | only `REAL_BODIES`/schema additive changes for #13/#14; #16 already exact-type; `Sneaky(...)` rejection unchanged; envelope gate unchanged |
| provenance carrying/ordering | #13/#16 are writes -> `(13,tick)` / `(16,tick)`, `==`; #14 is a read -> `(13,last_commit)` with `<`; seed rules untouched; `#13` apply asserts its own provenance |
| D3 not read | #20/#26 untouched; no `DebrisPool` consumption added to P4 |
| Q1 (D2 patch) | untouched; D1 slots are not patches |
| Q2 (source) | fixed dispatch, evidence-only, one-way `on_commit`; no schedule change |
| Q3 (cadence) | fixed per-tick cadence; state-based acceptance; no trigger gate |
| Q4 (#22 gate) | untouched |
| INV-GEOMETRY | D1 is operational; D2/D4 schema/writers unchanged; Lean purity theorems unaffected |

---

## 3. Derivation path for #13/#16 (and the #14 evidence)

### 3.1 Per-flow path

| Step | #13 Board Ops | #16 Cost/Error | #14 Slot State |
| --- | --- | --- | --- |
| **Source fid** | 13 (P4 -> D1 write; producing flow = emitting edge) | 16 (P4 -> P5; producing flow = emitting edge) | 13 (the historical write that bore the D1 state) |
| **Body type** | `BoardOpBody` | `CostErrorBody` | `SlotStateBody` |
| **Evidence** | `#12.StateDeltaBody` + `#14.SlotStateBody` | `#12.surprise` + `#14.nbytes` | D1 `BoardSlot` table + eviction log |
| **Derivation** | `Regulator.regulate(...)` branch on occupancy/pressure; digest of delta; bytes = 8 x len(delta) | cost = D1 allocated bytes; error = measured surprise | snapshot of slot identity/occupancy/bytes/oldest |
| **Envelope** | `Envelope(Provenance(13, tick), op)` | `Envelope(Provenance(16, tick), metrics)` | `Envelope(Provenance(13, last_commit_tick), state)` |
| **Channel rule** | non-read: `tick_born == emit_tick` | non-read: `==` | store read with non-None body: `<` |
| **Store effect** | executor applies it to D1 post-commit; sets `state_provenance=(13,tick)` | P5 consumes later (out of scope this slice) | D1 read for P4 next tick |
| **Coverage ledger** | `derived` (physics) / `synthetic` (default) | `derived` (physics) / `placeholder` removed | `derived` after first write |

**Evidence, not clock.** The only clock input is the write's birth tick.
The body contents that distinguish one tick from another are the delta
digest, the slot identity/occupancy, the measured surprise, and the
allocated bytes. Removing those inputs must change the outputs (§6).

### 3.2 Falsifiers (what would prove the derivation is fake)

| Flow | Falsifier |
| --- | --- |
| #13 | The same #12/#14 inputs always produce the same body (correct determinism), **but** changing the delta vector does not change `body_id`, or changing occupancy does not change the branch/`reason`/`slot_id`, or the op is the same verb on every state -> the body is constant-dressed. A policy bug that writes at capacity instead of evicting (store must raise) is also a falsifier. |
| #16 | `error != #12.surprise` (or not a documented function of it), or `cost` stays `0` while `#14.nbytes > 0`, or cost/error do not change when the slot state/evidence changes -> not derived. |
| #14 | D1 write/evict does not change the next #14 body; #14 is non-`None` before any #13 commit; `#14.provenance != (13, last_commit_tick)`; a same-tick read appears (`tick_born == emit_tick`) -> provenance/state bug. |
| Structural | `PhysicsSource` derives #13/#16 from `organ`/`digest` directly instead of `evidence[12]/[14]`; `derive_body` falls back to `None`/constant; a hardcoded numeric literal decides the branch -> fake derivation or back-channel. |

---

## 4. Explicit non-goals

1. **No trigger-gating (9b main track).** #13/#16 keep the fixed
   per-tick cadence; the policy is state-based but not event-gated. The
   trigger-gating round owns birth/revalidation cadence and may change
   the schedule as a different Lean law.
2. **No admission-policy change (9c).** No change to
   `amateur/physics/capacity.py`, `EvidenceBudget`, pre-adoption
   validation, or patch admission. D1 capacity is not an admission rule.
3. **No D1/D3 read consumers beyond what #13/#16 require.** #14 is
   derived because P4's declared inputs require it (strictly necessary);
   #11 (`D1 -> P3`) and #15 (`D1 -> P5`) stay `None`; #20/#26 (D3 reads)
   stay `None`. The D1 eviction log is maintained but has no #15 emitter.
4. **No D2/D3/D4 lifecycle change.** No patch, mat, debris, or #22 gate
   changes; no `MatBoard` signature change.
5. **No Lean/formal change.** No D1 model, no new theorem, no
   `formal/`/`tools/` edit; `gen_lean.py` remains a fixed point (verified).
6. **No persistence/restart.** D1 is in-memory like D2/D3/D4.
7. **No full `ElasticBoard` implementation.** `read_context` (#11) and
   `evictable_clusters` (#15) are not implemented; `BoardStore` is the
   slice store, not a claim that the D1 contract is complete (§8 Q9).
8. **No P5 consumption of #16.** P5's digest does not read #16 this
   slice; the flow is derived and emitted so the evidence exists, but its
   declared consumer remains future work (same honesty as #15/#20).

---

## 5. Impact analysis

### 5.1 Files (implementation, not this task)

| File | Change |
| --- | --- |
| `amateur/payloads.py` | `BoardSlot`, `BoardOpBody`, `SlotStateBody`, `BoardStore`, `BOARD_CAPACITY`; `REAL_BODIES[13]`, `REAL_BODIES[14]`; no D2/D4 changes |
| `amateur/regulate.py` | new pure P4 `Regulator` (`regulate`); no torch/organ imports |
| `amateur/dryrun.py` | `PLACEHOLDER_FLOWS` drops 13/16; `self.d1 = BoardStore()`; #14 store read; `derive_body(..., evidence=...)`; #13 apply in `_absorb_writes`; no scheme/schedule change |
| `amateur/physics/run.py` | `_derivers[13] = _board_op`, `[16] = _cost_error`; evidence mapping; no organ reads in these derivers; coverage unchanged mechanically |
| `tests/test_payloads.py` | body validation/hash fixtures for #13/#14; `TestBoardStore` (write/evict/erase/resize/capacity/provenance/reset/no-op) |
| `tests/test_regulate.py` (new) | P4 determinism, branch coverage, digest sensitivity, cost/error equals evidence, falsifier tests |
| `tests/test_provenance.py` | one #14 channel-rule test (strict store read) if not covered by the generic helper tests |
| `tests/test_physics_run.py` | coverage expectations follow `PLACEHOLDER_FLOWS`; `_body_type` gains 13/16; direct `derive_body` calls gain `evidence`; integration tests for D1/#14/#16 (state-based, `@requires_artifacts`) |
| `docs/TESTING.md` | Lead-owned: coverage table (#13/#14/#16), INV-PROVENANCE row, note that #14 is now a derived store read; correct the 11/26 vs 14/26 arithmetic |
| `docs/phase4-5-design.md` | Lead-owned: a wiring note/§16 recording the slice; not edited here |
| `formal/` / `tools/` | **none** |

### 5.2 Test impact (expected)

- **Existing tests that must change:** a handful of `tests/test_physics_run.py`
  call sites (`derive_body(..., evidence=...)`), `_body_type` for the new
  derived fids, and any direct assertion that #13/#16 are in
  `PLACEHOLDER_FLOWS` (the coverage tests compare against the constant, so
  they follow automatically). `tests/test_payloads.py` schema assertions
  follow `REAL_BODIES` automatically; new body fixtures are additions.
- **Existing tests that must not change:** D2/D4 schema/subclass tests,
  provenance rule tests, the accounting/liveness/fuzz/Lean lockstep
  layers, `test_contracts`, `test_engine`, `test_thermodynamics`,
  `test_recurrence`, `test_design_intent`.
- **No assertion may be weakened.** The new coverage rows are additions;
  the 313-test baseline (1 expected failure) must stay green.
- **New test count estimate:** ~10 regulator tests + ~7 D1 store/body
  tests + ~4 integration tests ≈ 21; exact count reported by the
  implementer.

### 5.3 Lean / proof impact: none, and why

The Lean model is content-blind (`opsOf`, scalar `St`, `List Nat` mat
epochs). There are **no provenance lemmas in `formal/DryRun.lean`** to
extend: provenance/derivation is enforced in Python (A2 envelope +
`test_provenance`) and D1 is not modelled. This slice does not change
`flow_ops`, the ledger, `d2WriteFids`, `d4WriteFids`, `MAT_CAPACITY`, or
the generator inputs, so **no theorem regeneration is needed**.
`python3 tools/gen_lean.py` must remain byte-identical to
`formal/DryRun.lean` (verified: identical at the time of this memo; a
theorem or doc marker must not be added in this slice). A formal D1
capacity/derivation proof would be a new model, not this prerequisite.

### 5.4 Coverage-ledger path to 26/26

| State | Derived | Synthetic/placeholder | Notes |
| --- | --- | --- | --- |
| Current (measured from code) | **14/26**: #3,#6,#7,#8,#10,#12,#17,#18,#19,#21,#22,#23,#24,#25 | 12: #1,#2,#4,#5,#9,#11,#13,#14,#15,#16,#20,#26 | TESTING says 11/26; code says 14 (§0 correction) |
| After this slice (physics mode) | **17/26** (+#13,#16,#14) | 9: #1,#2,#4,#5,#9,#11,#15,#20,#26 | #13/#16 source-derived; #14 executor read |
| Remaining prerequisites | | | #15/#11 -> D1 evictions/content slice; #20/#26 -> D3 read slice; #1/#4/#5/#9/#2 -> E1/P1/P2 slice |

The slice deliberately does **not** chase 26/26: three independent
prerequisite slices remain, and faking any of them would repeat the
problem this memo exists to prevent.

---

## 6. Risks and anti-"fake derivation" guards

| Attack | Why it would pass a naive test | Structural guard |
| --- | --- | --- |
| Constant body with a plausible type | `type(body) is BoardOpBody`, schema passes | Regulator sensitivity tests: mutate delta/occupancy/bytes; assert branch/digest/cost/error change; no constant fields |
| Clock-only derivation (`slot_id=tick`, `cost=occupancy`) | Every tick differs, so it looks dynamic | `body_id` is a digest of the delta; `cost/error` equal measured #14/#12 values; a test freezes the tick and varies evidence (must still differ) |
| Hidden organ/private state | Values look derived and match the run | `Regulator` has no organ import; `PhysicsSource._board_op/_cost_error` read only `evidence[12]/[14]`; unit tests construct the evidence by hand and never instantiate an organ |
| Reading D1 directly in the source | Bypasses #14, hides the evidence chain | No D1 handle in `PhysicsSource`; only the read-only evidence mapping is passed; test removes/None-fills `evidence[14]` -> assert |
| Silent fallback to `None`/constant | Makes the run green under a source bug | `PhysicsSource` dashes no fallback: absent evidence/unsupported fid -> `AssertionError` before any arrival; coverage-derived test asserts non-None exact type |
| Provenance laundering | A body derived from old evidence claims the current tick | Writes `==`; #14 read `<`; `BoardStore.apply` asserts `provenance == (13, op.tick)`; #14 provenance equals `state_provenance` |
| Fake eviction (evicting an id not present) | The policy "works" until #15 | `BoardStore` raises on missing/duplicate ids; `SlotStateBody` invariants (`occupied == len(slot_ids)`, oldest = first) |
| Schedule influence | Source changes what runs next | Q2-A fixed dispatch, `PLACEHOLDER_FLOWS` constant, no ops-table edits; `INV-SOURCE` tests unchanged |

**Q2-A cross-check.** The source remains content-only: it is given a
read-only evidence snapshot, cannot branch on the coverage ledger, cannot
see the D1 store, cannot create events, and cannot touch the schedule or
stage order. D1 mutation happens in the executor's post-commit absorption,
which is the same one-way direction as D2/D3/D4. The one contract
extension — `evidence` on `derive_body` — is additive and read-only; it
exists so the content derivation can cite the declared inflow bodies
instead of a private channel.

---

## 7. One-round step plan and acceptance criteria

Sized for one variable-step round; each step ends green with the previous
steps intact.

| Step | Work | Stop condition |
| --- | --- | --- |
| 1 | Add `BoardOpBody`/`SlotStateBody`/`BoardSlot`/`BoardStore` + `REAL_BODIES` entries; add body/store unit tests. No executor/source change. | `test_payloads` green; schema generic tests follow `REAL_BODIES`; no default-run change |
| 2 | Add `amateur/regulate.py` + `tests/test_regulate.py` (branch, determinism, digest, cost/error, falsifiers). Pure; no wiring. | New regulator tests green; no run behaviour change |
| 3 | Executor: `self.d1`, #14 read, `evidence` on `derive_body`, #13 apply in `_absorb_writes`, `PLACEHOLDER_FLOWS` drops 13/16. Update affected tests. | Default synthetic run unchanged (#13/#16 still None); focused executor tests green |
| 4 | Source: `_derivers[13]`/`[16]`, evidence-only P4, coverage; integration tests (requires artifacts). | Coverage shows 17/26 in physics mode; #14 strict provenance; #13/#16 exact bodies; no fallback |
| 5 | Verification + report: full suite, generator byte-identity, `git diff` audit, coverage table handoff to Lead. | See acceptance below |

**Acceptance criteria**

1. `LEAN_BIN=$PWD/.work/lean4/bin/lean python3 -m unittest discover -s tests -t .`
   is green with the same one recorded `expectedFailure` (313 + new tests).
2. `python3 tools/gen_lean.py` output is byte-identical to
   `formal/DryRun.lean`; `git diff --stat formal/ tools/` empty.
3. Coverage: physics-mode #13/#16 `derived`, #14 `derived` (store read);
   default synthetic still `synthetic`; the coverage ledger test covers
   every non-read flow; no flow is silently uncovered.
4. Evidence: #13 body changes with #12 delta and #14 occupancy; #16
   `error == #12.surprise` and `cost == #14.nbytes`; #14 changes after a
   D1 write/evict; seed/first-tick empty read is `(14,tick)` with body
   `None`.
5. Provenance: #13/#16 `tick_born == emit_tick`; #14 `tick_born <
   emit_tick` and equals `BoardStore.state_provenance`; no raw body
   escapes `_arrive`.
6. No test weakened; no D2/D4 schema or Lean change; no D3 read; no
   trigger-gating/admission change; no persistence claim.
7. `tests/test_payloads.py`/`test_provenance.py`/`test_physics_run.py`
   diffs are additive or mechanical (signature/coverage), stated in the
   report.

---

## 8. Open questions requiring Lead adjudication

**Q1. Evidence plumbing.** (Recommend read-only `evidence` mapping on
`derive_body`.)
- `evidence` mapping (recommended): explicit declared-inflow evidence,
  no D1 handle in the source, ~4 call-site updates; contract additive.
- D1 store handle in `PhysicsSource.__init__`: no protocol change, but
  the source can bypass #14 and read/write D1 directly — rejected as a
  back-channel risk.
- Batch source API: cleanest dependency ordering, but a larger Q2-A
  contract change and more test churn.

**Q2. D1 content scope.** (Recommend metadata + content digest.)
- Metadata + digest (recommended): minimal; #11 stays placeholder; slot
  identity/eviction state are real; content read deferred.
- Store the 64-float context: makes #11 nearly free but pulls a content
  schema and a P3 consumer into this slice, which the non-goals exclude.

**Q3. Is #14 in this slice?** (Recommend yes.)
- Yes: P4's declared inputs are #12/#14; deriving #13/#16 without the
  #14 read would require private state (fake-derivation risk).
- No: #13/#16 would have to use a direct D1 handle or a counter; both are
  weaker evidence and one is a back-channel.

**Q4. #13 channel schema.** (Recommend the existing `REAL_BODIES`
convention, `{BoardOpBody, None}`.)
- `{BoardOpBody, None}` (recommended): smallest change; synthetic mode
  keeps `None`; physics coverage test enforces non-None.
- `{BoardOpBody}` strict: better "writes carry bodies" discipline, but
  SyntheticSource must return a no-op body and default envelopes change;
  more test churn for no extra derivation guarantee.

**Q5. `#16.cost` definition.** (Recommend absolute allocated bytes.)
- `slots.nbytes` (recommended): directly derived from D1, no ratio
  constants; P5 can normalize later.
- Utilization ratio (`nbytes / (capacity * slot_bytes)`): comparable
  across capacities, but introduces a denominator constant and a design
  choice that belongs to P5's collapse policy.
- Occupancy count: simplest, but throws away the byte evidence.

**Q6. Does P4 consume #17 (collapse mandate) now?** (Recommend no.)
- No (recommended): #17 is a no-op sentinel until 9b; ignoring it keeps
  the derivation to #12/#14 and avoids an ordering dependency.
- Yes: mandate-driven eviction is more realistic, but requires #17 to be
  derived before #13 in the same source pass (or cached from `begin_tick`),
  i.e. a schedule/dependency change that 9b owns.

**Q7. D1 design constants.** Capacity 8 and no separate byte ceiling are
proposed as design choices (like D3 capacity 8). If the user wants a byte
ceiling or a different capacity, state the number; it does not change the
slice shape, only the store bounds and the capacity-evict cadence.

**Q8. Coverage arithmetic correction.** The current `docs/TESTING.md`
physics-mode text says 11/26; the code yields 14/26 (11 non-read + 3
store reads). Confirm that the Lead will correct the row to 14/26 (then
17/26 after this slice), rather than changing the code to match the doc.

**Q9. `BoardStore` vs the `ElasticBoard` ABC.** (Recommend the standalone
slice store.)
- Standalone `BoardStore` (recommended): no claim that #11/#15 exist;
  `stores.ElasticBoard` stays the full contract.
- Implement the ABC now with raising `read_context`/`evictable_clusters`
  stubs: makes `Loop` wiring closer but advertises methods that are not
  implemented; the stub exceptions are honest but add surface for no
  derivation gain.

**Q10. `ElasticBoard.apply(deltas)` ambiguity.** The abstract store
method cites #12, while the ledger's only D1 write edge is #13. Confirm
the concrete D1 slice routes **all** mutation through `#13`/`BoardStore.apply(op)`
and treats #12 as P4 input only (no second write path). This memo assumes
that reading; it is a small contract correction to record.

---

## Appendix A — proposed schema summary

| # | Body | Fields (frozen, hashable) | Producer | Channel rule |
| --- | --- | --- | --- | --- |
| 13 | `BoardOpBody` | `op: str` in {write,erase,resize,evict}; `slot_id: str`; `body_id: str\|None`; `nbytes: int`; `reason: str`; `tick: int` | P4 `Regulator`, source-derived | non-read write: `tick_born == emit_tick` |
| 14 | `SlotStateBody` | `capacity: int`; `occupied: int`; `nbytes: int`; `slot_ids: tuple[str,...]`; `oldest_slot_id: str\|None` | D1 store read, executor-derived | store read (non-None): `tick_born < emit_tick` |
| 16 | `CostErrorBody` (existing) | `cost: float`; `error: float` | P4 `Regulator`, source-derived | non-read: `tick_born == emit_tick` |

Validation guards: op enum; string types; non-negative ints; `write`
requires `body_id`/`nbytes > 0`; `resize` requires a capacity change;
`occupied == len(slot_ids) <= capacity`; `oldest_slot_id` equals
`slot_ids[0]` or `None`; all floats finite via the existing
`CostErrorBody`/`StateDeltaBody` validators.

## Appendix B — evidence-chain examples (what a test should see)

```text
tick 1:
  D1 empty -> #14 = None, (14,1)
  P4: cold-start -> #13 write(ctx-1-<digest>, body_id=<digest>, nbytes=512)
  #16 = CostError(cost=0, error=surprise_1)
  executor absorbs #13 -> D1 slot [ctx-1-...], state_provenance=(13,1)

tick 2:
  #14 = SlotState(capacity=8, occupied=1, nbytes=512,
                  slot_ids=("ctx-1-...",), oldest="ctx-1-..."),
         provenance=(13,1)  [1 < 2]
  P4: free-slot -> #13 write(ctx-2-<digest2>, ..., 512)
  #16 = CostError(cost=512, error=surprise_2)

... tick 9:
  #14 occupied=8 -> P4: capacity-evict -> #13 evict(oldest)
  #16 = CostError(cost=<sum bytes>, error=surprise_9)
```

No step uses a constant for `slot_id`, `body_id`, `cost`, or `error`;
the clock only names the write and its birth tick.

---

## 9. Lead adjudication (2026-11-20)

Verdict: **approved to implement**, with rulings below. The design is
accepted as the slice of record; §9 overrides §8 wherever they differ, and
adds three mandatory guards (§9.11) the memo did not require.

### 9.1 Q1 — evidence plumbing: `evidence` mapping

Approved as recommended. `derive_body(fid, tick, *, board, debris, evidence)`
takes a read-only mapping; #13/#16 may read only `evidence[12]` and
`evidence[14]`; a **missing key is an `AssertionError` in the derivation
pass, never a fallback and never a defaulted body**. The D1-handle option is
rejected: a source that can read D1 can bypass #14, and a bypassed #14 is
indistinguishable from a fabricated one. The batch-source API is rejected for
this slice: it is a Q2-A contract change bought for dependency ordering that a
16-line change to `_derive` already provides.

### 9.2 Q2 — D1 content: metadata + digest only

Approved as recommended. #11 stays a placeholder; no content schema, no P3
consumer, no byte-array retention. The store retains identity and eviction
state, which is exactly what P4 needs.

### 9.3 Q3 — #14 is in this slice, but as an **executor read**

Approved, with the implementation shape pinned: #14 is not a source deriver.
It follows #7/#8/#25 exactly — `_derive` reads the store, not the source:

- D1 empty → body `None`, provenance `(14, tick)` (self-origin), the same
  convention #7/#8/#25 already use for an empty store;
- D1 non-empty → `SlotStateBody`, provenance `(13, last_commit)`, strictly
  `< tick`.

This avoids inventing a genesis commit for tick 1 and makes the `derived`
class mean for #14 exactly what it already means for #7. Deriving #13/#16
without the #14 read is rejected outright: they would have to consult private
state, which is the definition of fake derivation on this codebase.

### 9.4 Q4 — #13 schema: `{BoardOpBody, None}`

Approved as recommended. `SyntheticSource.derive_body` already returns `None`
for fids outside its dispatch, so the synthetic default envelope is
**unchanged** by dropping 13/16 from `PLACEHOLDER_FLOWS`; the strict variant
would force a minted no-op body into the synthetic path for no derivation
gain. Physics coverage enforces non-`None` where it matters. The tests at
`tests/test_physics_run.py:184-211` pin the set symbolically and must be
updated in phase 2, not re-pinned by literal.

### 9.5 Q5 — `#16.cost`: absolute allocated bytes

Approved as recommended, with two pins:

- `nbytes` is computed from the represented record: `8 * len(delta)` (float64
  × latent dimension, 512 B at D=64). **No literal `512`** anywhere in the
  store or the regulator — the 512 in Appendix B is the value at D=64, not a
  constant to hard-code.
- The docstring must state that `cost` measures the *represented* bytes D1
  retains a handle to, not the store's own metadata footprint.

Utilization ratio and occupancy count are rejected: the denominator and the
normalisation policy are P5's collapse policy, which this slice does not own.

### 9.6 Q6 — #17 is not consumed this slice; carries a cross-round obligation

Approved as recommended. But this becomes an explicit obligation, not a
permanent exclusion: when 9b or a later round makes #17 a non-noop derived
mandate, the Regulator must either consume it (mandate-driven eviction) or the
deferral must be re-adjudicated in writing. A derived, non-noop mandate that
P4 silently ignores is not an acceptable end state. 9b's report must state
which of the two it is doing.

### 9.7 Q7 — D1 capacity 8, no byte ceiling

Approved. Capacity 8 is a design constant of the same kind as D3's 8. No byte
ceiling: with metadata + digest only there is no content bound to enforce; a
ceiling added now would be an unmeasured number. If the user later wants a
specific capacity, it changes the store constants, not the slice shape.

### 9.8 Q8 — coverage arithmetic: 14/26 confirmed; the doc is wrong

Confirmed by independent measurement (`NON_READ_FLOWS` = 18, placeholders = 7
→ 11 non-read derived, plus the three store reads #7/#8/#25 = **14/26**).
The `11/26` in `docs/TESTING.md` counted the non-read column and dropped the
reads. The Lead corrects the doc; the code stays. After this slice: **17/26**.

### 9.9 Q9 — standalone `BoardStore`

Approved as recommended. No `ElasticBoard` stubs: a method that raises is
still an advertised method, and this slice makes no claim about #11/#15.

### 9.10 Q10 — all D1 mutation through #13

Approved, and it is a contract correction to record: D1's only write edge is
#13. `stores.ElasticBoard.apply(deltas)` citing #12 is wrong — #12 is P4's
declared input, not a D1 write path. Correct the docstring in the same commit
that lands `BoardStore`. Additionally pinned: the commit happens **post-
arrival** in the executor's absorb step, so a refused #13 leaves D1
byte-identical (see 9.11a).

### 9.11 Mandatory guards the memo did not specify

- **(a) Refusal atomicity.** Force a #13 refusal (capacity violation reachable
  by test) and assert: D1 state digest byte-identical before/after, no
  eviction recorded, `last_commit` unchanged. A refusal must not be able to
  half-write the board.
- **(b) No-constant audit.** A test asserting every state-bearing field is
  traceable: `nbytes == 8 * len(delta)`, `cost == sum(slot.nbytes)`, `slot_id`
  and `body_id` digest-derived, `occupied == len(slot_ids)`. Nothing in
  #13/#14/#16 may equal a module-level literal standing in for state.
- **(c) Single writer.** Assert `BoardStore` is mutated only by `apply`, and
  that no #12 handling touches D1 (no second write path, per 9.10).

### 9.12 Ordering and write scopes

Phase 1 (disjoint, may start now): `amateur/payloads.py` (bodies + store),
new `amateur/regulate.py`, `tests/test_payloads.py`, `tests/test_regulate.py`.
Phase 2 (blocked on Round 9b / task-11, which touches the same executor
files): `amateur/dryrun.py`, `amateur/physics/run.py`,
`tests/test_physics_run.py`, `tests/test_provenance.py`. Rationale: both
rounds edit the derivation pass; serialising phase 2 costs less than
reconciling two concurrent derivations of `_derive`.

### 9.13 Measurement contract

Baseline is 313 tests OK (1 expected failure), generator byte-identical (Lead
re-verified: `formal/DryRun.lean` has **zero** provenance mentions and
`tools/gen_lean.py` output diffs clean). The slice must keep the generator
byte-identical and must not add or change Lean. D1 bounds and the regulator
are Python-tested, not proved, and the design record must say so in those
words.

### 9.14 Phase 1 review rulings (2026-11-20, after implementation)

Phase 1 reviewed against the code, not the report, and re-measured in an
isolated worktree at HEAD + the phase-1 files: **348 tests OK (1 expected
failure, 27 skipped), generator byte-identical**. The ADMISSION
`@expectedFailure` is intact in that tree, so the new files do not touch the
round's recorded failure.

1. **`stores.py` docstring correction: accepted.** It is required by §9.10;
   the omission was the Lead's write-scope list, not the implementer's
   deviation.
2. **"D1 empty" = never committed (`last_commit is None`). Confirmed.** A
   board emptied by erase/evict returns a real `SlotStateBody` with
   `occupied = 0` and the prior commit's provenance: that state is derived
   from commits, and P4 needs `occupied = 0` to take the free-slot branch.
   Phase 2 must **not** add `if occupied == 0: None`.
3. **`resize`: schema-legal, operator/test-only this slice.** `nbytes` is
   the new capacity; same-capacity and below-occupancy resizes are refused.
   The Regulator must never emit `resize`, and a test must pin that.
4. **`BoardSlot.nbytes > 0`: confirmed** — slots only arise from writes.
5. **New requirement — monotone commit order.** `oldest_slot_id` and
   `slot_ids` order currently come from dict insertion order, which equals
   born-tick order only if commits are monotone, and `apply` accepts a stale
   tick today. Add: reject `op.tick <= last_commit` ("#13 commits must
   strictly increase"; one `#13` per tick by the ledger), and have
   `slot_state()` assert the resident born-ticks are strictly increasing
   before publishing `oldest_slot_id`. `oldest` must be a property of
   evidence, not of insertion accident.
6. **`#16.cost` is the pre-op occupancy. Confirmed and to be documented.**
   `#16` is derived before the `#13` of the same tick arrives, so `cost` is
   what `#14` reported — the honest evidence available. The post-commit
   occupancy becomes visible as `#14` on the next tick. A "cost + nbytes"
   projection would be a prediction, not derivation, and is refused.
