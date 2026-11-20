# Provenance carrier — phase-2 adjudication memo

**Status:** decision-ready proposal. No production file changed; the
carrier is a standalone draft (`amateur/provenance.py`, `tests/test_provenance.py`).
**Decision needed:** pick the carrier shape (option A/B/C/D in §3) and
confirm the semantics in §2. The Lead integrates after adjudication.
**Phase 2 scope:** satisfy the *carrying* and *ordering* halves of the
physics-epoch gate. It cannot satisfy the *derivation* half: the two
geometry bodies that are synthetic today (#22, #23) stay synthetic until
physics replaces them (§6.6). Do not read this phase as gate closure.

---

## 1. The gate being satisfied

`docs/TESTING.md` §"Physics-epoch acceptance gates" (binding):

> every payload must carry `(source_fid, tick_born)` and be derived from
> actual node state — **no minting at emit time**. The current synthetic
> contents (`cry-{t}`, rotating confidences) are acceptable only in the
> dry run.

Decomposed into four obligations, so each can be adjudicated and tested
separately:

| Obligation | Meaning | Closed by this phase? |
| --- | --- | --- |
| O1 *carry* | every accepted payload has a provenance record | yes, if the carrier is chosen (§3) |
| O2 *ordering* | `tick_born <= emit_tick` at acceptance | yes, enforced at the gate |
| O3 *derivation* | body is a function of live node/store state, built at the derivation site | **no** — needs physics; #22/#23 remain synthetic |
| O4 *no minting* | the emit boundary constructs no payload body and no provenance | structurally enforceable at integration (§6.4); not provable by ordering alone (§2.3) |

The existing violation is one function with two minted bodies:
`amateur/dryrun.py::_payload_for`, called from `_arrive`, so every body is
built at the acceptance boundary — `cry-{tick}` for #23 and the
`(tick - 1) % 3` rotating confidence for #22. `MatBoard` itself does not
mint; it stores what `_arrive` hands it, and it currently strips stored
records to `(overlay_id, epoch)`, so it cannot yet hand provenance back on
a read.

---

## 2. Semantics (holds for every option)

### 2.1 What `source_fid` names: a producing flow, not a node

Recommended definition: **`source_fid` is the ledger flow id that produced
(bore) the value.**

| Value kind | `source_fid` | Example |
| --- | --- | --- |
| value written by a process | the emitting flow id | #22 sedimented invariant -> `22`; #24 ratcheted overlay -> `24` |
| value read out of a store | the historical write edge that created the stored record | #25 stale overlay born from the #24 ratchet -> `24`; a #23-born crystal read on #8 -> `23` |
| exogenous boundary input | the boundary edge itself | #1 Gradients -> `1`; #3 Perturbations -> `3` |

Why not the producing **node**: a node owns several outgoing edges (P1
owns #2, #4, #5, #6), so "P1" does not identify which typed carrier
produced the value, cannot be validated against `carrier_for`, and would
need a second field to disambiguate. A fid is the typed edge the value
actually travelled; it is checkable against the ledger with no extra
vocabulary. The counter-reading, "youngest causal input", is rejected for
this field: derivations have several inputs (#22 consumes #21 and #25),
so one fid cannot name the dependency set, and encoding a partial graph
would over-claim precision the schema does not have (§7 Q2 leaves the
door open if the user wants trace-grade provenance later).

`source_fid` must not be confused with `FlowEvent.FLOW_ID` (the edge
carrying the payload *now*). For writes they coincide; for reads they
deliberately differ, and that difference is the testable signature of a
read versus a re-mint (§6.4).

### 2.2 `tick_born` semantics

- A non-negative run tick; `0` is the pre-run seed (the D2 `MAT_SEED`
  overlay). There is no "unset" value: every record is explicit, so
  provenance cannot be silently omitted.
- Produced/write values: `tick_born == emit_tick` (born now).
- Store reads: `tick_born` is the stored record's birth tick and must be
  **strictly less** than the read's emit tick (a value cannot be read
  before it was stored; reads precede writes in the stage order, and
  writes are absorbed at the end of the tick).
- Derived-from-inputs values (physics epoch): convention
  `tick_born = max(input.tick_born)` — the value is as new as its youngest
  input. The single field cannot record the dependency set; the convention
  makes the stamp monotone and comparable.

### 2.3 The ordering invariant, and its honest limit

`tick_born <= emit_tick` is enforced at the single acceptance boundary:

- it rejects future-dating (a body claiming to be born after it is
  emitted);
- it catches clock or state leakage between runs;
- it is a tripwire, **not** a proof of derivation.

A value fabricated at emit time can claim `tick_born == emit_tick` and
pass. That is legitimate for a *write* (the value is genuinely born now),
so ordering alone cannot separate a derivation from a mint. The proof of
no minting is structural + continuity (§6.4); the memo and the draft tests
state this limit explicitly rather than letting a green ordering test
imply more than it proves.

### 2.4 Read continuity

For a store read, provenance is inherited, not re-created:

- #25's body and provenance are the current D2 head record (written by
  #23 or #24);
- #8 likewise reads the D2 head;
- #7 reads the latest #22-sedimented `BiasPrimitive`;
- #11/#14/#15 read D1 state produced by #13; #20/#26 read D3 state
  produced by #19.

This is what makes "derived from actual node state" checkable: the read's
`Provenance` must equal the stored record's `Provenance`. A re-minted read
would carry `(read_fid, emit_tick)` and fail both the strict ordering and
the equality. The draft pins this signature
(`tests/test_provenance.py::TestNoMintingTripwire`).

### 2.5 D4 and the philosophy clamp

"The core contains no trajectories. It contains the biases through which
trajectories become more or less probable before one occurs."

- A provenance record is a **scalar stamp** (`source_fid`, `tick_born`),
  not a trajectory, state-delta, or context. It is the same category of
  information as the already-adjudicated `MatOverlay.epoch`.
- D4's body must remain exactly `BiasPrimitive(bias_primitive, confidence)`
  — that is the "and only that" clause. The recommended envelope keeps the
  body type literally unchanged; the two integers live outside it.
  Option B (fields on `BiasPrimitive`) is the only option that changes the
  core's payload schema; if chosen, the user is amending an adjudicated
  sentence, not merely adding a helper (§3.2, §7 Q1).

### 2.6 Placeholder channels

Today 20 of 26 flows have no payload type at all (placeholder `None`); the
D2/D4 read channels may also be empty before their store fills. The envelope can still carry
provenance for those traversals, but it must not *claim* derivation: a
`None` body means "no value", and the provenance records only the
transport edge and emit tick. O3 applies only to non-`None` bodies. This
is a decision point: require provenance on all 26 traversals, or only on
non-`None` bodies (§7 Q3). Recommended: envelope on all 26 (uniform gate,
no future payload type can bypass it), with `body is None` explicitly
meaning "no derived value yet".

---

## 3. Options

### 3.0 At a glance

| | **A1** envelope as channel schema | **A2** envelope beside unchanged body schema (draft) | **B** per-type fields | **C** sidecar index | **D** event-level carrier fields |
| --- | --- | --- | --- | --- | --- |
| Carrier sits | around the body | around the body | inside each payload type | beside the payload | on `FlowEvent` |
| `PAYLOAD_SCHEMA` change | all 26 entries | **none** | none (two types gain fields) | none | none |
| Exact-type whitelist | must be re-expressed on the body | **unchanged**, gate checks body exact type | unchanged | unchanged (but not tied to provenance) | unchanged |
| Subclass rejection | survives if body-exact | **survives, tested** | survives | survives (orthogonal) | survives |
| Coverage of 26 flows | all | all | typed payloads only (`None` channels uncoverable) | all (as log) | all |
| Ordering enforcement | gate | gate | per-type validator / helper | at record time only | carrier construction |
| D4 body unchanged | yes | **yes** | no (`BiasPrimitive` gains ints) | yes | yes |
| Existing tests edited | ~7 | **5, mechanical unwrap** | ~4-5 fixtures/validators | 0 | 3 carrier-semantics methods |
| Lean change | none | none | none | none | none |
| Main risk | schema churn reopens typed-carrier text | envelope mistaken for new carrier (mitigate in review) | 20 `None`-only flows uncovered; core schema amended | provenance separable from payload; weaker than the gate wording | provenance detaches from the payload; gate says payload must carry |
| Draft status | not implemented | **implemented, 33 tests** | emulated in tests | not implemented | not implemented |

### 3.1 Option A — generic `Provenance`/`Envelope` wrapper

**A1 (literal reading of the brief).** `Envelope(provenance, body)` becomes
the payload type on every channel; `PAYLOAD_SCHEMA` entries become
`{Envelope}` (or `{Envelope, None}`), and every type check moves to the
inner body. Consequences: uniform single type; but every schema entry and
every type check changes, `test_payloads.test_d2_d4_channels_are_typed`
and all executor-integration tests change, and the adjudicated schema text
is rewritten — the highest churn and the only option that touches the
typed-carrier adjudication itself.

**A2 (recommended refinement, implemented in the draft).**
`PAYLOAD_SCHEMA` stays exactly what it is: the *body* whitelist. The
envelope is a transport gate beside it:

- channel payload = `Envelope(Provenance(source_fid, tick_born), body)`;
- `allowed_envelope(fid, value)` = `type(value) is Envelope` **and**
  `type(value.body) in PAYLOAD_SCHEMA[fid]`;
- `require_envelope(fid, value, emit_tick)` checks envelope type, body
  whitelist, then `tick_born <= emit_tick`, in that order;
- `provenance_of` / `derived_at` read envelopes, bare records, or per-type
  fields, so a staged migration (and even option B) can share the helper.

Exact-type-whitelist interaction: subclass rejection survives twice —
`type(envelope) is Envelope` rejects a subclassed envelope, and
`type(body) in schema` rejects `Sneaky(MatOverlay)` / `Smooth(BiasPrimitive)`
exactly as today. The draft tests pin both (`TestEnvelopeWhitelist`).
`source_fid`/`tick_born` are carried by the envelope; ordering is checked
once, at acceptance; D4's body is untouched; all 26 traversals carry
provenance including `body is None`.
Cost: 5 mechanical `.body` unwraps in `tests/test_payloads.py`, ~60-80
lines in `amateur/dryrun.py`; `amateur/payloads.py` needs **zero code
changes** (a docstring note only). No Lean change.
Residual risk: an envelope is an extra type on the channel; a reviewer can
mistake A2 for a new carrier. The memo's answer: the carrier (one class
per fid, `events._carrier`) is unchanged; the body whitelist and its
exact-type semantics are unchanged; the envelope is provenance metadata,
not a payload kind. This is why A2 is a refinement of A, not a reopening
of the typed-carrier decision.

### 3.2 Option B — per-type provenance fields

`MatOverlay(epoch, overlay_id, status, source_fid, tick_born)` and
`BiasPrimitive(bias_primitive, confidence, source_fid, tick_born)`, with
the same validation in each `__post_init__`; future payload types repeat
the two fields.

- Exact-type whitelist: unchanged by construction — `require_payload` still
  checks `type(payload) in PAYLOAD_SCHEMA[fid]`, so `Sneaky(MatOverlay)`
  remains rejected. The tests' fixtures gain two arguments; rejection is
  untouched.
- Ordering: enforced in each `__post_init__`? No — the record does not know
  its emit tick. It needs a helper at acceptance (`require_provenance`
  duck-types the fields) or a per-channel check, so the enforcement site is
  the same as A2; the difference is only where the integers live.
- Coverage: only the two typed payload types. The 20 `None`-only channels have
  no object to attach to; making them coverable would require inventing a
  placeholder payload type — which is the same work as A2 and loses A2's
  single gate.
- D4: `BiasPrimitive` gains temporal scalars, so the adjudicated sentence
  "`BiasPrimitive(bias_primitive, confidence)` and *only* that" is amended.
  The type is still trajectory-free (no sequence/state-delta/context), but
  the user would be changing the core's payload schema, not just the
  transport. This is the option's decisive cost.
- Read continuity: works, but the store must retain the fields through
  `MatBoard` (which currently strips to `(overlay_id, epoch)`); the same
  retention is needed under A2. No advantage.
- Migration: ~4-5 test fixtures/validators + `_payload_for` constructors;
  111 green achievable. But every new payload type is a fresh invitation
  to forget the fields — there is no single choke point.

### 3.3 Option C — sidecar provenance index

Payloads untouched; a `ProvenanceIndex` maps `(emit_tick, fid)` (or an
accepted-event ordinal) to `Provenance`; `_arrive` records on acceptance,
`MatBoard` records birth provenance for stored records so reads can chain.
Cheapest: zero schema/test churn, 111 green with no existing edits.

- Exact-type whitelist: untouched (orthogonal) — subclass rejection
  survives trivially.
- Ordering: can be checked when the index entry is written; the payload
  itself is not ordered.
- D4: body unchanged; strongest literal preservation.
- Failure mode: provenance is separable from the payload. A body minted at
  emit time can still be paired with a fabricated index entry, a payload
  can be copied out of the executor without provenance, and the
  "carry" obligation is satisfied only by convention. Reads need a second
  stored mapping (birth provenance) or the index becomes a second source
  of truth. This is the weakest reading of the binding gate. It is still
  useful as an **audit log** alongside A2 (recommended if the Lead wants
  an immutable provenance history), but not as the enforcement mechanism.

### 3.4 Option D — event-level carrier fields (considered for completeness)

`FlowEvent` already pins the traversal (one class per fid) and wraps the
payload. Add `provenance: Provenance` (or the two fields) to the carrier;
`_arrive` builds the event with provenance. `PAYLOAD_SCHEMA` is untouched,
all 26 events carry provenance, and the executor's payload log need not be
unwrapped.

- Exact-type whitelist: body whitelist unchanged; subclass rejection
  survives. Note the event class itself has no exact-type gate today, but
  `_arrive` uses `carrier_for(fid)`, so construction is fixed.
- Ordering: checked when `_arrive` builds the event.
- D4: body unchanged.
- Failure mode: provenance lives on the event, not the payload; extracting
  `event.payload` drops it. The gate's wording is "every payload must
  carry", which D satisfies only transitively. It is the least-churn
  by-construction option for the transport (3 carrier-semantics test edits
  + `events.py`), and a serious alternative if the user reads the gate
  transport-first. Its other cost is semantic: event equality/hash would
  now include provenance (arguably correct — same body, different origin
  should not compare equal — but it is a behavior change).

---

## 4. Recommendation

**Adopt A2**: a generic frozen `Provenance(source_fid, tick_born)` carried
inside a frozen `Envelope` at the acceptance boundary, with
`PAYLOAD_SCHEMA` left untouched as the exact-type *body* whitelist.

Three strongest consequences:

1. **One gate, all 26 flows, no payload type left behind.** Every accepted
   traversal — including the 20 placeholder `None` channels and every
   payload type added in the physics epoch — passes the same
   `require_envelope` check. B cannot cover the placeholder channels, C
   cannot enforce the pairing, and A1/D pay more churn or weaken the
   payload-level guarantee.
2. **The adjudicated typed-carrier decision and D4 purity are preserved
   literally.** `payloads.py` keeps the exact-type whitelist and
   `BiasPrimitive` keeps exactly its two adjudicated fields; the envelope
   carries two scalar stamps. The draft proves `Sneaky(MatOverlay)` and a
   subclassed envelope are still rejected, and that body type/hash
   semantics are unchanged. No existing adjudication (#18, two-tier
   geometry, census, typed carriers, reject-with-charge, theorem
   discipline) is reopened.
3. **Lowest integration risk per unit of enforcement.** The schema module
   is untouched; 5 existing test methods need a mechanical `.body` unwrap
   (not a loosening); the migration is four green-at-every-step phases
   with clean rollback points; Lean and the generator are untouched and
   verified byte-identical (§6.5). C is cheaper today but fails the gate's
   binding wording; B and D save equivalent or less effort while creating a
   larger semantic debt.

What A2 does **not** do: it does not make #22/#23 derived. The gate's
derivation half is physics work; the coverage audit in §5 marks those two
`synthetic` and §6.6 states the acceptance test for closing them.

---

## 5. Per-flow coverage audit (all 26 fids)

Current epoch = dry-run executor as checked in. "Coverage" is the
`synthetic | derived` column the Lead will fold into the INV matrix:
`derived` = the body is computed from live node/store state (or is the
stored record itself) at the moment of emission; `synthetic` = no body
(`None` placeholder) or minted content with no state dependency.
`source_fid` is the recommended producing-flow attribution.

| # | Flow (source -> target) | Body this epoch | Coverage | Derivation source in the physics epoch | `source_fid` |
| --- | --- | --- | --- | --- | --- |
| 1 | Gradients (E2 -> P1) | `None` | synthetic | E2 Gardener gradient state (exogenous) | 1 |
| 2 | Crystallized Output (P1 -> E2) | `None` | synthetic | P1 transduction of #1/#3 | 2 |
| 3 | Perturbations (E1 -> P1) | `None` | synthetic | E1 perturbation state (exogenous) | 3 |
| 4 | Active Probes (P1 -> E1) | `None` | synthetic | P1 probe policy over #1/#3 | 4 |
| 5 | Entropy Exhaust (P1 -> E1) | `None` | synthetic | P1 dissipated difference (#1/#3) | 5 |
| 6 | Prediction Error (P1 -> P2) | `None` | synthetic | P1 expectation vs. #1/#3 | 6 |
| 7 | Base Ops (read) (D4 -> P2) | `BiasPrimitive` from `board.d4` (empty at tick 1) | **derived** | D4 store, record sedimented by #22 | 22 |
| 8 | Mat Overlays (read) (D2 -> P2) | `MatOverlay` from `board.head()` | **derived** | D2 store, head born on #23/#24 | 23 or 24 |
| 9 | Routed Dynamics (P2 -> P3) | `None` | synthetic | P2 routing over #6/#7/#8/#26 | 9 |
| 10 | Resolved Trajectory (P3 -> P1) | `None` | synthetic | P3 observation loop over #9/#11 | 10 |
| 11 | Active Context (read) (D1 -> P3) | `None` | synthetic | D1 store, written by #13 | 13 |
| 12 | State Deltas & Surprise (P3 -> P4) | `None` | synthetic | P3 loop over #9/#11 | 12 |
| 13 | Write / Erase / Resize / Evict (P4 -> D1) | `None` | synthetic | P4 regulation over #12/#14/#17 | 13 |
| 14 | Slot State (read) (D1 -> P4) | `None` | synthetic | D1 store, written by #13 | 13 |
| 15 | Evicted Clusters (read) (D1 -> P5) | `None` | synthetic | D1 evictions caused by #13 | 13 |
| 16 | Cost vs. Error Metrics (P4 -> P5) | `None` | synthetic | P4 regulation over #12/#14/#17 | 16 |
| 17 | Collapse Mandate & Freed Slots (P5 -> P4) | `None` | synthetic | P5 digestion over #15/#16/#18/#20 | 17 |
| 18 | Emergent Crystals (P3 -> P5) | `None` | synthetic | P3 loop over #9/#11 (candidates) | 18 |
| 19 | Digested Debris (P5 -> D3) | `None` | synthetic | P5 digestion over #15/#16/#18/#20 | 19 |
| 20 | Consumed Scaffolds (read) (D3 -> P5) | `None` | synthetic | D3 store, written by #19 | 19 |
| 21 | Structural Motifs (P5 -> P6) | `None` | synthetic | P5 extraction over #15/#16/#18/#20 | 21 |
| 22 | Sedimented Invariants (P6 -> D4) | rotating `BiasPrimitive` | **synthetic** (minted: `(tick-1) % 3`, `0.4/0.6/0.8`) | P6 ratchet over #21 motifs and #25 stale read | 22 |
| 23 | Surviving Crystals (P6 -> D2) | `MatOverlay("cry-{tick}")` | **synthetic** (minted literal) | P6 ratchet over #21 motifs and #25 stale read | 23 |
| 24 | Ratcheted Overlays (P6 -> D2) | `MatOverlay(tick, head_id, "active")` | **derived** (head identity from #25; epoch is the revalidation tick) | D2 head read on #25 + current epoch | 24 |
| 25 | Stale Overlays (read) (D2 -> P6) | `MatOverlay` from `board.head()` | **derived** | D2 store, head born on #23/#24 | 23 or 24 |
| 26 | Avoidance Trails (read) (D3 -> P2) | `None` | synthetic | D3 store, written by #19 | 19 |

### 5.1 Summary for the INV matrix

| Category | Count | Detail |
| --- | --- | --- |
| **derived** this epoch | **4 / 26** | #7, #8, #24, #25 (all D2/D4 paths; #24's *identity* is derived, its epoch is the revalidation tick) |
| **synthetic** this epoch | **22 / 26** | 20 placeholder `None` channels + #22 `BiasPrimitive` + #23 `MatOverlay` |
| geometry channels (D2/D4) | 6 | 4 derived, **2 synthetic** (#22, #23) |
| store reads | 8 | 3 derived (#7, #8, #25); 5 blocked on unimplemented D1/D3 stores (#11, #14, #15, #20, #26) |
| store writes | 5 | 1 derived (#24); 2 synthetic (#22, #23); 2 placeholder (#13, #19) |
| non-store interior + entity flows | 13 | all placeholder `None` (synthetic by absence) |

**Read this column correctly:** a `derived` read of a `synthetic` origin
is still locally non-minting (it reads state) but globally synthetic. The
gate's derivation half is about where the value is *created*: #22 and #23
are the only two currently-created geometry values, and both are minted.
No other flow creates a body this epoch, so no other flow can currently
violate O3 — they are empty, not honest.

---

## 6. Migration plan (keeps 111 green at every phase)

### 6.1 Phase sequence

| Phase | Change | Existing tests edited | Green after | Rollback point |
| --- | --- | --- | --- | --- |
| P0 (this task) | add `amateur/provenance.py`, `tests/test_provenance.py`, this memo | 0 | 111 + 33 = 144 | delete two files |
| P1 (intake) | `_arrive` accepts and validates envelopes; `MatBoard` returns body+provenance on reads; write provenance supplied as `(fid, tick)`; `_payload_for` survives only for the two write bodies until P2 | 5 mechanical `.body` unwraps in `test_payloads.py` | 144 + new | revert commit; P0 remains |
| P2 (derivation-site move) | delete `_payload_for`; write bodies built by the ratchet step (still synthetic constants for #22/#23, flagged); reads inherit stored provenance; structural + continuity tests enabled | 0 additional | 144 + new | revert to P1 (executor still mints, but carrier and read continuity intact) |
| P3 (physics gate closure) | #22 derives from #21/#25, #23 derives from #21/#25; counterfactual tests; O3 closed for the geometry channels | 0 | 144 + new | revert to P2 (synthetic #22/#23 flagged open) |
| P4 (broad capture) | replace remaining `None` bodies as each process/store is implemented, one flow at a time | per flow | grows | per flow |

Lead owns git; the memo only proposes the commit boundaries.

### 6.2 File-by-file diff plan (A2)

**`amateur/payloads.py` — no code change.** Optional docstring addition
("the body schema; provenance is the transport envelope"). Keeping it
untouched is the point of A2: the exact-type whitelist and its tests are
the adjudicated artifact and must not drift.

**`amateur/provenance.py` — new (already drafted).** `Provenance`,
`Envelope`, `require_provenance`, `derived_at`, `provenance_of`,
`allowed_body`, `allowed_envelope`, `require_envelope`, `FLOW_IDS` derived
from the ledger.

**`amateur/dryrun.py` — the integration (P1/P2).** Concretely:

1. **P1** keeps `_payload_for` for the two write bodies only, but the
   provenance travels independently: `MatBoard.head()` /
   `latest_bias()` return `(body, provenance)` so reads inherit, and
   write provenance is supplied as `(fid, tick)` by the caller. **P2**
   deletes `_payload_for`; its three jobs move to where the state lives:
   - reads (#7, #8, #25): the board methods hand back `(body, provenance)`
     and the caller emits that pair unchanged;
   - writes (#22, #23, #24): built by the P6 ratchet step from #21/#25
     inputs at the derivation site, then handed to `_arrive`. The bodies
     of #22/#23 are still synthetic constants at P2 and are flagged as
     such; P3 replaces them with derived values.
2. `_arrive(fid, body, provenance, copies=1)`: no longer constructs a
   body. It builds `Envelope(provenance, body)`, calls
   `require_envelope(fid, envelope, self._tick_no)`, appends the envelope
   to the mailbox, and records `(tick, fid, envelope)` in `emissions`.
3. `MatBoard`: `_residents` from `(overlay_id, epoch)` to
   `(overlay_id, epoch, provenance)`; `birth`/`revalidate` carry the
   write's provenance; `head()` returns the record; `latest_bias()`
   returns `(bias, provenance)`; `sediment` stores the provenance. Seed:
   `MAT_SEED` gets `Provenance(23, 0)` (see §7 Q5).
4. Injection plan: the harness is the boundary producer for #1/#3 and the
   not-yet-implemented store reads; it supplies `Provenance(fid, tick)`
   for empty bodies only. A non-`None` body from a store always inherits
   the stored provenance.
5. `emissions`: switch to `(tick, fid, envelope)`; tests unwrap `.body`.
   (Alternative: keep bodies and add a parallel provenance log — zero
   existing test edits, weaker audit visibility; §7 Q6.)

**`tests/test_payloads.py` — 5 mechanical edits.** `TestExecutorIntegration`
methods that touch payload attributes unwrap `.body`:
`test_inv_emptiness_no_trajectory_ever_reaches_d4`,
`test_d2_channels_carry_overlays`,
`test_stale_read_pairs_with_same_tick_ratchet_write`,
`test_epochs_strictly_increase_per_overlay`,
`test_evicted_overlays_were_previously_read_stale`. No assertion is
weakened; each continues to assert the same body-level fact, and the new
provenance tests assert the added one. `test_d4_core_accumulates_only_bias_primitives`
and the schema tests are untouched.

**`tests/test_provenance.py` — new (drafted) + P1 additions.** Add
integration tests: every emission log entry has a carrier, ordering holds,
read provenance equals stored provenance, `_arrive` source contains no
body constructor, `_payload_for` is gone.

**`docs/TESTING.md` — Lead-owned.** Replace the gate's "deferred by
adjudication" sentence with the adjudicated option; add one paragraph
under "Typed payload schema" describing the envelope; extend the
INV-EMPTINESS row ("body whitelist by construction + envelope gate") and
the INV matrix with the §5.1 coverage summary.

**`formal/DryRun.lean`, `tools/gen_lean.py` — no change (P1-P4).** The
Lean model is content-blind (§6.5); regeneration is a verified no-op.

### 6.3 Which tests change, and why

| Test | Change under A2 | Why it is not a loosening |
| --- | --- | --- |
| `test_payloads.TestExecutorIntegration` (5 methods) | `payload.field` -> `payload.body.field` | same body assertion; now exercises the envelope |
| `test_payloads` schema/validation/subclass tests | none | `PAYLOAD_SCHEMA` is unchanged |
| `test_design_intent`, `test_contracts`, `test_dryrun`, `test_accounting`, `test_fuzz` | none | no payload attribute access |
| `test_lean_lockstep` | none | markers pin write fids/capacity, unchanged |
| new `tests/test_provenance.py` | new | strengthens: carrier, ordering, continuity, exact-type |

Under **B**, the same five methods mostly need no unwrap, but ~4-5
constructor/fixture sites and the validation tests change (signature
change, not loosening). Under **C**, no existing test changes; new tests
are the only enforcement. Under **D**, 3 `TestCarrierSemantics` methods
change (constructor + equality/hash semantics).

### 6.4 How to prove "no minting at emit time"

Four layered checks, of which ordering is only the first:

1. **Structural (necessary):** `_arrive` constructs no body and no
   provenance. Enforce with an AST/source test:
   `inspect.getsource(DryRun._arrive)` contains none of `MatOverlay(`,
   `BiasPrimitive(`, `Provenance(`; `hasattr(DryRun, "_payload_for")` is
   false. A grep test is cheap and catches the exact failure mode this
   phase removes.
2. **Derivation-site (the real rule):** the body is built where the state
   is read. Test by substituting the state: monkeypatch
   `MatBoard.head` to return a sentinel record and assert #8/#25 emit
   exactly that body; monkeypatch `latest_bias` likewise for #7; mutate
   the #21 motif input for #22/#23 in P3 and assert the body changes.
3. **Continuity (the anti-minting signature):**
   - every store read with a non-`None` body satisfies
     `tick_born < emit_tick`;
   - its provenance equals the stored record's provenance;
   - #25 and #24 pair on `overlay_id` in the same tick (existing test),
     now also on inherited provenance for the read;
   - #7's body is the exact object sedimented by #22 (identity or
     equality + equal provenance).
   A value minted at emit time carries `(fid, emit_tick)`, so a store read
   fails the strict ordering and the continuity equality. This is the check
   that ordering cannot provide for writes.
4. **Counterfactual (integration-level):** two runs with different
   `MAT_SEED` produce different #8/#25 bodies; two runs with different #21
   inputs produce different #22/#23 bodies (P3). A minted constant is
   invariant under both.

Honest limit: for **writes** `tick_born == emit_tick` is legitimate, so
only (1)+(2)+(4) distinguish derivation from minting. The gate is
declared passed for a flow only when its body test exists, not when the
ordering assertion passes.

### 6.5 Lean impact

The Lean model is content-blind: `opsOf` words, scalar `St` counters, and
`List Nat` resident epochs. It cannot see a payload value, so no theorem
can or should prove `(source_fid, tick_born)` correctness. Consequences:

- `formal/DryRun.lean` and `tools/gen_lean.py` need **no change** for
  A2/B/C/D. `geometry_write_channel_purity` is a kind-level fact
  (`d4Allowed = [bias]`, `d2Allowed = [overlay]`) and stays true; the
  envelope adds no payload kind. `geometry_writes_anchored` pins
  `d4WriteFids = [22]`, `d2WriteFids = [23, 24]`; unchanged.
  `mat_*` theorems are about epoch lists; unchanged.
- The only at-risk text markers are in
  `test_lean_lockstep.test_payload_schema_markers_match_python`
  (`def d4WriteFids : List Nat := [22]`,
  `def d2WriteFids : List Nat := [23, 24]`, `rev.length > 4`). None of the
  options touches them. `test_master_theorems_present` only asserts
  presence, so adding a theorem would not break it — but adding one for
  provenance would prove only `Nat` ordering, not derivation, and would
  manufacture a false impression of gate closure. **Recommendation: no
  Lean change.** Keep the honesty line that the Lean phase proves
  plumbing, not payload semantics.
- Regeneration is a verified fixed point: `python3 tools/gen_lean.py`
  reproduces `formal/DryRun.lean` byte-for-byte (checked in this task), so
  if the Lead must regenerate for unrelated reasons, the diff is empty.
  If a provenance marker is ever added to the generator, add it to the
  marker test at the same time.

### 6.6 Gate sequence (what "closed" will mean)

The physics-epoch gate closes only when, for **every** payload body:
O1 carried, O2 ordered, O3 derived (no synthetic origin), O4 no
emit-time construction. P1-P2 close O1/O2 and make the O4 boundary
structural for the geometry channels; O3 stays open for #22/#23 and is
auditable per flow via §5. P3 closes O3 for #22/#23; the other 20 flows are
placeholder-empty and inherit O3 as their physics lands. The INV matrix
should show `4/26 derived` now and move each flow to `derived` only with a
body-derivation test, never by assertion.

---

## 7. Open questions requiring adjudication

Each is phrased as a choice; the parenthetical is the recommendation.

**Q1. Carrier shape.** (A2, envelope beside the unchanged body schema.)
- A2: one gate, all 26 flows, schema untouched; 5 mechanical test edits;
  needs a one-line review note that the envelope is not a new carrier.
- A1: uniform channel type but rewrites all 26 schema entries and the
  typed-carrier text; highest churn.
- B: minimal type-distance for the two typed payloads, but 20 `None`-only
  channels stay uncoverable and D4's "and only that" schema is amended.
- C: zero schema churn, but provenance is separable from the payload —
  it does not enforce the binding "payload must carry" wording.
- D: payload-level gate satisfied transitively only; least transport
  churn; provenance lost when a payload is extracted from its event.

**Q2. What `source_fid` names.** (The producing flow id.)
- Producing flow (recommended): 1..26, checkable against the ledger,
  distinguishes reads (historical write) from writes (self).
- Youngest causal input: richer trace, but a single fid cannot name a
  multi-input dependency set (#22 has #21 and #25), so it over-claims.
- Producing node: loses the typed-edge attribution; P1 alone owns four
  outgoing flows and cannot identify the carrier.

**Q3. Do the placeholder `None` channels carry provenance?** (Yes,
envelope on all 26.)
- Yes: one invariant over every accepted traversal; no future payload
  type can bypass the gate; `body is None` explicitly means "no derived
  value".
- Only non-`None` bodies: less noise today, but the gate then has a
  per-flow exception list and future payload types can slip through it.

**Q4. How strict is a read's birth?** (Strict: `tick_born < emit_tick`
for non-`None` store reads; `tick_born == emit_tick` for writes.)
- Strict: catches a read that re-mints at emit time; matches the stage
  order (reads precede writes, writes absorbed at end of tick).
- Universal `<=`: simpler invariant, but a re-minted read passes; the
  gate loses its strongest detection signature.

**Q5. Seed provenance.** `MAT_SEED = ("ovr-0", 0)` has no producing
ledger flow.
- Attribute it to the D2 birth channel: `Provenance(23, 0)` (recommended
  — keeps validation at 1..26, no sentinel, documented as the harness
  seed).
- Add a sentinel `source_fid = 0` for harness-origin values: cleaner
  semantics, but relaxes the 1..26 validation and adds a second origin
  kind the ledger does not have.

**Q6. `emissions` shape.** (Envelopes, 5 test unwraps.)
- Envelopes: the audit log is the accepted payload exactly as carried;
  tests traverse the real structure; 5 mechanical edits.
- Body view + parallel provenance log: zero existing edits, but the log
  no longer shows what the channel actually carried; audit visibility
  depends on two lists staying in step.

**Q7. Lean.** (No change.)
- No change: honest — the model is content-blind and cannot prove
  derivation.
- Add a content-blind ordering theorem: provable, but proves only Nat
  ordering and risks reading as gate closure.

**Q8. Gate sign-off sequencing.** (Record partial closure, not passed.)
- Accept that P1-P2 close carrying/ordering/anti-minting structurally and
  leave #22/#23 `synthetic` in the INV matrix; gate remains open until
  their derivation tests exist.
- Declare the gate passed on the carrier alone: not supportable — the
  binding sentence includes "derived from actual node state", and #22/#23
  are minted today.

---

## Appendix A — draft API (`amateur/provenance.py`)

| Symbol | Contract |
| --- | --- |
| `Provenance(source_fid, tick_born)` | frozen; `source_fid` in ledger 1..26; `tick_born` int >= 0 |
| `Envelope(provenance, body)` | frozen; exact `Provenance`; body's type/hash/equality untouched |
| `provenance_of(value)` | envelope -> record, bare record, or per-type fields; else `TypeError` |
| `require_provenance(value, emit_tick)` | `tick_born <= emit_tick`; `TypeError` if absent, `ValueError` if future-dated |
| `derived_at(payload, tick)` | `require_provenance` under the derivation-site name |
| `allowed_body(fid, body)` | `type(body) in PAYLOAD_SCHEMA[fid]` (exact type) |
| `allowed_envelope(fid, value)` | `type(value) is Envelope` and body exact-type member |
| `require_envelope(fid, value, emit_tick)` | envelope -> body whitelist -> ordering; returns the envelope |
| `FLOW_IDS` | ledger flow ids, derived from `FLOW_LEDGER` |

## Appendix B — verification evidence

| Check | Result |
| --- | --- |
| Baseline `LEAN_BIN=$PWD/.work/lean4/bin/lean python3 -m unittest discover -s tests -t .` | 111 green |
| After P0 draft (`tests/test_provenance.py`) | 144 green (111 + 33 new) |
| Shared-tree re-check once sibling phase tests appeared | 170 tests OK, 1 skipped (`test_lewm.TestDatasetVoE`, PushT dataset absent — phase 1, not phase 2) |
| `python3 -m unittest tests.test_provenance -v` | 33 green |
| `python3 tools/gen_lean.py > .work/prov/DryRun.gen.lean` vs `formal/DryRun.lean` | byte-identical (fixed point) |
| Files touched by P0 | `amateur/provenance.py`, `tests/test_provenance.py`, `docs/provenance-proposal.md` (`.work/prov/` scratch) |

**Constraints honored:** `amateur/payloads.py`, `amateur/dryrun.py`,
`docs/TESTING.md`, `formal/DryRun.lean`, `tools/gen_lean.py`, and all
existing tests are untouched; no git commit/push; CPU only.
