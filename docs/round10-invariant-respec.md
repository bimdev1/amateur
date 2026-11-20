# Round 10 invariant re-specification (proposal)

**Status:** proposal for user adjudication. This file is the only artifact
written by task-16; `docs/TESTING.md`, `docs/phase4-5-design.md`, the
ledger, events, payloads, executor, tests, and `formal/` are owned by the
Round-10 implementation track (physics-integrator) and were not edited
here. Code is still catching up with the corrected spec; this memo reasons
against the corrected ledger, not against the old code shape.

**Basis:** `v0.10-untested.md` (Round-10 spec correction, 27 flows) and the
corrected `amateur/ledger.py` in the worktree. The void readings are re-specified below
(four invariants, the D2 write whitelist, and the crystal lexicon). The old wording is quoted, the reason it is void is
stated, and each replacement carries a falsifier and a proof/test status.

**Independent census check.** I recomputed the census from the corrected
ledger with a node-distinct simple-cycle enumeration (canonical rotation,
parallel edges collapsed) and reproduce the Lead's figures exactly:
**26 simple cycles, 24 interior, 2 boundary; membership D1=12, D2=11,
D3=15, D4=4; D4 minimum cycle length 5; D3 has no 2-cycle; D2 contains
D2→P2→P3→D2.** The edge-distinct count would be 27 because E1↔P1 has two
parallel return edges (#4, #5); the adjudicated convention is node-distinct
and that is the convention used here. This arithmetic is stated because the
old census was a hard-coded list and the new one must be a derived property.

---

## 1. INV-GEOMETRY — three tiers, with D2 as a nursery

### 1.1 Old wording (void)

`docs/TESTING.md` (old matrix, row INV-GEOMETRY):

> adjudicated two-tier reading: structural (writer sets) + proven
> (type-system purity); D2/D4 are the parametric geometry written only by
> 6.0, D1/D3 operational by design

and `tests/test_invariants.py` `TestGeometry` asserted the writer map
`{"D1": {"P4"}, "D2": {"P6"}, "D3": {"P5"}, "D4": {"P6"}}` with D2's writes
`{23, 24}`. `formal/DryRun.lean` encoded the same premise as
`d2Allowed = [PayloadKind.overlay]`, `d2WriteFids = [23, 24]`,
`d4Allowed = [PayloadKind.bias]`, `d4WriteFids = [22]`, and the theorem
`geometry_write_channel_purity` ("forbidden payload kinds are absent from
both geometry channels").

### 1.2 Why the corrected graph voids it

| Fact | Old | Corrected |
| --- | --- | --- |
| D2 writers | P6 only (`#23`, `#24`) | **P3 (`#18`) and P6 (`#24`)** |
| #23 | P6→D2 write | **D2→P6 read** |
| #18 | P3→P5 (to digestion) | **P3→D2 (nursery birth)** |
| D2 reads | #8, #25 | **#8, #23, #25** (three readers with different roles) |
| D4 | P6-only, parametric | unchanged (P6 `#22` only) |
| D1 / D3 | operational | unchanged (D1←P4 `#13`; D3←P5 `#19`), D3 now has #27 exit |

The old two-tier reading and every theorem built on D2 being ratchet-only
are false under the corrected spec. The D2 write whitelist `{23, 24}` is
therefore **void** and must be replaced (new writers `{18, 24}`, #23 a
read).

### 1.3 Proposed wording

**Three-tier store classification (structural, derived from the ledger):**

| Tier | Store | Writer set | What it may hold | Geometry status |
| --- | --- | --- | --- | --- |
| **Parametric** | D4 Base Core | `{P6 #22}` | `BiasPrimitive` only | The only parametric geometry; ratchet-only; trajectory-free by type whitelist |
| **Nursery** | D2 Multi-Layer Mat | `{P3 #18 candidate birth, P6 #24 ratchet write}` | `MatOverlay` records with `kind ∈ {candidate, ratcheted}` and `status ∈ {active, stale}` | Provisional candidates + ratcheted overlays coexist; only ratcheted records are geometry-grade |
| **Operational** | D1 Elastic Board, D3 Debris Pool | D1 `{P4 #13}`, D3 `{P5 #19}` | operational state (context slots, debris) | not geometry; no mat/patch purity claim; D3 has #27 exit |

**Payload-purity consequence — what D2 may hold and how candidates are
kept from being sedimented geometry:**

1. `MatOverlay` gains a **`kind`** discriminator (`"candidate"` |
   `"ratcheted"`) beside `epoch`/`overlay_id`/`status`. The exact-type
   channel whitelist still applies at the class level; the body validator
   pins the kind to the edge:
   - `#18` (P3→D2) writes **only** `kind="candidate"`, `status="active"`;
   - `#24` (P6→D2) writes **only** `kind="ratcheted"`, `status="active"`;
   - `#8` (D2→P2) reads **only active ratcheted records** (candidates are
     not routing bias; this is the proposed filter — see §7 Q1);
   - `#23` (D2→P6) reads the candidates that survived in the nursery (the
     selector for promotion);
   - `#25` (D2→P5) reads stale records of either kind for digestion.
2. **A candidate is never patch-bearing.** `MatBoard.attach_patch`
   asserts `record.kind == "ratcheted"`; a candidate cannot acquire a
   predictor patch before promotion. `assert_patch_bounds(require_complete)`
   counts ratcheted records only; a resident candidate is explicitly
   exempt, not an error.
3. **Only ratcheted records may be sedimented.** `#22` consumes the #23
   survivor selection and writes `BiasPrimitive` to D4; no candidate path
   to D4 exists.
4. **Only P6 writes D4**, and D4 accepts only `BiasPrimitive`. The
   trajectory-free core guarantee is unchanged and is now the *only*
   store-level purity theorem that remains true as stated.

**Invariant, one sentence:** *D2 is a nursery in which candidate records
and ratcheted overlays may coexist, but a candidate is never treated as
sedimented geometry: it is kind/status tagged, never patch-bearing, never
read by P2 as mat bias, and never written to D4; promotion to ratcheted
happens only through #24.*

### 1.4 Falsifier

Any of: a D2 writer outside `{P3, P6}`; a candidate record with an attached
patch; a #8 read that includes a candidate; a candidate in `board.d4`; a
`#18` body not `kind=candidate`; a `#24` body not `kind=ratcheted`; a D4
writer outside `{P6}`; a D1/D3 record treated as geometry (patch attached,
or a D4-style purity claim applied); a #23 read that cannot distinguish
survivors from ratcheted records.

### 1.5 Proof / test status

| Claim | Status |
| --- | --- |
| Three-tier writer sets (D4 `{P6}`, D2 `{P3,P6}`, D1 `{P4}`, D3 `{P5}`) | **Python structural test** from the ledger (`TestGeometry` rewrite); independent of execution |
| `#18` candidate-only / `#24` ratcheted-only | **Python exact-type + body-validator test**; schema change is implementation work |
| Candidate cannot be patch-bearing; only ratcheted patch-counted | **Python guard test** on `MatBoard.attach_patch`/`assert_patch_bounds` |
| D4 parametric purity (`#22`→`BiasPrimitive`, no trajectory/delta/context) | **Provable in Lean** as a D4-only theorem; the old D2 half of `geometry_write_channel_purity` is void |
| D2 nursery kind purity | **Python type/validator test** for now (`TestGeometryWriters`, `TestNurseryKinds`, `TestCandidateCannotBePatched`); a Lean formulation needs the adjudicated `MatOverlay` schema and is not proposed in this round |

**Assumed / needs adjudication:** the #8 candidate filter and the exact
`#23` read filter; whether `kind` is a field or a separate exact-type pair
(§7 Q1). The proposal does not weaken D4's trajectory-free by-construction
claim.

---

## 2. INV-CENSUS — a derived structural fingerprint, not a hard-coded list

### 2.1 Old wording (void)

Old `docs/TESTING.md` row INV-CENSUS:

> resolved (raw membership, adjudicated); census: |Cycles(D1)|=7 (entangled
> hub), |Cycles(D2)|=4, |Cycles(D3)|=4, |Cycles(D4)|=3 with zero 2-/3-hop
> loops — D4's structural insulation

and `tests/test_recurrence.py` hard-coded `EXPECTED_CYCLES` with exactly 17
interior node-canonical cycles.

### 2.2 Why the corrected graph voids it

The census is a function of the ledger. Seven corrected edges change both
the cycle set and the membership counts; the old numbers cannot be
"updated by editing constants" while keeping the old meaning, and a
hard-coded cycle list would be a transcription of a particular reading,
which is exactly the failure mode Round 10 is correcting. The old reading
is void.

### 2.3 Proposed wording

**Definition (must be part of the wording):** a *node-distinct simple
directed cycle* is a sequence of ≥2 distinct nodes `n0..n(k-1)` with an
edge `n_i -> n_(i+1 mod k)` for every step; parallel edges between the same
pair of nodes collapse to one cycle; each cycle is counted once under
canonical rotation (rotate to the lexicographically smallest node
sequence). The census is **computed from `FLOW_LEDGER` at test time**; the
only literals in the test are the expected aggregate values, never a
hard-coded list of cycles.

**Derived census of the corrected ledger (computed, independently
reproduced):**

| Property | Value |
| --- | --- |
| total simple cycles | **26** |
| interior cycles (`{P1..P6,D1..D4}` only) | **24** |
| boundary cycles (touch E1/E2) | **2** (`E1↔P1`, `E2↔P1`) |
| cycles containing D1 | **12** |
| cycles containing D2 | **11** |
| cycles containing D3 | **15** |
| cycles containing D4 | **4** |
| 2-cycles | D1↔P4, D2↔P6 (interior); E1↔P1, E2↔P1 (boundary) |
| D3 2-cycle | **none** (D3 minimum cycle length 4) |
| D4 shortest cycle | **5** (no ≤3-hop loop) |
| D2 nursery loop | **D2→P2→P3→D2** (edges #8, #9, #18) |

**What the census is evidence for:** structural richness and non-degeneracy
— every store participates in at least one recurrence (D1 12, D2 11, D3 15,
D4 4), D2 has the nursery loop that makes candidates reachable from the
observation loop, D3 remains the entangled digester without a bare 2-cycle,
and D4 is structurally insulated (shortest loop 5). It is **not** evidence
for dynamics, correctness, or liveness; it is a topology fingerprint that
changes iff the ledger changes.

**Invariant, one sentence:** *The ledger's derived node-distinct cycle
census is exactly the corrected fingerprint above; a ledger edit that
changes any aggregate or shape property is a spec change and must fail the
census test.*

### 2.4 Falsifier

Any discrepancy between the computed census and the table above; a store
with zero cycles; a D4 cycle of length ≤3; a D3 2-cycle; a missing
D2→P2→P3→D2 loop; a count that only matches because a hard-coded list was
updated instead of recomputed from the ledger.

### 2.5 Proof / test status

| Claim | Status |
| --- | --- |
| Census values | **Python derived test**: enumerate from `FLOW_LEDGER`, canonicalize, assert aggregates/shape properties; remove the old hard-coded `EXPECTED_CYCLES` list |
| Graph topology | Not modeled in Lean; no Lean change is justified by this invariant |
| "evidence for structural richness" | Assumed interpretation; the numbers themselves are computed, the evidential reading is a design claim |

---

## 3. INV-CYCLE-DELAY — stage inversion + store-read injection, not cycle count

### 3.1 Old wording (void)

Old `docs/TESTING.md` row INV-CYCLE-DELAY and `dryrun`/test docstrings:

> the two in-tick cycles (#10 Resolved Trajectory, #17 Collapse Mandate)
> are one-tick-delayed mailboxes; INV-CYCLE-DELAY: proven + cycle check
> pinned — the two delayed cycles themselves are design

The justification was cycle-based: the old graph had a direct P4↔P5 cycle
through #16, so #10/#17 were described as "the two in-tick cycles".

### 3.2 Why the corrected graph voids it

#16 moved from P4→P5 to **P3→P5**. The direct P4↔P5 cycle is gone, so the
old cycle-based sentence no longer names the right reason. The delay set
{#10, #17} survives; the justification does not.

### 3.3 Proposed wording

**Stage order:** `P1 < P2 < P3 < P4 < P5 < P6` (the fixed spine order).
**Backward process edge:** a process→process edge whose target stage is
earlier than its source stage.

**Delay set = {#10, #17}.** They are exactly the backward process edges:

| Edge | Direction | Consumer stage < producer stage | Delay |
| --- | --- | --- | --- |
| #10 Resolved Trajectory | P3 → P1 | 1 < 3 | one-tick mailbox |
| #17 Collapse Mandate & Freed Slots | P5 → P4 | 4 < 5 | one-tick mailbox |

Every other process→process edge is stage-forward:
`#6 P1→P2`, `#9 P2→P3`, `#12 P3→P4`, `#16 P3→P5`. Entity edges are the
ports (#1/#3 in, #2/#4/#5 out). All **store reads** are injected at tick
start from state committed in earlier ticks and therefore need no mailbox
delay:
`#7, #8, #11, #14, #15, #20, #21, #23, #25, #26, #27`.

**Cycle-cut lemma (derived, not hard-coded).** Every one of the **24
interior cycles** contains at least one *cut*: a store-read injection or
#10/#17. On the corrected ledger:

| Cycles | Count | Cut |
| --- | --- | --- |
| store-read only | **20** | a t-start store read |
| store-read + #17 | **3** | both |
| #10 only | **1** (`P1→P2→P3→P1`, edges #6, #9, #10) | #10 |
| no cut | **0** | — |

No interior cycle contains both #10 and #17, so **at most one delayed edge
per cycle**; the 24 cycles do not need 24 delays, they share the same two
backward edges plus the store-read injections.

**Sufficiency.** Delay #10 and #17 by one tick; inject every store read at
tick start from the prior committed state. Then all remaining in-tick
dependencies respect the stage order P1..P6: forward process edges emit
before their consumers, and store reads are available before stage 1. A
single stage-topological pass per tick is therefore causal, and every
interior cycle has at least one cut (cycle-cut lemma), so no cycle demands
same-tick backward traversal.

**Minimality.** Each of #10 and #17 is individually necessary: its consumer
stage precedes its producer stage under the fixed spine, so a same-tick
hand-off would require consuming before producing. Both appear in at least
one interior cycle (#10 in one, #17 in three), so removing either delay
breaks a cycle's causality. No third delay is needed: every other
process→process edge is stage-forward, and every store edge is a t-start
read injection.

**Invariant, one sentence:** *The one-tick delayed set is exactly the two
stage-backward process edges #10 and #17; every other interior dependency
is either stage-forward or a tick-start store-read injection, so every one
of the 24 interior cycles is cut by at most one delayed edge and no third
delay exists.*

### 3.4 Falsifier

A process→process edge other than #10/#17 with target stage < source stage;
an interior cycle with no store read and no #10/#17; same-tick consumption
of #10 or #17 (INV-CAUSAL failure); a mailbox at depth >1 outside
{#10,#17}; a proof or test that treats the delay set as a per-cycle
resource rather than per-edge.

### 3.5 Proof / test status

| Claim | Status |
| --- | --- |
| Only #10/#17 are stage-backward | **Python structural test** from the ledger and spine |
| Every interior cycle has a cut (20/3/1/0 split) | **Python derived test** (cycle enumeration + edge-set membership) |
| Sufficiency / minimality | **Python structural argument + tests**: remove #10/#17 and exhibit the broken backward edges; no Lean topology theorem is proposed |
| Executor profile | **Empirical**: in-transit `{#10: 1, #17: 1}`; INV-CAUSAL structural test |
| Lean | `boxes_master`/`cycle_warm_*` remain schedule-relative and consume a generated 27-flow table; they prove the plumbing profile, **not** the census or the cycle-cut lemma. Old "cycle justification" wording is withdrawn |

---

## 4. INV-EXHAUST — sinks, the dissipation chain, and #27

### 4.1 Old wording (void)

Old reading (v0.10 disambiguation, `test_thermodynamics`, INV-EXHAUST row):

> Dissipated Entropy edge deleted (was D3 -> 1.0). Degraded throughput does
> not circulate: it terminates at the boundary exhaust (1.0 -> E1), while
> only distinctions recur — residue already re-enters via Avoidance Trails
> (#26), Surviving Crystals (#23), and Consumed Scaffolds (#20).

with tests asserting "no residue edge back into transduction"
(`not (D3 -> P1)`) and "Entropy Exhaust is the only degenerate outflow".

### 4.2 Why the corrected graph voids it

#27 `D3 -> P1` **Dissipated Entropy (read)** is restored. The old
no-residue-edge test is false; the old "only entropy-named flow" test is
false because #27 also contains "Entropy". The old reading also named #23
and #20 with their old endpoints (#23 was P6→D2; #20 was D3→P5), so its
list of recurrences is stale. The old INV-EXHAUST wording is void.

### 4.3 Proposed structure (independent of the #27 reading)

**Terminal sinks** (target is an entity): **#2** Crystallized Output
(P1→E2), **#4** Active Probes (P1→E1), **#5** Entropy Exhaust (P1→E1).
No entity→interior edge exists except the fresh inputs #1/#3; entities do
not return consumed material (the blanket port structure).

**Dissipation chain:** D3 holds digested debris (#19 in). Its degraded
throughput read is **#27 D3→P1**; P1 can then emit the terminal exhaust
**#5 P1→E1** (and its lateral outputs #2/#4). #27 is therefore an
*upstream read* of dissipated material, not a sink; #5 is the terminal
exhaust. The chain is `D3 --#27--> P1 --#5--> E1`, with #2/#4 as the other
terminal boundary outputs.

**Circulating state vs dissipating throughput:**

- *Circulating state* = distinctions/records that are re-read and reused:
  D1 context (#11/#14), D2 overlays/candidates (#8/#23/#25), D3 debris and
  trails (#20/#21/#26), D4 bias (#7). Re-entry through a store read is
  state recurrence.
- *Dissipating throughput* = quantity that has been degraded by
  digestion into D3 and leaves toward an entity boundary without being
  re-stored as state. #27 is the only D3→P1 edge; its content must be
  classified before INV-EXHAUST can be asserted. #26 (avoidance trail) is
  state/bias; #27 (dissipated entropy) is the candidate dissipation path.

The invariant to adjudicate: *degraded D3 throughput that crosses #27 must
terminate at a terminal sink (#2/#4/#5) and must not re-enter a store as
state; distinctions may recur, dissipated throughput may not.*

### 4.4 The three candidate readings of #27 (user adjudicates; no choice made)

| Reading | #27 is | Identity A (`attempted = accepted + refused`) | Identity B (`accepted = consumed + sunk + depth`) | Liveness | Sink profile | Provenance/carrier | Lean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Flow** | an event/read injected at tick start, consumed by P1 | applies: one #27 attempt per tick (accept/refuse on the capacity-1 mailbox) | applies: #27 accepted is consumed by P1; no sunk, depth 0; the *content* may be routed to #5 or (if allowed) #6 | 27/27 flows live; emitted once per tick | unchanged: terminal sinks #2/#4/#5; #27 is interior | A2 envelope; `source_fid=19` (D3 write that bore the stored record), `tick_born < emit_tick`; typed body needed | 27-arm op table; A/B lemmas per flow |
| **Charge** | a scalar deficit/excess-like charge at P1/D3, no event | A/B apply only to the 26 **event** flows; #27 is a charge, not a counted attempt | unchanged for events; the charge has separate counter(s), never mixed with refuse/miss | liveness covers events only; #27 is not an emitted event | unchanged: #2/#4/#5; charge is not a sink | no payload/envelope/provenance (charges are scalar, exactly like refuse/miss) | 26-arm event table + scalar charge accounting; Lean table not extended with #27 |
| **Scalar drain** | a per-tick scalar amount drained from a D3 pool to P1/E1, no event | A/B unchanged for events; drain has its own conservation identity | unchanged for events; drain may be recorded as a scalar sink/dissipation term | liveness excludes #27 (or reports a drain count, not an event) | terminal sinks unchanged; a scalar drain term is added (not a flow sink) | no payload/envelope/provenance | requires a scalar-pool/drain model; not the current event Lean |

**What the user's choice changes:** whether #27 is in the event op table
and liveness (flow), whether it is a charge with no payload (charge), or a
scalar drain with a conservation law (drain); whether the corrected census
is *also* an event-cycle census (flow) or only a topology census (charge/
drain); and whether #27 needs a typed body/provenance. The three readings
are not interchangeable for A/B and liveness, so INV-EXHAUST cannot be
pinned until one is chosen.

**Falsifier (per reading):** flow — a cycle carrying #27 material back into
a store as state without a terminal sink; A/B fail when #27 is counted;
liveness ≠27. charge — a refused/missed #27 being counted as an event; A/B
being applied to it; a charge summed with refuse/miss. drain — no
conservation identity between D3 decrement, #27 drain, and #5 exhaust; a
drain counted as an event. Across all readings: a second terminal
"Entropy" flow counted as exhaust; a non-terminal sink; any path from D3
material to a store after #27.

### 4.5 Proof / test status

| Claim | Status |
| --- | --- |
| Terminal sinks are exactly #2/#4/#5; no entity return except #1/#3 | **Python structural test** (ledger ports) |
| Reachability from D3 to a terminal sink exists (via #27→P1→#5) | **Python graph test** |
| A/B under the chosen #27 reading | **Provable in Lean only if #27 is an event flow** (27-op table); for charge/drain, the event identities are unchanged and the charge/drain needs a separate scalar accounting lemma/model |
| Dissipation-vs-state classification | **Assumed pending user adjudication**; this memo deliberately does not choose |

---

## 5. Crystal lexicon rewrite

### 5.1 Old lexicon (void)

The old pinned lexicon (`v0.10-untested.md`, superseded text, and
`tests/test_design_intent.TestCrystalLexicon`):

> *Emergent Crystal* = candidate (3.0 -> 5.0); *Structural Motif* =
> extracted rule (5.0 -> 6.0); *Surviving Crystal* = vetted, ratcheted
> structure (6.0 -> D2).

Void because #18 now enters **D2**, the ratchet reads D2 (#23) and mines
D3 (#21), P5 no longer emits or receives crystals, and #24 is the D2
write-back. The old three terms name the wrong edges.

### 5.2 Proposed lexicon (each term pinned to a corrected edge)

| Term | Corrected edge | Referent | D2 kind/status | Distinct from |
| --- | --- | --- | --- | --- |
| **Emergent Crystal** | **#18 P3→D2** | a candidate born into the nursery by the observation loop | `kind=candidate`, `status=active` | Structural Motif, Ratcheted Overlay |
| **Nursery candidate** | (record kind; born by #18) | the D2 record for an emergent crystal | `kind=candidate` | ratcheted overlay; sedimented invariant |
| **Surviving Crystal** | **#23 D2→P6 (read)** | the D2 resident(s) the ratchet reads as survivors; a **selection/read relation**, not a stored kind | read only; no new kind | Ratcheted Overlay (#24), Sedimented Invariant (#22) |
| **Structural Motif** | **#21 D3→P6 (read)** | the rule the ratchet extracts from D3 debris; no longer a P5→P6 flow | read only (D3 body) | Emergent Crystal, Digested Debris |
| **Ratcheted Overlay** | **#24 P6→D2** | the promoted/certified record written back to the nursery | `kind=ratcheted`, `status=active`; patchable | Surviving Crystal (read), candidate |
| **Sedimented Invariant** | **#22 P6→D4** | the parametric-geometry write (bias primitive) | D4 body `BiasPrimitive` | Ratcheted Overlay (D2), candidate |
| **Digested Debris** | **#19 P5→D3** | debris material deposited in the pool | D3 body | Structural Motif (read) |
| **Stale Overlay** | **#25 D2→P5 (read)** | a stale D2 record handed to digestion | read only | Surviving Crystal (#23), Mat Overlay (#8) |
| **Mat Overlay** | **#8 D2→P2 (read)** | the routing-bias read of active ratcheted records | read only (ratcheted, active) | Stale Overlay, candidate |
| **Consumed Scaffold** | **#20 D3→P4 (read)** | D3 material recycled into regulation | read only | Structural Motif, Digested Debris |
| **Avoidance Trail** | **#26 D3→P2 (read)** | D3 state as routing bias | read only | Dissipated Entropy (#27) |
| **Dissipated Entropy** | **#27 D3→P1 (read)** | the degraded-throughput read into transduction (reading pending §4) | read only / charge / drain (pending) | Entropy Exhaust (#5, terminal) |

### 5.3 Names with no distinct referent (must be flagged, not silently reused)

1. **"Surviving Crystal"** now names a **read** (#23), not a stored kind.
   The old referent — "vetted, ratcheted structure 6.0→D2" — is now
   **Ratcheted Overlay** (#24). If the user wants a stored survivor kind,
   it must be defined as a D2 `status` and pinned; otherwise the lexicon
   should say "survivor read" explicitly.
2. **"Emergent Crystal"** is the #18 event; the D2 record kind should be
   called **candidate** in payload/schema prose to avoid conflating event
   and record.
3. **"Structural Motif"** is a D3→P6 read; the old "extracted rule out of
   5.0" is gone. Any text that says "5.0 emits motifs" is stale.
4. **"Crystallized Output"** (#2, P1→E2) is an entity output, not a D2
   crystal; it must not be folded into this lexicon.
5. **"Mat Overlays"** (#8 read) vs **"Ratcheted Overlays"** (#24 write)
   vs **"Stale Overlays"** (#25 read) are three different things; only
   ratcheted active records are #8 material.
6. **"Dissipated Entropy"** (#27 read) vs **"Entropy Exhaust"** (#5
   terminal) must remain distinct; the old "only exhaust" phrasing is
   false.

### 5.4 Falsifier

A term with no corrected edge; two terms naming the same edge; a term
pinned to the removed `5.0→6.0` edge; "Surviving Crystal" used as a stored
kind without a defined status; #18 called a motif; #24 called a survivor
read; #21 described as P5 output; #27 conflated with #5.

### 5.5 Proof / test status

**Python/ledger test:** a term→edge map asserting source/target and read
markers (#18 target D2; #23 source D2; #21 source D3; #24 target D2; #22
target D4; #20 target P4; #25 target P5; #27 target P1), the absence of a
5.0→6.0 crystal edge, and the D2 kind tags. Names are not Lean content;
no Lean change is proposed for the lexicon.

---

## 6. Prior adjudications left untouched (nothing here relaxes them)

| Adjudication | Status in this proposal |
| --- | --- |
| **Reject-with-charge** and its four binding conditions (capacity 1; first-arrival-wins with ledger order; scalar excess separate from deficit miss, never summed; identities A/B preserved under refusal) | **untouched.** #27's reading (§4) may add a charge or drain, but it may not change mailbox admission, refusal semantics, or A/B for the existing event flows |
| **Theorem discipline** (no axioms/`sorry`; generated Lean table; explicit schedule hypotheses; policy-relative vs cadence-free classification) | **untouched.** The new census/cut properties are Python/structural; Lean regeneration for 27 flows is implementation work, not a change to the discipline |
| **Cadence quarantine** (uniform cadence is plumbing; lifecycle is store/organ-owned until trigger-gating) | **untouched.** No trigger-gating is proposed here; #27's cadence is part of the reading decision |
| **Provenance gate** (A2 envelope, carrying/ordering; no minting; #13/#16 deferred) | **untouched.** If #27 is a flow, it inherits the store-read rule; if charge/drain, it carries no payload, which must be stated rather than silently minted |
| **Typed carriers** (exact-type per-fid body whitelist; D4 `BiasPrimitive`; envelope transport) | **discipline untouched**, but the **D2 write whitelist is void** as a prior adjudication (new writers `{18,24}`; #23 a read). The replacement is the D2 nursery kind whitelist of §1, not a relaxation of exact typing |
| **Q4 zero-confidence #22** | **untouched.** #22 remains the derived gate output; D4 sediment acceptance unchanged |
| **Encoder frozen / bit-identical** | **untouched.** No physics-model change is proposed |

The voided items are exactly those the user named: the four invariants
INV-GEOMETRY, INV-CENSUS, INV-CYCLE-DELAY, and INV-EXHAUST, plus the D2
write whitelist and the crystal lexicon. Nothing in this proposal reopens reject-with-charge,
theorem discipline, cadence quarantine, provenance, Q4, or the frozen
encoder.

---

## 7. Decisions needed from the user

1. **D2 nursery representation.** Field `kind ∈ {candidate, ratcheted}` on
   `MatOverlay` (proposed) vs two exact types (`CandidateOverlay`,
   `RatchetedOverlay`); and the exact read filters: #8 sees only active
   ratcheted records (proposed), #23 reads active candidates, #25 reads
   stale of either kind. The choice fixes the payload schema and the
   candidate/patch test.
2. **#27 reading.** Flow (27/27 events, A/B and provenance apply) vs
   charge (26 events + scalar charge) vs scalar drain (26 events + drain
   conservation). This blocks INV-EXHAUST, the sink profile wording, and
   whether #27 needs a typed carrier.
3. **Census convention.** Confirm node-distinct (26 total / 24 interior;
   the Lead's figures, reproduced here) as the adjudicated count; the
   edge-distinct alternative is 27/24 because E1↔P1 has parallel returns.
   Also confirm the census remains an invariant at all, versus a derived
   fingerprint with no status.
4. **INV-CYCLE-DELAY wording.** Accept the stage-inversion + store-read
   injection justification and the edge-level delay set {#10,#17}; keep
   the INV-CYCLE-DELAY id, or rename it to an edge-level `INV-DELAY` to
   stop future cycle-based reasoning.
5. **Surviving Crystal.** Keep "Surviving Crystal" as the #23 read name
   (with the old stored-kind meaning explicitly withdrawn), or define a
   stored `surviving` status; in either case, #24 Ratcheted Overlay and
   #22 Sedimented Invariant remain distinct.
6. **Implementation sequencing.** The corrected ledger is already in the
   worktree; the schema/tests/Lean updates remain physics-integrator's.
   This memo only names the invariants and their tests.

## Appendix — reproduction of the census

Node-distinct simple cycles, canonical rotation, from the corrected
`FLOW_LEDGER` (27 flows):

- total **26**; interior **24**; boundary **2** (`E1↔P1`, `E2↔P1`);
- membership **D1=12, D2=11, D3=15, D4=4**;
- 2-cycles: D1↔P4, D2↔P6 (interior); E1↔P1, E2↔P1 (boundary);
- D3 has no 2-cycle (minimum length 4); D4 minimum length 5;
- D2 nursery loop: `D2→P2→P3→D2` (edges #8, #9, #18);
- delay-cut split over the 24 interior cycles: 20 store-read only, 3
  store-read + #17, 1 #10 only, 0 uncut; no cycle contains both #10 and
  #17.
