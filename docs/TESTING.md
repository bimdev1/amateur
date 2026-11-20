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
rejected). Flow ids derive from the ledger itself: `D2_FLOWS = {8, 23, 24,
25}`, `D4_FLOWS = {7, 22}`, with ratchet-owned writes `D2_WRITES = {23, 24}`
and `D4_WRITES = {22}`.

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
is the ratchet gate's output, #23/#24 are the organ's records), but the
default `DryRun` keeps `SyntheticSource` and the derivation half remains
**partial** overall — #2/#4/#5/#9/#13/#16 and all D1/D3 reads are still
placeholders, and P5 does not consume #15/#16/#20.

The mat lifecycle is one closed policy — mirrored byte-for-byte by the Lean
`matTick`/`matRun` recursion and by the executor's `MatBoard`: seed one
overlay (`"ovr-0"`, epoch 0); each tick *t* append a birth (#23, epoch *t*),
revalidate the head at epoch *t* (its stale snapshot was just read by #25 at
the same stage), then evict the head iff the board exceeds capacity 4.
Consequences proved for all ticks in Lean: `mat_operational` (closed form
`replicate 2 (t+2) ++ replicate 2 (t+3)`), `mat_fresh` (every resident epoch
is *t* or *t−1* — nothing ossifies), `mat_bounded` (≤ 4 residents).

`test_payloads.py` covers the schema, the subclass/trajectory rejection, the
lifecycle, and 200-tick executor integration (no trajectory ever reaches D4;
#25→#24 same-tick pairing on `overlay_id`). `test_provenance.py` covers the
carrier, the exact-type interaction, the per-channel birth rule, executor
read continuity (#7 identity, #8/#25 inherited provenance, seed `(23, 0)`),
and the structural anti-mint evidence (explicitly labelled evidence, not a
proof of derivation).

### Provenance coverage by flow (round 8, measured)

`synthetic` = no body (`None` placeholder) or a body with no state
dependency; `derived` = the body is the stored record or a function of live
node/store state at emission. The `source_fid` column is the producing-flow
attribution carried in the envelope.

| # | Flow | Body this epoch | Coverage | `source_fid` |
| --- | --- | --- | --- | --- |
| 1 | Gradients (E2→P1) | `None` | synthetic | 1 |
| 2 | Crystallized Output (P1→E2) | `None` | synthetic | 2 |
| 3 | Perturbations (E1→P1) | `None` | synthetic | 3 |
| 4 | Active Probes (P1→E1) | `None` | synthetic | 4 |
| 5 | Entropy Exhaust (P1→E1) | `None` | synthetic | 5 |
| 6 | Prediction Error (P1→P2) | `None` | synthetic | 6 |
| 7 | Base Ops (D4→P2) | stored `BiasPrimitive` | **derived** | 22 |
| 8 | Mat Overlays (D2→P2) | stored `MatOverlay` | **derived** | 23/24 |
| 9 | Routed Dynamics (P2→P3) | `None` | synthetic | 9 |
| 10 | Resolved Trajectory (P3→P1) | `None` | synthetic | 10 |
| 11 | Active Context (D1→P3) | `None` | synthetic | 11 |
| 12 | State Deltas & Surprise (P3→P4) | `None` | synthetic | 12 |
| 13 | Board Ops (P4→D1) | `None` | synthetic | 13 |
| 14 | Slot State (D1→P4) | `None` | synthetic | 14 |
| 15 | Evicted Clusters (D1→P5) | `None` | synthetic | 15 |
| 16 | Cost vs. Error (P4→P5) | `None` | synthetic | 16 |
| 17 | Collapse Mandate (P5→P4) | `None` | synthetic | 17 |
| 18 | Emergent Crystals (P3→P5) | `None` | synthetic | 18 |
| 19 | Digested Debris (P5→D3) | `None` | synthetic | 19 |
| 20 | Consumed Scaffolds (D3→P5) | `None` | synthetic | 20 |
| 21 | Structural Motifs (P5→P6) | `None` | synthetic | 21 |
| 22 | Sedimented Invariants (P6→D4) | rotating `BiasPrimitive` | **synthetic** (minted content) | 22 |
| 23 | Surviving Crystals (P6→D2) | `cry-{tick}` `MatOverlay` | **synthetic** (minted content) | 23 |
| 24 | Ratcheted Overlays (P6→D2) | head identity + tick epoch | **derived** | 24 |
| 25 | Stale Overlays (D2→P6) | stored `MatOverlay` | **derived** | 23/24 |
| 26 | Avoidance Trails (D3→P2) | `None` | synthetic | 26 |

Summary for the default **synthetic dry run** (`SyntheticSource`):
**4/26 derived** (#7, #8, #24, #25); 22/26 synthetic (20 `None`
placeholders + #22 + #23). All store reads with a non-`None` body are
derived; D1/D3 reads stay synthetic because those stores are unimplemented
in the dry run.

Physics mode (`PhysicsRun` / `PhysicsSource`, Q1–Q4 = A) moves ten more
flows to `derived`, so **11/26 derived** in physics mode:

| # | Physics-mode body | Class |
| --- | --- | --- |
| 3 | perturbation frame | `FrameBody` |
| 6 | surprise + 64-d residual | `PredictionErrorBody` |
| 10 | 4-step latent rollout | `LatentRolloutBody` |
| 12 | 64-d latent delta + surprise | `StateDeltaBody` |
| 17 | digest collapse mandate | `CollapseMandateBody` |
| 18 | emergent crystal residual | `EmergentCrystalBody` |
| 19 | digested debris centroid | `DigestedDebrisBody` |
| 21 | structural motif | `StructuralMotifBody` |
| 22 | the real ratchet gate's output (blocked = confidence 0; round 9a passes on real evidence) | `BiasPrimitive` |
| 23/24 | organ birth / revalidation record | `MatOverlay` (unchanged) |

Still placeholders in physics mode, by Lead decision: #1 (E2 gradient),
#2 (crystallized output), #4/#5 (E1 exhaust), #9 (routed dynamics), #13
(D1 write), #16 (P4 cost/error — P4 is unimplemented). D1 reads
(#11/#14/#15) and D3 reads (#20/#26) remain placeholders: D1 is
unimplemented and D3 is written but not read this round. The trigger-gated
flows carry exact-type **no-op sentinels** on non-event ticks (`ovr-noop`,
`debris-none`, `motif-none`, `none`), and #22 carries the Q4 gate body
(confidence 0 when blocked; round 9a's capacity-bounded patch passes on
real evidence, so physics-mode #22 can carry a positive confidence and D4
can sediment real invariants); the sentinels are constants, not derived
content, and are never retained in D2/D3/D4. The coverage ledger is plain test-only data — no
executor behaviour branches on it. See
[phase4-5-design.md](phase4-5-design.md) §14.

## Layer 2 — Pointwise recurrence, proved (`amateur/dryrun.py`, `formal/DryRun.lean`, `tools/gen_lean.py`, `test_lean_lockstep.py`)

Because each flow's mailbox sees only its own op word, the whole executor
reduces to a per-flow recurrence. `step_flow(fid, box)` is that pure step;
the Lean file is generated from the same table and proves its long-run
shape for *all* ticks, not just sampled ones. The channel semantics are
the round-7 adjudicated ones (capacity one, first-arrival-wins,
reject-with-charge; deficits and excesses are distinct scalar charges,
never summed):

| Theorem | Statement | Cadence |
| --- | --- | --- |
| `boxes_bounded` | every mailbox depth <= 1 at every tick boundary (cold start) | tick-induction over the dry-run op table (plumbing cadence) |
| `boxes_master`  | for `t >= 1`, depth profile is exactly `{10: 1, 17: 1}`, zero elsewhere |
| `misses_master` | cumulative misses are exactly one on each cycle, frozen after tick 1 |
| `sunk_master`   | `sIter t i = t` for the sinks (#2, #4, #5), 0 elsewhere |
| `refused_master` | cumulative refusals are exactly zero for all ticks >= 1 under the canonical schedule | schedule-pinned (plumbing cadence) |
| `fold_attempt` / `applyOp_attempt` | identity A, word-generic over any op sequence: refused + used grows by exactly the word's emit weight (`attempted = accepted + refused`), event by event | cadence-free (enduring) |
| `fold_take` / `applyOp_take` | word-generic: consumed + miss grows by the word's consume weight; a consume on an empty box is a miss and removes nothing | cadence-free (enduring) |
| `fold_event` / `applyOp_event` | identity B, word-generic: accepted (`used`) = consumed + sunk + delta-depth, carried through every event; refused copies never enter | cadence-free (enduring) |
| `fold_append` | folds compose over concatenated words (`flowFold (l1 ++ l2) st = flowFold l2 (flowFold l1 st)`) | cadence-free (enduring) |
| `fold_box_bounded` / `step_contraction` | INV-BOUNDS intra-tick (adjudicated): from any depth <= 1, depth <= 1 after *every prefix* of the event sequence | cadence-free (enduring) |
| `emit_full_box` | multiplicity k onto a full box: the held event stays, all k extra copies refused as scalar charges, nothing buffered | cadence-free (enduring) |
| `emit_empty_box` | multiplicity k + 1 onto an empty box: first arrival accepted, k refused | cadence-free (enduring) |
| `cycle_warm_no_refusal` | cycle check pinned as a lemma: with the ordering hypothesis `opsOf i = [Op.consume, Op.emit]` lifted explicitly, consume empties the delay box before emit refills it -- refusal never fires on #10/#17 | policy-relative, ordering hypothesis explicit |
| `cycle_burst_refuses_extras` | burst of k + 1 copies on a cycle flow, then its word: depth restored to 1, refusals grown by exactly k -- extras refused, nothing hovering | policy-relative, ordering hypothesis explicit |
| `flow_attempts_raw` / `flow_take_raw` / `flow_event` | identities A/B and the take-split instantiated per flow per tick from the cold start | cadence-free (enduring) |
| `emit_liveness` | per-op table: every flow's word emits exactly once (fids 1..26) |
| `consume_sink_split` | per-op table: every flow either consumes once or is a sink (fids 1..26) |
| `attempts_at_master` | identity A at the master profile: attempted = accepted = 1, refused = 0, per flow | schedule-pinned at the master profile |
| `balance_at_master` | identity B at the master profile: accepted = consumed + sunk + depth, per flow per tick (delta-depth = box' - fbox i) | schedule-pinned at the master profile |
| `geometry_write_channel_purity` | the payload kinds excluded from geometry (trajectory, delta, context) are absent from *both* D2 and D4 channel allow-lists | type-level, cadence-free (enduring) |
| `geometry_writes_anchored` | D4 writes are exactly flow #22 and D2 writes exactly {#23, #24}, all ratchet (6.0) consume ops | topology-level, cadence-free (enduring) |
| `matTick_length` | one matTick step preserves the capacity bound (given a compliant board) | step law, cadence-free (enduring) |
| `mat_operational_policy` | closed form `run (t+3) = replicate 2 (t+2) ++ replicate 2 (t+3)` for **every** stream obeying the update law (`run 0 = [0]`, one matTick per tick) | policy-relative, schedule explicit |
| `mat_fresh_policy` | INV-NO-OSSIFY for every stream obeying the update law: every resident epoch equals *t* or *t−1* | policy-relative, schedule explicit |
| `mat_bounded_policy` | capacity bound for every stream obeying the update law | policy-relative, schedule explicit |
| `mat_operational` / `mat_fresh` / `mat_bounded` | the same three statements, specialized to the dry-run minter (`matRun`) | corollaries of the policy theorems; the minter provably satisfies the law |

Proof architecture: per-flow *warm* facts (one tick from zero) and *step*
facts (one tick from the master profile) — the concrete fid arms discharge
by kernel `rfl`, the case split over fids is one 27-arm match plus a tail
arm — then short tick-inductions plus a fold induction over op words. No
axioms, no `sorry`; core Lean 4 only.

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
  `geometry_writes_anchored`, `matTick_length`, and the word-generic
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
| INV-LIVENESS | proven + empirical + structural **in its current per-flow, per-tick form** | Lean `emit_liveness`; `test_invariants.TestLiveness` | plumbing-only; enduring form: liveness scoped to the always-on spine, #22/#23/#24 trigger-gated in the physics epoch |
| INV-CYCLE-DELAY | proven + cycle check pinned | Lean `boxes_master`; the cycle check stands as `cycle_warm_no_refusal` (ordering hypothesis explicit) and `cycle_burst_refuses_extras` (extras refused, nothing hovering) | plumbing-only (the depth profile is schedule-relative; the two delayed cycles themselves are design) |
| INV-CONSERVE | two exact identities per flow per tick, proven word-generically + empirical | Lean `fold_attempt`/`fold_event` (any op sequence), `attempts_at_master`/`balance_at_master`; `test_accounting` identities A/B; overflow preserves both verbatim under reject-with-charge | enduring |
| INV-EXHAUST | proven | Lean `sunk_master` | plumbing-only (linear growth = per-tick schedule) |
| INV-CAUSAL | structural | `test_invariants.TestCausal` | enduring |
| INV-GEOMETRY | adjudicated two-tier reading: structural (writer sets) + proven (type-system purity) | `test_invariants.TestGeometry`; Lean `geometry_write_channel_purity`, `geometry_writes_anchored`; D2/D4 are the parametric geometry written only by 6.0, D1/D3 operational by design | enduring |
| INV-NO-OSSIFY | typed + proven (schedule explicit) + empirical | Lean `mat_fresh_policy`/`mat_fresh`, `mat_bounded_policy`/`mat_bounded`; `test_payloads` lifecycle/integration | enduring (statement); current proof policy-relative to the mat law, hypothesis explicit |
| INV-EMPTINESS | typed by construction + proven | `MatOverlay`/`BiasPrimitive` exact-type whitelist; Lean `geometry_write_channel_purity`; `test_payloads` trajectory-rejection | enduring |
| INV-PROVENANCE (round 8) | carrying + ordering closed, structural anti-mint checked; derivation **partial** | A2 `Envelope(Provenance(source_fid, tick_born), body)`; `require_channel_provenance`; `test_provenance` (48 tests incl. executor continuity); `test_payloads.TestEmissionProvenance`; coverage table above | enduring (statement); #2/#4/#5/#9/#13/#16 stay placeholders and D3 is not read yet |
| INV-PATCH (round 8; round 9a) | Python-asserted, **tested-not-proven** (Lean bounds records only) | `MatBoard` patch table keyed `overlay_id` (Q1-A): record and patch evict atomically on the capacity, byte-ceiling and organ paths; `MAX_PATCH_BYTES=262,144` per patch, `MAX_PATCH_TOTAL_BYTES=1,048,576` per table. Round 9a applies the evidence-derived bound **first**: `K ≤ 5·N·64` (`amateur/physics/capacity.py`), low-rank `head.2` patch `K(r)=192r+64`, 20,224 B at N=16 (rank 26); dense head+norm (34,240) and full (210,920) are inadmissible; `test_physics_capacity`, `test_physics_run.TestPatchAccounting` | physics epoch; substance bound not proved, patch map is in-memory (persistence absent) |
| INV-SOURCE (round 8) | structural + tested (no silent fallback) | content-only `PayloadSource` (Q2-A); fixed `PLACEHOLDER_FLOWS` dispatch that never reads the ledger; `PhysicsSource.derive_body` raises before any arrival; a source failure leaves every charge counter at 0; `test_physics_run.TestSourceSelection`/`TestCoverageLedger`/`TestSourceFailureIsNotACharge` | enduring (contract); physics producer partial |
| INV-ATOMIC (round 8; round 9a) | tested (fail-closed) | record+patch atomicity on every path: capacity, byte ceiling, organ-driven evict; round 9a refuses `patch="full"` at the capacity gate **before any fit/overlay**, and the outer byte-ceiling path is exercised with a synthetic oversized patch (asserted in the derivation pass before any arrival, no state change); `MatBoard` guard rails (validate-before-pop revalidation, duplicate-birth rejection, exactly-one `evict_record`); `test_payloads.TestMatBoardGuardRails`/`TestMatBoardPatches`, `test_physics_run.TestFullPatchModeFailsClosed` | enduring (Python); Lean bounds records only |
| INV-CAPACITY (round 9a) | derivation + tests (arithmetic; not proved in Lean) | `K ≤ γ·N·D` with γ=5.0, N = real evidence transitions used, D=64 (`amateur/physics/capacity.py`); last-layer low-rank patch `K(r)=192r+64`, `r_max=min(floor((γND−D)/192),64)`, `nbytes=4K`; N=4 → rank 6 / K=1,216 / 4,864 B; N=16 → rank 26 / K=5,056 / 20,224 B; N≥64 → rank 64 / K=12,352 / 49,408 B; dense 34,240 and full 210,920 refused at every N; outer byte ceilings are storage bounds only; `test_physics_capacity` (11) | physics epoch; γ=5.0 is pre-registered design slack, not measured; not proved in Lean |
| INV-VALIDATION (round 9a) | tested (state-based; no-peek asserted) | pre-adoption proposal at frame 16; paired candidate-vs-base validation on held-out ticks 17–18 of the same stream; birth only if `candidate_mean < base_mean` strictly, decision at 18; rejection record reason `interference` with both means; no birth/activation on reject; `test_physics_organ.TestPreAdoptionValidation`, `test_physics_run.TestPreAdoptionValidation` | physics epoch; `validation_window=2` == `stale_patience` (documented); one candidate at a time; trigger-gating (9b) not in scope |
| INV-RESET (round 8; round 9a) | tested | `PhysicsOrgan.reset_p6_state()` clears decisions/pending/cooldown/gate evidence/overlay table/fit history; `PhysicsRun.reset()` calls it; two consecutive `acceptance()` runs are identical (round 9a: proposal 16, birth 18, stale 31, evict 35, all-active 1.0461437156029878, flip-only 0.9665724706513396, 11 D4 sediments, clean end state); `test_physics_run.TestResetRepeatability` | enduring (harness contract); needed for the registered trace to be repeatable |
| INV-ZERO-CONF (round 8; round 9a) | tested by serialized-bytes equality + identity B | #22 is the **derived** gate output (Q4-A): it passes only when an overlay is active and cumulative adapted < base evidence exists, confidence `clamp(1-adapted/base,0,1)`; production publishes evidence after `stale_patience` (2) applied ticks, disclosed. Round 9a's capacity-bounded patch is genuinely beneficial while active, so the gate passes on **real** evidence: 11 real sediments (frames 21–31), 25 confidence-0 no-ops, `11+25=36`; the retained prefix is byte-identical and object-identical (no test double), and the gate blocks again after eviction. `test_physics_run.TestZeroConfidenceGate` (9 tests) | physics epoch; the passing branch is now real, not injected |
| INV-BLANKET | structural + layer-4 | `test_invariants.TestBlanket`, `test_design_intent`, `test_thermodynamics` | enduring |
| INV-THROUGHPUT | covered | liveness + `test_every_entity_cycle_carries_throughput` | plumbing-only in current form (inherits LIVENESS's cadence); enduring form follows the spine-scoped liveness |
| INV-CENSUS | resolved (raw membership, adjudicated) | `test_recurrence` census: \|Cycles(D1)\|=7 (entangled hub), \|Cycles(D2)\|=4, \|Cycles(D3)\|=4, \|Cycles(D4)\|=3 with zero 2-/3-hop loops — D4's structural insulation | enduring (pure topology) |
| D4 sediment cadence (#22 one per tick), #23 birth cadence | factual in the dry run | `test_payloads` lifecycle | plumbing-only (harness artifact; 6.0 is an interference-checked gate, not a metronome) |
| #18 edge (3.0 -> 5.0) | RESOLVED, lexicon pinned, falsifier on record | `v0.10-untested.md` disambiguation; `test_design_intent.TestCrystalLexicon` | enduring (closed) |

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
on full mailboxes, liveness (all 26 flows accepted exactly once per tick
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
- per tick of every seed, for all 26 flows: intra-tick INV-BOUNDS (`max_depth` <= 1
  at every event boundary, mailbox depth <= 1 at every tick boundary);
  identity A and identity B exactly; schedule-shift `attempt == 1`
  outside the plan; scalar agreement with canonical `step_flow`;
- round-7 results: per seed attempted ~32,821–33,139, accepted =
  26,000 exactly (= 26 flows x 1000 ticks), refused ~6,821–7,139, missed
  = 2 exactly (the cycle warm-ups); over the whole 12-seed sweep
  attempted = 395,794, accepted = 312,000, refused = 83,794, missed =
  24. Adversarial pressure registers in every seed (refused > 0) while
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
  #19/#21 and the organ's gate/records (#22/#23/#24), so the original
  #22/#23 objection is closed; the gate stays **PARTIAL** because #2/#4/
  #5/#9/#13/#16 are still placeholders, #15/#16/#20 are not consumed by
  P5, and D3 is written but not read.
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

Protocol: one executor stream (`PhysicsRun`, seed 7), 8 normal ticks,
gravity flip, 20 post-flip ticks, flip back, 8 recovery ticks. The organ
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

Protocol: one executor stream (`PhysicsRun`, seed 7), 8 normal ticks,
gravity flip, 20 post-flip ticks, flip back, 8 recovery ticks. The
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
| adapted / base, active **flipped** ticks (19–27) | 0.25469933084115787 / 0.2635077436765035 = **0.9665724706513396** |
| adapted / base, **all** active ticks | 0.3444968502626556 / 0.3293016486402069 = **1.0461437156029878** |
| adapted / base, active recovered ticks (28–32) | 0.5061323852213514 / 0.44773067757487295 = **1.1304393702991506** |
| stale / evict | finalized tick 31 / executor tick 35 (record+patch together) |
| end state | D2 residents and patches `()`, patch bytes 0 |
| D4 | 11 real gate sediments (frames 21–31), 25 confidence-0 no-ops; retained prefix byte-identical and object-identical (no test double) |
| encoder | bit-identical after the fit (`torch.equal`) |
| rejection trace (seed 0) | validations 17–18: 0.19090842455625534 vs 0.13387499749660492 = **1.4260200046771003**; `interference` at 18; no birth/activation; D2 empty, patch bytes 0; D4 byte-identical |
| reassessed bank fit | 0.8225682511949177 (old 0.690× quarantined: dense-patch artifact) |
| suite / Lean | 312 tests, OK (1 expected failure); `gen_lean` byte-identical |

Honest reading: the validation gate passes on the registered trace and the
organ's stale/evict rule removes the patch on the flip-back harm — the
"stale/evict on a failure trace" requirement is met by the real trace. The
acceptance criterion's wording is now material: the literal "active flip
window" reading is 0.9666 (< 1.0), while the previously registered
all-active reading is 1.0461 (> 1.0). The strict `expectedFailure` is kept
on the registered all-active metric until the user adjudicates; selecting
another seed after seeing the result would be seed-shopping and was not
done (13 of the first 31 seeds reject, 18 birth; the rejection test uses
seed 0, the first natural seed, not a re-registered acceptance). Full
derivation, per-tick trace and honesty notes in
[phase4-5-design.md](phase4-5-design.md) §15.

## What each layer catches that the others cannot

- Layer 1 catches wiring regressions that no run would exhibit soon (or
  ever, from a cold start).
- Layer 2 catches semantic drift between the documented op table and both
  implementations, and proves limits (`forall t`) that sampling cannot.
- Layer 3 catches executor-only state (counters, drains, evictions,
  payload emissions) that the pointwise model does not cover.
- Layer 4 catches code that satisfies the ledger but betrays the text.
