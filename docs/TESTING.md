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
#25→#24 same-tick pairing on `overlay_id`).

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

- **Payload provenance (deferred by adjudication, gated here):** every
  payload must carry `(source_fid, tick_born)` and be derived from actual
  node state — no minting at emit time. The current synthetic contents
  (`cry-{t}`, rotating confidences) are acceptable only in the dry run.
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

## What each layer catches that the others cannot

- Layer 1 catches wiring regressions that no run would exhibit soon (or
  ever, from a cold start).
- Layer 2 catches semantic drift between the documented op table and both
  implementations, and proves limits (`forall t`) that sampling cannot.
- Layer 3 catches executor-only state (counters, drains, evictions,
  payload emissions) that the pointwise model does not cover.
- Layer 4 catches code that satisfies the ledger but betrays the text.
