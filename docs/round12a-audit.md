# Round 12a — coverage audit of the six placeholders

Prereg §7 (`docs/round12a-prereg.md:252-256`) asks for the six flows
`{1,2,4,5,9,13}` to be classified as **blocked on an unimplemented node** or
**not yet derived**, with evidence. This document is that audit. It is
analysis only.

**Scope ruling, stated once and binding for this round.** 12a is an audit,
not a schema expansion. Every one of the six would need a *body type* to be
derived, and a body type is a schema change (`PAYLOAD_SCHEMA`,
`amateur/payloads.py:792`), which 12a forbids. So nothing here is
implemented, no schema entry is added, no store/Lean/`run.py` surface moves.
The five flows classified *not yet derived* are queued for 12b/13 (§5).

## 0. Vocabulary and method

| Term | Meaning here |
| --- | --- |
| **Blocked on an unimplemented node** | The producing node has no implementation in any executable layer — only an abstract base class, or nothing at all — so no content value exists to derive from. Implementing a node must precede deriving its flow. |
| **Not yet derived** | The producing and/or consuming node exists in executable form, and the content the flow would carry is already computed and reachable. What is missing is a body type, a source method, a dispatch slot, and a schema entry. |
| **Exercised** | Executed by an executor run (`DryRun.tick` / `PhysicsRun.tick`), not merely unit-tested. The distinction is load-bearing below: `Regulator` and `BoardStore` have tests but never run. |

Evidence columns are `file:line` at HEAD (`7681c8f`) plus the design-doc
section. "Schema today" is `PAYLOAD_SCHEMA[fid]` (`amateur/payloads.py:792`,
built by `_schema()` at `:777` from `WRITE_BODY`:737, `READ_BODY`:744,
`REAL_BODIES`:758, else `_PLACEHOLDER`:726).

## 1. The audit table

| # | Edge and flow (`amateur/ledger.py`) | Schema today | Producer (node) | Consumer (node) | Exercised in a run? | Content value available? | Classification | Exact missing artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `E2 -> P1` "Gradients" (`:52`) | `{None}` (`_PLACEHOLDER`) | E2: `Gardener` ABC only (`amateur/entities.py:38-47`), abstract `gradient()` (`:42`); never instantiated (`amateur/engine.py:40` is a dead `Loop` field) | P1 stage exists (`amateur/dryrun.py:101`) and consumes a `None` | P1 stage yes; E2 **no** | **No** — the driving gravity exists as organ state (`amateur/physics/organ.py:2152` `flip_gravity`), but no node method turns it into a flow body | **blocked on an unimplemented node (E2)** | E2 implementation (`Gardener.gradient()`), a `GradientsBody`, a source method, a `PAYLOAD_SCHEMA` entry |
| 2 | `P1 -> E2` "Crystallized Output" (`:53`) | `{None}` | P1 stage exists and derives real content (#6 at `amateur/physics/run.py:326`) — no "transduce" output value is computed | E2 sink only: drained by `_drain_sinks` (`amateur/dryrun.py:806`); `Gardener.receive` (`entities.py:46`) never called | Sink drain yes; delivery **no** | Partly (observation latent/residual), but the *output* semantics are undefined and #2 must not reuse a D2 crystal kind (`docs/round11-record.md:26`) | **not yet derived** | `CrystallizedOutputBody`, P1 source method, schema entry |
| 4 | `P1 -> E1` "Active Probes" (`:55`) | `{None}` | P1 stage exists; the probe action **is** in the executor (`RunTick.action`, `amateur/physics/run.py:615`; chosen `:804-809`, passed to `organ.step`, `:809`) and reaches the source as `Observation.action` (`amateur/physics/organ.py:626`) | E1 sink only (`_drain_sinks`); `ExternalReality.absorb_probes` (`entities.py:29`) never called | Sink drain yes; delivery **no** | **Yes** — the action integer exists per tick | **not yet derived** | `ActiveProbesBody`, P1 source method, schema entry |
| 5 | `P1 -> E1` "Entropy Exhaust" (`:56`) | `{None}` | P1 stage exists; dissipated magnitudes exist per tick (`#27` cohort spreads, `amateur/payloads.py:565`; `#12` surprise, `run.py:341`) | E1 sink only; `absorb_exhaust` (`entities.py:33`) never called | Sink drain yes; delivery **no** | **Yes, but under-defined** — see the duplication flag in §2.5 | **not yet derived** | `EntropyExhaustBody` with its own content definition, P1 source method, schema entry |
| 9 | `P2 -> P3` "Routed Dynamics" (`:60`) | `{None}` | P2 stage exists (`dryrun.py:102`); `Router` ABC (`amateur/processes.py:65-82`) has **no implementation** | P3 stage exists (`dryrun.py:103`) and consumes a `None` | Both stages yes; content **no** | **Yes** — #6 residual plus the #7/#8/#26 reads are all derived that tick (`dryrun.py:477-535`) | **not yet derived** | `RoutedDynamicsBody` + a routing implementation, P2 source method, schema entry |
| 13 | `P4 -> D1` "Write / Erase / Resize / Evict" (`:64`) | **`{BoardOpBody, None}`** (`REAL_BODIES`, `payloads.py:763`) — not a placeholder | P4: `Regulator.regulate` implemented (`amateur/regulate.py:78-145`), tested (`tests/test_regulate.py:61+`); never called by an executor | D1: `BoardStore` implemented (`payloads.py:1615`), `apply` (`:1668`), `slot_state` (`:1744`); tested (`tests/test_regulate.py:210+`); not instantiated by an executor | **No** — only unit tests | **Yes** — `#14` is derivable from `BoardStore.slot_state()`; `#12` exists at stage 4.0 | **not yet derived (transport only)** | A D1 instance in `DryRun`, a `#14` derivation, a `#13` dispatch entry, a `#13` absorber |

Summary: **1 blocked on an unimplemented node** (#1, E2), **5 not yet
derived** (#2, #4, #5, #9, #13). Of the five, #13 is the strongest case: both nodes exist and are tested, and
both bodies it needs at the D1 boundary (#13 `BoardOpBody`, #14
`SlotStateBody`) already exist — only the wiring is missing.

## 2. Notes per flow

### 2.1 `#1` is the only genuine node dependency

E2 exists as an interface and nothing else: `Gardener.gradient()` is
`@abstractmethod` (`entities.py:42`), implemented nowhere, and the only
place a `Gardener` appears is the declarative, unimplemented `Loop`
(`engine.py:31-60`). The physics path fuses the environment into the organ
(`organ.step`, `amateur/physics/organ.py:2146`), so E1 has an organ analogue but E2
has none: the gravity *state* exists (`flip_gravity`, `:2152`), its *node*
does not. Deriving #1 therefore means implementing an exported entity first —
which is exactly what the classification "blocked on an unimplemented node"
is for. Note for 12b/13: because #1 is the driving input, the round that
derives it must also state who produces the exogenous sequence.

### 2.2 `#2`, `#4`, `#5` share one missing body type each, not a missing node

All three are P1 outputs and P1 runs every tick. The content exists in a
weaker form than the flow's name implies: the action for #4 is an integer the
executor already records, the dissipation for #5 is in the `#27`/`#12`
values, and the transduced observation for #2 is the organ latent. What does
not exist is an *exact-type body* and the per-flow content semantics —
substantive work, but not node work. `#2` additionally must not reuse
`EmergentCrystalBody`: ruling (D) fixes #2 as the blanket's output to E2, not
a D2 crystal (`round11-record.md:26`, `ledger.py:53`).

### 2.3 `#9` has the body gap plus a missing policy

P2 exists as a stage and consumes #6/#7/#8/#26 every tick, so the routed-error
input is available. Two artifacts are missing, not one: the body type and a
routing implementation (`processes.py:65-82` is an ABC). That makes #9 the
most expensive of the five, because a routing policy is a modelling decision
about P2, not a serializer.

### 2.4 `#13`: both nodes exist, so the "blocked" comment is false

`dryrun.py:546` says `# --- D1 reads (frozen; #13 blocked)`. At HEAD that is
false in both directions:

- D1 exists: `BoardStore` (`payloads.py:1615`) with the single-writer
  `apply` (`:1668`) and the `#14` read `slot_state` (`:1744`).
- P4 exists: `Regulator.regulate` (`regulate.py:78`) returns `(#13, #16)`
  from `#12` and `#14`, with refusal-atomicity and monotone-commit tests
  (`tests/test_regulate.py:210+`).

What is actually missing is transport, in four pieces:

1. no D1 instance is owned by `DryRun` (it owns D2/D3/D4 only);
2. `#14` is hardcoded to `None` (`dryrun.py:546-547`), so the regulator
   would see a cold start forever;
3. `#13` is in `PLACEHOLDER_FLOWS` (`dryrun.py:132`), so the executor never
   asks the source for it, and the physics dispatch would *assert* on it
   (`amateur/physics/run.py:236-252`);
4. `_absorb_writes` (`dryrun.py:661`) has handlers for #18/#19/#22/#24 and
   none for #13 — `for fid in WRITE_FLOWS` (`:682`) consumes a #13 envelope
   and drops it without touching a store, so a derived #13 would be silently
   inert.

Piece 4 is the one to watch: adding a body type and a dispatch entry without
an absorber would produce a half-wired flow that looks derived in the
coverage ledger and does nothing.

### 2.5 Content-definition flag on `#5`

`#5` is the system's terminal waste edge (`round10-invariant-respec.md:329-330`),
and its declared derivation is "P1 dissipated difference over #1/#3"
(`provenance-proposal.md:334`). But `#27` already carries D3's dissipated
entropy (`payloads.py:565`), and `#12` carries the same tick's surprise. A
naive `EntropyExhaustBody(magnitudes=drainable spreads)` would mint on #5 the
number #27 already minted, making two derived flows share one quantity — the
Round-8 gate-stub failure mode in reverse. The body must define *what P1
exhausts that is not already carried elsewhere* before it is added; that is a
12b/13 content decision, recorded here so it is not discovered as a "derived"
coincidence later.

## 3. False-dependency check (the #16 failure mode)

The #16 failure mode: a flow labelled *blocked on D1* when the value was
derivable from an already-live flow (`#12`) at P3. The label was real one
round earlier and false at HEAD. Ask of each of the six: is the node really
absent, or is only the transport missing?

| # | Claimed dependency | Verdict |
| --- | --- | --- |
| 1 | E2 (Gardener) produces the gradient | **Real.** E2 exists only as an ABC; no implementation, no instance, no analogue. |
| 2 | P1 output | **Transport/definition only.** P1 runs; the output value is the organ observation, not a node dependency. |
| 4 | P1 probe policy | **Transport only.** The action is already in `RunTick`. |
| 5 | P1 dissipation | **Transport + a content definition** (§2.5). Not a node dependency. |
| 9 | P2 routing | **Transport + a policy.** P2 runs; the missing piece is a routing rule, which is a model, not a node that "does not exist". |
| 13 | "D1 is frozen, so #13 blocked" | **False at HEAD — circular.** D1 exists (`BoardStore`), and `#14` is derivable from it; the reads are frozen *because* #13 is not wired, and #13 is not wired *because* the reads are frozen. The loop is broken by owning a `BoardStore` in `DryRun`, not by implementing anything new. |

So: one of six is a true node dependency (#1); four are body-type/transport
work with a live producer (#2, #4, #5, #9); one is pure wiring between two
already-tested nodes (#13).

## 4. Where the code contradicts the docs

1. `docs/TESTING.md:119` pins #13's body as `BoardOpBody` while its physics
   column says "placeholder (#13 blocked)". The pin is real
   (`payloads.py:763`); the absence of a producer is a wiring fact, not a
   schema fact. The two columns read as if a body and a producer were the same
   thing.
2. `docs/TESTING.md:139` — "D1 reads stay `None` because #13 is blocked and
   D1 is frozen" — states the circular dependency of §3 as a justification.
3. `docs/provenance-13-16-design.md` §1.2 grounds 1 and 2 ("No D1 state
   exists", "No P4 policy exists") are false at HEAD: `BoardStore`
   (`payloads.py:1615`) and `Regulator` (`regulate.py:69`) exist and are
   tested. Those grounds predate the implemented slice; the memo's §2-§3 plan
   remains valid, its §1.2 justification does not.
4. `docs/provenance-13-16-design.md` §1.1 places #16 at `P4 -> P5`. Round 11
   corrected the ledger to `P3 -> P5` (`ledger.py:67`), and the live #16 is
   derived at P3 (`run.py:346`).
5. `amateur/regulate.py:1-29` calls the regulator "the content-only producer
   of #13 and #16" with `#16.cost` = pre-op `#14` occupancy, but no executor
   calls it. The live #16 is `cost = ‖#12.delta‖₂` (`run.py:346-357`). Two
   `cost` semantics exist for one flow; the live one is the tested-and-pinned
   one (`TESTING.md:123`), the documented one is the unused one.
6. `dryrun.py:546` — "D1 reads (frozen; #13 blocked)" — §2.4 above.
7. `DebrisPool.promote` (`payloads.py:1430`) has no executor caller: only
   `tests/test_payloads.py` calls it. The issuance account's `promoted` term
   is therefore structurally 0 in every run (measured: 0 of 4000 deposits on
   the 1000-tick synthetic profile). The Round-11 contract presents the
   issuance account as the ruling's live account; today only its
   resident/drained/evicted terms move, and no D3 consumer ever issues a
   cohort. Not fixed here (it is not a placeholder flow), recorded because
   the account is now a named surface.

## 5. Queue for 12b/13 (not implemented here)

Ordered by cost, lowest first, all requiring a body type (a schema change):

1. **#13 + #14 + #11/#15** — one body each at the D1 boundary, plus a
   `BoardStore` in `DryRun` and a #13 absorber. The only five flows that
   complete a closed circuit with code that already exists and is tested.
2. **#4** then **#5** — one P1 output each; #5 needs its content defined
   first (§2.5).
3. **#2** — needs a semantics decision (what the blanket hands back to E2).
4. **#9** — needs a routing policy, the only substantive modelling work left
   in the six.
5. **#1** — needs E2 implemented. Last, because it is the only one that
   cannot be reached by serializing a value the executor already computes.
