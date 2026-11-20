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
shape for *all* ticks, not just sampled ones:

| Theorem | Statement |
| --- | --- |
| `boxes_bounded` | every mailbox depth <= 1 at every tick |
| `boxes_master`  | for `t >= 1`, depth profile is exactly `{10: 1, 17: 1}`, zero elsewhere |
| `misses_master` | cumulative misses are exactly one on each cycle, frozen after tick 1 |
| `sunk_master`   | `sIter t i = t` for the sinks (#2, #4, #5), 0 elsewhere |
| `applyOp_step` / `fold_balance` / `flow_balance` | per-flow conservation: box' + sunk' + attempted consumes is fixed by the word's emit weight |
| `emit_liveness` | per-op table: every flow's word emits exactly once (fids 1..26) |
| `consume_sink_split` | per-op table: every flow either consumes once or is a sink (fids 1..26) |
| `balance_at_master` | at the master profile, box' + sunk' + consumes = fbox i + 1 per flow |
| `geometry_write_channel_purity` | the payload kinds excluded from geometry (trajectory, delta, context) are absent from *both* D2 and D4 channel allow-lists |
| `geometry_writes_anchored` | D4 writes are exactly flow #22 and D2 writes exactly {#23, #24}, all ratchet (6.0) consume ops |
| `mat_operational` | closed form: `matRun (t+3) = replicate 2 (t+2) ++ replicate 2 (t+3)` |
| `mat_fresh` | INV-NO-OSSIFY: every resident epoch in `matRun t` (t ≥ 1) equals *t* or *t−1* |
| `mat_bounded` | `(matRun t).length ≤ 4` at every tick |

Proof architecture: per-flow *warm* facts (one tick from zero) and *step*
facts (one tick from the master profile) — the concrete fid arms discharge
by kernel `rfl`, the case split over fids is one 27-arm match plus a tail
arm — then short tick-inductions plus a fold induction over op words. No
axioms, no `sorry`; core Lean 4 only.

## Master invariant matrix (INV-*)

How each user-supplied master invariant is currently verified:

| INV-id | Status | Where |
| --- | --- | --- |
| INV-BOUNDS | proven | Lean `boxes_bounded` |
| INV-LIVENESS | proven + empirical + structural | Lean `emit_liveness`; `test_invariants.TestLiveness` |
| INV-CYCLE-DELAY | proven | Lean `boxes_master` |
| INV-CONSERVE | proven per-flow + empirical system-wide | Lean `flow_balance`, `balance_at_master`; `test_accounting` identity |
| INV-EXHAUST | proven | Lean `sunk_master` |
| INV-CAUSAL | structural | `test_invariants.TestCausal` |
| INV-GEOMETRY | adjudicated two-tier reading: structural (writer sets) + proven (type-system purity) | `test_invariants.TestGeometry`; Lean `geometry_write_channel_purity`, `geometry_writes_anchored`; D2/D4 are the parametric geometry written only by 6.0, D1/D3 operational by design |
| INV-NO-OSSIFY | typed + proven + empirical | Lean `mat_fresh`, `mat_bounded`; `test_payloads` lifecycle/integration |
| INV-EMPTINESS | typed by construction + proven | `MatOverlay`/`BiasPrimitive` exact-type whitelist; Lean `geometry_write_channel_purity`; `test_payloads` trajectory-rejection |
| INV-BLANKET | structural + layer-4 | `test_invariants.TestBlanket`, `test_design_intent`, `test_thermodynamics` |
| INV-THROUGHPUT | covered | liveness + `test_every_entity_cycle_carries_throughput` |
| INV-CENSUS | resolved (raw membership, adjudicated) | `test_recurrence` census: \|Cycles(D1)\|=7 (entangled hub), \|Cycles(D2)\|=4, \|Cycles(D3)\|=4, \|Cycles(D4)\|=3 with zero 2-/3-hop loops — D4's structural insulation |

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

## Layer 3 — Empirical bookkeeping (`test_dryrun.py`, `test_accounting.py`, `test_invariants.py`, `test_payloads.py`)

The executor (`DryRun`) is checked against conservation-style laws over
real runs: liveness (all 26 flows fire every tick), the conservation
identity `emitted = consumed + sunk + delta in-transit` (misses remove
nothing and are budgeted separately), stationarity (per-tick stats
identical from tick 2; tick 1 differs only in the two cycle warm-up misses
and the matching consumption deficit), and the master invariants
re-derived empirically. A final assertion replays `step_flow` for 32 ticks
and confirms the pure scalar model tracks the executor tick for tick.

## Layer 4 — Design intent (`test_design_intent.py`, `test_contracts.py`, `test_engine.py`)

Prose-level invariants from the "Emptiness" text and the ABC contracts:
timescale separation, trajectory-free core, debris discipline, carrier
semantics, node vocabulary. These bind the *intended* behavior, not just
the implemented behavior.

## What each layer catches that the others cannot

- Layer 1 catches wiring regressions that no run would exhibit soon (or
  ever, from a cold start).
- Layer 2 catches semantic drift between the documented op table and both
  implementations, and proves limits (`forall t`) that sampling cannot.
- Layer 3 catches executor-only state (counters, drains, evictions,
  payload emissions) that the pointwise model does not cover.
- Layer 4 catches code that satisfies the ledger but betrays the text.
