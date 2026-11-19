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

Proof architecture: per-flow *warm* facts (one tick from zero) and *step*
facts (one tick from the master profile) — the concrete fid arms discharge
by kernel `rfl`, the case split over fids is one 27-arm match plus a tail
arm — then four short tick-inductions. No axioms, no `sorry`; core Lean 4
only.

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

## Layer 3 — Empirical bookkeeping (`test_dryrun.py`, `test_accounting.py`)

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
- Layer 3 catches executor-only state (counters, drains) that the
  pointwise model does not cover.
- Layer 4 catches code that satisfies the ledger but betrays the text.
