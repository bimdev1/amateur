"""Adversarial property fuzz over the reject-with-charge channel.

Scope (user-pinned): injection multiplicities k >= 1 only, on the E1
entity channels. Zero-injection (starvation) semantics are deferred to
the physics epoch together with trigger-gating -- "without throughput,
the geometry remains without unfolding" makes an idle tick a legitimate
state, not a liveness violation, and the enduring INV-LIVENESS is
re-scoped then. This fuzz distribution is plumbing, not the spec.

Refusal coupling (design intent, physics epoch): refusal rate at the
blanket is the world arriving faster than we can digest; sustained
excess must drive collapse pressure at 4.0/5.0 and dissipated refusal
must exit as entropy exhaust. At this epoch charges are pure accounting
with no feedback into dynamics -- recorded here so it is built, not
rediscovered.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .dryrun import FLOW_OPS, PER_TICK_INJECTIONS, DryRun, step_flow
from .ledger import FLOW_LEDGER

_INJECTABLE = (1, 3)


@dataclass(frozen=True)
class FuzzResult:
    """One adversarial trajectory plus its charge totals."""

    seed: int
    ticks: int
    max_k: int
    attempted: int
    accepted: int
    refused: int
    missed: int
    peak_k: dict[int, int]


def run_fuzz(seed: int, ticks: int, max_k: int, check: bool = True) -> FuzzResult:
    """Randomized E1-injection multiplicities with per-tick checks.

    Asserted every tick, per flow (the adjudicated identities):
    - identity A: attempted = accepted + refused;
    - identity B: accepted = consumed + delta-sunk + delta-depth;
    - INV-BOUNDS intra-tick: depth <= 1 at tick end and at every event
      boundary of the whole run (`DryRun.max_depth` monitors each
      arrival and consumption, O(1) per event, charges only);
    - flows outside the plan keep exactly one attempt per tick;
    - the canonical scalar recurrence tracks executor depths exactly:
      under reject-with-charge, multiplicity cannot alter dynamics.
    """
    if ticks < 1:
        raise ValueError("ticks must be >= 1")
    if max_k < 1:
        raise ValueError("k must be >= 1; starvation is a physics-epoch decision")
    rng = random.Random(seed)
    run = DryRun()
    flow_ids = [f.fid for f in FLOW_LEDGER]
    scalar = dict.fromkeys(flow_ids, 0)
    peak = dict.fromkeys(_INJECTABLE, 0)
    for _ in range(ticks):
        plan = {fid: rng.randint(1, max_k) for fid in _INJECTABLE}
        for fid, k in plan.items():
            peak[fid] = max(peak[fid], k)
        before_depth = {fid: len(run.mailboxes[fid]) for fid in flow_ids}
        before_attempts = run.attempted.copy()
        before_accepted = run.accepted.copy()
        before_refused = run.refused.copy()
        before_consumed = run.consumed.copy()
        before_sunk = run.sunk.copy()
        run.tick(injections=plan)
        after_depth = {fid: len(run.mailboxes[fid]) for fid in flow_ids}
        for fid in flow_ids:
            attempt = run.attempted[fid] - before_attempts.get(fid, 0)
            accept = run.accepted[fid] - before_accepted.get(fid, 0)
            refuse = run.refused[fid] - before_refused.get(fid, 0)
            consume = run.consumed[fid] - before_consumed.get(fid, 0)
            sunk = run.sunk[fid] - before_sunk.get(fid, 0)
            drift = after_depth[fid] - before_depth[fid]
            if check:
                assert after_depth[fid] <= 1, f"tick boundary depth #{fid}"
                assert run.max_depth[fid] <= 1, f"intra-tick depth #{fid}"
                assert attempt == accept + refuse, f"identity A #{fid}"
                assert accept == consume + sunk + drift, f"identity B #{fid}"
                if fid not in plan:
                    assert attempt == 1, f"schedule shift at #{fid}"
                delta = step_flow(fid, scalar[fid])
                scalar[fid] = delta.box
                assert delta.box == after_depth[fid], f"scalar drift #{fid}"
        if check:
            assert all(len(box) <= 1 for box in run.mailboxes.values())
    return FuzzResult(
        seed=seed,
        ticks=ticks,
        max_k=max_k,
        attempted=sum(run.attempted.values()),
        accepted=sum(run.accepted.values()),
        refused=sum(run.refused.values()),
        missed=sum(run.missed.values()),
        peak_k=peak,
    )
