"""P4 minimal regulator: the D1/P4 slice of the #13/#16 derivation.

Pure policy, no side effects, no organ access, no D2/D3/D4 state.  It is
the content-only producer of #13 (`BoardOpBody`) and #16 (`CostErrorBody`)
from exactly two pieces of declared evidence:

- ``#12`` :class:`StateDeltaBody` — P3's state delta and measured surprise;
- ``#14`` :class:`SlotStateBody` — D1's slot identity/occupancy snapshot.

The policy is deliberately small: cold start or a free slot writes a new
context slot; a full board evicts the oldest.  ``resize``/``erase`` are
schema-legal but not emitted by this slice.  ``#17``
(``CollapseMandateBody``) is **not consumed**: mandates stay fixed-cadence
no-ops until the 9b trigger-gating round, and a non-noop #17 later carries
the cross-round obligation that this regulator must then consume it or the
deferral is re-adjudicated in writing (Lead adjudication 2026-11-20 §9.6).

Everything here is Python-tested, not proved.  In particular, ``nbytes`` is
computed as ``8 * len(delta)`` and ``cost`` is the sum of the stored slots'
represented bytes; neither is a hard-coded latent-width byte count or any
other state literal (§9.5, §9.11b).  The clock only names a write slot; it
does not decide the branch or the content digest.

``#16.cost`` is the **pre-op** occupancy reported by ``#14``: this
regulator runs before the same tick's ``#13`` arrives in the executor, so
the post-commit occupancy becomes visible as the next tick's ``#14``.  A
"cost + nbytes" projection would be a prediction, not derivation, and is
deliberately not made (§9.14.6).
"""
from __future__ import annotations

import hashlib
import struct

from .payloads import (
    BoardOpBody,
    CostErrorBody,
    SlotStateBody,
    StateDeltaBody,
)

#: Digest width for the content-addressed body id / slot name.
_DIGEST_SIZE = 16

#: Slack width of the slot identity slice; a label, not a state value.
_SLOT_DIGEST_CHARS = 12


def content_digest(delta: StateDeltaBody) -> str:
    """Deterministic digest of the represented #12 delta vector.

    Big-endian float64 serialization makes the digest stable across
    platforms and runs; it is the evidence-derived ``body_id`` and the
    slot-name suffix (not a tick or a constant).
    """
    packed = struct.pack(f">{len(delta.delta)}d", *delta.delta)
    return hashlib.blake2b(packed, digest_size=_DIGEST_SIZE).hexdigest()


def represented_bytes(delta: StateDeltaBody) -> int:
    """Bytes D1 is said to retain for this record: ``8 * len(delta)``.

    Computed from the evidence, not from a literal (the design appendix's
    latent-width byte count is a value at D=64, not a constant; §9.5).
    """
    return 8 * len(delta.delta)


class Regulator:
    """P4 minimal policy; stateless and deterministic.

    ``regulate`` returns ``(#13 body, #16 body)`` in one call so both
    outputs see the same evidence snapshot.  It never mutates the store and
    never touches #17; the caller (the physics source) reads only
    ``evidence[12]`` and ``evidence[14]`` (Q1-A).
    """

    def regulate(
        self,
        delta: StateDeltaBody,
        slots: SlotStateBody | None,
        tick: int,
    ) -> tuple[BoardOpBody, CostErrorBody]:
        """Derive (#13, #16) from the #12 and #14 evidence.

        Only ``write`` and ``evict`` are ever emitted; ``resize``/``erase``
        are schema-legal operator verbs but are not reachable from this
        policy (Lead adjudication §9.14.3).  ``#16.cost`` is the pre-op
        ``#14`` byte total, not a post-op projection (§9.14.6).
        """
        if type(delta) is not StateDeltaBody:
            raise TypeError(
                f"P4 needs an exact StateDeltaBody, got {type(delta).__name__}"
            )
        if slots is not None and type(slots) is not SlotStateBody:
            raise TypeError(
                f"P4 needs a SlotStateBody or None, got {type(slots).__name__}"
            )
        if type(tick) is not int or isinstance(tick, bool):
            raise TypeError(f"tick must be an int, got {tick!r}")
        if tick < 0:
            raise ValueError("tick must be non-negative")

        digest = content_digest(delta)
        nbytes = represented_bytes(delta)
        slot_name = f"ctx-{tick}-{digest[:_SLOT_DIGEST_CHARS]}"

        if slots is None:
            op = BoardOpBody(
                op="write",
                slot_id=slot_name,
                body_id=digest,
                nbytes=nbytes,
                reason="cold-start",
                tick=tick,
            )
            cost = 0.0
        elif slots.occupied < slots.capacity:
            op = BoardOpBody(
                op="write",
                slot_id=slot_name,
                body_id=digest,
                nbytes=nbytes,
                reason="free-slot",
                tick=tick,
            )
            cost = float(slots.nbytes)
        else:
            oldest = slots.oldest_slot_id
            if oldest is None:  # pragma: no cover - SlotStateBody invariant
                raise AssertionError(
                    "full D1 state has no oldest_slot_id; invariant broken"
                )
            op = BoardOpBody(
                op="evict",
                slot_id=oldest,
                body_id=None,
                nbytes=0,
                reason="capacity-evict",
                tick=tick,
            )
            cost = float(slots.nbytes)

        metrics = CostErrorBody(cost=cost, error=float(delta.surprise))
        return op, metrics
