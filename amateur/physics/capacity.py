"""Evidence-bounded motif capacity (Round 9a).

The user's binding condition for round 9a: **a patch's stored size scales
with the evidence that justifies it** -- "four frames earns a sentence, not
a manifesto".  This module is the derivation, kept as pure arithmetic so the
bound can be tested without a model.

Derivation (rate-distortion / MDL, pre-registered)
--------------------------------------------------
A motif is a hypothesis ``theta`` fitted to evidence ``E``.  The minimum
description length principle says we may spend storage on the hypothesis in
proportion to the description length of the evidence it explains:

    L(theta) <= gamma * L(E)

with a uniform 32-bit encoding of every stored parameter and every evidence
value:

    L(E) = N * D * 32 bits      (N evidence transitions, D latents each)
    L(theta) = K * 32 bits      (K stored parameters)

so the parameter budget is

    K <= gamma * N * D.

``N`` is the number of **real evidence transitions actually used by the
fit** -- counted from the data, never the requested window length and never
the number of motifs.  ``D`` is the model's latent width (64 for the phase-3
checkpoint).  ``gamma = 5.0`` is the pre-registered design slack: a
*honesty note*, not a measurement (see the design memo, Round 9a notes).

Admissible patch form (fixed by the Lead contract)
--------------------------------------------------
The patch adapts the predictor's **last head layer only** -- ``head.2``
weight ``W`` in ``R^{D x H}`` plus a bias in ``R^{D}`` -- and is stored in
low-rank factored form rather than densely:

    W_delta = u @ v.T,   u: (D, r),  v: (H, r),  bias: (D,)

    K(r)      = r * (D + H) + D      = 192 r + 64   (D=64, H=128)
    nbytes(r) = 4 * K(r)             = 768 r + 256
    r_max     = min(floor((gamma*N*D - D) / (D + H)), min(D, H))

The outer storage ceilings in ``amateur.payloads`` (262,144 B per patch,
1,048,576 B per table) remain **storage bounds only**; this budget is the
relative, evidence-derived bound that is applied first.

The dense forms are inadmissible at every evidence size reachable in this
slice: the rank ceiling caps the admissible patch at ``K(64) = 12,352``
parameters, while the dense head+norm patch is 34,240 parameters and the
full-predictor patch is 210,920.  They are a *different patch class*, not a
larger budget, so :meth:`EvidenceBudget.admit_params` refuses both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEFAULT_GAMMA",
    "EvidenceBudget",
    "MAX_ADMISSIBLE_PARAMS_NOTE",
]

#: The pre-registered design slack (a design choice, not a measurement).
DEFAULT_GAMMA = 5.0

#: The phase-3 checkpoint's latent width and predictor head input width.
DEFAULT_LATENT_DIM = 64
DEFAULT_HEAD_IN = 128

#: ``min(D, H)`` for the phase-3 model: the largest admissible rank.
MAX_ADMISSIBLE_RANK = min(DEFAULT_LATENT_DIM, DEFAULT_HEAD_IN)

#: ``K(MAX_ADMISSIBLE_RANK)`` for the phase-3 model (documentation only).
MAX_ADMISSIBLE_PARAMS_NOTE = 192 * MAX_ADMISSIBLE_RANK + 64


@dataclass(frozen=True)
class EvidenceBudget:
    """The evidence-derived storage budget for one motif candidate.

    ``n_transitions`` is the number of real evidence transitions the fit
    actually consumed.  The dataclass is frozen: a budget is derived once
    per fit candidate and never mutated (no peeking, no retro-fitting).
    """

    n_transitions: int
    latent_dim: int = DEFAULT_LATENT_DIM
    head_in: int = DEFAULT_HEAD_IN
    gamma: float = DEFAULT_GAMMA

    def __post_init__(self) -> None:
        if self.n_transitions < 0:
            raise ValueError("n_transitions must be non-negative")
        if self.latent_dim < 1 or self.head_in < 1:
            raise ValueError("latent_dim and head_in must be positive")
        if not math.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("gamma must be a finite positive number")

    # ------------------------------------------------------------ evidence
    @property
    def evidence_bits(self) -> float:
        """``L(E) = N * D * 32`` (uniform 32-bit encoding)."""
        return float(self.n_transitions * self.latent_dim * 32)

    @property
    def capacity_params(self) -> float:
        """``K <= gamma * N * D``: the admissible stored-parameter budget."""
        return float(self.gamma) * float(self.n_transitions) * float(self.latent_dim)

    @property
    def capacity_bits(self) -> float:
        """``L(theta) <= gamma * L(E)`` expressed in bits."""
        return self.capacity_params * 32.0

    # ------------------------------------------------------------ patch form
    @property
    def per_rank_params(self) -> int:
        """``D + H`` parameters added per rank step."""
        return self.latent_dim + self.head_in

    @property
    def rank_cap(self) -> int:
        """``min(D, H)``: a rank cannot exceed the matrix's smaller side."""
        return min(self.latent_dim, self.head_in)

    def params_for_rank(self, rank: int) -> int:
        """``K(r) = r(D + H) + D``."""
        rank = int(rank)
        if rank < 0:
            raise ValueError("rank must be non-negative")
        return rank * self.per_rank_params + self.latent_dim

    def nbytes_for_rank(self, rank: int) -> int:
        """``4 * K(r)`` (float32 factors plus bias)."""
        return 4 * self.params_for_rank(rank)

    @property
    def max_rank(self) -> int:
        """``min(floor((gamma*N*D - D) / (D + H)), min(D, H))``."""
        numerator = self.capacity_params - float(self.latent_dim)
        if numerator < 0.0:
            return 0
        rank = int(math.floor(numerator / float(self.per_rank_params)))
        return max(0, min(rank, self.rank_cap))

    @property
    def max_params(self) -> int:
        """``K(max_rank)``; ``0`` when no rank is admissible.

        ``rank < 1`` means no candidate at all (a bias-only patch is not a
        motif adaptation), so the reported maximum is ``0`` rather than the
        bias width -- an empty budget must not look like a 64-parameter one.
        """
        if self.max_rank < 1:
            return 0
        return self.params_for_rank(self.max_rank)

    @property
    def max_nbytes(self) -> int:
        return 4 * self.max_params

    # ------------------------------------------------------------- admission
    def admit_params(self, param_count: int) -> bool:
        """True iff ``param_count`` is an admissible stored patch size.

        Two conditions, both required:

        1. the MDL bound ``K <= gamma*N*D`` (``capacity_params``); and
        2. the fixed-representation ceiling ``max_params`` (rank <=
           ``rank_cap`` on the last head layer).

        Condition 2 is what refuses the dense forms: the admissible patch
        class is the capacity-bounded low-rank last-layer delta, so a dense
        head+norm (34,240 params) or full-predictor (210,920 params) patch
        exceeds the representation ceiling at every evidence size, not just
        at small ``N``.
        """
        params = int(param_count)
        if params < 0:
            raise ValueError("param_count must be non-negative")
        return params <= self.max_params and float(params) <= self.capacity_params

    def admit_rank(self, rank: int) -> bool:
        """True iff ``rank <= max_rank``."""
        return 0 <= int(rank) <= self.max_rank

    def describe(self) -> dict[str, float]:
        """The budget as plain numbers (used in records and reports)."""
        return {
            "n_transitions": float(self.n_transitions),
            "gamma": float(self.gamma),
            "latent_dim": float(self.latent_dim),
            "head_in": float(self.head_in),
            "capacity_params": float(self.capacity_params),
            "max_rank": float(self.max_rank),
            "max_params": float(self.max_params),
            "max_nbytes": float(self.max_nbytes),
        }
