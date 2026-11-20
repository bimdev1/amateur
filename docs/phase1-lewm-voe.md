# Phase 1 — LeWM organ and the violation-of-expectation (VoE) reproduction

Status: **complete**. The organ is stood up and verified against the reference
implementation; the VoE measurement was run on the full published PushT
dataset on CPU. Every number below was produced by a command listed in
§8; nothing is projected, extrapolated, or copied from the paper.

Scope: this is an isolated organ behind future P1/P3. It has no coupling to the
event ledger, no executor imports, and no runtime network access. The encoder
is frozen everywhere; any adaptation is a later-phase decision.

---

## 0. Summary

| Question | Answer |
| --- | --- |
| Does the organ load the published checkpoint exactly? | Yes — 303 keys, strict in both directions, 18,042,672 state-dict elements |
| Is it the same model as the reference implementation? | Yes to fp32 noise — `encode` and `predict` agree to max abs diff `1.43e-06` (activations `absmax 2.62`), `SIGReg` bit-identical |
| Teleportation spike (24 episodes) | **1.9444 ± 0.0951** vs baseline **0.0097** → **×199.5**; 24/24 episodes up; paired `t = 20.25` (df 23) |
| Block-colour spike (24 episodes) | **0.7684 ± 0.0475** vs baseline **0.0097** → **×78.9**; 24/24 episodes up; paired `t = 15.89` (df 23) |
| Where is the spike? | target step 6 in both conditions — the step whose *target frame* is the first perturbed one; peak step = 6 for both |
| What happens next? | teleport decays to baseline (`0.0093` at step 7); colour stays elevated (`0.15–0.18`) for the rest of the window |
| CPU wall time | 61.6 s for 24 episodes × 3 conditions × 9 targets (8 threads, encoder + predictor, load excluded 0.3 s) |

Ordering reproduces the paper's claim: physical violation ≫ visual violation ≫
unperturbed. One deviation from the paper's *statistical* sentence is recorded
in note 7: our colour manipulation is much stronger than the paper's
description implies, and our colour effect is significant.

---

## 1. The organ (`amateur/lewm.py`)

Single file, imports only `torch` and the standard library (`einops` is spelled
out as `view`/`transpose`, `stable_pretraining` / `stable_worldmodel` / `hydra` /
`lightning` are not imported anywhere).

### 1.1 Public surface

| Function | Signature | Notes |
| --- | --- | --- |
| `LewM.load_checkpoint(dir)` | → model | builds from `config.json`, then `load_state_dict(..., strict=True)`, `eval()`, `requires_grad_(False)` |
| `LewM.from_config(cfg)` | → model | config mapping or path; no weights needed |
| `encode(frames)` | `[B,T,3,224,224]` or `[B,3,224,224]` → `[B,T,192]` | `pixels.float()` → flatten `B·T` → ViT with `interpolate_pos_encoding=True` → CLS token → `projector`. A 4-D input keeps a length-1 time axis (`[B,1,192]`) so the contract never changes shape |
| `predict(z, actions)` | `[B,T,192]`, `[B,T,10]` → `[B,T,192]` | action encoder → autoregressive predictor → `pred_proj`; slot `i` predicts frame `i+1`; `T ≤ num_frames = 3` (the positional table has 3 entries, exactly as `JEPA.rollout` truncates) |
| `surprise(z_hat, z_next)` | → `(per_sample [B], mean scalar)` | mean of squared error over time and latent axes per sample, then mean over batch; no square root, i.e. the paper's MSE prediction quantity. `[B,D]` also accepted (time axis optional) |
| `surprise_per_step(z_hat, z_next)` | → `[B,T]` | per-target squared error, used for the VoE curves |
| `imagenet_normalize(frames)` | uint8 `[0,255]` → ImageNet-normalised float | the encoder's input convention; `encode` *rejects* uint8 rather than silently mis-scaling it |
| `normalize_actions(actions, mean, std)` | z-score | the training convention, nothing more |
| `SIGReg` | `(T,B,D)` → scalar | the regulariser, exposed for tests; stochastic by construction (fresh projections) |

All of the above are `@torch.no_grad()`. `LewM.train()` is overridden to a
no-op that keeps every submodule in eval mode, so a stray `.train()` cannot
move the `BatchNorm1d` running statistics that the checkpoint supplies.

Unit note: `encode` performs no resize. The published PushT frames are already
224×224, so the training-time `Resize(224)` is the identity; the organ takes
what the training pipeline produced.

### 1.2 What the checkpoint contains

| Quantity | Value | How measured |
| --- | --- | --- |
| state-dict entries | 303 | `len(torch.load(weights.pt))` |
| total tensor elements (parameters + buffers) | 18,042,672 | `sum(v.numel())` over the state dict |
| `model.parameters()` | 18,034,478 | `sum(p.numel() for p in model.parameters())` |
| buffers | 8,194 | BatchNorm running mean/var in `projector` and `pred_proj` |
| dtype | float32 everywhere | |

The organ's key set equals the checkpoint's key set exactly (no missing, no
unexpected), and every shape agrees. Architecture fixed by `config.json`:
ViT-tiny encoder (patch 14, 224 px, dim 192, 12 layers, 3 heads, MLP 768, CLS
token, `layer_norm_eps = 1e-12`, no mask token, no pooler); predictor
(`num_frames 3`, `depth 6`, `heads 16`, `dim_head 64`, `mlp_dim 2048`, AdaLN-zero
conditional blocks, causal attention, dropout 0.1 inactive at eval);
action `Embedder(10 → 768 → 192)`; two `MLP(192 → 2048 → 192)` projectors with
`BatchNorm1d(2048)`.

### 1.3 Fidelity verification against the reference stack

The organ was compared against the *real* reference: HuggingFace `ViTModel`
built exactly as `stable_pretraining.backbone.utils.vit_hf(size="tiny",
patch_size=14, image_size=224, pretrained=False, use_mask_token=False)` builds
it (`add_pooling_layer=False`, `config.interpolate_pos_encoding=True`), wrapped
in the unmodified `.work/le-wm/jepa.py` + `.work/le-wm/module.py` classes, with
the published weights loaded. Command in §8.

| Quantity | Reference | Organ | Difference |
| --- | --- | --- | --- |
| `encode` on `[2,4,3,224,224]` | `(2,4,192)` | `(2,4,192)` | max `1.431e-06`, mean `2.560e-07` (activations `absmax 2.620`) |
| `predict` on `[2,3,192]` + `[2,3,10]` | `(2,3,192)` | `(2,3,192)` | max `1.431e-06`, mean `2.501e-07` (activations `absmax 2.402`) |
| `SIGReg` (same seed) | `1.57506, 1.609455, 1.559502` | identical | `0` |

The residual is fp32 accumulation order: the reference `ViTModel` uses SDPA
attention while the organ writes the explicit `softmax(QKᵀ/√d)V` product (the
pre-SDPA HF path). At `absmax ≈ 2.6`, `1.4e-6` is ~`5e-7` relative — far below
any surprise value in §5, where the signal is 0.01 → 1.9.

Tests: `tests/test_lewm.py`, 27 tests, run in ~7–13 s; they pin the load
contract, shapes, determinism, frozen weights, the surprise reduction, and —
on synthetic PushT-like frames — that a teleport raises surprise ≥10× the
unperturbed value while a colour change raises it less.

---

## 2. The dataset

`.work/data/pusht_expert_train.h5`, decompressed from the published
`pusht_expert_train.h5.zst` (13,136,247,974 bytes → 46,300,921,856 bytes,
`zstd -d`, 93 s).

### 2.1 Schema (verified, not assumed)

| Dataset | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `pixels` | (2,336,736, 224, 224, 3) | uint8 | chunked 100 rows; HDF5 filter **32001** |
| `action` | (2,336,736, 2) | float32 | per-env-step action, one row per frame |
| `state` | (2,336,736, 7) | float32 | `[agent_x, agent_y, block_x, block_y, block_angle, …]`, world span `[0,512]` |
| `proprio` | (2,336,736, 4) | float32 | unused here |
| `episode_idx`, `step_idx`, `ep_offset`, `ep_len` | (2,336,736) / (18,685) | int | |

18,685 episodes; lengths min 49 / median 123 / max 246; rows are contiguous per
episode, `row = ep_offset[ep] + step`. Verified on 200 sampled episodes: no
violations.

Frame geometry was checked against `state`: the agent mask's centroid tracks
`state[0:2]` with `pixel = coordinate · 224/512` on both axes (no vertical
flip), to sub-pixel agreement over a whole episode. That is what licenses the
block mask and the displacement metric used below.

### 2.2 Pixel filter 32001

`pixels` uses HDF5 filter id **32001**, which is HDF5's **Blosc1**
meta-compressor with the **LZ4** codec (16-byte header carrying
`nbytes/blocksize/cbytes`, then one LZ4 block per 131,072-byte chunk), not raw
zstd. Two ways to read it were verified **byte-identical** on two chunks:

1. `import hdf5plugin` before `h5py.File` → plain indexing works. **This is the
   path the tests use** (falling back to the next one, then to a skip).
2. `Dataset.id.read_direct_chunk(...)` returns the still-filtered bytes, decoded
   by a ~50-line Blosc1 reader against system `liblz4` via `ctypes`
   (`decode_blosc1` in `tests/test_lewm.py`).

Without either, h5py raises
`OSError: Can't synchronously read data (can't open directory .../plugin)`.

### 2.3 Image preprocessing (must match training)

The reference pipeline is `ToImage(ImageNet stats)` then `Resize(224)`. Frames
are already 224×224, so:

```
input = (pixels / 255 - ImageNet_mean) / ImageNet_std
ImageNet_mean = (0.485, 0.456, 0.406)
ImageNet_std  = (0.229, 0.224, 0.225)
```

### 2.4 Actions (must match training)

Training sets `action_encoder.input_dim = frameskip · action_dim = 5 · 2 = 10`,
so the model consumes the **next 5 environment actions flattened**, and the
column is z-scored with statistics over the whole chunked column
(`get_column_normalizer`). Measured over all 2,261,996 valid chunks:

```
mean = [-0.0083,  0.0071, -0.0085,  0.0061, -0.0084,  0.0061,
        -0.0082,  0.0065, -0.0080,  0.0069]
std  = [ 0.2111,  0.2093,  0.2105,  0.2086,  0.2106,  0.2086,
         0.2104,  0.2084,  0.2098,  0.2077]
```

The VoE conditions reuse these exact statistics.

---

## 3. VoE protocol

### 3.1 What the paper did, and why we do something adjacent

From §5.2 and App. F.3 of arXiv:2603.19312: in PushT "the agent is controlled by
a random policy biased toward interacting with the block"; three trajectories
are generated in the **simulator** — (1) unperturbed, (2) "the color of the
block changes abruptly", (3) "both the agent and the block are teleported to
random positions at the perturbation timestep". Surprise is the discrepancy
between the predicted future observation and the observed future.

We do not have the PushT simulator, and the released repository contains no VoE
script (upstream HEAD `8edfeb3` tracks only `jepa.py`, `module.py`, `train.py`,
`eval.py`, `utils.py`, configs). So the **measurement** is reproduced and the
**trajectory source** is substituted: we replay expert episodes from the
published training set and construct the same three conditions by editing that
replay. Section 6 enumerates the consequences.

### 3.2 Episode selection

A condition is `W = 12` frames at frameskip 5, so it needs raw rows
`5·(W−1)+1 = 56`; the teleport condition reads a further `5·JUMP` rows.
With `JUMP = 12` the requirement is **121 raw rows**, which 9,797 of the 18,685
episodes satisfy. 24 episodes are drawn with
`numpy.random.default_rng(0).choice` (without replacement) from that pool:
`[127, 363, 703, 1341, 3128, 4657, 4736, 5029, 9574, 9844, 10271, 10431,
11189, 11441, 11478, 11605, 12025, 12797, 14521, 14556, 15182, 16607, 17136,
17682]`.

### 3.3 The three conditions (exact construction)

For episode `e` with base row `b = ep_offset[e]`, condition frame `i` is raw row
`b + 5i`, i.e. the same subsampling the training loader uses. Let `k = 6` be the
perturbation index in condition coordinates.

| Condition | Frames | Actions |
| --- | --- | --- |
| **unperturbed** | `frames[i] = frame(b + 5i)`, `i = 0..11` | `a[i] = chunk(b + 5i)`, `i = 0..11` |
| **teleport** | `frames[i] = frame(b + 5i)` for `i < k`; `frame(b + 5(i+JUMP))` for `i ≥ k` | same splice on the action chunks at the same boundary |
| **colour** | unperturbed frames, with every block pixel replaced by `(200,45,45)` from `i ≥ k` on | unperturbed actions |

Teleport is a discontinuity *into the same episode's later trajectory*: the
state jumps forward 12 frameskip steps (60 raw frames) while every frame after
the jump is a genuine, self-consistent observation of the same task with the
same goal. Splicing actions at the same boundary keeps the post-jump dynamics
consistent, so the *only* unexplained event is the jump itself. Choosing the
same episode also removes a confound the cross-episode version would have: the
goal `T` ghost is redrawn per episode (its centroid varies by a few pixels
between episodes), so a cross-episode splice would silently teleport the goal
as well.

Block colour change: the block renders as a slate `T` of RGB ≈ `(144,162,184)`.
The mask is `‖rgb − (144,162,184)‖₂ < 22` (≈ 950–1050 px, asserted non-empty),
**dilated by 2 px** so the anti-aliased fringe is replaced too — otherwise a
grey outline survives and blunts the visual change. From `i ≥ k` the mask is
filled with `(200,45,45)`, a saturated red distinct from the block (slate), the
goal (light green) and the agent (blue).

### 3.4 The measurement

For each target index `t ∈ {3,…,11}` and each condition:

```
z      = encode(frames)                        # [1,12,192], encoder frozen
z_hat  = predict(z[t-3:t], a[t-3:t])[:, -1]    # 3-frame causal context
surprise(t) = mean over 192 dims of (z_hat - z[t])²
```

This is one step of the reference `JEPA.rollout` with `history_size = 3`: the
last predictor slot is the prediction of the next frame, and the context is the
last three latent/action pairs. Nothing is re-encoded per window — the whole
condition is encoded once and sliced, which is exact because the predictor is
position-relative (the positional table is indexed `0..T−1`) and causal.

Because the predictor has only 3 positional slots, a longer context is not
available; `W = 12` was chosen to give 3 pre-perturbation targets (`t = 3,4,5`),
the perturbation target (`t = 6`) and 5 post-perturbation targets (`t = 7..11`).

### 3.5 What is controlled

* At the measured step `t = k`, the context windows and the action inputs are
  **identical** to the unperturbed condition (indices `3..5 < k`). The only
  difference is the observed frame `z[k]`. The comparison is therefore a clean
  paired test of "same actions, same history, different observation".
* Frames at `t = 3,4,5` are byte-identical across all three conditions, so their
  surprise columns must be identical — the tables below show exactly that, which
  is an internal consistency check on the harness.
* The colour condition changes no positions; the teleport changes positions and
  no appearance.

---

## 4. Results (24 episodes, 9,797-episode pool, 8 CPU threads)

### 4.1 Per-target surprise curves

Mean ± s.e.m. across 24 episodes; `t` is the target frame index in condition
coordinates, `k = 6` is the perturbation step. Values are squared latent
prediction error (192-dim mean).

| `t` | 3 | 4 | 5 | **6 (k)** | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unperturbed | 0.0086 ± 0.0039 | 0.0082 ± 0.0023 | 0.0124 ± 0.0050 | **0.0074 ± 0.0009** | 0.0076 ± 0.0011 | 0.0073 ± 0.0009 | 0.0070 ± 0.0010 | 0.0081 ± 0.0017 | 0.0091 ± 0.0020 |
| teleport | 0.0086 ± 0.0039 | 0.0082 ± 0.0023 | 0.0124 ± 0.0050 | **1.9444 ± 0.0951** | 0.0093 ± 0.0011 | 0.0104 ± 0.0033 | 0.0061 ± 0.0010 | 0.0077 ± 0.0011 | 0.0124 ± 0.0039 |
| colour change | 0.0086 ± 0.0039 | 0.0082 ± 0.0023 | 0.0124 ± 0.0050 | **0.7684 ± 0.0475** | 0.1706 ± 0.0346 | 0.1817 ± 0.0350 | 0.1537 ± 0.0244 | 0.1483 ± 0.0206 | 0.1661 ± 0.0232 |

Full-precision arrays, per-episode spikes and s.e.m. vectors:
`.work/lewm/voe_results.json`.

Shape of the result:

* **Teleport is a one-step spike.** 1.9444 at `t = k` (262× the unperturbed
  value at the same target), back to 0.0093 at `t = k+1`, i.e. baseline. The model is surprised exactly
  once, then re-synchronises on the new state — the signature the paper plots.
* **Colour change is a spike plus a plateau.** 0.7684 at `t = k`, then
  0.15–0.18 for the rest of the window (≈ 16–20× baseline). The block is
  permanently off-manifold, so the model never fully re-synchronises inside the
  8-frame window; the physical violation, by contrast, becomes predictable as
  soon as it is observed.
* Ordering at `t = k`: teleport `1.9444` ≫ colour `0.7684` ≫ unperturbed
  `0.0074`.

### 4.2 Summary per condition

Baseline = mean over the shared pre-perturbation targets `t = 3,4,5`
(= 0.0097 for every condition, identical by construction). Paired statistics
compare each condition against the unperturbed condition on the same episode at
the same target `t = k`.

| Condition | Baseline | Spike at `t = k` (mean ± s.e.m.) | ×baseline | Peak step | `t = k+1` | Paired Δ (mean ± s.e.m.) | Paired `t` (df 23) | Episodes up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unperturbed | 0.0097 | 0.0074 ± 0.0009 | — | 5 | 0.0076 | — | — | — |
| teleport | 0.0097 | **1.9444 ± 0.0951** | **×199.5** | **6** | 0.0093 | +1.9371 ± 0.0957 | **20.25** | 24/24 |
| colour change | 0.0097 | **0.7684 ± 0.0475** | **×78.9** | **6** | 0.1706 | +0.7610 ± 0.0479 | **15.89** | 24/24 |

Median per-episode spike ratio against the same episode's unperturbed value:
teleport ×294, colour ×130. Worst-case paired delta: teleport `+0.5667`, colour
`+0.3979` — both far above zero, i.e. no episode contradicts the effect.

For orientation only: `|t| > 2.069` would be `p < 0.05` two-sided with df = 23
under a t-distribution. No p-value is computed here because a t-distribution CDF
is not in the standard library; the paired `t` statistics and the episode counts
are reported instead (note 7).

### 4.3 Teleport magnitude (state-jump size)

The teleport's size is a measured property of the replay, not a free parameter
we tuned. Block/agent displacement between the pre-jump and post-jump frames,
in image pixels (`pixel = world · 224/512`):

| Quantity | mean | min | max |
| --- | --- | --- | --- |
| block displacement | 43.4 px | 0.0 px | 79.6 px |
| agent displacement | 68.6 px | 27.1 px | 175.7 px |

22 of 24 episodes move the block by >5 px; in the two that do not, the block was
stationary while the agent still moved (27–176 px). The spike does not depend on
the block moving — the agent's jump alone is enough.

### 4.4 Dose–response (fixed episode set)

Same 12 episodes (`min-rows = 161` so the pool is identical for every row of the
table), varying only `JUMP`. The colour condition is independent of `JUMP`, so
its value must be identical across rows — it is, which validates that the
episode set and everything else are truly fixed.

| `JUMP` (frameskip steps) | block Δ | agent Δ | teleport spike at `t = k` | colour spike at `t = k` | paired `t` (teleport) |
| --- | --- | --- | --- | --- | --- |
| 4 | 22.2 px | 57.8 px | 1.1719 ± 0.1719 | 1.0993 ± 0.1329 | 6.71 |
| 12 | 51.7 px | 102.1 px | 1.9068 ± 0.1486 | 1.0993 ± 0.1329 | 12.49 |
| 20 | 54.6 px | 91.1 px | 1.8733 ± 0.1378 | 1.0993 ± 0.1329 | 13.22 |

The spike rises with the size of the state jump and then saturates: at 1.87–1.91
it is ~1.6× the per-dimension latent variance measured on a real episode
(`z` std 1.09 → variance 1.18, measured separately), i.e. the prediction error is
already larger than a typical latent fluctuation. We report the three points and
do not fit a functional form; the curve is not monotone in `JUMP` beyond ≈50 px.
Baseline in these runs: 0.0034.

### 4.5 CPU wall time

| Run | Episodes | Conditions | Targets | Wall time | Breakdown |
| --- | --- | --- | --- | --- | --- |
| **main result** | 24 | 3 | 9 | **61.62 s** | model load 0.30 s, data select 1.24 s, encode+predict loop 60.07 s |
| sweep `JUMP=4` | 12 | 3 | 9 | 20.03 s | |
| sweep `JUMP=12` | 12 | 3 | 9 | 19.76 s | |
| sweep `JUMP=20` | 12 | 3 | 9 | 31.24 s | |

The pass is 864 frame-encodes (12 frames × 3 conditions × 24 episodes) plus 648
predictor calls (9 targets × 3 conditions × 24 episodes) — ≈14 frame-encodes/s
end-to-end on ViT-tiny at 224 px, fp32, 8 threads. The encoder dominates; each
predictor call is a 3-token sequence.
Timings vary by up to ~50 % with machine load (the 31.24 s row ran with another
job in flight), so treat them as order-of-magnitude.

---

## 5. Tests

| Command | Result |
| --- | --- |
| `python3 -m unittest tests.test_lewm -v` | **27 tests, OK**, ~7–13 s (dataset test runs, not skipped) |
| `LEAN_BIN=$PWD/.work/lean4/bin/lean python3 -m unittest discover -s tests -t .` | **219 tests, OK** in 23.3 s — the pre-existing 111-test baseline is included and green; the rest are this phase's 27 plus other teammates' tests |

What `tests/test_lewm.py` pins:

1. checkpoint: 303 keys, 18,042,672 state-dict elements, 18,034,478 parameters,
   exact key/shape equality, strict load rejects a missing **and** an unexpected
   key, `config.json` ↔ state-dict agreement (patch geometry, 12 encoder layers,
   3 context frames, 6 conditional blocks, AdaLN `6·192`, 16 heads × 64,
   action encoder `10→768→192`, `BatchNorm1d(2048)` in both projectors);
2. `encode`/`predict`: shapes, determinism (`torch.equal` on repeat), batch
   grouping invariance, 4-D input keeps the time axis, rejection of uint8 /
   wrong channels / non-square / empty / `T > num_frames` / wrong action width;
3. `surprise`: exactly zero for identical latents, matches the documented
   reduction, strictly positive and strictly rising with perturbation size,
   `surprise_per_step` consistent with the scalar reduction;
4. frozen weights: no parameter tracks gradients, `train()` is a no-op, a
   forward pass moves neither weights nor BatchNorm statistics;
5. synthetic VoE (no dataset needed): four conditions built from deterministic
   synthetic PushT-like frames; teleport ≥10× the unperturbed value at the
   perturbed step, colour above unperturbed but below teleport, colour decays
   next step while the teleport leaves a trace;
6. dataset VoE (skips with an explicit reason when the 46 GB file or `h5py` is
   absent): block mask tracks `state` `(x, y)` within 20/30 px, and the
   unperturbed-vs-teleport-vs-colour ordering holds on 3 real episodes.

---

## 6. Deviations from the published protocol

1. **Trajectory source: replay, not simulation.** The paper rolls out PushT with
   a random policy biased toward the block; we replay expert episodes. The
   surprise *measurement* is identical; the observation distribution is
   expert-data, not random-policy. Both are inside the training distribution.
2. **Teleport = temporal splice, not a random state reset.** We jump to the same
   episode's later trajectory (60 raw frames ahead) instead of a uniformly random
   pose. Rationale: the goal ghost is redrawn per episode, so a cross-episode
   splice would also teleport the goal. The achieved jump is measured (43 px
   block, 69 px agent, medians in §4.3), not assumed.
3. **Colour change = solid repaint.** The paper does not specify the colour
   mapping or the size of the change; ours is aggressive (every block pixel
   dilated by 2 px, repainted to a saturated red). This is the most likely reason
   our colour spike is large and significant while the paper reports it as
   weaker and non-significant (note 7).
4. **Window is 12 frames / 9 measured targets.** The predictor has 3 positional
   slots, so each surprise value uses a 3-frame causal context; the paper does
   not state its VoE window length.
5. **No statistical test library.** Paired `t` statistics and episode counts are
   reported; no `p`-values (a t CDF is not in the standard library and adding
   `scipy` was out of scope).
6. **CPU only.** No GPU is present, so the whole VoE pass is CPU fp32, 8 threads.

---

## 7. Honesty notes

1. **Measured, not assumed — checkpoint identity.** The organ's key set and
   every tensor shape are byte-identical to the checkpoint; the load is strict in
   both directions. The 18,042,672 figure counts *all* tensors in the state dict
   (parameters + BatchNorm buffers); `model.parameters()` is 18,034,478. Both are
   reported so the two conventions cannot be confused.
2. **Measured — the organ is the reference model, not a look-alike.** `encode`
   and `predict` were compared to the unmodified reference stack (HF `ViTModel`
   + `.work/le-wm/jepa.py`/`module.py`) with the published weights: max abs
   difference `1.43e-06` on activations of `absmax ≈ 2.6`. `SIGReg` is
   bit-identical under the same seed. The residual is fp32 attention-path
   ordering (SDPA vs explicit matmul), three orders of magnitude below the VoE
   signal.
3. **Assumed-with-evidence — dataset semantics.** Row layout
   (`row = ep_offset + step`), frameskip-5 subsampling, the action-chunk
   convention and the `pixel = world · 224/512` mapping were each checked
   against the file's own `state`/`ep_len` columns (agent mask follows
   `state[0:2]` to sub-pixel precision over an episode; 200/200 sampled
   episodes obey the layout). The *training loader's* internal chunking is
   inferred from `train.py` (`frameskip: 5`, `input_dim = frameskip · 2 = 10`),
   not observed directly, because `stable_worldmodel` was not installed.
4. **Measured — the surprise reduction.** `surprise` is the mean squared error
   over the 192 latent dimensions (then over the batch), no square root. This is
   the paper's prediction-loss quantity, but the paper does not print the exact
   VoE aggregation, so the absolute numbers here are comparable to each other and
   to the model's own prediction loss, not directly to numbers read off a figure.
5. **Measured — the spike is paired and controlled.** At `t = k` the context and
   actions are identical to the unperturbed condition; the only difference is the
   observed frame. Pre-perturbation columns are byte-identical across conditions
   (a harness check). 24/24 episodes move in the same direction for both
   perturbations.
6. **Measured — teleport and colour differ in decay, not just magnitude.**
   Teleport returns to baseline at `t = k+1` (0.0093 vs 0.0097); colour stays at
   16–20× baseline for the rest of the window. We do not interpret this as
   "physical vs visual" beyond the ordering, because the two manipulations also
   differ in how long they persist in the input (one frame vs a permanent
   repaint) — that is a confound we did not isolate.
7. **Deviation — the colour result is *stronger* than the paper's.** The paper's
   Fig. 8 caption reports the teleportation increase as significant (paired
   t-test, p < 0.01) "but weaker and non-significant for color changes"; ours is
   significant (t = 15.89, 24/24 episodes up). Two candidate explanations, both
   unchecked: (a) our repaint is a much larger visual change than theirs;
   (b) we test the single change frame per episode, which is where the transient
   lives, while their plotted statistic may average over a window. This is a
   reproduction of the *measurement on replay*, not a replication of their
   manipulation, and the colour condition should be treated as "a visual
   violation is detected", not as "the paper's colour effect reproduces".
8. **Assumed — goal ghost is per-episode.** The light-green `T` in the frames is
   read as the goal marker (its centroid is nearly constant within an episode
   and varies by a few pixels between episodes). The teleport design avoids
   relying on this being exactly true by staying inside one episode.
9. **Measured — no gradient anywhere.** All parameters are frozen and every
   public function runs under `no_grad`; a test asserts weights and BatchNorm
   statistics are unchanged by forward passes, and `train()` is a no-op. The
   encoder is the measurement basis and was never adapted.
10. **Not done — no executor coupling, no retrieval/MPC.** This phase exposes
    latents and surprise only; no planning, no cost model, no P1/P3 wiring, and
    no protocol claim beyond the numbers in §4.
11. **Not done — no p-values, no confidence intervals on the ratios.** Only
    paired t statistics, s.e.m. and episode counts are reported. The ×199.5 and
    ×78.9 ratios are point estimates from 24 episodes.
12. **Byte-level correction to a working assumption:** filter id 32001 is Blosc1
    (LZ4 codec), not raw zstd. This was established from the filter's own header
    and confirmed by decoding two chunks byte-identically with an independent
    Blosc1 reader and with `hdf5plugin`.

---

## 8. Reproduction

```sh
# 0. dataset (13.1 GB archive -> 46.3 GB HDF5, ~93 s)
zstd -d .work/data/pusht_expert_train.h5.zst -o .work/data/pusht_expert_train.h5

# 1. organ + unit tests (no dataset needed for 26 of 27 tests)
python3 -m unittest tests.test_lewm -v
LEAN_BIN=$PWD/.work/lean4/bin/lean python3 -m unittest discover -s tests -t .

# 2. fidelity vs the real reference implementation (dev-time only)
PYTHONPATH=$PWD/.work/pydeps-tf4:$PWD/.work/pydeps:$PWD/.work/le-wm \
    python3 .work/lewm/verify_reference.py

# 3. VoE (the numbers in §4.1-4.3)
PYTHONPATH=$PWD/.work/pydeps python3 .work/lewm/voe.py --episodes 24
#    -> .work/lewm/voe_results.json

# 4. dose-response, fixed episode set (§4.4)
for j in 4 12 20; do
  PYTHONPATH=$PWD/.work/pydeps python3 .work/lewm/voe.py \
      --episodes 12 --jump $j --min-rows 161 --out .work/lewm/voe_fixed_jump$j.json
done
```

`--episodes 24` is deterministic given `--seed 0` and the eligibility rule
(`ep_len ≥ 5·(W+JUMP)+1`).

---

## 9. Deferred to a later phase

* Any adaptation or fine-tuning of the encoder — the checkpoint is the
  measurement basis, full stop.
* Surprise as a control signal: thresholds, calibration, and the coupling into
  P1/P3's observation loop.
* A background-only pixel-change control (repaint a non-object region of the same
  area), which is the missing control that would separate "the model detects
  object change" from "the model detects any large pixel change" (note 6).
* Longer-horizon surprise (multi-step rollouts) rather than the single-step
  measurement used here.
