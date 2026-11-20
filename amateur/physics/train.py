"""Train a micro-LeWM on the flippable 2-D world (CPU, reproducible).

Protocol (the phase-3 boundary slice)
-------------------------------------
1. Roll out the environment with a fixed seed.  By default the fitting set
   is *in-distribution*: gravity sign +1 (downward) for every episode.
   ``--train-gravity mixed`` fits on both signs instead (a control, not the
   headline protocol).
2. Fit the CNN encoder + GRU predictor with the LeWM two-loss objective
   ``MSE(z_hat_{t+1}, z_{t+1}) + lambda * SIGReg(z)`` on contiguous windows
   of the rollouts.
3. Freeze the encoder (binding ruling: overlays adapt the predictor only)
   and checkpoint.
4. Evaluate surprise (mean squared latent prediction error per transition)
   on held-out episodes in three regimes:
   * in-distribution: gravity +1 for the whole episode;
   * mid-episode flip: identical seeds to the in-distribution set, gravity
     flipped at ``--flip-at`` (the phase-5 acceptance event);
   * whole-episode flipped: gravity -1 from reset (the model has never seen
     this dynamics law).
5. Write ``checkpoint.pt`` and ``stats.json`` to ``--out-dir``
   (``.work/forge/`` by default).

Reproduce the committed numbers with::

    python3 -m amateur.physics.train --out-dir .work/forge --seed 0

Use ``--help`` for all knobs.  Artifacts are not committed; regenerate with
the command above and with ``docs/phase3-boundary-slice.md`` §Reproduction.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import (
    RolloutBatch,
    concat_rollouts,
    frames_tensor,
    generate_rollouts,
)
from .env import PhysicsEnv, WorldConfig
from .micro_lewm import MicroLeWM, MicroLeWMConfig

DEFAULT_OUT_DIR = ".work/forge"


# --------------------------------------------------------------------- helpers
def _chunks(n: int, size: int) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


def _flat(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("cannot take quantiles of an empty array")
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "q90": float(np.quantile(flat, 0.90)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(flat.max()),
    }


def _surprise_arrays(
    model: MicroLeWM,
    batch: RolloutBatch,
    *,
    chunk_size: int = 8,
) -> dict[str, np.ndarray]:
    """Per-transition surprise and identity residual, ``(N, T-1)`` each."""
    frames = frames_tensor(batch)
    actions = torch.from_numpy(batch.actions).long()
    surprises: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    with torch.no_grad():
        for indices in _chunks(len(batch), chunk_size):
            frames_chunk = frames[indices]
            actions_chunk = actions[indices]
            latents = model.encode(frames_chunk)
            prediction = model.predict(latents, actions_chunk)
            error = (prediction - latents[:, 1:]).pow(2).mean(dim=-1)
            identity = (latents[:, 1:] - latents[:, :-1]).pow(2).mean(dim=-1)
            surprises.append(error.cpu().numpy())
            identities.append(identity.cpu().numpy())
    return {
        "surprise": np.concatenate(surprises, axis=0),
        "identity": np.concatenate(identities, axis=0),
    }


def _latent_stats(model: MicroLeWM, batch: RolloutBatch, *, chunk_size: int = 8) -> dict[str, Any]:
    frames = frames_tensor(batch)
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for indices in _chunks(len(batch), chunk_size):
            latents.append(model.encode(frames[indices]).cpu().numpy())
    stacked = np.concatenate(latents, axis=0).reshape(-1, model.config.latent_dim)
    per_dim = stacked.std(axis=0).astype(np.float64)
    return {
        "samples": int(stacked.shape[0]),
        "per_dim_std": [float(v) for v in per_dim],
        "std_mean": float(per_dim.mean()),
        "std_min": float(per_dim.min()),
        "std_max": float(per_dim.max()),
        "frac_dims_std_gt_0.05": float((per_dim > 0.05).mean()),
        "frac_dims_std_gt_0.25": float((per_dim > 0.25).mean()),
        "mean_abs": float(np.abs(stacked).mean()),
        "mean_norm": float(np.linalg.norm(stacked, axis=1).mean()),
    }


def _curve_summary(values: list[float], *, points: int = 60) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    stride = max(1, len(array) // points)
    curve = [
        {"step": int(i + 1), "value": float(array[i])}
        for i in range(0, len(array), stride)
    ]
    last_window = max(1, len(array) // 10)
    return {
        "first": float(array[0]),
        "final": float(array[-1]),
        "min": float(array.min()),
        "min_step": int(array.argmin()) + 1,
        "mean_last_10pct": float(array[-last_window:].mean()),
        "curve": curve,
    }


def _per_step_means(array: np.ndarray) -> list[float]:
    return [float(v) for v in array.mean(axis=0)]


# --------------------------------------------------------------------- protocol
def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    wall_start = time.perf_counter()

    env = PhysicsEnv(WorldConfig(frame_size=int(args.frame_size)))
    config = MicroLeWMConfig(
        frame_size=int(args.frame_size),
        latent_dim=int(args.latent_dim),
        encoder_widths=tuple(int(w) for w in args.encoder_widths),
        predictor_hidden=int(args.predictor_hidden),
        predictor_layers=int(args.predictor_layers),
        sigreg_knots=int(args.sigreg_knots),
        sigreg_projections=int(args.sigreg_projections),
        sigreg_weight=float(args.sigreg_weight),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
    )
    model = MicroLeWM(config)

    t0 = time.perf_counter()
    train_normal: RolloutBatch | None = None
    train_flipped: RolloutBatch | None = None
    if args.train_gravity == "mixed":
        normal_count = int(args.episodes) // 2
        flipped_count = int(args.episodes) - normal_count
        train_normal = generate_rollouts(
            env, normal_count, int(args.horizon), seed=int(args.seed) + 11, gravity_sign=1
        )
        train_flipped = generate_rollouts(
            env, flipped_count, int(args.horizon), seed=int(args.seed) + 12, gravity_sign=-1
        )
        train_set = concat_rollouts([train_normal, train_flipped])
    else:
        train_set = generate_rollouts(
            env, int(args.episodes), int(args.horizon), seed=int(args.seed) + 11, gravity_sign=1
        )
    val_set = generate_rollouts(
        env, int(args.val_episodes), int(args.horizon), seed=int(args.seed) + 21, gravity_sign=1
    )

    # Held-out evaluation: one episode set, rolled out three ways.  The normal
    # and mid-flip sets use *identical seeds*, so any difference is caused by
    # the dynamics flip alone.
    eval_seed = int(args.seed) + 31
    eval_normal = generate_rollouts(
        env, int(args.eval_episodes), int(args.horizon), seed=eval_seed, gravity_sign=1
    )
    eval_flip = generate_rollouts(
        env,
        int(args.eval_episodes),
        int(args.horizon),
        seed=eval_seed,
        gravity_sign=1,
        flip_at=int(args.flip_at),
    )
    eval_whole_flip = generate_rollouts(
        env,
        int(args.eval_episodes),
        int(args.horizon),
        seed=eval_seed,
        gravity_sign=-1,
    )
    timings["rollout_generation_seconds"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    history = model.fit(
        train_set,
        steps=int(args.steps),
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        eval_rollouts=val_set,
        eval_every=int(args.eval_every),
        freeze_after=True,
        log=not args.quiet,
    )
    timings["training_seconds"] = float(history["wall_seconds"])
    timings["training_ms_per_step"] = float(history["ms_per_step"])
    assert model.encoder_frozen, "fit must freeze the encoder"

    # Fixed window on validation data: model vs identity reference.
    window_frames, window_actions = _first_window(val_set, int(args.seq_len), int(args.batch_size))
    with torch.no_grad():
        window_latents = model.encode(window_frames)
        window_pred = model.predict(window_latents, window_actions)
        val_pred_mse = float(torch.nn.functional.mse_loss(window_pred, window_latents[:, 1:]))
        val_identity_mse = float(
            torch.nn.functional.mse_loss(window_latents[:, 1:], window_latents[:, :-1])
        )

    t0 = time.perf_counter()
    normal_arrays = _surprise_arrays(model, eval_normal, chunk_size=int(args.eval_chunk))
    flip_arrays = _surprise_arrays(model, eval_flip, chunk_size=int(args.eval_chunk))
    whole_arrays = _surprise_arrays(model, eval_whole_flip, chunk_size=int(args.eval_chunk))
    latent_stats = _latent_stats(model, eval_normal, chunk_size=int(args.eval_chunk))
    timings["evaluation_seconds"] = float(time.perf_counter() - t0)
    timings["total_seconds"] = float(time.perf_counter() - wall_start)

    flip_at = int(args.flip_at)
    normal = normal_arrays["surprise"]
    normal_identity = normal_arrays["identity"]
    flipped = flip_arrays["surprise"]
    flipped_identity = flip_arrays["identity"]
    whole = whole_arrays["surprise"]

    # Post-flip predictions start at index flip_at (prediction of frame
    # flip_at + 1 from frame flip_at); indices are transition indices.
    post_flip = flipped[:, flip_at:]
    pre_flip = flipped[:, :flip_at]
    immediate = flipped[:, flip_at : flip_at + 4]
    # Fair reference: the same transition indices on the unflipped twin.
    normal_matched = normal[:, flip_at:]
    normal_immediate = normal[:, flip_at : flip_at + 4]

    separation = {
        "normal_mean": float(normal.mean()),
        "normal_median": float(np.median(normal)),
        "post_flip_mean": float(post_flip.mean()),
        "post_flip_median": float(np.median(post_flip)),
        "immediate4_mean": float(immediate.mean()),
        "normal_matched_mean": float(normal_matched.mean()),
        "normal_immediate4_mean": float(normal_immediate.mean()),
        "post_flip_over_normal_mean": float(post_flip.mean() / normal.mean()),
        "post_flip_over_matched_mean": float(post_flip.mean() / normal_matched.mean()),
        "immediate4_over_normal_mean": float(immediate.mean() / normal.mean()),
        "immediate4_over_matched_mean": float(immediate.mean() / normal_immediate.mean()),
        "post_flip_over_normal_median": float(np.median(post_flip) / np.median(normal)),
        "whole_flip_over_normal_mean": float(whole.mean() / normal.mean()),
        "model_over_identity_normal": float(normal.mean() / float(np.mean(normal_identity))),
    }

    stats: dict[str, Any] = {
        "artifact": "phase3-boundary-slice",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": int(args.seed),
        "protocol": {
            "environment": "amateur.physics.env.PhysicsEnv",
            "frame_size": int(args.frame_size),
            "horizon": int(args.horizon),
            "train_gravity": args.train_gravity,
            "train_episodes": int(args.episodes),
            "flip_at": flip_at,
            "eval_episodes": int(args.eval_episodes),
            "eval_seeds": "same seeds for normal / mid-flip / whole-flip sets",
            "encoder_frozen_after_fit": True,
            "surprise_definition": "mean over latent dims of (z_hat_{t+1} - z_{t+1})^2",
            "reproduce": (
                "python3 -m amateur.physics.train "
                f"--out-dir {args.out_dir} --seed {args.seed}"
            ),
        },
        "model_config": config.to_dict(),
        "param_counts": model.param_counts(),
        "training": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "wall_seconds": float(history["wall_seconds"]),
            "ms_per_step": float(history["ms_per_step"]),
            "loss": _curve_summary(history["loss"]),
            "pred_mse": _curve_summary(history["pred_mse"]),
            "sigreg": _curve_summary(history["sigreg"]),
            "eval": {
                "steps": [int(s) for s in history["eval_steps"]],
                "pred_mse": [float(v) for v in history["eval_pred_mse"]],
            },
            "val_window": {
                "pred_mse": val_pred_mse,
                "identity_mse": val_identity_mse,
                "model_over_identity": float(val_pred_mse / val_identity_mse),
            },
        },
        "latent": latent_stats,
        "evaluation": {
            "in_distribution": {
                "surprise": _quantiles(normal),
                "identity_residual": _quantiles(normal_identity),
                "per_step_mean": _per_step_means(normal),
            },
            "flipped_mid_episode": {
                "flip_at": flip_at,
                "all": _quantiles(flipped),
                "pre_flip": _quantiles(pre_flip),
                "post_flip": _quantiles(post_flip),
                "immediate4": _quantiles(immediate),
                "identity_residual_post_flip": _quantiles(flipped_identity[:, flip_at:]),
                "per_step_mean": _per_step_means(flipped),
            },
            "flipped_whole_episode": {
                "surprise": _quantiles(whole),
                "identity_residual": _quantiles(whole_arrays["identity"]),
                "per_step_mean": _per_step_means(whole),
            },
            "separation": separation,
        },
        "timings": timings,
        "environment_info": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": int(torch.get_num_threads()),
            "cpu": _cpu_name(),
        },
    }

    checkpoint = model.save(
        out_dir / "checkpoint.pt",
        extra={
            "seed": int(args.seed),
            "protocol": stats["protocol"],
            "separation": separation,
            "train_gravity": args.train_gravity,
        },
    )
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    stats["artifacts"] = {"checkpoint": str(checkpoint), "stats": str(stats_path)}
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    if not args.quiet:
        print("\n=== phase-3 boundary slice ===")
        print("params:", model.param_counts())
        print(
            "training:", f"{history['wall_seconds']:.1f}s",
            f"({history['ms_per_step']:.1f} ms/step, {args.steps} steps)",
        )
        print(
            "loss:", f"{history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}",
            "| pred MSE:", f"{history['pred_mse'][0]:.6f} -> {history['pred_mse'][-1]:.6f}",
            "| sigreg:", f"{history['sigreg'][0]:.4f} -> {history['sigreg'][-1]:.4f}",
        )
        print(
            "latent std:", f"mean {latent_stats['std_mean']:.3f}",
            f"min {latent_stats['std_min']:.3f}",
            f"frac>0.05 {latent_stats['frac_dims_std_gt_0.05']:.2f}",
        )
        print(
            "surprise normal:", f"mean {separation['normal_mean']:.4f}",
            f"median {separation['normal_median']:.4f}",
        )
        print(
            "surprise post-flip:", f"mean {separation['post_flip_mean']:.4f}",
            f"immediate4 {separation['immediate4_mean']:.4f}",
            f"-> {separation['post_flip_over_normal_mean']:.2f}x normal "
            f"({separation['post_flip_over_matched_mean']:.2f}x phase-matched)",
        )
        print("artifacts:", checkpoint, stats_path)
    return stats


def _first_window(batch: RolloutBatch, seq_len: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .data import sample_window_batch

    return sample_window_batch(
        batch,
        min(seq_len, batch.horizon),
        min(batch_size, len(batch)),
        np.random.default_rng(0),
    )


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a micro-LeWM on the flippable 2-D world.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="artifact directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--flip-at", type=int, default=8)
    parser.add_argument("--train-gravity", choices=("normal", "mixed"), default="normal")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-chunk", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--encoder-widths", type=int, nargs=3, default=(16, 32, 32))
    parser.add_argument("--predictor-hidden", type=int, default=128)
    parser.add_argument("--predictor-layers", type=int, default=2)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-projections", type=int, default=256)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
