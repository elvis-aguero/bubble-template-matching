#!/usr/bin/env python3
"""
E0-B — Cross-image pipeline evaluation (leave-one-session-out).

Hypothesis to falsify:
  "The current pipeline (per-level NCC + cross-scale NMS + calibration) achieves
   median relL1 < LOO oracle median relL1 on held-out images, meaning
   detect-then-count is extracting per-image information the oracle cannot."

Design:
  - 5 LOSO folds (sessions: C1S0014, C1S0019, C1S0024, C1S0004, C1S0010)
  - For each fold: train pipeline on all other sessions, predict on held-out session.
  - Compute relL1 per test image using the same binning as debug_pipeline.py.
  - Compare per-image relL1 against the E0-A LOO oracle (median histogram).
  - Report: per-image relL1, fold-level stats, overall median/mean, comparison to oracle.
  - Exclude images with n_gt < 100 from summary statistics (relL1 metric pathology).

USAGE: python scripts/experiments/eval_pipeline_e0b.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, load_image, parse_annotations, get_session_id
from bubble_histogram.pipeline import BubblePipeline

MAX_IMG_MEAN = 0.6
MIN_N_GT     = 100

# LOO oracle values from E0-A (median histogram, stable images)
# Key: image stem suffix that uniquely identifies the image; value: relL1_median
E0A_ORACLE = {
    "C1S0014_img006001": 0.536,
    "C1S0014_img009542": 0.705,
    "C1S0014_img018008": 0.781,
    "C1S0014_img018351": 0.502,
    "C1S0019_img003593": 0.543,
    "C1S0019_img011890": 0.432,
    "C1S0024_img014500": 0.548,
    "C1S0004_IMG_S0001004509": 0.755,
    "C1S0004_IMG_S0001005070": 1.462,
    "C1S0004_IMG_S0001012062": 0.609,
    "C1S0010_IMG_S0001005432": 0.740,
    "C1S0010_IMG_S0001019655": 1.172,
}


def bin_gt(result, bubbles):
    """Bin GT bubbles into pipeline radius bins; return pred, gt arrays and relL1."""
    radii = np.array(result["radius_px"])
    pred  = np.array(result["expected_count"])
    log_r = np.log(radii)
    half = (log_r[1] - log_r[0]) / 2 if len(log_r) > 1 else 0.1
    edges = np.exp(np.concatenate([
        [log_r[0] - half],
        (log_r[:-1] + log_r[1:]) / 2,
        [log_r[-1] + half],
    ]))
    gt = np.histogram([b.radius for b in bubbles], bins=edges)[0].astype(float)
    rel_l1 = float(np.abs(pred - gt).sum() / max(gt.sum(), 1))
    return radii, pred, gt, rel_l1


def lookup_oracle(img_stem: str) -> float | None:
    for key, val in E0A_ORACLE.items():
        if key in img_stem:
            return val
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out", type=Path, default=Path("output/e0b_pipeline_eval.png"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Identify all sessions present in the data directory
    all_img_paths = sorted((data_dir / "images").glob("*.png"))
    sessions = sorted(set(get_session_id(p.name) for p in all_img_paths))
    print(f"Sessions: {sessions}")

    cfg = PipelineConfig()

    all_results = []  # accumulate per-image results across folds

    for val_session in sessions:
        print(f"\n{'='*65}")
        print(f"FOLD: val_session={val_session}")
        ds = AnnotatedDataset(data_dir, val_session=val_session)
        n_train = len(ds.template_images)
        n_test  = len(ds.test_images)
        print(f"  Train: {n_train} images,  Test: {n_test} images")

        if n_train == 0 or n_test == 0:
            print("  Skipping (empty train or test set)")
            continue

        # Filter tractable images (skip photometrically dead)
        tractable_train = [p for p in ds.template_images
                           if load_image(p).mean() < MAX_IMG_MEAN]
        tractable_test  = [p for p in ds.test_images]  # keep all test images to show unstable

        if len(tractable_train) == 0:
            print("  Skipping (no tractable train images)")
            continue

        # Override dataset splits to only use tractable training images
        ds.template_images    = tractable_train
        ds.calibration_images = tractable_train

        print(f"  Tractable train: {len(tractable_train)}")
        print(f"  Training pipeline...", flush=True)
        pipeline = BubblePipeline(cfg)
        pipeline.train(ds)
        print(f"  Training done.")

        # Predict on each test image
        for img_path in tractable_test:
            lbl_path = data_dir / "labels" / (img_path.stem + ".json")
            if not lbl_path.exists():
                continue
            img = load_image(img_path)
            if img.mean() >= MAX_IMG_MEAN:
                print(f"  Skipping {img_path.stem[-24:]} (dead frame, mean={img.mean():.3f})")
                continue
            bubbles = parse_annotations(lbl_path)
            n_gt = len(bubbles)

            result = pipeline.predict(img)
            _, pred, gt, rl1 = bin_gt(result, bubbles)
            oracle_rl1 = lookup_oracle(img_path.stem)
            stable = n_gt >= MIN_N_GT

            all_results.append({
                "stem":       img_path.stem,
                "session":    val_session,
                "img_mean":   float(img.mean()),
                "n_gt":       n_gt,
                "pred_total": float(pred.sum()),
                "gt_total":   float(gt.sum()),
                "relL1":      rl1,
                "oracle":     oracle_rl1,
                "stable":     stable,
            })

            oracle_str = f"{oracle_rl1:.3f}" if oracle_rl1 is not None else "  n/a"
            delta_str  = (f"{rl1 - oracle_rl1:+.3f}" if oracle_rl1 is not None else "   n/a")
            flag = "" if stable else " (unstable)"
            print(f"  {img_path.stem[-32:]:<32}  n_gt={n_gt:>4}  "
                  f"relL1={rl1:.3f}  oracle={oracle_str}  Δ={delta_str}{flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    stable = [r for r in all_results if r["stable"]]
    print(f"\n{'='*65}")
    print(f"All images: {len(all_results)},  Stable (n_gt≥{MIN_N_GT}): {len(stable)}")

    if not stable:
        print("No stable images. Cannot produce summary.")
        return

    pipe_vals   = np.array([r["relL1"]  for r in stable])
    oracle_vals = np.array([r["oracle"] for r in stable if r["oracle"] is not None])

    print(f"\nPipeline relL1 (stable images):")
    print(f"  median={np.median(pipe_vals):.3f}  mean={pipe_vals.mean():.3f}  "
          f"std={pipe_vals.std():.3f}  [{pipe_vals.min():.3f}, {pipe_vals.max():.3f}]")

    if len(oracle_vals) > 0:
        print(f"\nE0-A oracle relL1 (matching stable images):")
        print(f"  median={np.median(oracle_vals):.3f}  mean={oracle_vals.mean():.3f}  "
              f"std={oracle_vals.std():.3f}  [{oracle_vals.min():.3f}, {oracle_vals.max():.3f}]")

        paired = [(r["relL1"], r["oracle"]) for r in stable if r["oracle"] is not None]
        n_pipeline_wins  = sum(p < o for p, o in paired)
        n_oracle_wins    = sum(p > o for p, o in paired)
        n_tied           = sum(p == o for p, o in paired)
        print(f"\nPer-image wins: pipeline={n_pipeline_wins}, oracle={n_oracle_wins}, tied={n_tied} "
              f"(out of {len(paired)})")

        print(f"\n{'='*65}")
        print("VERDICT:")
        pipe_med   = float(np.median(pipe_vals))
        oracle_med = float(np.median(oracle_vals))
        gap = pipe_med - oracle_med
        print(f"  Pipeline median relL1: {pipe_med:.3f}")
        print(f"  Oracle median relL1:   {oracle_med:.3f}")
        print(f"  Gap (pipeline − oracle): {gap:+.3f}")
        if gap < -0.05:
            print("  Pipeline BEATS oracle → detect-then-count extracts per-image information.")
            print("  Architecture is valid; feature quality is the bottleneck.")
        elif gap < 0.10:
            print("  Pipeline ≈ oracle (within 0.10). No meaningful advantage for either.")
            print("  Detect-then-count may not be necessary; distribution regression worth testing.")
        else:
            print(f"  Pipeline WORSE than oracle by {gap:.3f}.")
            print("  Detect-then-count is not outperforming a GT-oracle lookup table.")
            print("  Architecture is not adding measurable value over a distributional prior.")

    # ── Table ─────────────────────────────────────────────────────────────────
    print(f"\n{'Image':<38}  {'n_gt':>5}  {'pipe':>6}  {'oracle':>8}  {'Δ':>7}  session")
    for r in sorted(all_results, key=lambda x: x["session"]):
        oracle_str = f"{r['oracle']:.3f}" if r["oracle"] is not None else "   n/a"
        delta = r["relL1"] - r["oracle"] if r["oracle"] is not None else float("nan")
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "    n/a"
        flag = "" if r["stable"] else " *"
        print(f"  {r['stem'][-36:]:<36}  {r['n_gt']:>5}  {r['relL1']:>6.3f}  "
              f"{oracle_str:>8}  {delta_str:>7}  {r['session']}{flag}")
    print("  * unstable (n_gt < 100, excluded from summary)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: per-image pipeline vs oracle bar chart (stable only)
    ax = axes[0]
    stable_sorted = sorted(stable, key=lambda x: x["session"])
    names = [r["stem"][-20:] for r in stable_sorted]
    x = np.arange(len(stable_sorted))
    p_vals = [r["relL1"]  for r in stable_sorted]
    o_vals = [r["oracle"] if r["oracle"] is not None else np.nan for r in stable_sorted]
    ax.bar(x - 0.2, p_vals, width=0.4, color="steelblue", alpha=0.8, label="Pipeline")
    ax.bar(x + 0.2, o_vals, width=0.4, color="seagreen",  alpha=0.8, label="Oracle (E0-A)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("relL1")
    ax.set_title(f"Pipeline vs Oracle per image (stable, n_gt≥{MIN_N_GT})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Middle: scatter pipeline vs oracle
    ax = axes[1]
    paired_stable = [(r["relL1"], r["oracle"]) for r in stable if r["oracle"] is not None]
    if paired_stable:
        px_, ox_ = zip(*paired_stable)
        ax.scatter(ox_, px_, s=60, alpha=0.8, color="steelblue", edgecolors="none")
        lims = [0, max(max(px_), max(ox_)) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=1, label="pipeline=oracle")
        ax.fill_between(lims, lims, [1.0]*2, alpha=0.05, color="tomato",
                        label="pipeline worse")
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Oracle relL1 (E0-A)")
    ax.set_ylabel("Pipeline relL1 (E0-B)")
    ax.set_title("Pipeline vs Oracle scatter")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: delta (pipeline − oracle) distribution
    ax = axes[2]
    deltas = [r["relL1"] - r["oracle"] for r in stable if r["oracle"] is not None]
    if deltas:
        bins = np.linspace(min(deltas) - 0.1, max(deltas) + 0.1, 15)
        ax.hist(deltas, bins=bins, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="pipeline=oracle")
        ax.axvline(np.median(deltas), color="steelblue", linestyle=":",
                   linewidth=1.5, label=f"median={np.median(deltas):+.3f}")
    ax.set_xlabel("Δ relL1 (pipeline − oracle)")
    ax.set_ylabel("Count")
    ax.set_title("Pipeline vs Oracle gap distribution\n"
                 "(negative = pipeline better, positive = oracle better)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"E0-B — Cross-image pipeline evaluation (LOSO, {len(all_results)} images)",
                 fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
