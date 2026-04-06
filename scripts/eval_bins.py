"""
Per-bin relL1 decomposition across seeds.

For each pyramid level (size bin), reports the average absolute prediction error
relative to the true total — tells us which size bins drive the overall relL1.

Usage:
    python scripts/eval_bins.py --data seed_v04 --seeds 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.pipeline import BubblePipeline


MIN_TEST_BUBBLES = 50
TEMPLATE_FRAC    = 0.30
CALIB_FRAC       = 0.65


def _bin_annotations(radii_px: np.ndarray, ann_radii: np.ndarray) -> np.ndarray:
    """Assign annotated bubble radii to pyramid level bins. Returns count per bin."""
    log_r = np.log(radii_px)
    half = (log_r[1] - log_r[0]) / 2 if len(log_r) > 1 else 0.1
    edges = np.exp(np.concatenate([
        [log_r[0] - half],
        (log_r[:-1] + log_r[1:]) / 2,
        [log_r[-1] + half],
    ]))
    counts, _ = np.histogram(ann_radii, bins=edges)
    return counts.astype(float)


def run_seed(seed: int, data_root: Path, config: PipelineConfig):
    """Return (radius_px, pred_counts, true_counts, n_test_bubbles) for one seed."""
    dataset = AnnotatedDataset(
        data_root,
        template_frac=TEMPLATE_FRAC,
        calibration_frac=CALIB_FRAC,
        seed=seed,
    )

    test_paths = [
        p for p in dataset.test_images
        if len(dataset.load_sample(p).bubbles) >= MIN_TEST_BUBBLES
    ]
    if not test_paths:
        return None

    pipeline = BubblePipeline(config)
    pipeline.train(dataset)

    all_pred = []
    all_true = []
    n_total  = 0
    radii_px = None

    for p in test_paths:
        sample = dataset.load_sample(p)
        result = pipeline.predict(sample.image)

        r_px   = np.array(result["radius_px"])
        pred   = np.array(result["expected_count"])
        ann    = np.array([b.radius for b in sample.bubbles])
        true   = _bin_annotations(r_px, ann)

        if radii_px is None:
            radii_px = r_px

        all_pred.append(pred)
        all_true.append(true)
        n_total += len(ann)

    pred_sum = np.sum(all_pred, axis=0)
    true_sum = np.sum(all_true, axis=0)
    return radii_px, pred_sum, true_sum, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="seed_v04")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out",   default="output/bin_errors.png")
    args = parser.parse_args()

    data_root = Path(args.data)
    # Best approach: pixel calibration (f⁻ from random pixels, pixel prior) + LM prediction
    config = PipelineConfig(local_maxima_calibration=False, predict_local_maxima=True)

    # Accumulate per-bin errors across seeds
    per_seed_abs_err = []   # list of (n_bins,) arrays: |pred - true|
    per_seed_true    = []   # list of (n_bins,) arrays: true counts
    n_total_all      = []
    radii_px         = None
    seeds_used       = []

    for seed in range(args.seeds):
        print(f"seed {seed:2d}...", flush=True, end=" ")
        result = run_seed(seed, data_root, config)
        if result is None:
            print("skipped (no valid test image)")
            continue
        r_px, pred, true, n = result
        seeds_used.append(seed)
        radii_px = r_px
        per_seed_abs_err.append(np.abs(pred - true))
        per_seed_true.append(true)
        n_total_all.append(n)
        ratio = pred.sum() / true.sum() if true.sum() > 0 else float("nan")
        rel1  = np.abs(pred - true).sum() / true.sum() if true.sum() > 0 else float("nan")
        print(f"ratio={ratio:.2f}x  relL1={rel1:.3f}")

    print(f"\nSeeds used: {seeds_used}")

    abs_err = np.stack(per_seed_abs_err)  # (n_seeds, n_bins)
    true_arr = np.stack(per_seed_true)    # (n_seeds, n_bins)

    # Per-bin mean |pred - true| / mean(N_true_total)
    mean_n_true = np.mean(n_total_all)
    mean_abs_err_per_bin = abs_err.mean(axis=0)              # avg across seeds
    rel_per_bin = mean_abs_err_per_bin / mean_n_true         # fractional contribution

    # Also: per-bin true fraction (how much of the distribution is here)
    mean_true_per_bin = true_arr.mean(axis=0)
    true_frac = mean_true_per_bin / mean_true_per_bin.sum()

    # Print table
    print(f"\n{'radius_px':>10}  {'true_frac':>9}  {'mean|err|':>9}  {'rel_contrib':>11}  {'ratio err/true':>14}")
    print("-" * 62)
    for i, r in enumerate(radii_px):
        print(f"{r:10.2f}  {true_frac[i]:9.3f}  {mean_abs_err_per_bin[i]:9.2f}  "
              f"{rel_per_bin[i]:11.4f}  "
              f"{(mean_abs_err_per_bin[i] / max(mean_true_per_bin[i], 1e-6)):14.3f}")

    # Cumulative relL1 from largest contributor down
    order = np.argsort(rel_per_bin)[::-1]
    cumrel = np.cumsum(rel_per_bin[order])
    print(f"\nTop bins by contribution (cumulative relL1 = {rel_per_bin.sum():.4f}):")
    print(f"  {'bin i':>6}  {'radius_px':>10}  {'contribution':>12}  {'cumulative':>10}")
    for rank, i in enumerate(order[:10]):
        print(f"  {i:6d}  {radii_px[i]:10.2f}  {rel_per_bin[i]:12.4f}  {cumrel[rank]:10.4f}")

    # ---- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: predicted vs true mean histogram
    ax = axes[0]
    mean_pred = np.stack([p for p in per_seed_abs_err]).mean(axis=0)  # placeholder reuse
    # recompute properly
    pred_sums = []
    for seed_i, (ae, tr) in enumerate(zip(per_seed_abs_err, per_seed_true)):
        # we don't have pred directly; reconstruct from ae+true (signed error unknown)
        pass
    # Use true and abs-error bars instead
    x = np.arange(len(radii_px))
    ax.bar(x, mean_true_per_bin, color="steelblue", alpha=0.7, label="Mean true count per bin")
    ax.bar(x, mean_abs_err_per_bin, bottom=0, color="tomato", alpha=0.5,
           label="Mean |pred − true| per bin")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{r:.1f}" for r in radii_px[::2]], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Effective radius (px)")
    ax.set_ylabel("Mean count (across seeds)")
    ax.set_title("True counts vs absolute prediction error per size bin")
    ax.legend()

    # Bottom: relative contribution to relL1
    ax2 = axes[1]
    ax2.bar(x, rel_per_bin, color="darkorange", alpha=0.8)
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([f"{r:.1f}" for r in radii_px[::2]], rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Effective radius (px)")
    ax2.set_ylabel("Contribution to relL1")
    ax2.set_title(f"Per-bin relL1 contribution  (total = {rel_per_bin.sum():.3f})")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
