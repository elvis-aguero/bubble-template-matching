#!/usr/bin/env python3
"""
O1 — Profile NCC response curve across scale levels per GT bubble.

Hypothesis (PAL E7): The NCC peak falls consistently at delta = -2 to -3
(finer than the correct level), meaning the averaged template encodes an
effective canonical radius of ~3.5-4px rather than the assumed 5px.

Falsification: if the median peak delta is near 0, the template scale
offset hypothesis is wrong.

USAGE: python scripts/profile_ncc_response.py [pipeline.pkl] [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.calibration import _local_maxima, _lm_min_dist
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps
from bubble_histogram.pipeline import BubblePipeline


def best_lm_score_at_level(cache, lv, cy, cx, radius):
    """Max NCC score among LMs within bubble.radius at level lv."""
    ys, xs, sc = cache[lv]
    if not len(sc):
        return np.nan
    dists = np.hypot(ys - cy, xs - cx)
    in_r = dists < radius
    return float(sc[in_r].max()) if in_r.any() else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=Path, default=Path("output/pipeline.pkl"), nargs="?")
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--n-bubbles", type=int, default=40,
                        help="Number of GT bubbles to profile (sampled across sizes)")
    parser.add_argument("--out", type=Path, default=Path("output/ncc_response_profile.png"))
    args = parser.parse_args()

    pipeline = BubblePipeline.load(args.pipeline)
    cfg = pipeline.config
    ds = AnnotatedDataset(args.data_dir, template_frac=0.30, calibration_frac=0.65, seed=42)

    # Use the first calibration image (same as debug_pipeline.py)
    image_path = list(ds.calibration_images)[0]
    sample = ds.load_sample(image_path)
    ncc_results = compute_ncc_maps(sample.image, pipeline.templates, cfg)
    n_levels = len(ncc_results)
    eff_radii = np.array([er for er, _ in ncc_results])
    min_d = _lm_min_dist(cfg)

    # Pre-cache per-level LMs in original-image coordinates
    cache = []
    for lv, (er, sm) in enumerate(ncc_results):
        alp = cfg.template_size / (2.0 * cfg.template_context_factor * er)
        pks = _local_maxima(sm, min_d)
        sc = sm[pks[:, 0], pks[:, 1]] if len(pks) else np.array([])
        ys = pks[:, 0] / alp if len(pks) else np.array([])
        xs = pks[:, 1] / alp if len(pks) else np.array([])
        cache.append((ys, xs, sc))

    # Filter bubbles whose correct level is covered by the pyramid
    bubbles = [b for b in sample.bubbles
               if eff_radii.min() <= b.radius <= eff_radii.max()]

    # Sample evenly across log-radius space so we get coverage of all sizes
    if len(bubbles) > args.n_bubbles:
        log_r = np.log([b.radius for b in bubbles])
        bins = np.linspace(log_r.min(), log_r.max(), args.n_bubbles + 1)
        selected = []
        for i in range(args.n_bubbles):
            in_bin = [b for b, lr in zip(bubbles, log_r) if bins[i] <= lr < bins[i + 1]]
            if in_bin:
                selected.append(in_bin[len(in_bin) // 2])  # pick middle of bin
        bubbles = selected

    print(f"Profiling {len(bubbles)} GT bubbles across {n_levels} levels...")

    # For each bubble: score at each level, centred on delta=0
    delta_range = 6   # ±6 levels around correct level
    delta_cols = np.arange(-delta_range, delta_range + 1)
    curves = []          # one row per bubble: scores indexed by delta
    correct_levels = []
    bubble_radii = []

    for b in bubbles:
        correct_lv = int(np.argmin(np.abs(eff_radii - b.radius)))
        row = []
        for d in delta_cols:
            lv = correct_lv + d
            if 0 <= lv < n_levels:
                row.append(best_lm_score_at_level(cache, lv, b.cy, b.cx, b.radius))
            else:
                row.append(np.nan)
        curves.append(row)
        correct_levels.append(correct_lv)
        bubble_radii.append(b.radius)

    curves = np.array(curves, dtype=np.float32)   # (n_bubbles, n_deltas)

    # ------------------------------------------------------------------ #
    # Print text summary
    # ------------------------------------------------------------------ #
    print(f"\n{'delta':>6}  {'mean_score':>10}  {'median':>8}  {'n_valid':>8}")
    for i, d in enumerate(delta_cols):
        col = curves[:, i]
        valid = col[~np.isnan(col)]
        print(f"  {d:+3d}  {np.nanmean(col):10.3f}  {np.nanmedian(col):8.3f}  {len(valid):8d}")

    # Where does each bubble's curve peak?
    peak_deltas = []
    for row in curves:
        valid_mask = ~np.isnan(row)
        if valid_mask.any():
            peak_deltas.append(delta_cols[np.nanargmax(row)])
    peak_deltas = np.array(peak_deltas)
    print(f"\nPeak delta per bubble:")
    print(f"  mean={peak_deltas.mean():+.2f}  median={np.median(peak_deltas):+.1f}  "
          f"p25={np.percentile(peak_deltas,25):+.1f}  p75={np.percentile(peak_deltas,75):+.1f}")
    print(f"  Histogram: ", end="")
    for d in delta_cols:
        print(f"  {d:+d}:{(peak_deltas==d).sum()}", end="")
    print()

    if np.median(peak_deltas) <= -1.5:
        print("\n  → Hypothesis SURVIVES: peak is at finer scale than expected "
              f"(median delta={np.median(peak_deltas):+.1f})")
        print(f"     Effective canonical_r ≈ {5.0 * (0.9 ** abs(np.median(peak_deltas))):.2f} px "
              f"(assumed 5.00 px)")
    else:
        print(f"\n  → Hypothesis FALSIFIED: peak is near correct level "
              f"(median delta={np.median(peak_deltas):+.1f})")

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean ± std response curve
    ax = axes[0]
    mean_curve = np.nanmean(curves, axis=0)
    std_curve  = np.nanstd(curves, axis=0)
    ax.plot(delta_cols, mean_curve, "o-", color="steelblue", linewidth=2, label="mean")
    ax.fill_between(delta_cols, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="correct level (delta=0)")
    ax.axvline(np.median(peak_deltas), color="orange", linestyle=":",
               linewidth=1.5, label=f"median peak (delta={np.median(peak_deltas):+.1f})")
    ax.set_xlabel("delta = level − correct_level  (negative = finer scale)")
    ax.set_ylabel("Max NCC score within bubble.radius")
    ax.set_title("Mean NCC response curve across scale levels\n"
                 f"({len(bubbles)} GT bubbles, image: {image_path.name})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: histogram of peak deltas, coloured by bubble size quartile
    ax = axes[1]
    radii = np.array(bubble_radii)
    q33, q66 = np.percentile(radii, 33), np.percentile(radii, 66)
    masks = [radii < q33, (radii >= q33) & (radii < q66), radii >= q66]
    labels = [f"small r<{q33:.1f}px", f"medium r={q33:.1f}–{q66:.1f}px", f"large r>{q66:.1f}px"]
    colors = ["steelblue", "seagreen", "tomato"]
    for mask, label, color in zip(masks, labels, colors):
        if mask.any():
            ax.hist(peak_deltas[mask], bins=delta_cols - 0.5,
                    alpha=0.6, color=color, label=label, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="correct level")
    ax.axvline(np.median(peak_deltas), color="orange", linestyle=":",
               linewidth=1.5, label=f"median={np.median(peak_deltas):+.1f}")
    ax.set_xlabel("delta at peak NCC score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of peak-delta by bubble size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
