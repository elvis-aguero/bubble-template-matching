#!/usr/bin/env python3
"""
E0-A — Oracle lower bound: leave-one-out GT histogram predictor.

Hypothesis to falsify:
  "The LOO GT-histogram oracle achieves relL1 ≤ 0.70 (median, n_gt≥100 images),
   establishing that cross-image histogram consistency is high enough for
   distribution-regression approaches to be viable alternatives to
   detect-then-count."

Design:
  - For each image i, compute BOTH the mean AND median of GT histograms from
    all other images (leave-one-out). Median is the L1-minimizing estimator.
  - relL1(pred, gt) = sum|pred_bin - gt_bin| / sum(gt_bin)
  - GT histogram bins = same radius grid as PipelineConfig.
  - Exclude images with n_gt < 100: relL1 is not comparable across drastically
    different population sizes (n_gt=14 → relL1 can exceed 30 for any predictor).
  - Summary statistic: MEDIAN relL1 over stable images (n_gt ≥ 100). Mean and
    std are reported but NOT used for verdicts due to outlier sensitivity.
  - Session confounding note: C1S0024 (1 image) is the pipeline's development
    test image; LOO mean may be cross-session extrapolation.

IMPORTANT: This is an oracle baseline — it requires GT labels from all other
images. A new deployment image has no GT oracle. Do not conflate LOO relL1
with what a practical estimator could achieve.

Pipeline comparison note: the pipeline best (0.950) was evaluated on a single
image (C1S0024) that is likely the primary development test image. Cross-image
pipeline evaluation is required before any comparison is meaningful.

USAGE: python scripts/experiments/baseline_e0a.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import load_image, parse_annotations

MAX_IMG_MEAN  = 0.6    # exclude saturated/dead frames
MIN_N_GT      = 100   # exclude images with too few bubbles (relL1 metric pathology)


def build_radius_bins(cfg: PipelineConfig) -> np.ndarray:
    """Build canonical radius array (one per pyramid level)."""
    radii = []
    r = cfg.min_radius
    while r <= cfg.max_radius * 1.001:
        radii.append(r)
        r /= cfg.scale_factor   # finer levels have smaller effective radii going up
    # Actually the pyramid goes from min to max by multiplying: start at min_radius
    # and divide by scale_factor each level means radii GROW. Let's rebuild correctly.
    # Level 0: original image, effective radius = canonical_radius (5px by convention)
    # We need: for each level l, eff_r = canonical_r / scale_factor^l
    # But the pipeline uses: for each scale step, image shrinks by scale_factor
    # So at level l, pixel radius = bubble.radius * scale_factor^l
    # The bins are the effective radii in original-image pixel units.
    # From config: the pyramid covers min_radius to max_radius in original-image px.
    radii = []
    r = cfg.min_radius
    while r <= cfg.max_radius * 1.001:
        radii.append(r)
        r = r / cfg.scale_factor   # next coarser level: effective radius is larger
    return np.array(radii)


def assign_to_bin(bubble_radius: float, bin_radii: np.ndarray) -> int:
    """Return index of nearest bin to bubble_radius (log-space)."""
    return int(np.argmin(np.abs(np.log(bin_radii) - np.log(bubble_radius))))


def gt_histogram(bubbles, bin_radii: np.ndarray) -> np.ndarray:
    """Count GT bubbles per radius bin."""
    counts = np.zeros(len(bin_radii), dtype=float)
    for b in bubbles:
        if b.radius < bin_radii[0] * 0.5 or b.radius > bin_radii[-1] * 2:
            continue  # out of range
        idx = assign_to_bin(b.radius, bin_radii)
        counts[idx] += 1
    return counts


def rel_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    """relL1 = sum|pred - gt| / sum(gt). Returns nan if gt is all-zero."""
    total_gt = gt.sum()
    if total_gt == 0:
        return np.nan
    return float(np.abs(pred - gt).sum() / total_gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out", type=Path, default=Path("output/e0a_trivial_baseline.png"))
    args = parser.parse_args()

    cfg = PipelineConfig()
    bin_radii = build_radius_bins(cfg)
    print(f"Radius bins: {len(bin_radii)} levels, "
          f"r={bin_radii[0]:.2f}–{bin_radii[-1]:.2f}px")

    # ── Load all tractable annotated images ───────────────────────────────────
    all_paths = sorted((Path(args.data_dir) / "images").glob("*.png"))
    samples = []
    for img_path in all_paths:
        lbl_path = Path(args.data_dir) / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        img = load_image(img_path)
        if img.mean() >= MAX_IMG_MEAN:
            continue  # skip photometrically dead frames
        bubbles = parse_annotations(lbl_path)
        hist = gt_histogram(bubbles, bin_radii)
        samples.append({
            "path": img_path,
            "img_mean": float(img.mean()),
            "n_bubbles": len(bubbles),
            "hist": hist,
        })

    n = len(samples)
    print(f"\nLoaded {n} tractable annotated images")
    if n < 2:
        print("Need at least 2 images for LOO baseline. Exiting.")
        return

    # ── Leave-one-out relL1 ───────────────────────────────────────────────────
    all_hists = np.stack([s["hist"] for s in samples])  # (n, n_bins)
    results = []

    print(f"\n{'Image':<40}  {'n_gt':>5}  {'img_mean':>9}  "
          f"{'relL1_mean':>11}  {'relL1_median':>13}  {'relL1_zeros':>12}  {'stable':>6}")
    for i, s in enumerate(samples):
        gt = s["hist"]
        loo_mask = np.ones(n, dtype=bool)
        loo_mask[i] = False
        loo_mean   = all_hists[loo_mask].mean(axis=0)
        loo_median = np.median(all_hists[loo_mask], axis=0)  # L1-minimizing estimator

        rl1_mean   = rel_l1(loo_mean,            gt)
        rl1_median = rel_l1(loo_median,          gt)
        rl1_zeros  = rel_l1(np.zeros_like(gt),   gt)
        stable     = s["n_bubbles"] >= MIN_N_GT

        results.append({
            "name": s["path"].stem[-36:],
            "img_mean": s["img_mean"],
            "n_bubbles": s["n_bubbles"],
            "gt": gt,
            "loo_mean_pred":   loo_mean,
            "loo_median_pred": loo_median,
            "relL1_mean":   rl1_mean,
            "relL1_median": rl1_median,
            "relL1_zeros":  rl1_zeros,
            "stable":       stable,
        })
        flag = "  ✓" if stable else "  (unstable)"
        print(f"  {results[-1]['name']:<38}  {s['n_bubbles']:>5}  "
              f"{s['img_mean']:>9.3f}  {rl1_mean:>11.3f}  {rl1_median:>13.3f}  "
              f"{rl1_zeros:>12.3f}{flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    stable = [r for r in results if r["stable"] and not np.isnan(r["relL1_median"])]
    unstable = [r for r in results if not r["stable"]]

    pipeline_dev_image = "ZeroG_FlightDay_Test_C1S0024_img014500"
    pipeline_best_relL1 = 0.950  # single-image evaluation on C1S0024, likely selection-biased

    print(f"\n{'='*70}")
    print(f"Stable images (n_gt ≥ {MIN_N_GT}): {len(stable)}/{len(results)}")
    print(f"Excluded (n_gt < {MIN_N_GT}): {[r['name'][-24:] for r in unstable]}")

    if stable:
        med_vals  = np.array([r["relL1_median"] for r in stable])
        mean_vals = np.array([r["relL1_mean"]   for r in stable])
        print(f"\nLOO MEDIAN histogram (L1-optimal oracle, stable images):")
        print(f"  median relL1 = {np.median(med_vals):.3f}  "
              f"mean = {med_vals.mean():.3f}  "
              f"std = {med_vals.std():.3f}  "
              f"[{med_vals.min():.3f}, {med_vals.max():.3f}]")
        print(f"\nLOO MEAN histogram (suboptimal oracle, for comparison):")
        print(f"  median relL1 = {np.median(mean_vals):.3f}  "
              f"mean = {mean_vals.mean():.3f}  "
              f"std = {mean_vals.std():.3f}")

        # Direct comparison on development test image only
        dev = next((r for r in results
                    if pipeline_dev_image in r["name"].replace("-", "_").replace(" ", "_")), None)
        if dev is None:
            # fuzzy match
            dev = next((r for r in results if "C1S0024" in r["name"]), None)

        print(f"\n{'='*70}")
        print(f"DIRECT COMPARISON on dev-test image ({pipeline_dev_image}):")
        if dev:
            print(f"  LOO median oracle:  {dev['relL1_median']:.3f}")
            print(f"  LOO mean oracle:    {dev['relL1_mean']:.3f}")
            print(f"  Pipeline best:      {pipeline_best_relL1:.3f}  "
                  f"[CAUTION: n=1 eval, likely selection-biased on this image]")
            gap = dev["relL1_median"] - pipeline_best_relL1
            print(f"  Oracle − pipeline:  {gap:+.3f}  "
                  f"({'oracle better' if gap < 0 else 'pipeline better'})")
        else:
            print("  (dev-test image not found in sample)")

        print(f"\n{'='*70}")
        print("VERDICT:")
        med = float(np.median(med_vals))
        if med <= 0.50:
            print(f"  Oracle relL1={med:.3f} ≤ 0.50: cross-image distributions are CONSISTENT.")
            print("  Distribution regression approaches are likely viable.")
        elif med <= 0.70:
            print(f"  Oracle relL1={med:.3f} in (0.50, 0.70]: MODERATE cross-image consistency.")
            print("  Distribution regression may add value but is not straightforward.")
        else:
            print(f"  Oracle relL1={med:.3f} > 0.70: cross-image distributions are HETEROGENEOUS.")
            print("  Even an oracle cannot predict a new image's histogram well from others.")
            print("  Per-image feature estimation is required — image content must be read.")

        print(f"\n  IMPORTANT CAVEAT: Pipeline evaluated on n=1 image (likely biased).")
        print(f"  Cross-image pipeline evaluation required before any architecture comparison.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: per-image relL1 bar chart (median oracle, stable images only)
    ax = axes[0]
    plot_results = [r for r in results if r["stable"]]
    names_plot = [r["name"][-20:] for r in plot_results]
    x = np.arange(len(plot_results))
    med_plot  = np.array([r["relL1_median"] for r in plot_results])
    mean_plot = np.array([r["relL1_mean"]   for r in plot_results])
    ax.bar(x - 0.2, mean_plot,  width=0.4, color="steelblue", alpha=0.7, label="LOO mean oracle")
    ax.bar(x + 0.2, med_plot,   width=0.4, color="seagreen",  alpha=0.7, label="LOO median oracle (L1-opt)")
    ax.axhline(pipeline_best_relL1, color="tomato", linestyle="--", linewidth=2,
               label=f"Pipeline (n=1 dev image, {pipeline_best_relL1:.3f})")
    ax.axhline(float(np.median(med_plot)), color="seagreen", linestyle=":", linewidth=1.5,
               label=f"Median oracle median={np.median(med_plot):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names_plot, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("relL1")
    ax.set_title(f"Per-image relL1 (stable images, n_gt≥{MIN_N_GT})\nLOO mean vs median oracle")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Middle: histogram shape comparison for best/worst stable cases
    ax = axes[1]
    if plot_results:
        worst_idx = int(np.argmax(med_plot))
        best_idx  = int(np.argmin(med_plot))
        for idx, label, color in [(worst_idx, "worst", "tomato"),
                                   (best_idx,  "best",  "steelblue")]:
            r = plot_results[idx]
            log_bins = np.log10(bin_radii)
            ax.step(log_bins, r["gt"],             where="mid", color=color,
                    linewidth=2, label=f"{label} GT (median_oracle={r['relL1_median']:.2f})")
            ax.step(log_bins, r["loo_median_pred"], where="mid", color=color,
                    linewidth=1.5, linestyle="--", alpha=0.7, label=f"{label} LOO median pred")
        tick_vals = [3, 5, 10, 20, 50]
        ax.set_xticks(np.log10(tick_vals))
        ax.set_xticklabels([str(v) for v in tick_vals])
        ax.set_xlabel("Bubble radius (px)")
        ax.set_ylabel("Count")
        ax.set_title("Best/worst case: GT vs LOO median prediction")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Right: relL1 vs img_mean scatter (all images, colour by stable/unstable)
    ax = axes[2]
    for r in results:
        color = "steelblue" if r["stable"] else "lightgray"
        ax.scatter(r["img_mean"], r["relL1_median"], s=50, alpha=0.8,
                   color=color, edgecolors="none")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="steelblue", label=f"stable (n_gt≥{MIN_N_GT})"),
                        Patch(color="lightgray",  label=f"unstable (excluded)")],
              fontsize=9)
    ax.axhline(pipeline_best_relL1, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Pipeline best ({pipeline_best_relL1:.3f})")
    ax.set_xlabel("Image mean intensity")
    ax.set_ylabel("relL1 (LOO median oracle)")
    ax.set_title("Oracle relL1 vs photometric regime")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"E0-A — LOO GT-histogram oracle  ({n} images, "
                 f"{len(plot_results)} stable)",
                 fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
