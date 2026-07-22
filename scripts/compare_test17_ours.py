#!/usr/bin/env python3
"""
Port of the collaborator's MATLAB imfindcircles script (Madeline Federle,
2025-10-31) to Python, with our NCC-pyramid pipeline standing in for
imfindcircles.

IMPORTANT CAVEATS (read before trusting any number this script prints)
-----------------------------------------------------------------------
1. NO GROUND TRUTH. There are no bubble annotations for this dataset — the
   MATLAB script was itself never validated against manual counts, only
   eyeballed. Every plot here is DESCRIPTIVE, not an accuracy comparison.
   Do not report a relL1 or any "beats Hough" claim from this script; there
   is nothing to compute it against.
2. OUT OF DOMAIN. Our pipeline (template + calibrator) is trained on
   seed_v04 — a different apparatus, camera, and bubble population than
   whatever rig produced this footage. The collaborator's own comment
   ("bubble initial = 47.88 mm" at ~8.4 px/mm => ~202px initial radius)
   puts the bubble population here far outside seed_v04's annotated range
   (radius 0.81-263.6px, but median 7.0px — this dataset's bubbles are
   larger than all but the extreme tail of what the template was built
   from). --max-radius below widens the PYRAMID so these bubbles are at
   least representable, but the TEMPLATE APPEARANCE is not re-fit for this
   apparatus. Treat every result as exploratory, not a validated estimate.
3. SOFT COUNTS, NOT CIRCLES. Our pipeline outputs an expected bubble count
   per log-spaced radius bin (a histogram), not (x, y, r) detections. There
   is no equivalent to imfindcircles' viscircles overlay. "Mean/median
   diameter" and "bubble count" per frame below are computed as
   expected-count-weighted statistics over the predicted histogram, not
   over individual detected circles.

USAGE
-----
  python scripts/compare_test17_ours.py <data_dir> [options]

<data_dir> must contain the .bmp frames directly (e.g. the frame files
named f"{experim}{i:05d}.bmp"), matching the MATLAB `path` variable.

  python scripts/compare_test17_ours.py ZeroG_FlightDay_Test17_Edits/ \\
      --experim Test17_S00010 --scale-px-per-mm 8.4252 \\
      --crop 228 188 630 630 --first-frame 650 --last-frame 1100 \\
      --fps 1250 --step 5 --max-radius 260

Outputs (in --out-dir, default output/test17/):
  global_distribution.png   histogram + lognormal fit, all frames pooled
  mean_diameter_vs_time.png exponential-decay fit (matches MATLAB figure)
  median_diameter_vs_time.png  linear trend (matches MATLAB figure)
  bubble_count_vs_time.png
  pdf_evolution_heatmap.png predicted size-PDF over time
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, load_image
from bubble_histogram.pipeline import BubblePipeline


def load_and_crop_bmp(path: Path, crop_xywh: tuple[int, int, int, int] | None) -> np.ndarray:
    """Load a grayscale BMP frame and optionally crop it, mirroring MATLAB's
    imread+imcrop. crop_xywh matches MATLAB's `rect = [xmin ymin width height]`
    convention; pass None to use the full frame (e.g. when no crop rect is
    known for this apparatus).
    """
    img = Image.open(path).convert("L")
    if crop_xywh is not None:
        x, y, w, h = crop_xywh
        img = img.crop((x, y, x + w, y + h))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def build_pipeline(cfg: PipelineConfig, seed_dir: Path) -> BubblePipeline:
    """Train fresh on all of seed_v04 (no held-out split — we're not scoring
    accuracy on seed_v04 here, just reusing its template/calibrator)."""
    dataset = AnnotatedDataset(seed_dir)  # no split kwargs => everything used for train+calibration
    pipeline = BubblePipeline(cfg)
    pipeline.train(dataset)
    return pipeline


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    return float((values * weights).sum() / total) if total > 0 else float("nan")


def weighted_std(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float(np.sqrt((weights * (values - mean) ** 2).sum() / total))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, 0.5 * total))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def exp_decay(t, d_inf, d0, tau):
    return d_inf + (d0 - d_inf) * np.exp(-t / tau)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="directory containing the .bmp frames")
    parser.add_argument("--experim", type=str, default="Test17_S00010",
                         help="filename prefix before the zero-padded frame index")
    parser.add_argument("--frame-digits", type=int, default=5,
                         help="zero-padding width of the frame index in filenames "
                              "(5 for the original Test17 script, 6 for FASTCAM Nova "
                              "Photron output like ZeroG_Test3_Opt3)")
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"),
                         help="annotated dataset used to train template+calibrator")
    parser.add_argument("--scale-px-per-mm", type=float, default=None,
                         help="if omitted, sizes are reported/plotted in PIXELS instead of mm "
                              "(no fabricated scale — supply this once you have a real "
                              "calibration for this apparatus)")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                         default=None, help="MATLAB imcrop-style [x y w h]; omit to use full frame")
    parser.add_argument("--first-frame", type=int, default=650)
    parser.add_argument("--last-frame", type=int, default=1100)
    parser.add_argument("--fps", type=float, default=1250)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--frames", type=str, default=None,
                         help="comma-separated explicit frame indices, overrides "
                              "--first-frame/--last-frame/--step (use for sparse sampling "
                              "of a huge video instead of a dense contiguous range)")
    parser.add_argument("--max-radius", type=float, default=260.0,
                         help="pyramid max radius in CROPPED-image px; widen to cover this "
                              "apparatus's bubble sizes (default covers ~47.88mm initial "
                              "diameter at the default scale, with margin)")
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/test17/"))
    args = parser.parse_args()

    cfg = PipelineConfig(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        local_maxima_calibration=True,  # avoids the train/predict covariate-shift bug in the
                                         # pixel-calibration path (see docs/experiments.md review)
    )
    print(f"Training pipeline on {args.seed_dir} "
          f"(min_radius={cfg.min_radius}, max_radius={cfg.max_radius})...")
    pipeline = build_pipeline(cfg, args.seed_dir)
    print("Training done.")

    unit = "mm" if args.scale_px_per_mm is not None else "px"
    if args.frames is not None:
        frame_indices = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        frame_indices = list(range(args.first_frame, args.last_frame + 1, args.step))
    t0 = frame_indices[0]

    time_s, mean_diam, median_diam, std_diam, bubble_count = [], [], [], [], []
    per_frame_counts = []  # list of expected_count arrays, one per frame
    diam_centers = None
    missing = []

    for i in frame_indices:
        fpath = args.data_dir / f"{args.experim}{i:0{args.frame_digits}d}.bmp"
        if not fpath.exists():
            missing.append(fpath.name)
            continue
        img = load_and_crop_bmp(fpath, tuple(args.crop) if args.crop else None)
        result = pipeline.predict(img)
        radius_px = np.array(result["radius_px"])
        counts = np.array(result["expected_count"])
        if diam_centers is None:
            diam_centers = 2 * radius_px / args.scale_px_per_mm if args.scale_px_per_mm else 2 * radius_px

        t = (i - t0) / args.fps
        time_s.append(t)
        total = counts.sum()
        bubble_count.append(total)
        if total > 0:
            m = weighted_mean(diam_centers, counts)
            mean_diam.append(m)
            median_diam.append(weighted_median(diam_centers, counts))
            std_diam.append(weighted_std(diam_centers, counts, m))
        else:
            mean_diam.append(np.nan)
            median_diam.append(np.nan)
            std_diam.append(np.nan)
        per_frame_counts.append(counts)
        print(f"  frame {i:>6}  t={t:7.3f}s  expected_count_total={total:8.2f}  "
              f"mean_diam={mean_diam[-1]:.3f}{unit}")

    if missing:
        print(f"\nWARNING: {len(missing)} frames not found, skipped "
              f"(e.g. {missing[:3]})")

    if not per_frame_counts:
        print("No frames loaded — check --data-dir / --experim / frame range. Exiting.")
        return

    time_s = np.array(time_s)
    mean_diam = np.array(mean_diam)
    median_diam = np.array(median_diam)
    bubble_count = np.array(bubble_count)
    all_counts = np.stack(per_frame_counts)  # (n_frames, n_bins)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # GLOBAL DISTRIBUTION (all frames pooled) — analogue of MATLAB's histogram
    # + fitdist(A1,'Lognormal'), but weighted (soft counts) instead of raw
    # per-bubble samples.
    # ========================================================================
    total_counts_per_bin = all_counts.sum(axis=0)
    w = total_counts_per_bin
    if w.sum() > 0:
        log_d = np.log(diam_centers)
        mu = weighted_mean(log_d, w)
        sigma = weighted_std(log_d, w, mu)

        d_fit = np.linspace(diam_centers.min() * 0.1, diam_centers.max() * 1.5, 1000)
        pdf_fit = (1.0 / (d_fit * sigma * np.sqrt(2 * np.pi))
                   * np.exp(-(np.log(d_fit) - mu) ** 2 / (2 * sigma ** 2)))

        threshold = 0.01 * pdf_fit.max()
        idx_candidates = np.where(pdf_fit > threshold)[0]
        estimated_min_diameter = float(d_fit[idx_candidates[0]]) if len(idx_candidates) else float("nan")
        print(f"\nWeighted-lognormal fit: mu={mu:.4f} sigma={sigma:.4f}")
        print(f"Estimated smallest bubble diameter (from fit) = {estimated_min_diameter:.4f} {unit}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(diam_centers, total_counts_per_bin,
               width=np.diff(np.log(diam_centers)).mean() * diam_centers * 0.8
               if len(diam_centers) > 1 else 0.1,
               color="steelblue", alpha=0.7, label="Predicted (expected count, all frames pooled)")
        ax2 = ax.twinx()
        ax2.plot(d_fit, pdf_fit, "r", linewidth=2, label="Weighted lognormal fit")
        pdf_min = np.interp(estimated_min_diameter, d_fit, pdf_fit)
        ax2.plot(estimated_min_diameter, pdf_min, "ko", markerfacecolor="k", markersize=8,
                 label=f"Estimated min d = {estimated_min_diameter:.3f} {unit}")
        ax2.set_ylabel("PDF (fit)")
        ax.set_xlabel(f"Bubble diameter ({unit})")
        ax.set_ylabel("Expected count (summed over frames)")
        ax.set_xlim(0, diam_centers.max())
        ax.set_title("Overall predicted bubble size distribution — our algorithm\n"
                      "(NO ground truth available — descriptive only)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "global_distribution.png", dpi=150)
        plt.close(fig)

    # ========================================================================
    # MEAN DIAMETER vs TIME + exponential decay fit
    # ========================================================================
    valid = ~np.isnan(mean_diam)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_s, mean_diam, linewidth=2, label="Data")
    if valid.sum() >= 3:
        t_fit, d_fit_vals = time_s[valid], mean_diam[valid]
        try:
            p0 = [d_fit_vals.min(), d_fit_vals[0], (t_fit.max() - t_fit.min()) / 4 or 1.0]
            popt, _ = curve_fit(exp_decay, t_fit, d_fit_vals, p0=p0, maxfev=10000)
            d_inf, d0, tau = popt
            t_smooth = np.linspace(t_fit.min(), t_fit.max(), 300)
            ax.plot(t_smooth, exp_decay(t_smooth, *popt), "k", linewidth=2,
                    label="Exponential best fit")
            print("\n---- Exponential fit parameters (our algorithm) ----")
            print(f"  D0   = {d0:.4f} {unit}")
            print(f"  Dinf = {d_inf:.4f} {unit}")
            print(f"  tau  = {tau:.4f} s")
        except RuntimeError as e:
            print(f"\nExponential fit did not converge: {e}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Mean diameter ({unit})")
    ax.set_title("Mean bubble diameter vs time — our algorithm\n(no ground truth available)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "mean_diameter_vs_time.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # MEDIAN DIAMETER vs TIME + linear trend
    # ========================================================================
    valid = ~np.isnan(median_diam)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_s, median_diam, linewidth=2, label="Median diameter")
    if valid.sum() >= 2:
        p = np.polyfit(time_s[valid], median_diam[valid], 1)
        trend = np.polyval(p, time_s)
        ax.plot(time_s, trend, "k--", linewidth=2, label="Linear trend")
        ax.text(0.05 * time_s.max(), 0.9 * np.nanmax(median_diam),
                f"Slope = {p[0]:.4e} {unit}/s", fontsize=12)
        print(f"\nMedian diameter linear trend slope = {p[0]:.4e} {unit}/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Median diameter ({unit})")
    ax.set_title("Median bubble diameter vs time — our algorithm\n(no ground truth available)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "median_diameter_vs_time.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # BUBBLE COUNT (expected, summed over bins) vs TIME
    # ========================================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_s, bubble_count, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Expected bubble count (summed over size bins)")
    ax.set_title("Bubble count vs time — our algorithm\n(soft/expected count, not discrete detections)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "bubble_count_vs_time.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # PDF EVOLUTION HEATMAP
    # Note: bins are our pipeline's native log-spaced radius grid converted to
    # mm diameter — NOT the linear 0-4mm/40-bin grid MATLAB used. Axis is
    # log-scaled to keep bin visual width meaningful.
    # ========================================================================
    bin_sums = all_counts.sum(axis=1, keepdims=True)
    density = np.divide(all_counts, bin_sums, out=np.zeros_like(all_counts), where=bin_sums > 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(time_s, diam_centers, density.T, shading="nearest", cmap="hot")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Bubble diameter ({unit}, log scale)")
    ax.set_title("Predicted bubble size distribution over time — our algorithm\n"
                 "(per-frame normalized; no ground truth available)")
    fig.colorbar(im, ax=ax, label="P(diameter bin | frame)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "pdf_evolution_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nAll plots written to {args.out_dir}/")
    print("Reminder: no ground truth exists for this dataset. These are descriptive "
          "outputs from an out-of-domain application of the seed_v04-trained pipeline, "
          "not a validated accuracy comparison against imfindcircles.")


if __name__ == "__main__":
    main()
