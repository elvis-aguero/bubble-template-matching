#!/usr/bin/env python3
"""
Faithful reproduction of MATLAB Figures 2, 5, 6 (Madeline Federle's script),
Hough vs. ours, on the same frames.

Fidelity notes (read before trusting the plots):
- Fig 2 pools ONE diameter sample per Hough-detected circle across all frames
  (matches MATLAB's A1 = pooled mmdiam{}), using imfindcircles' literal
  radius bounds [1,30]px from the original script -- NOT the wider range used
  for "ours" elsewhere in this repo. Our algorithm has no per-object samples;
  its series is the expected-count-per-bin summed across frames instead.
  The MATLAB script's fit curve is never actually drawn (commented out) and
  its y-axis is mislabeled 'PDF' when the histogram is count-normalized --
  reproduced as a plain count axis here, not as a (non-existent) density.
- Fig 5 compares a genuine discrete count (Hough) against a soft/expected
  count (ours) -- axis labeled to make that distinction explicit.
- Fig 6 uses linear (not log) diameter bins and proper per-frame density
  normalization for both series, matching MATLAB's histcounts(...,'pdf').

No ground truth exists for this dataset -- descriptive only.

USAGE
-----
  python scripts/matlab_figs_2_5_6.py <local_frames_dir> [options]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.pipeline import BubblePipeline

plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 16

CANNY_HIGH = 50
DP = 1
MIN_DIST = 8
PARAM2 = 20
HOUGH_MIN_R = 1     # literal imfindcircles(photo,[1 30],...) from the original script
HOUGH_MAX_R = 30


def img_to_uint8(img: np.ndarray) -> np.ndarray:
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def run_hough(img: np.ndarray) -> np.ndarray:
    img_u8 = img_to_uint8(img)
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_DIST,
        param1=CANNY_HIGH, param2=PARAM2, minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    if circles is None:
        return np.array([])
    return circles[0][:, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--experim", type=str, default="ZeroG_Test3_Opt3")
    parser.add_argument("--frame-digits", type=int, default=6)
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"))
    parser.add_argument("--first-frame", type=int, default=650)
    parser.add_argument("--last-frame", type=int, default=1100)
    parser.add_argument("--fps", type=float, default=1250)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--ours-min-radius", type=float, default=3.0)
    parser.add_argument("--ours-max-radius", type=float, default=70.0)
    parser.add_argument("--heatmap-bins", type=int, default=40)
    parser.add_argument("--heatmap-max-diam", type=float, default=150.0,
                         help="px; matches MATLAB's fixed linspace(0,4,40)-style range, "
                              "scaled to this dataset since no mm calibration exists")
    parser.add_argument("--out-dir", type=Path, default=Path("output/test17_zerog_opt3/"))
    args = parser.parse_args()

    cfg = PipelineConfig(min_radius=args.ours_min_radius, max_radius=args.ours_max_radius,
                         local_maxima_calibration=True)
    print(f"Training our pipeline on {args.seed_dir}...")
    dataset = AnnotatedDataset(args.seed_dir)
    pipeline = BubblePipeline(cfg)
    pipeline.train(dataset)
    print("Training done.")

    frame_indices = list(range(args.first_frame, args.last_frame + 1, args.step))
    time_s = []
    hough_all_diam = []          # pooled, one sample per detected circle (Fig 2)
    hough_count_per_frame = []   # Fig 5
    hough_diam_per_frame = []    # list of arrays, for Fig 6
    ours_bin_diam_px = None      # our pyramid's diameter bin centers (fixed across frames)
    ours_counts_per_frame = []   # list of per-bin expected-count arrays, for Figs 2/5/6

    for i in frame_indices:
        fpath = args.data_dir / f"{args.experim}{i:0{args.frame_digits}d}.bmp"
        if not fpath.exists():
            continue
        img = np.array(Image.open(fpath).convert("L")).astype(np.float32) / 255.0
        t = (i - args.first_frame) / args.fps
        time_s.append(t)

        # --- ours ---
        result = pipeline.predict(img)
        radius_px = np.array(result["radius_px"])
        counts = np.array(result["expected_count"])
        if ours_bin_diam_px is None:
            ours_bin_diam_px = 2 * radius_px
        ours_counts_per_frame.append(counts)

        # --- Hough ---
        hough_radii = run_hough(img)
        hough_diam = 2 * hough_radii
        hough_all_diam.append(hough_diam)
        hough_count_per_frame.append(len(hough_diam))
        hough_diam_per_frame.append(hough_diam)

        print(f"  frame {i:>6}  t={t:6.3f}s  ours_expected_total={counts.sum():7.2f}  "
              f"hough_detected_count={len(hough_diam):5d}")

    time_s = np.array(time_s)
    hough_all_diam_flat = np.concatenate(hough_all_diam) if hough_all_diam else np.array([])
    ours_counts_arr = np.stack(ours_counts_per_frame)          # (n_frames, n_bins)
    hough_count_per_frame = np.array(hough_count_per_frame)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # FIG 2 — Overall Bubble Size Distribution
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if len(hough_all_diam_flat) > 0:
        ax.hist(hough_all_diam_flat, bins=50, color="tomato", alpha=0.6,
                label=f"Hough - Data ({len(hough_all_diam_flat)} detections pooled)")
    ours_total_per_bin = ours_counts_arr.sum(axis=0)
    ax2 = ax.twinx()
    ax2.bar(ours_bin_diam_px, ours_total_per_bin,
            width=np.diff(ours_bin_diam_px).mean() if len(ours_bin_diam_px) > 1 else 1.0,
            color="steelblue", alpha=0.6,
            label="Ours - Data (expected count, summed over frames)")
    ax.set_xlabel("Bubble Diameter (px)")
    ax.set_ylabel("Count (Hough)")
    ax2.set_ylabel("Expected count (ours)")
    ax.set_title("Overall Bubble Size Distribution")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig2_overall_distribution.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIG 5 — Bubble Count vs Time
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(time_s, hough_count_per_frame, color="tomato", label="Hough - Detected Count")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of Bubbles")
    ax.set_title("Bubble Count vs Time")
    ax2 = ax.twinx()
    ax2.plot(time_s, ours_counts_arr.sum(axis=1), color="steelblue", label="Ours - Expected Count")
    ax2.set_ylabel("Expected count (ours)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig5_bubble_count_vs_time.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIG 6 — Bubble Size Distribution Over Time (heatmap), linear bins
    # ========================================================================
    bins = np.linspace(0, args.heatmap_max_diam, args.heatmap_bins)
    bin_left_edges = bins[:-1]

    # Hough: real per-frame density histogram (matches histcounts(...,'pdf'))
    hough_pdf_time = np.zeros((len(bins) - 1, len(time_s)))
    for k, d in enumerate(hough_diam_per_frame):
        if len(d) > 0:
            hough_pdf_time[:, k] = np.histogram(d, bins=bins, density=True)[0]

    # Ours: resample our native (log-spaced pyramid) bins onto the same linear
    # grid via linear interpolation of counts, then density-normalize the same way.
    ours_pdf_time = np.zeros((len(bins) - 1, len(time_s)))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    for k in range(len(time_s)):
        counts_k = ours_counts_per_frame[k]
        total_k = counts_k.sum()
        if total_k > 0:
            interp_counts = np.interp(bin_centers, ours_bin_diam_px, counts_k, left=0, right=0)
            ours_pdf_time[:, k] = interp_counts / (total_k * bin_width)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    im0 = axes[0].imshow(hough_pdf_time, aspect="auto", origin="lower", cmap="hot",
                          extent=[time_s.min(), time_s.max(), bin_left_edges.min(), bin_left_edges.max()])
    axes[0].set_title("Hough")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Bubble Diameter (px)")
    fig.colorbar(im0, ax=axes[0], label="PDF")

    im1 = axes[1].imshow(ours_pdf_time, aspect="auto", origin="lower", cmap="hot",
                          extent=[time_s.min(), time_s.max(), bin_left_edges.min(), bin_left_edges.max()])
    axes[1].set_title("Ours")
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(im1, ax=axes[1], label="PDF")

    fig.suptitle("Bubble Size Distribution Over Time")
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig6_pdf_evolution_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved fig2, fig5, fig6 to {args.out_dir}/")


if __name__ == "__main__":
    main()
