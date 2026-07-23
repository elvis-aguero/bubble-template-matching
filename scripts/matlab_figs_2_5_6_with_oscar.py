#!/usr/bin/env python3
"""
Faithful reproduction of MATLAB Figures 2, 5, 6 (Madeline Federle's script)
on the ZeroG_Test3_Opt3 video (91 frames, frames 650-1100 step 5, fps=1250),
now with Oscar's classical detector and FRST+SAM3 hybrid overlaid alongside
Hough and our own pipeline.

No ground truth exists for this dataset -- descriptive only, and Hough is
used as the reference series in each panel since it is the collaborator's
own baseline (not because it is more correct).

Requires pre-computed Oscar detections for the same 91 frames:
  --oscar-json-dir   detected_<stem>.json  (classical, [[cx,cy,r], ...])
  --hybrid-json-dir  <stem>_analysis.json  (FRST+SAM3 hybrid, instances[].radius_equiv_px)

USAGE
-----
  python scripts/matlab_figs_2_5_6_with_oscar.py <local_frames_dir> \\
      --oscar-json-dir /tmp/oscar_pull/detect_bubbles_video \\
      --hybrid-json-dir /tmp/oscar_pull/hybrid_out_video
"""
import argparse
import json
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
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

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


def load_oscar_json(json_dir: Path, stem: str) -> np.ndarray:
    path = json_dir / f"detected_{stem}.json"
    if not path.exists():
        return np.array([])
    data = json.loads(path.read_text())
    return np.array([2 * r for (_, _, r) in data])


def load_hybrid_json(json_dir: Path, stem: str) -> np.ndarray:
    path = json_dir / f"{stem}_analysis.json"
    if not path.exists():
        return np.array([])
    data = json.loads(path.read_text())
    return np.array([2 * inst["radius_equiv_px"] for inst in data["instances"]])


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
    parser.add_argument("--oscar-json-dir", type=Path, required=True)
    parser.add_argument("--hybrid-json-dir", type=Path, required=True)
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
    hough_all_diam = []
    hough_count_per_frame = []
    hough_diam_per_frame = []
    oscar_diam_per_frame = []
    hybrid_diam_per_frame = []
    ours_bin_diam_px = None
    ours_counts_per_frame = []

    for i in frame_indices:
        stem = f"{args.experim}{i:0{args.frame_digits}d}"
        fpath = args.data_dir / f"{stem}.bmp"
        if not fpath.exists():
            continue
        img = np.array(Image.open(fpath).convert("L")).astype(np.float32) / 255.0
        t = (i - args.first_frame) / args.fps
        time_s.append(t)

        result = pipeline.predict(img)
        radius_px = np.array(result["radius_px"])
        counts = np.array(result["expected_count"])
        if ours_bin_diam_px is None:
            ours_bin_diam_px = 2 * radius_px
        ours_counts_per_frame.append(counts)

        hough_diam = 2 * run_hough(img)
        hough_all_diam.append(hough_diam)
        hough_count_per_frame.append(len(hough_diam))
        hough_diam_per_frame.append(hough_diam)

        oscar_diam = load_oscar_json(args.oscar_json_dir, stem)
        hybrid_diam = load_hybrid_json(args.hybrid_json_dir, stem)
        oscar_diam_per_frame.append(oscar_diam)
        hybrid_diam_per_frame.append(hybrid_diam)

        print(f"  frame {i:>6}  t={t:6.3f}s  ours={counts.sum():7.2f}  "
              f"hough={len(hough_diam):5d}  oscar={len(oscar_diam):5d}  hybrid={len(hybrid_diam):5d}")

    time_s = np.array(time_s)
    hough_all_diam_flat = np.concatenate(hough_all_diam) if hough_all_diam else np.array([])
    oscar_all_diam_flat = np.concatenate(oscar_diam_per_frame) if oscar_diam_per_frame else np.array([])
    hybrid_all_diam_flat = np.concatenate(hybrid_diam_per_frame) if hybrid_diam_per_frame else np.array([])
    ours_counts_arr = np.stack(ours_counts_per_frame)
    hough_count_per_frame = np.array(hough_count_per_frame)
    oscar_count_per_frame = np.array([len(d) for d in oscar_diam_per_frame])
    hybrid_count_per_frame = np.array([len(d) for d in hybrid_diam_per_frame])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hough_label = f"Hough transform (collaborator's baseline, {len(hough_all_diam_flat)} pooled)"

    # ========================================================================
    # FIG 2 — Overall Bubble Size Distribution, 4-panel: Hough alone, then
    # Hough vs. each other method
    # ========================================================================
    bins = np.linspace(0, args.heatmap_max_diam, 50)
    ours_total_per_bin = ours_counts_arr.sum(axis=0)
    panels2 = [
        (oscar_all_diam_flat, "mediumseagreen",
         f"Classical detector (adaptive threshold + contour, Oscar HPC, {len(oscar_all_diam_flat)})"),
        (hybrid_all_diam_flat, "darkorchid",
         f"FRST + SAM3 hybrid (state-of-the-art, Oscar HPC, {len(hybrid_all_diam_flat)})"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].hist(hough_all_diam_flat, bins=bins, color="tomato", alpha=0.6, label=hough_label)
    axes[0].set_ylabel("Count", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    for ax, (data, color, label) in zip(axes[1:3], panels2):
        ax.hist(hough_all_diam_flat, bins=bins, color="tomato", alpha=0.4, label=hough_label)
        ax.hist(data, bins=bins, color=color, alpha=0.5, label=label)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4.hist(hough_all_diam_flat, bins=bins, color="tomato", alpha=0.4, label=hough_label)
    ax4b = ax4.twinx()
    ax4b.bar(ours_bin_diam_px, ours_total_per_bin,
             width=np.diff(ours_bin_diam_px).mean() if len(ours_bin_diam_px) > 1 else 1.0,
             color="steelblue", alpha=0.5, label="Template matching (this work, LOSO-trained)")
    ax4.set_ylabel("Count (Hough)", fontsize=9)
    ax4b.set_ylabel("Expected count\n(template matching)", fontsize=8)
    l1, lb1 = ax4.get_legend_handles_labels()
    l2, lb2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Bubble Diameter (px)")
    ax4.set_xlim(0, args.heatmap_max_diam)

    fig.suptitle("Overall Bubble Size Distribution — ZeroG_Test3_Opt3 video, 91 frames (no GT)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "video_fig2_overall_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # ========================================================================
    # FIG 5 — Bubble Count vs Time, 4-panel
    # ========================================================================
    panels5 = [
        (oscar_count_per_frame, "mediumseagreen",
         "Classical detector (adaptive threshold + contour, Oscar HPC)"),
        (hybrid_count_per_frame, "darkorchid",
         "FRST + SAM3 hybrid (state-of-the-art, Oscar HPC)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_s, hough_count_per_frame, "o-", color="tomato", label=hough_label, markersize=3)
    axes[0].set_ylabel("Count", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    for ax, (counts, color, label) in zip(axes[1:3], panels5):
        ax.plot(time_s, hough_count_per_frame, "o-", color="tomato", alpha=0.6,
                label=hough_label, markersize=3)
        ax.plot(time_s, counts, "o-", color=color, label=label, markersize=3)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4.plot(time_s, hough_count_per_frame, "o-", color="tomato", alpha=0.6,
             label=hough_label, markersize=3)
    ax4b = ax4.twinx()
    ax4b.plot(time_s, ours_counts_arr.sum(axis=1), "o-", color="steelblue",
              label="Template matching (this work, LOSO-trained)", markersize=3)
    ax4.set_ylabel("Count (Hough)", fontsize=9)
    ax4b.set_ylabel("Expected count\n(template matching)", fontsize=8)
    l1, lb1 = ax4.get_legend_handles_labels()
    l2, lb2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Time (s)")

    fig.suptitle("Bubble Count vs Time — ZeroG_Test3_Opt3 video, 91 frames (no GT)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "video_fig5_bubble_count_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)

    # ========================================================================
    # FIG 6 — Bubble Size Distribution Over Time (heatmap), linear bins, 4 panels
    # ========================================================================
    bins = np.linspace(0, args.heatmap_max_diam, args.heatmap_bins)
    bin_left_edges = bins[:-1]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    def density_matrix(diam_lists):
        mat = np.zeros((len(bins) - 1, len(diam_lists)))
        for k, d in enumerate(diam_lists):
            if len(d) > 0:
                mat[:, k] = np.histogram(d, bins=bins, density=True)[0]
        return mat

    hough_pdf_time = density_matrix(hough_diam_per_frame)
    oscar_pdf_time = density_matrix(oscar_diam_per_frame)
    hybrid_pdf_time = density_matrix(hybrid_diam_per_frame)

    ours_pdf_time = np.zeros((len(bins) - 1, len(time_s)))
    for k in range(len(time_s)):
        counts_k = ours_counts_per_frame[k]
        total_k = counts_k.sum()
        if total_k > 0:
            interp_counts = np.interp(bin_centers, ours_bin_diam_px, counts_k, left=0, right=0)
            ours_pdf_time[:, k] = interp_counts / (total_k * bin_width)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    extent = [time_s.min(), time_s.max(), bin_left_edges.min(), bin_left_edges.max()]
    for ax, mat, title in zip(
        axes, [hough_pdf_time, oscar_pdf_time, hybrid_pdf_time, ours_pdf_time],
        ["Hough transform\n(collaborator's baseline)",
         "Classical detector\n(adaptive threshold, Oscar HPC)",
         "FRST + SAM3 hybrid\n(state-of-the-art, Oscar HPC)",
         "Template matching\n(this work, LOSO)"],
    ):
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="hot", extent=extent)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)")
        fig.colorbar(im, ax=ax, label="PDF", fraction=0.046)
    axes[0].set_ylabel("Bubble Diameter (px)")
    fig.suptitle("Bubble Size Distribution Over Time — ZeroG_Test3_Opt3 video, 91 frames (no GT)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "video_fig6_pdf_evolution_heatmap.pdf")
    plt.close(fig)

    print(f"\nSaved video_fig2/5/6 to {args.out_dir}/")


if __name__ == "__main__":
    main()
