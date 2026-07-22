#!/usr/bin/env python3
"""
Fig 2/5/6-style comparison with REAL ground truth, on session C1S0014 from
seed_v04 (4 annotated images: frames 6001, 9542, 18008, 18351) -- the only
place in this project with actual bubble annotations. Not a video: 4 sparse
real frames from the same physical recording, x-axis is FRAME INDEX (no fps
is recorded for this apparatus, so no time-in-seconds axis is fabricated).

Our pipeline is trained LOSO (session C1S0014 held out), matching the
validated E0-B protocol -- "ours" is never evaluated on data it trained on.
Hough uses our pipeline's own native radius range (3-50px), not the Test17
script's [1,30]px (a different apparatus/camera).

USAGE
-----
  python scripts/seed_v04_gt_comparison.py [--seed-dir seed_v04/] [--session C1S0014]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, get_session_id
from bubble_histogram.pipeline import BubblePipeline

plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 15
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 15

CANNY_HIGH = 50
DP = 1
MIN_DIST = 8
PARAM2 = 20


def img_to_uint8(img: np.ndarray) -> np.ndarray:
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def run_hough(img: np.ndarray, min_r: int, max_r: int) -> np.ndarray:
    img_u8 = img_to_uint8(img)
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_DIST,
        param1=CANNY_HIGH, param2=PARAM2, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return np.array([])
    return circles[0][:, 2]


def frame_index_of(path: Path) -> int:
    import re
    m = re.search(r"(\d{6,})\D*$", path.stem)
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"))
    parser.add_argument("--session", type=str, default="C1S0014")
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--max-radius", type=float, default=50.0)
    parser.add_argument("--heatmap-bins", type=int, default=25)
    parser.add_argument("--heatmap-max-diam", type=float, default=60.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/test17_zerog_opt3/"))
    args = parser.parse_args()

    cfg = PipelineConfig(min_radius=args.min_radius, max_radius=args.max_radius,
                         local_maxima_calibration=True)
    print(f"Training pipeline LOSO, holding out session {args.session}...")
    dataset = AnnotatedDataset(args.seed_dir, val_session=args.session)
    pipeline = BubblePipeline(cfg)
    pipeline.train(dataset)
    print(f"Training done. Test images (session {args.session}): "
          f"{[p.name for p in dataset.test_images]}")

    test_images = sorted(dataset.test_images, key=frame_index_of)
    frame_idx = []
    gt_diam_per_frame, hough_diam_per_frame, ours_counts_per_frame = [], [], []
    gt_count, hough_count, ours_count = [], [], []
    ours_bin_diam_px = None

    for img_path in test_images:
        sample = dataset.load_sample(img_path)
        fi = frame_index_of(img_path)
        frame_idx.append(fi)

        gt_diam = np.array([2 * b.radius for b in sample.bubbles])
        gt_diam_per_frame.append(gt_diam)
        gt_count.append(len(gt_diam))

        result = pipeline.predict(sample.image)
        radius_px = np.array(result["radius_px"])
        counts = np.array(result["expected_count"])
        if ours_bin_diam_px is None:
            ours_bin_diam_px = 2 * radius_px
        ours_counts_per_frame.append(counts)
        ours_count.append(counts.sum())

        hough_radii = run_hough(sample.image, min_r=int(args.min_radius), max_r=int(args.max_radius))
        hough_diam = 2 * hough_radii
        hough_diam_per_frame.append(hough_diam)
        hough_count.append(len(hough_diam))

        print(f"  frame {fi:>7}  n_gt={len(gt_diam):4d}  "
              f"ours_expected={counts.sum():7.2f}  hough_detected={len(hough_diam):4d}")

    frame_idx = np.array(frame_idx)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # FIG 2 — Overall distribution, GT vs Hough vs Ours, pooled across the 4 frames
    # ========================================================================
    gt_all = np.concatenate(gt_diam_per_frame)
    hough_all = np.concatenate(hough_diam_per_frame)
    ours_total_per_bin = np.stack(ours_counts_per_frame).sum(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(0, args.heatmap_max_diam, 30)
    ax.hist(gt_all, bins=bins, color="black", histtype="step", linewidth=2.5,
            label=f"Ground truth ({len(gt_all)} bubbles, {len(test_images)} frames)")
    ax.hist(hough_all, bins=bins, color="tomato", alpha=0.5,
            label=f"Hough ({len(hough_all)} detections)")
    ax2 = ax.twinx()
    ax2.bar(ours_bin_diam_px, ours_total_per_bin,
            width=np.diff(ours_bin_diam_px).mean() if len(ours_bin_diam_px) > 1 else 1.0,
            color="steelblue", alpha=0.5, label="Ours (expected count)")
    ax.set_xlabel("Bubble Diameter (px)")
    ax.set_ylabel("Count (GT, Hough)")
    ax2.set_ylabel("Expected count (ours)")
    ax.set_xlim(0, args.heatmap_max_diam)
    ax.set_title(f"Overall Bubble Size Distribution — session {args.session} (real GT)")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_fig2_overall_distribution.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIG 5 — Count vs frame index, GT vs Hough vs Ours
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(frame_idx, gt_count, "o-", color="black", label="Ground truth")
    ax.plot(frame_idx, hough_count, "o-", color="tomato", label="Hough")
    ax.plot(frame_idx, ours_count, "o-", color="steelblue", label="Ours (expected)")
    ax.set_xlabel(f"Frame index (session {args.session}, no fps recorded for this apparatus)")
    ax.set_ylabel("Number of Bubbles")
    ax.set_title("Bubble Count vs Frame Index (real GT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_fig5_count_vs_frame.png", dpi=150)
    plt.close(fig)
    print("\nCount comparison:")
    for fi, g, h, o in zip(frame_idx, gt_count, hough_count, ours_count):
        print(f"  frame {fi:>7}  GT={g:4d}  Hough={h:4d}  Ours={o:7.2f}")

    # ========================================================================
    # FIG 6 — heatmap-style panels (only 4 columns -- honestly sparse)
    # ========================================================================
    bins = np.linspace(0, args.heatmap_max_diam, args.heatmap_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    def density_matrix(diam_lists):
        mat = np.zeros((len(bins) - 1, len(diam_lists)))
        for k, d in enumerate(diam_lists):
            if len(d) > 0:
                mat[:, k] = np.histogram(d, bins=bins, density=True)[0]
        return mat

    gt_mat = density_matrix(gt_diam_per_frame)
    hough_mat = density_matrix(hough_diam_per_frame)
    ours_mat = np.zeros((len(bins) - 1, len(frame_idx)))
    for k, counts_k in enumerate(ours_counts_per_frame):
        total_k = counts_k.sum()
        if total_k > 0:
            interp_counts = np.interp(bin_centers, ours_bin_diam_px, counts_k, left=0, right=0)
            ours_mat[:, k] = interp_counts / (total_k * bin_width)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, mat, title in zip(axes, [gt_mat, hough_mat, ours_mat], ["Ground Truth", "Hough", "Ours"]):
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="hot",
                        extent=[0, len(frame_idx) - 1, 0, args.heatmap_max_diam])
        ax.set_xticks(range(len(frame_idx)))
        ax.set_xticklabels(frame_idx, rotation=45, fontsize=10)
        ax.set_title(title)
        ax.set_xlabel("Frame index")
        fig.colorbar(im, ax=ax, label="PDF", fraction=0.046)
    axes[0].set_ylabel("Bubble Diameter (px)")
    fig.suptitle(f"Bubble Size Distribution — session {args.session}, {len(frame_idx)} real annotated frames "
                 "(sparse, not a continuous video)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_fig6_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved gt_fig2, gt_fig5, gt_fig6 to {args.out_dir}/")


if __name__ == "__main__":
    main()
