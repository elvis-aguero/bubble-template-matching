#!/usr/bin/env python3
"""
Fig 2/5/6-style comparison with REAL ground truth, on session C1S0014 from
seed_v04, now including a fourth series: a classical (non-ML) detector from
the collaborator's Oscar "Bubble-tracking" repo --
`bubbly_flows/tests/src/deterministic/detect_bubbles.py` (adaptive threshold +
contour, area/circularity/solidity/intensity filtered, radius = sqrt(area/pi)).

Freshly rerun on Oscar (2026-07-14) against the exact same 4 annotated frames,
NOT the stale sample output that shipped in the repo's own output/ dir --
results were pulled down right after running, per instruction to not trust
previously-stated numbers.

A companion classical detector, FRST (`classical_test.py`), only outputs
detection centers with no per-bubble radius -- it can contribute a count-vs-frame
comparison (Fig 5) but not a size distribution (Fig 2/6).

Our pipeline is trained LOSO (session C1S0014 held out) as before. Hough uses
our pipeline's native radius range (3-50px).

USAGE
-----
  python scripts/seed_v04_gt_comparison_with_oscar.py \\
      --oscar-json-dir /tmp/oscar_pull/detect_bubbles_out \\
      --frst-counts 297,393,269,307
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.pipeline import BubblePipeline

plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 14

CANNY_HIGH = 50
DP = 1
MIN_DIST = 8
PARAM2 = 20

FRAME_STEMS = ["img006001", "img009542", "img018008", "img018351"]
FRAME_IDX = [6001, 9542, 18008, 18351]


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


def load_oscar_detections(json_dir: Path, stem: str) -> np.ndarray:
    path = json_dir / f"detected_ZeroG_FlightDay_Test_C1S0014_{stem}.json"
    data = json.loads(path.read_text())
    return np.array([2 * r for (_, _, r) in data])  # diameters


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"))
    parser.add_argument("--session", type=str, default="C1S0014")
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--max-radius", type=float, default=50.0)
    parser.add_argument("--oscar-json-dir", type=Path, required=True)
    parser.add_argument("--frst-counts", type=str, default=None,
                         help="comma-separated FRST detected-center counts, "
                              "same frame order as FRAME_STEMS (count-only, no size)")
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

    test_images = {p.stem.split("_")[-1]: p for p in dataset.test_images}
    frame_idx, gt_diam_per_frame, hough_diam_per_frame = [], [], []
    ours_counts_per_frame, oscar_diam_per_frame = [], []
    gt_count, hough_count, ours_count, oscar_count = [], [], [], []
    ours_bin_diam_px = None

    for stem, fi in zip(FRAME_STEMS, FRAME_IDX):
        img_path = test_images[stem]
        sample = dataset.load_sample(img_path)
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

        hough_diam = 2 * run_hough(sample.image, min_r=int(args.min_radius), max_r=int(args.max_radius))
        hough_diam_per_frame.append(hough_diam)
        hough_count.append(len(hough_diam))

        oscar_diam = load_oscar_detections(args.oscar_json_dir, stem)
        oscar_diam_per_frame.append(oscar_diam)
        oscar_count.append(len(oscar_diam))

        print(f"  frame {fi:>7}  GT={len(gt_diam):4d}  Ours={counts.sum():7.2f}  "
              f"Hough={len(hough_diam):4d}  Oscar-deterministic={len(oscar_diam):4d}")

    frst_counts = None
    if args.frst_counts:
        frst_counts = [int(x) for x in args.frst_counts.split(",")]

    frame_idx = np.array(frame_idx)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # FIG 2 — Overall distribution, pooled across the 4 frames
    # ========================================================================
    gt_all = np.concatenate(gt_diam_per_frame)
    hough_all = np.concatenate(hough_diam_per_frame)
    oscar_all = np.concatenate(oscar_diam_per_frame)
    ours_total_per_bin = np.stack(ours_counts_per_frame).sum(axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, args.heatmap_max_diam, 30)
    ax.hist(gt_all, bins=bins, color="black", histtype="step", linewidth=2.5,
            label=f"Ground truth ({len(gt_all)} bubbles)")
    ax.hist(hough_all, bins=bins, color="tomato", alpha=0.45,
            label=f"Hough ({len(hough_all)} detections)")
    ax.hist(oscar_all, bins=bins, color="mediumseagreen", alpha=0.45,
            label=f"Oscar deterministic ({len(oscar_all)} detections)")
    ax2 = ax.twinx()
    ax2.bar(ours_bin_diam_px, ours_total_per_bin,
            width=np.diff(ours_bin_diam_px).mean() if len(ours_bin_diam_px) > 1 else 1.0,
            color="steelblue", alpha=0.45, label="Ours (expected count)")
    ax.set_xlabel("Bubble Diameter (px)")
    ax.set_ylabel("Count (GT, Hough, Oscar)")
    ax2.set_ylabel("Expected count (ours)")
    ax.set_xlim(0, args.heatmap_max_diam)
    ax.set_title(f"Overall Bubble Size Distribution — session {args.session} (real GT)")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_fig2_with_oscar.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIG 5 — Count vs frame index
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(frame_idx, gt_count, "o-", color="black", label="Ground truth")
    ax.plot(frame_idx, hough_count, "o-", color="tomato", label="Hough")
    ax.plot(frame_idx, oscar_count, "o-", color="mediumseagreen", label="Oscar deterministic")
    ax.plot(frame_idx, ours_count, "o-", color="steelblue", label="Ours (expected)")
    if frst_counts:
        ax.plot(frame_idx, frst_counts, "o--", color="goldenrod", label="Oscar FRST (count only, no size)")
    ax.set_xlabel(f"Frame index (session {args.session}, no fps recorded for this apparatus)")
    ax.set_ylabel("Number of Bubbles")
    ax.set_title("Bubble Count vs Frame Index (real GT)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_fig5_with_oscar.png", dpi=150)
    plt.close(fig)

    print("\nCount comparison:")
    header = f"  {'frame':>7}  {'GT':>5}  {'Hough':>6}  {'Oscar':>6}  {'Ours':>8}"
    if frst_counts:
        header += f"  {'FRST':>5}"
    print(header)
    for k, fi in enumerate(frame_idx):
        row = f"  {fi:>7}  {gt_count[k]:>5}  {hough_count[k]:>6}  {oscar_count[k]:>6}  {ours_count[k]:>8.2f}"
        if frst_counts:
            row += f"  {frst_counts[k]:>5}"
        print(row)

    # ========================================================================
    # FIG 6 — heatmap panels, 4 sparse frames
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
    oscar_mat = density_matrix(oscar_diam_per_frame)
    ours_mat = np.zeros((len(bins) - 1, len(frame_idx)))
    for k, counts_k in enumerate(ours_counts_per_frame):
        total_k = counts_k.sum()
        if total_k > 0:
            interp_counts = np.interp(bin_centers, ours_bin_diam_px, counts_k, left=0, right=0)
            ours_mat[:, k] = interp_counts / (total_k * bin_width)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
    for ax, mat, title in zip(axes, [gt_mat, hough_mat, oscar_mat, ours_mat],
                               ["Ground Truth", "Hough", "Oscar deterministic", "Ours"]):
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="hot",
                        extent=[0, len(frame_idx) - 1, 0, args.heatmap_max_diam])
        ax.set_xticks(range(len(frame_idx)))
        ax.set_xticklabels(frame_idx, rotation=45, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Frame index")
        fig.colorbar(im, ax=ax, label="PDF", fraction=0.046)
    axes[0].set_ylabel("Bubble Diameter (px)")
    fig.suptitle(f"Bubble Size Distribution — session {args.session}, {len(frame_idx)} real annotated frames")
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_fig6_with_oscar.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved gt_fig2/5/6_with_oscar to {args.out_dir}/")


if __name__ == "__main__":
    main()
