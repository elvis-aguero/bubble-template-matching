#!/usr/bin/env python3
"""
Segmentation snapshot on one GT image: ground truth annotations, Hough
transform, Oscar classical detector, Oscar FRST+SAM3 hybrid, and our
template-matching pipeline (LOSO-trained, holding out this image's session).

Saves one PNG per method (not a single combined figure) so they can be
embedded individually in docs/experiments_summary.md. Circle overlays are
drawn at native detection scale so images are visually comparable. This is
a qualitative complement to the relL1 numbers in that doc, not a metric.

USAGE
-----
  python scripts/segmentation_snapshot.py \\
      --image-stem ZeroG_FlightDay_Test_C1S0014_img006001 \\
      --oscar-json-dir /tmp/oscar_pull/detect_bubbles_out_14 \\
      --hybrid-json-dir /tmp/oscar_pull/hybrid_out_14
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
from bubble_histogram.data import AnnotatedDataset, get_session_id
from bubble_histogram.pipeline import BubblePipeline
from bubble_histogram.calibration import nms_3d
from bubble_histogram.ncc import compute_ncc_maps

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

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
        return np.empty((0, 3))
    return circles[0]  # (cx, cy, r)


def draw_circles(ax, circles, color, linewidth=0.8):
    for cx, cy, r in circles:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=color,
                                 linewidth=linewidth, alpha=0.9))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"))
    parser.add_argument("--image-stem", type=str,
                         default="ZeroG_FlightDay_Test_C1S0014_img006001")
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--max-radius", type=float, default=50.0)
    parser.add_argument("--oscar-json-dir", type=Path, required=True)
    parser.add_argument("--hybrid-json-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path,
                         default=Path("docs/assets/"))
    args = parser.parse_args()

    all_paths = sorted((args.seed_dir / "images").glob("*.png"))
    img_path = next(p for p in all_paths if p.stem == args.image_stem)
    val_session = get_session_id(img_path.name)

    cfg = PipelineConfig(min_radius=args.min_radius, max_radius=args.max_radius,
                         local_maxima_calibration=True)
    dataset = AnnotatedDataset(args.seed_dir, val_session=val_session)
    pipeline = BubblePipeline(cfg)
    pipeline.train(dataset)

    sample = dataset.load_sample(img_path)
    img = sample.image
    gt_circles = np.array([[b.cx, b.cy, b.radius] for b in sample.bubbles])

    hough_circles = run_hough(img, int(args.min_radius), int(args.max_radius))

    oscar_path = args.oscar_json_dir / f"detected_{args.image_stem}.json"
    oscar_data = json.loads(oscar_path.read_text())
    oscar_circles = np.array([[cx, cy, r] for (cx, cy, r) in oscar_data])

    hybrid_path = args.hybrid_json_dir / f"{args.image_stem}_analysis.json"
    hybrid_data = json.loads(hybrid_path.read_text())
    hybrid_circles = np.array([
        [inst["centroid_xy"][0], inst["centroid_xy"][1], inst["radius_equiv_px"]]
        for inst in hybrid_data["instances"]
    ])

    ncc_results = compute_ncc_maps(img, pipeline.templates, pipeline.config)
    peaks = nms_3d(ncc_results, pipeline.config)
    n_show = min(len(peaks), len(gt_circles) * 2 if len(gt_circles) else 500)
    ours_circles = np.array([[x, y, r] for (score, level, y, x, r) in peaks[:n_show]])

    panels = [
        ("gt", "Ground truth (manual annotation)", gt_circles, "black"),
        ("hough", "Hough transform (collaborator's baseline)", hough_circles, "tomato"),
        ("oscar_classical", "Classical detector (adaptive threshold, Oscar HPC)", oscar_circles, "mediumseagreen"),
        ("oscar_hybrid", "FRST + SAM3 hybrid (state-of-the-art, Oscar HPC)", hybrid_circles, "darkorchid"),
        ("ours", f"Template matching (this work, LOSO, top {n_show} of {len(peaks)})", ours_circles, "steelblue"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, title, circles, color in panels:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        draw_circles(ax, circles, color)
        ax.set_title(f"{title}\n(n={len(circles)})", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        out_path = args.out_dir / f"segmentation_{args.image_stem}_{key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
