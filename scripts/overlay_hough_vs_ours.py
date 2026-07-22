#!/usr/bin/env python3
"""
Minimal overlay: collaborator's Hough-circle approach vs. our NCC-pyramid
pipeline, mean bubble diameter vs time, same frames, same time axis.

NO ground truth exists for this dataset — this is not an accuracy comparison,
just the two predictions side by side.

Hough parameters reused as-is from this repo's own scripts/experiments/profile_hough_e12.py
(full-image cv2.HoughCircles, dark-on-bright via inverted contrast-stretch,
GaussianBlur(5,5), param1=50, param2=20, minDist=8) — the closest available
Python equivalent to MATLAB's imfindcircles('ObjectPolarity','dark','Sensitivity',0.90).

USAGE
-----
  python scripts/overlay_hough_vs_ours.py <data_dir> [options]
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

CANNY_HIGH = 50
DP = 1
MIN_DIST = 8
PARAM2 = 20


def load_and_crop_bmp(path: Path, crop_xywh) -> np.ndarray:
    img = Image.open(path).convert("L")
    if crop_xywh is not None:
        x, y, w, h = crop_xywh
        img = img.crop((x, y, x + w, y + h))
    return np.array(img).astype(np.float32) / 255.0


def img_to_uint8(img: np.ndarray) -> np.ndarray:
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def run_hough(img: np.ndarray, min_r: int, max_r: int) -> np.ndarray:
    """Return array of detected radii (px), empty if none found."""
    img_u8 = img_to_uint8(img)
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_DIST,
        param1=CANNY_HIGH, param2=PARAM2, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return np.array([])
    return circles[0][:, 2]  # radii


def weighted_mean(values, weights):
    total = weights.sum()
    return float((values * weights).sum() / total) if total > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--experim", type=str, default="ZeroG_Test3_Opt3")
    parser.add_argument("--frame-digits", type=int, default=6)
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"))
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    parser.add_argument("--fps", type=float, default=1250)
    parser.add_argument("--frames", type=str,
                         default="200,500,1000,1500,2000,2500,3000,4000,5000,6000,7000")
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--max-radius", type=float, default=70.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/test17_zerog_opt3/"))
    args = parser.parse_args()

    frame_indices = [int(x) for x in args.frames.split(",") if x.strip()]
    t0 = frame_indices[0]

    cfg = PipelineConfig(min_radius=args.min_radius, max_radius=args.max_radius,
                         local_maxima_calibration=True)
    print(f"Training our pipeline on {args.seed_dir}...")
    dataset = AnnotatedDataset(args.seed_dir)
    pipeline = BubblePipeline(cfg)
    pipeline.train(dataset)
    print("Training done.")

    time_s, ours_mean_diam, hough_mean_diam = [], [], []

    for i in frame_indices:
        fpath = args.data_dir / f"{args.experim}{i:0{args.frame_digits}d}.bmp"
        if not fpath.exists():
            print(f"  frame {i}: MISSING, skipped")
            continue
        img = load_and_crop_bmp(fpath, args.crop)
        t = (i - t0) / args.fps
        time_s.append(t)

        # --- ours ---
        result = pipeline.predict(img)
        radius_px = np.array(result["radius_px"])
        counts = np.array(result["expected_count"])
        diam_px = 2 * radius_px
        ours_mean_diam.append(weighted_mean(diam_px, counts))

        # --- Hough ---
        hough_radii = run_hough(img, min_r=int(args.min_radius), max_r=int(args.max_radius))
        hough_diam = 2 * hough_radii
        hough_mean_diam.append(float(hough_diam.mean()) if len(hough_diam) else float("nan"))

        print(f"  frame {i:>6}  t={t:6.3f}s  ours_mean_diam={ours_mean_diam[-1]:7.2f}px "
              f"({counts.sum():.1f} expected count)   "
              f"hough_mean_diam={hough_mean_diam[-1]:7.2f}px ({len(hough_radii)} circles)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_s, hough_mean_diam, "o-", color="tomato", linewidth=2, label="Hough (collaborator's method)")
    ax.plot(time_s, ours_mean_diam, "o-", color="steelblue", linewidth=2, label="Ours (NCC pyramid)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean bubble diameter (px)")
    ax.set_title("Hough vs. ours — mean bubble diameter over time\n(no ground truth available)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = args.out_dir / "hough_vs_ours_mean_diameter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
