#!/usr/bin/env python3
"""
Faithful port of the collaborator's MATLAB "Mean Diameter vs Time WITH TREND"
figure (Madeline Federle, 2025-10-31) — same title, axes, exponential-fit
convention, same 91-consecutive-frame/4ms-step sampling design — with BOTH
Hough (collaborator's method) and ours (NCC pyramid) plotted on it.

Model: D(t) = Dinf + (D0 - Dinf)*exp(-t/tau), same as MATLAB's fittype,
same initial-guess heuristic (D0=first point, Dinf=min(data), tau=(range)/4).

NO ground truth exists for this dataset (see prior discussion) — this
replicates their FIGURE, not a validated accuracy claim.

USAGE
-----
  python scripts/matlab_style_overlay.py <local_frames_dir> [options]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.pipeline import BubblePipeline

# ---- MATLAB rcParams equivalent ----
# set(groot,'defaultLineLineWidth',2)
# set(groot,'defaultAxesFontName','Times New Roman'); defaultTextFontName same
# set(groot,'defaultAxesFontSize',18); defaultTextFontSize same
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18

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


def weighted_mean(values, weights):
    total = weights.sum()
    return float((values * weights).sum() / total) if total > 0 else float("nan")


def exp_decay(t, d_inf, d0, tau):
    return d_inf + (d0 - d_inf) * np.exp(-t / tau)


def fit_exp(t_fit, d_fit):
    """Same fittype + initial-guess heuristic as the MATLAB script."""
    d0_guess = d_fit[0]
    dinf_guess = d_fit.min()
    tau_guess = (t_fit.max() - t_fit.min()) / 4 or 1.0
    popt, _ = curve_fit(exp_decay, t_fit, d_fit,
                         p0=[dinf_guess, d0_guess, tau_guess], maxfev=10000)
    return popt  # (d_inf, d0, tau)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="LOCAL directory with the frame files "
                                                       "(copy off the network mount first)")
    parser.add_argument("--experim", type=str, default="ZeroG_Test3_Opt3")
    parser.add_argument("--frame-digits", type=int, default=6)
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"))
    parser.add_argument("--first-frame", type=int, default=650)
    parser.add_argument("--last-frame", type=int, default=1100)
    parser.add_argument("--fps", type=float, default=1250)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--max-radius", type=float, default=70.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/test17_zerog_opt3/"))
    args = parser.parse_args()

    cfg = PipelineConfig(min_radius=args.min_radius, max_radius=args.max_radius,
                         local_maxima_calibration=True)
    print(f"Training our pipeline on {args.seed_dir}...")
    dataset = AnnotatedDataset(args.seed_dir)
    pipeline = BubblePipeline(cfg)
    pipeline.train(dataset)
    print("Training done.")

    frame_indices = list(range(args.first_frame, args.last_frame + 1, args.step))
    time_s, ours_diam, hough_diam = [], [], []
    missing = 0

    for i in frame_indices:
        fpath = args.data_dir / f"{args.experim}{i:0{args.frame_digits}d}.bmp"
        if not fpath.exists():
            missing += 1
            continue
        img = np.array(Image.open(fpath).convert("L")).astype(np.float32) / 255.0
        t = (i - args.first_frame) / args.fps
        time_s.append(t)

        result = pipeline.predict(img)
        radius_px = np.array(result["radius_px"])
        counts = np.array(result["expected_count"])
        ours_diam.append(weighted_mean(2 * radius_px, counts))

        hough_radii = run_hough(img, min_r=int(args.min_radius), max_r=int(args.max_radius))
        hough_d = 2 * hough_radii
        hough_diam.append(float(hough_d.mean()) if len(hough_d) else float("nan"))

        print(f"  frame {i:>6}  t={t:6.3f}s  ours={ours_diam[-1]:6.2f}px  "
              f"hough={hough_diam[-1]:6.2f}px ({len(hough_radii)} circles)")

    if missing:
        print(f"WARNING: {missing} frames missing from {args.data_dir}")

    time_s = np.array(time_s)
    ours_diam = np.array(ours_diam)
    hough_diam = np.array(hough_diam)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # --- Hough series ---
    ax.plot(time_s, hough_diam, color="tab:red", label="Hough - Data")
    valid = ~np.isnan(hough_diam)
    if valid.sum() >= 3:
        try:
            d_inf, d0, tau = fit_exp(time_s[valid], hough_diam[valid])
            t_smooth = np.linspace(time_s[valid].min(), time_s[valid].max(), 300)
            ax.plot(t_smooth, exp_decay(t_smooth, d_inf, d0, tau), "k--",
                    label="Hough - Exponential Fit")
            print(f"\nHough exp fit: D0={d0:.3f} Dinf={d_inf:.3f} tau={tau:.4f}s")
        except RuntimeError as e:
            print(f"Hough exponential fit did not converge: {e}")

    # --- Ours series ---
    ax.plot(time_s, ours_diam, color="tab:blue", label="Ours - Data")
    valid = ~np.isnan(ours_diam)
    if valid.sum() >= 3:
        try:
            d_inf, d0, tau = fit_exp(time_s[valid], ours_diam[valid])
            t_smooth = np.linspace(time_s[valid].min(), time_s[valid].max(), 300)
            ax.plot(t_smooth, exp_decay(t_smooth, d_inf, d0, tau), "k:",
                    label="Ours - Exponential Fit")
            print(f"Ours exp fit:  D0={d0:.3f} Dinf={d_inf:.3f} tau={tau:.4f}s")
        except RuntimeError as e:
            print(f"Ours exponential fit did not converge: {e}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Diameter (px)")  # px, not mm: no scale calibration available for this apparatus
    ax.set_title("Mean Bubble Diameter vs Time")
    ax.legend(loc="best", fontsize=13)
    ax.grid(True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "matlab_style_mean_diameter_vs_time.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
