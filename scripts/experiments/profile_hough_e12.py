#!/usr/bin/env python3
"""
E12 — Hough circle transform diagnostic.

Hypothesis to falsify:
  "HoughCircles on gradient magnitude produces per-bubble radius estimates
   within ≤2 pyramid-levels of the GT correct level for ≥70% of GT bubbles
   (r≥8px), with a background false-positive rate of ≤5 per image at the
   same detection threshold."

Design:
  - Run on FULL image (not patches) — tests the realistic pipeline scenario.
  - Sweep accumulator threshold param2 ∈ {10, 20, 30} to test robustness.
    Falsification must hold (or fail) across the sweep, not just for one tuning.
  - Match Hough circles to GT bubbles: nearest Hough detection within
    0.5×bubble.radius of GT center.
  - Radius accuracy: detected radius vs GT radius, converted to pyramid levels.
  - Background FP: Hough detections with no GT bubble within 3×detected_radius.
  - Stratify by morphology (dark-rim vs filled-dark vs other) using E11 Step 1
    morphology classifications where available; fall back to "unknown" otherwise.

Two-level failure:
  Level 1 (sensitivity): detection rate at ≤2 levels < 70% → scale selectivity fails
  Level 2 (specificity): background FP > 5 per image at the threshold giving ≥70% DR

Pyramid level offset formula:
  delta = log(r_detected / r_gt) / log(scale_factor)
  where scale_factor = 0.9 (PipelineConfig default)

USAGE: python scripts/experiments/profile_hough_e12.py [data_dir]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import load_image, parse_annotations

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_RADIUS       = 8.0    # exclude sub-pixel-reliable bubbles
MAX_RADIUS       = 60.0   # slightly above dataset max
MATCH_DIST_FRAC  = 0.5    # match within 0.5×r_gt of GT center
LEVEL_TOLERANCE  = 2      # ≤2 pyramid levels = "correct scale"
FP_DIST_FRAC     = 3.0    # FP if no GT within 3×r_detected
MIN_IMG_MEAN     = 0.0
MAX_IMG_MEAN     = 0.6

PARAM2_SWEEP     = [10, 20, 30]     # accumulator threshold: lower = more detections
CANNY_HIGH       = 50               # Canny high threshold (param1)
DP               = 1                # accumulator resolution = 1×image resolution
MIN_DIST         = 8                # min center-to-center distance between detections


def px_to_levels(r_det: float, r_gt: float, scale_factor: float) -> float:
    """Convert radius ratio to pyramid level offset."""
    if r_det <= 0 or r_gt <= 0:
        return np.nan
    return np.log(r_det / r_gt) / np.log(scale_factor)


def img_to_uint8(img: np.ndarray) -> np.ndarray:
    """Float [0,1] → uint8 [0,255] with per-image contrast stretch."""
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    stretched = (img - mn) / (mx - mn)
    return (stretched * 255).astype(np.uint8)


def run_hough(img_u8: np.ndarray, param2: int) -> list[tuple[float, float, float]]:
    """Return list of (cx, cy, r) detections from HoughCircles."""
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=CANNY_HIGH,
        param2=param2,
        minRadius=int(MIN_RADIUS),
        maxRadius=int(MAX_RADIUS),
    )
    if circles is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]


def match_detections(detections, bubbles, scale_factor):
    """
    For each GT bubble, find the nearest Hough detection within MATCH_DIST_FRAC×r_gt.
    Returns list of dicts with keys: r_gt, r_det (nan if unmatched), level_offset.
    """
    results = []
    for b in bubbles:
        if b.radius < MIN_RADIUS:
            continue
        best_dist = np.inf
        best_r_det = np.nan
        for cx, cy, r in detections:
            dist = np.sqrt((cx - b.cx) ** 2 + (cy - b.cy) ** 2)
            if dist < MATCH_DIST_FRAC * b.radius and dist < best_dist:
                best_dist = dist
                best_r_det = r
        offset = px_to_levels(best_r_det, b.radius, scale_factor)
        results.append({
            "r_gt": b.radius,
            "r_det": best_r_det,
            "level_offset": offset,
            "matched": not np.isnan(best_r_det),
        })
    return results


def count_fp(detections, bubbles) -> int:
    """Count detections with no GT bubble within FP_DIST_FRAC×r_detected."""
    fp = 0
    for cx, cy, r in detections:
        close = any(
            np.sqrt((cx - b.cx) ** 2 + (cy - b.cy) ** 2) < FP_DIST_FRAC * r
            for b in bubbles
        )
        if not close:
            fp += 1
    return fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out", type=Path, default=Path("output/e12_hough_diagnostic.png"))
    args = parser.parse_args()

    cfg = PipelineConfig()
    sf  = cfg.scale_factor

    all_paths = sorted((Path(args.data_dir) / "images").glob("*.png"))
    tractable = [(p, load_image(p)) for p in all_paths
                 if MIN_IMG_MEAN <= load_image(p).mean() < MAX_IMG_MEAN]
    print(f"Images: {len(tractable)}/{len(all_paths)}")

    # Per param2: collect match results and FP counts across all images
    sweep_results = {p2: {"matches": [], "fp_per_image": [], "n_det": []}
                     for p2 in PARAM2_SWEEP}

    for img_path, img_f in tractable:
        lbl_path = Path(args.data_dir) / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        bubbles = parse_annotations(lbl_path)
        valid_bubbles = [b for b in bubbles if b.radius >= MIN_RADIUS]
        if not valid_bubbles:
            continue

        img_u8 = img_to_uint8(img_f)
        print(f"  {img_path.stem[-28:]}  mean={img_f.mean():.3f}  "
              f"n_gt={len(valid_bubbles)}")

        for p2 in PARAM2_SWEEP:
            dets = run_hough(img_u8, p2)
            matches = match_detections(dets, valid_bubbles, sf)
            fp = count_fp(dets, bubbles)  # FP vs ALL bubbles (inc. small)

            sweep_results[p2]["matches"].extend(matches)
            sweep_results[p2]["fp_per_image"].append(fp)
            sweep_results[p2]["n_det"].append(len(dets))

            n_matched = sum(m["matched"] for m in matches)
            n_in_tol  = sum(m["matched"] and abs(m["level_offset"]) <= LEVEL_TOLERANCE
                            for m in matches)
            print(f"    param2={p2:2d}: det={len(dets):4d}  "
                  f"matched={n_matched}/{len(matches)}  "
                  f"in_tol={n_in_tol}/{len(matches)}  fp={fp}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'param2':>7}  {'n_bubbles':>10}  {'DR_matched':>12}  "
          f"{'DR_in_tol(≤2lv)':>17}  {'mean_FP/img':>13}")
    for p2 in PARAM2_SWEEP:
        m = sweep_results[p2]["matches"]
        if not m:
            continue
        n = len(m)
        dr_match = sum(x["matched"] for x in m) / n
        dr_tol   = sum(x["matched"] and abs(x["level_offset"]) <= LEVEL_TOLERANCE
                       for x in m) / n
        mean_fp  = np.mean(sweep_results[p2]["fp_per_image"])
        print(f"  {p2:>5}  {n:>10}  {dr_match:>12.3f}  {dr_tol:>17.3f}  {mean_fp:>13.1f}")

        # Verdict
        if dr_tol >= 0.70 and mean_fp <= 5:
            verdict = "VIABLE at this threshold"
        elif dr_tol >= 0.70:
            verdict = f"DR OK but FP={mean_fp:.1f} > 5 → specificity fails"
        elif mean_fp <= 5:
            verdict = f"FP OK but DR={dr_tol:.2f} < 0.70 → sensitivity fails"
        else:
            verdict = f"FALSIFIED: DR={dr_tol:.2f} AND FP={mean_fp:.1f}"
        print(f"          → {verdict}")

    # Overall verdict
    print(f"\n{'='*70}")
    best_tol = max(PARAM2_SWEEP,
                   key=lambda p2: (
                       sum(x["matched"] and abs(x["level_offset"]) <= LEVEL_TOLERANCE
                           for x in sweep_results[p2]["matches"])
                       / max(len(sweep_results[p2]["matches"]), 1)
                   ))
    m = sweep_results[best_tol]["matches"]
    n = len(m)
    dr_best = sum(x["matched"] and abs(x["level_offset"]) <= LEVEL_TOLERANCE
                  for x in m) / n
    fp_best = np.mean(sweep_results[best_tol]["fp_per_image"])
    print(f"Best param2={best_tol}: DR_tol={dr_best:.3f}, FP/img={fp_best:.1f}")
    if dr_best >= 0.70 and fp_best <= 5:
        print("HYPOTHESIS SURVIVES at best tuning → Hough viable")
    elif dr_best >= 0.70:
        print(f"FALSIFIED (specificity): DR passes but FP={fp_best:.1f} > 5")
    else:
        print(f"FALSIFIED (sensitivity): DR={dr_best:.3f} < 0.70 even at best tuning")

    # ── Level offset distribution ─────────────────────────────────────────────
    print(f"\nLevel offset distribution (all matched, best param2={best_tol}):")
    offsets = [x["level_offset"] for x in sweep_results[best_tol]["matches"]
               if x["matched"] and not np.isnan(x["level_offset"])]
    if offsets:
        offsets = np.array(offsets)
        print(f"  mean={offsets.mean():+.2f}  median={np.median(offsets):+.2f}  "
              f"std={offsets.std():.2f}  p25={np.percentile(offsets,25):+.1f}  "
              f"p75={np.percentile(offsets,75):+.1f}")
        in1 = (np.abs(offsets) <= 1).mean()
        in2 = (np.abs(offsets) <= 2).mean()
        in3 = (np.abs(offsets) <= 3).mean()
        print(f"  within ±1 levels: {in1:.1%}  ±2 levels: {in2:.1%}  ±3 levels: {in3:.1%}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: DR and FP vs param2
    ax = axes[0]
    dr_vals  = []
    fp_vals  = []
    det_vals = []
    for p2 in PARAM2_SWEEP:
        m = sweep_results[p2]["matches"]
        n = len(m)
        dr_vals.append(sum(x["matched"] and abs(x["level_offset"]) <= LEVEL_TOLERANCE
                           for x in m) / max(n, 1))
        fp_vals.append(np.mean(sweep_results[p2]["fp_per_image"]))
        det_vals.append(np.mean(sweep_results[p2]["n_det"]))
    ax2 = ax.twinx()
    ax.plot(PARAM2_SWEEP, dr_vals, "o-", color="steelblue", linewidth=2, label="DR (≤2 levels)")
    ax.axhline(0.70, color="steelblue", linestyle=":", linewidth=1.2, label="DR=0.70 target")
    ax2.plot(PARAM2_SWEEP, fp_vals, "s--", color="tomato", linewidth=2, label="FP/image")
    ax2.axhline(5, color="tomato", linestyle=":", linewidth=1.2, label="FP=5 target")
    ax.set_xlabel("param2 (accumulator threshold)")
    ax.set_ylabel("Detection rate", color="steelblue")
    ax2.set_ylabel("Mean FP per image", color="tomato")
    ax.set_title("DR vs FP tradeoff across param2 sweep")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Middle: level offset histogram (best param2)
    ax = axes[1]
    if offsets is not None and len(offsets) > 0:
        bins = np.arange(offsets.min() - 0.5, offsets.max() + 1.5, 1.0)
        ax.hist(offsets, bins=bins, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="exact match")
        for lv in [-LEVEL_TOLERANCE, LEVEL_TOLERANCE]:
            ax.axvline(lv, color="orange", linestyle="--", linewidth=1.2,
                       label=f"±{LEVEL_TOLERANCE} levels" if lv > 0 else None)
        ax.set_xlabel("Detected radius − GT radius (pyramid levels)")
        ax.set_ylabel("Count")
        ax.set_title(f"Radius error distribution (param2={best_tol})\n"
                     f"±2 level DR = {in2:.1%}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Right: radius accuracy scatter (r_det vs r_gt)
    ax = axes[2]
    matched = [x for x in sweep_results[best_tol]["matches"] if x["matched"]]
    if matched:
        r_gt  = np.array([x["r_gt"]  for x in matched])
        r_det = np.array([x["r_det"] for x in matched])
        err   = np.abs(r_gt - r_det)
        sc = ax.scatter(r_gt, r_det, c=err, cmap="RdYlGn_r", s=20, alpha=0.7,
                        vmin=0, vmax=r_gt.max() * 0.3)
        plt.colorbar(sc, ax=ax, label="|r_det − r_gt| (px)")
        rng = [min(r_gt.min(), r_det.min()) - 2, max(r_gt.max(), r_det.max()) + 2]
        ax.plot(rng, rng, "k--", linewidth=1, label="perfect radius")
        for lv in [-LEVEL_TOLERANCE, LEVEL_TOLERANCE]:
            factor = cfg.scale_factor ** (-lv)
            ax.plot(rng, [r * factor for r in rng], ":",
                    color="orange", linewidth=1,
                    label=f"±{LEVEL_TOLERANCE} levels" if lv > 0 else None)
        ax.set_xlim(rng); ax.set_ylim(rng)
        ax.set_xlabel("GT radius (px)")
        ax.set_ylabel("Detected radius (px)")
        ax.set_title(f"Radius accuracy scatter (param2={best_tol})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"E12 — Hough circle transform diagnostic  "
                 f"({len(tractable)} images)", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
