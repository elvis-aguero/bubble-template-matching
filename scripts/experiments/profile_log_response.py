#!/usr/bin/env python3
"""
E10 — LoG scale-space response diagnostic.

Hypothesis: LoG has a genuine scale-selective peak near delta=0 for GT bubbles
AND that peak is significantly higher than LoG at background locations at the
same scale.

Design: sample GT bubbles from all non-saturated images (img_mean < 150) to
cover multiple photometric regimes. For each bubble:
  - Measure max scale-normalized LoG response within bubble.radius at every
    pyramid level (sensitivity curve, mirroring E8 for NCC).
  - At the correct level, sample LoG at 20 random background locations
    (> 3*radius from any GT bubble) for the specificity comparison.

Falsification:
  1. Sensitivity curve monotone → LoG wrong feature, look elsewhere.
  2. Peak near delta=0 but background equally high → specificity fails.
  3. Peak near delta=0 AND background substantially lower → LoG viable.

USAGE: python scripts/profile_log_response.py [data_dir] [--n-bubbles N]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import build_pyramid


MAX_IMG_MEAN = 150   # exclude saturated/dead frames above this threshold


def log_map_at_level(scaled_img: np.ndarray, canonical_r: float) -> np.ndarray:
    """Scale-normalised LoG: sigma^2 * L(sigma), sigma = canonical_r / sqrt(2)."""
    sigma = canonical_r / np.sqrt(2)
    img_f = scaled_img.astype(np.float64)
    raw = gaussian_laplace(img_f, sigma=sigma)
    return (sigma ** 2 * raw).astype(np.float32)


def best_log_in_radius(log_map, scale, cy, cx, radius_orig):
    """Max |LoG| within bubble.radius (original-image coords) at given scale."""
    h, w = log_map.shape
    cy_s = cy * scale
    cx_s = cx * scale
    r_s  = radius_orig * scale

    y0, y1 = max(0, int(cy_s - r_s)), min(h, int(cy_s + r_s) + 1)
    x0, x1 = max(0, int(cx_s - r_s)), min(w, int(cx_s + r_s) + 1)
    if y0 >= y1 or x0 >= x1:
        return np.nan

    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt((yy - cy_s) ** 2 + (xx - cx_s) ** 2)
    mask = dist < r_s
    if not mask.any():
        return np.nan
    return float(np.abs(log_map[yy[mask], xx[mask]]).max())


def sample_background_log(log_map, scale, bubbles, canonical_r, rng, n=20):
    """Sample LoG at random locations > 3*radius from any GT bubble."""
    h, w = log_map.shape
    vals = []
    attempts = 0
    while len(vals) < n and attempts < 2000:
        attempts += 1
        py = rng.integers(0, h)
        px = rng.integers(0, w)
        # convert to original image coordinates
        py_orig = py / scale
        px_orig = px / scale
        too_close = any(
            np.sqrt((py_orig - b.cy) ** 2 + (px_orig - b.cx) ** 2) < 3 * b.radius
            for b in bubbles
        )
        if not too_close:
            vals.append(abs(float(log_map[py, px])))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--n-bubbles", type=int, default=60,
                        help="Total GT bubbles to profile across all images")
    parser.add_argument("--out", type=Path, default=Path("output/log_response_profile.png"))
    args = parser.parse_args()

    cfg = PipelineConfig()
    canonical_r = cfg.template_size / (2.0 * cfg.template_context_factor)

    ds = AnnotatedDataset(args.data_dir, template_frac=0.30, calibration_frac=0.65, seed=42)

    # Collect all non-saturated images (across all splits — this is a diagnostic)
    from bubble_histogram.data import load_image
    all_paths = sorted((Path(args.data_dir) / "images").glob("*.png"))
    tractable = []
    for p in all_paths:
        img = load_image(p)
        if img.mean() < MAX_IMG_MEAN:
            tractable.append(p)
    print(f"Non-saturated images: {len(tractable)}/{len(all_paths)}")

    # Sample bubbles evenly across tractable images
    rng = np.random.default_rng(seed=42)
    per_image = max(1, args.n_bubbles // len(tractable))

    delta_range = 6
    delta_cols = np.arange(-delta_range, delta_range + 1)

    sensitivity_rows = []   # one row per bubble: LoG at each delta
    correct_level_bubble = []   # LoG at correct level for each bubble
    correct_level_bg = []       # background LoG at correct level for each bubble
    bubble_radii = []
    image_names = []

    for img_path in tractable:
        lbl_path = Path(args.data_dir) / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue

        from bubble_histogram.data import parse_annotations
        img = load_image(img_path)
        bubbles = parse_annotations(lbl_path)
        if not bubbles:
            continue

        pyramid = build_pyramid(img, cfg)
        eff_radii = np.array([er for _, _, er in pyramid])
        scales    = np.array([img.shape[0] / s.shape[0] if s.shape[0] > 0 else 1.0
                               for _, s, _ in pyramid])
        # scale[lv] = original_height / scaled_height ≈ 1/alpha
        # Actually: scale of the scaled image relative to original = scaled_h / orig_h
        img_scales = np.array([s.shape[0] / img.shape[0] for _, s, _ in pyramid])

        # Pre-compute LoG maps for each level
        log_maps = []
        for _, scaled, _ in pyramid:
            log_maps.append(log_map_at_level(scaled, canonical_r))

        # Filter bubbles to those covered by pyramid
        valid = [b for b in bubbles
                 if eff_radii.min() <= b.radius <= eff_radii.max()
                 and b.cy > b.radius and b.cx > b.radius
                 and b.cy < img.shape[0] - b.radius
                 and b.cx < img.shape[1] - b.radius]
        if not valid:
            continue

        # Sample evenly across log-radius space
        log_r = np.log([b.radius for b in valid])
        bins = np.linspace(log_r.min(), log_r.max(), per_image + 1)
        selected = []
        for i in range(per_image):
            in_bin = [b for b, lr in zip(valid, log_r) if bins[i] <= lr < bins[i+1]]
            if in_bin:
                selected.append(in_bin[len(in_bin) // 2])
        if not selected:
            selected = valid[:per_image]

        print(f"  {img_path.stem[-25:]}  mean={img.mean():.0f}  "
              f"n_bubbles={len(bubbles)}  profiling={len(selected)}")

        for b in selected:
            correct_lv = int(np.argmin(np.abs(eff_radii - b.radius)))
            row = []
            for d in delta_cols:
                lv = correct_lv + d
                if 0 <= lv < len(log_maps):
                    val = best_log_in_radius(log_maps[lv], img_scales[lv], b.cy, b.cx, b.radius)
                else:
                    val = np.nan
                row.append(val)
            sensitivity_rows.append(row)

            # Specificity: correct level bubble vs background
            correct_val = row[delta_range]  # delta=0 column
            correct_level_bubble.append(correct_val)

            bg_vals = sample_background_log(
                log_maps[correct_lv], img_scales[correct_lv], bubbles, canonical_r, rng
            )
            correct_level_bg.extend(bg_vals)
            bubble_radii.append(b.radius)
            image_names.append(img_path.stem[-20:])

    if not sensitivity_rows:
        print("No data collected.")
        return

    curves = np.array(sensitivity_rows, dtype=np.float32)
    print(f"\nTotal bubbles profiled: {len(curves)}")
    print(f"Background samples: {len(correct_level_bg)}")

    # ── Text summary ──────────────────────────────────────────────────────────
    print(f"\n{'delta':>6}  {'mean_|LoG|':>11}  {'median':>8}  {'n_valid':>8}")
    for i, d in enumerate(delta_cols):
        col = curves[:, i]
        valid = col[~np.isnan(col)]
        print(f"  {d:+3d}  {np.nanmean(col):11.4f}  {np.nanmedian(col):8.4f}  {len(valid):8d}")

    peak_deltas = []
    for row in curves:
        valid_mask = ~np.isnan(row)
        if valid_mask.any():
            peak_deltas.append(delta_cols[np.nanargmax(row)])
    peak_deltas = np.array(peak_deltas)
    print(f"\nPeak delta per bubble:")
    print(f"  mean={peak_deltas.mean():+.2f}  median={np.median(peak_deltas):+.1f}  "
          f"p25={np.percentile(peak_deltas,25):+.1f}  p75={np.percentile(peak_deltas,75):+.1f}")

    bubble_at_correct = np.array([v for v in correct_level_bubble if not np.isnan(v)])
    bg_arr = np.array(correct_level_bg)
    print(f"\nSpecificity at correct level:")
    print(f"  Bubble |LoG|:     mean={bubble_at_correct.mean():.4f}  "
          f"median={np.median(bubble_at_correct):.4f}")
    print(f"  Background |LoG|: mean={bg_arr.mean():.4f}  "
          f"median={np.median(bg_arr):.4f}")
    if bg_arr.mean() > 0:
        print(f"  Signal/noise ratio (means): {bubble_at_correct.mean()/bg_arr.mean():.2f}×")

    # ── Verdict ───────────────────────────────────────────────────────────────
    med_peak = np.median(peak_deltas)
    snr = bubble_at_correct.mean() / max(bg_arr.mean(), 1e-9)
    print(f"\n{'='*55}")
    if abs(med_peak) <= 1.5 and snr >= 2.0:
        print(f"HYPOTHESIS SURVIVES: peak near delta=0 (median={med_peak:+.1f}), "
              f"SNR={snr:.1f}× → LoG viable")
    elif abs(med_peak) > 1.5:
        print(f"FALSIFIED (sensitivity): no peak near delta=0 "
              f"(median peak={med_peak:+.1f}) → wrong feature")
    else:
        print(f"FALSIFIED (specificity): peak near delta=0 but "
              f"SNR={snr:.1f}× < 2 → background too similar")
    print(f"{'='*55}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: mean sensitivity curve
    ax = axes[0]
    mean_c = np.nanmean(curves, axis=0)
    std_c  = np.nanstd(curves, axis=0)
    ax.plot(delta_cols, mean_c, "o-", color="steelblue", linewidth=2, label="mean |LoG|")
    ax.fill_between(delta_cols, mean_c - std_c, mean_c + std_c,
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="correct level")
    ax.axvline(med_peak, color="orange", linestyle=":",
               linewidth=1.5, label=f"median peak (Δ={med_peak:+.1f})")
    ax.set_xlabel("delta = level − correct_level  (negative = finer)")
    ax.set_ylabel("Max |LoG| within bubble.radius")
    ax.set_title(f"LoG sensitivity curve\n({len(curves)} bubbles, {len(tractable)} images)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Middle: peak delta histogram
    ax = axes[1]
    radii = np.array(bubble_radii)
    q33, q66 = np.percentile(radii, 33), np.percentile(radii, 66)
    for mask, label, color in [
        (radii < q33, f"small r<{q33:.1f}px", "steelblue"),
        ((radii >= q33) & (radii < q66), f"medium", "seagreen"),
        (radii >= q66, f"large r>{q66:.1f}px", "tomato"),
    ]:
        if mask.any():
            ax.hist(peak_deltas[mask], bins=delta_cols - 0.5,
                    alpha=0.6, color=color, label=label, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(med_peak, color="orange", linestyle=":", linewidth=1.5)
    ax.set_xlabel("delta at peak |LoG|")
    ax.set_ylabel("Count")
    ax.set_title("Peak-delta distribution by bubble size")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: specificity — bubble vs background at correct level
    ax = axes[2]
    ax.hist(bg_arr, bins=40, density=True, alpha=0.6,
            color="salmon", label=f"Background ({len(bg_arr)} samples)")
    ax.hist(bubble_at_correct, bins=40, density=True, alpha=0.6,
            color="steelblue", label=f"Bubble centers ({len(bubble_at_correct)} samples)")
    ax.set_xlabel("|LoG| at correct pyramid level")
    ax.set_ylabel("Density")
    ax.set_title(f"Specificity: bubble vs background\nSNR = {snr:.1f}×")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
