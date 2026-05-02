#!/usr/bin/env python3
"""
E11 Step 2 — Corrected LoG discriminability test.

Fixes all measurement defects identified in E10:
  - Tests two sigma models: sigma = R/sqrt(2) (blob) and sigma = 2px constant (ring/rim)
  - Measures LoG at BOTH center pixel AND rim pixel (r ≈ R) per bubble
  - Background uses max(|LoG|) in a 3px disk (symmetric with bubble metric)
  - Background measured at ALL delta levels to expose SNR(delta)
  - Excludes r < 8px bubbles from scale-space peak claims
  - Image filter corrected to img.mean() < 0.6 (float scale)

Hypothesis: at least one (location, sigma) combination produces a |LoG| sensitivity
curve that peaks at or near delta=0, with SNR >= 2x at that level.

Falsification:
  - All curves monotone → LoG cannot form a scale-space peak; look elsewhere
  - Peak near delta=0 but symmetric SNR < 2x → specificity fails
  - Peak near delta=0 AND SNR >= 2x → LoG viable

USAGE: python scripts/experiments/profile_log_e11.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import load_image, parse_annotations
from bubble_histogram.ncc import build_pyramid


MAX_IMG_MEAN = 0.6     # float [0,1] threshold
MIN_RADIUS   = 8.0     # exclude sub-pixel-bin bubbles from claims

SIGMA_BLOB_FACTOR = 1.0 / np.sqrt(2)   # sigma = R/sqrt(2) — blob model
SIGMA_RIM_CONST   = 2.0                 # sigma = 2px constant — ring/rim model

BG_DISK_R = 3          # radius (px) for background max aggregation
N_BG      = 20         # background samples per bubble at each delta
DELTA_RANGE = 6


def log_map(scaled_img: np.ndarray, sigma: float) -> np.ndarray:
    """Scale-normalised LoG: sigma^2 * L(sigma)."""
    raw = gaussian_laplace(scaled_img.astype(np.float64), sigma=sigma)
    return (sigma ** 2 * raw).astype(np.float32)


def center_log(lmap: np.ndarray, img_scale: float, cy: float, cx: float) -> float:
    """LoG at the bubble center pixel in the scaled image."""
    h, w = lmap.shape
    py = int(round(cy * img_scale))
    px = int(round(cx * img_scale))
    if 0 <= py < h and 0 <= px < w:
        return abs(float(lmap[py, px]))
    return np.nan


def rim_log(lmap: np.ndarray, img_scale: float, cy: float, cx: float,
            radius_orig: float) -> float:
    """Max |LoG| on the 1-pixel-wide annulus at r = bubble.radius."""
    h, w = lmap.shape
    r_s = radius_orig * img_scale
    cx_s = cx * img_scale
    cy_s = cy * img_scale
    margin = int(r_s) + 2
    y0, y1 = max(0, int(cy_s) - margin), min(h, int(cy_s) + margin + 1)
    x0, x1 = max(0, int(cx_s) - margin), min(w, int(cx_s) + margin + 1)
    if y0 >= y1 or x0 >= x1:
        return np.nan
    ys, xs = np.arange(y0, y1), np.arange(x0, x1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt((yy - cy_s) ** 2 + (xx - cx_s) ** 2)
    mask = (dist >= r_s * 0.85) & (dist <= r_s * 1.15)
    if not mask.any():
        return np.nan
    return float(np.abs(lmap[yy[mask], xx[mask]]).max())


def background_max(lmap: np.ndarray, img_scale: float, bubbles, rng,
                   disk_r: int = BG_DISK_R, n: int = N_BG) -> list[float]:
    """Max |LoG| in BG_DISK_R-radius disks at random locations > 3R from any bubble."""
    h, w = lmap.shape
    vals = []
    attempts = 0
    while len(vals) < n and attempts < 2000:
        attempts += 1
        py = rng.integers(disk_r, h - disk_r)
        px = rng.integers(disk_r, w - disk_r)
        py_orig = py / img_scale
        px_orig = px / img_scale
        too_close = any(
            np.sqrt((py_orig - b.cy) ** 2 + (px_orig - b.cx) ** 2) < 3 * b.radius
            for b in bubbles
        )
        if not too_close:
            ys = np.arange(py - disk_r, py + disk_r + 1)
            xs = np.arange(px - disk_r, px + disk_r + 1)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            yy = yy.clip(0, h - 1)
            xx = xx.clip(0, w - 1)
            vals.append(float(np.abs(lmap[yy, xx]).max()))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--n-bubbles", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("output/e11_log_discriminability.png"))
    args = parser.parse_args()

    cfg = PipelineConfig()
    canonical_r = cfg.template_size / (2.0 * cfg.template_context_factor)

    all_paths = sorted((Path(args.data_dir) / "images").glob("*.png"))
    tractable = [p for p in all_paths
                 if load_image(p).mean() < MAX_IMG_MEAN]
    print(f"Tractable images: {len(tractable)}/{len(all_paths)}")

    rng = np.random.default_rng(42)
    per_image = max(1, args.n_bubbles // len(tractable))

    delta_cols = np.arange(-DELTA_RANGE, DELTA_RANGE + 1)
    # 4 curves: (center|rim) × (blob|rim sigma)
    keys = ["center_blob", "rim_blob", "center_rim_sigma", "rim_rim_sigma"]
    # per-bubble sensitivity rows
    rows = {k: [] for k in keys}
    # per-delta background samples
    bg_blob      = {d: [] for d in delta_cols}
    bg_rim_sigma = {d: [] for d in delta_cols}

    for img_path in tractable:
        lbl_path = Path(args.data_dir) / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        img = load_image(img_path)
        bubbles = parse_annotations(lbl_path)
        if not bubbles:
            continue

        pyramid = build_pyramid(img, cfg)
        eff_radii  = np.array([er for _, _, er in pyramid])
        img_scales = np.array([s.shape[0] / img.shape[0] for _, s, _ in pyramid])

        valid = [b for b in bubbles
                 if b.radius >= MIN_RADIUS
                 and eff_radii.min() <= b.radius <= eff_radii.max()
                 and b.cy > b.radius * 1.5 and b.cx > b.radius * 1.5
                 and b.cy < img.shape[0] - b.radius * 1.5
                 and b.cx < img.shape[1] - b.radius * 1.5]
        if not valid:
            continue

        log_r = np.log([b.radius for b in valid])
        bins = np.linspace(log_r.min(), log_r.max(), per_image + 1)
        selected = []
        for i in range(per_image):
            in_bin = [b for b, lr in zip(valid, log_r) if bins[i] <= lr < bins[i + 1]]
            if in_bin:
                selected.append(in_bin[len(in_bin) // 2])
        if not selected:
            selected = valid[:per_image]

        print(f"  {img_path.stem[-25:]}  mean={img.mean():.3f}  profiling={len(selected)}")

        # Pre-compute per-level LoG maps for both sigma models
        lmaps_blob     = []
        lmaps_rim_sig  = []
        for _, scaled, _ in pyramid:
            lmaps_blob.append(log_map(scaled, canonical_r * SIGMA_BLOB_FACTOR))
            lmaps_rim_sig.append(log_map(scaled, SIGMA_RIM_CONST))

        for b in selected:
            correct_lv = int(np.argmin(np.abs(eff_radii - b.radius)))

            row = {k: [] for k in keys}
            for d in delta_cols:
                lv = correct_lv + d
                if 0 <= lv < len(pyramid):
                    sc = img_scales[lv]
                    row["center_blob"].append(center_log(lmaps_blob[lv], sc, b.cy, b.cx))
                    row["rim_blob"].append(rim_log(lmaps_blob[lv], sc, b.cy, b.cx, b.radius))
                    row["center_rim_sigma"].append(center_log(lmaps_rim_sig[lv], sc, b.cy, b.cx))
                    row["rim_rim_sigma"].append(rim_log(lmaps_rim_sig[lv], sc, b.cy, b.cx, b.radius))
                else:
                    for k in keys:
                        row[k].append(np.nan)

            for k in keys:
                rows[k].append(row[k])

            # Background: at ALL delta levels
            for d in delta_cols:
                lv = correct_lv + d
                if 0 <= lv < len(pyramid):
                    sc = img_scales[lv]
                    bg_blob[d].extend(
                        background_max(lmaps_blob[lv], sc, bubbles, rng))
                    bg_rim_sigma[d].extend(
                        background_max(lmaps_rim_sig[lv], sc, bubbles, rng))

    n_bubbles = len(rows["center_blob"])
    if n_bubbles == 0:
        print("No data collected.")
        return

    # Convert to arrays
    curves = {k: np.array(rows[k], dtype=np.float32) for k in keys}
    print(f"\nBubbles profiled: {n_bubbles}")

    # ── Text summary ──────────────────────────────────────────────────────────
    print(f"\n{'delta':>5}  {'ctr_blob':>10}  {'rim_blob':>10}  {'ctr_rim':>10}  {'rim_rim':>10}  "
          f"{'bg_blob':>9}  {'bg_rim':>9}")
    for i, d in enumerate(delta_cols):
        bg_b = np.array(bg_blob[d])
        bg_r = np.array(bg_rim_sigma[d])
        print(f"  {d:+3d}  "
              f"{np.nanmean(curves['center_blob'][:, i]):10.4f}  "
              f"{np.nanmean(curves['rim_blob'][:, i]):10.4f}  "
              f"{np.nanmean(curves['center_rim_sigma'][:, i]):10.4f}  "
              f"{np.nanmean(curves['rim_rim_sigma'][:, i]):10.4f}  "
              f"{bg_b.mean() if len(bg_b) else np.nan:9.4f}  "
              f"{bg_r.mean() if len(bg_r) else np.nan:9.4f}")

    # Peak delta per bubble per curve
    print(f"\n{'Curve':<20}  {'mean_peak_delta':>15}  {'median':>8}  {'p25':>6}  {'p75':>6}")
    for k in keys:
        peak_deltas = []
        for row in curves[k]:
            valid = ~np.isnan(row)
            if valid.any():
                peak_deltas.append(delta_cols[np.nanargmax(row)])
        if peak_deltas:
            pd = np.array(peak_deltas)
            print(f"  {k:<18}  {pd.mean():+15.2f}  {np.median(pd):+8.1f}  "
                  f"{np.percentile(pd, 25):+6.1f}  {np.percentile(pd, 75):+6.1f}")

    # SNR at delta=0
    delta0_idx = DELTA_RANGE  # delta_cols[DELTA_RANGE] == 0
    print(f"\nSNR at delta=0 (bubble mean / background mean, symmetric 3px disk):")
    for k, bg_dict in [("center_blob", bg_blob), ("rim_blob", bg_blob),
                       ("center_rim_sigma", bg_rim_sigma), ("rim_rim_sigma", bg_rim_sigma)]:
        bubble_vals = curves[k][:, delta0_idx]
        bubble_vals = bubble_vals[~np.isnan(bubble_vals)]
        bg_vals = np.array(bg_dict[0])
        snr = bubble_vals.mean() / max(bg_vals.mean(), 1e-9) if len(bg_vals) > 0 else np.nan
        print(f"  {k:<22}  bubble={bubble_vals.mean():.4f}  bg={bg_vals.mean():.4f}  SNR={snr:.2f}×")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    viable = []
    for k in keys:
        peak_deltas = []
        for row in curves[k]:
            v = ~np.isnan(row)
            if v.any():
                peak_deltas.append(delta_cols[np.nanargmax(row)])
        if not peak_deltas:
            continue
        med_peak = np.median(peak_deltas)
        bubble_vals = curves[k][:, delta0_idx]
        bubble_vals = bubble_vals[~np.isnan(bubble_vals)]
        bg_dict = bg_blob if "blob" in k else bg_rim_sigma
        bg_vals = np.array(bg_dict[0])
        snr = bubble_vals.mean() / max(bg_vals.mean(), 1e-9) if len(bg_vals) > 0 else 0.0
        if abs(med_peak) <= 1.5 and snr >= 2.0:
            print(f"  VIABLE  {k}: median_peak={med_peak:+.1f}, SNR={snr:.2f}×")
            viable.append(k)
        elif abs(med_peak) > 1.5:
            print(f"  FALSIFIED(sensitivity)  {k}: median_peak={med_peak:+.1f}")
        else:
            print(f"  FALSIFIED(specificity)  {k}: median_peak={med_peak:+.1f}, SNR={snr:.2f}×")
    if not viable:
        print("ALL four curves falsified → LoG is not a viable feature for this dataset.")
    print(f"{'='*60}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {"center_blob": "steelblue", "rim_blob": "tomato",
              "center_rim_sigma": "seagreen", "rim_rim_sigma": "darkorange"}
    labels = {"center_blob": "center, σ=R/√2 (blob)",
              "rim_blob": "rim, σ=R/√2 (blob)",
              "center_rim_sigma": "center, σ=2px (ring)",
              "rim_rim_sigma": "rim, σ=2px (ring)"}

    # Row 0: sensitivity curves, blob sigma
    for ax, k_pair, title in [
        (axes[0, 0], ["center_blob", "rim_blob"], "Sensitivity — σ=R/√2 (blob model)"),
        (axes[0, 1], ["center_rim_sigma", "rim_rim_sigma"], "Sensitivity — σ=2px (ring model)"),
    ]:
        for k in k_pair:
            mean_c = np.nanmean(curves[k], axis=0)
            std_c  = np.nanstd(curves[k], axis=0)
            ax.plot(delta_cols, mean_c, "o-", color=colors[k], linewidth=2,
                    markersize=5, label=labels[k])
            ax.fill_between(delta_cols, mean_c - std_c, mean_c + std_c,
                            alpha=0.15, color=colors[k])
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="correct level")
        ax.set_xlabel("delta (level − correct)")
        ax.set_ylabel("|LoG| (scale-normalised)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 0 col 2: SNR(delta) curves
    ax = axes[0, 2]
    for k, bg_dict in [("center_blob", bg_blob), ("rim_rim_sigma", bg_rim_sigma)]:
        snr_curve = []
        for d in delta_cols:
            bubble_col = curves[k][:, list(delta_cols).index(d)]
            bubble_col = bubble_col[~np.isnan(bubble_col)]
            bg_arr = np.array(bg_dict[d])
            snr = bubble_col.mean() / max(bg_arr.mean(), 1e-9) if len(bg_arr) else np.nan
            snr_curve.append(snr)
        ax.plot(delta_cols, snr_curve, "o-", color=colors[k], linewidth=2,
                markersize=5, label=labels[k])
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1, label="SNR=2 threshold")
    ax.set_xlabel("delta")
    ax.set_ylabel("SNR (bubble mean / bg mean)")
    ax.set_title("SNR(delta) — key curves only")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: per-bubble peak-delta histograms
    for ax, k_pair, title in [
        (axes[1, 0], ["center_blob", "rim_blob"], "Peak delta — σ=R/√2"),
        (axes[1, 1], ["center_rim_sigma", "rim_rim_sigma"], "Peak delta — σ=2px"),
    ]:
        for k in k_pair:
            pd = [delta_cols[np.nanargmax(row)]
                  for row in curves[k] if not np.all(np.isnan(row))]
            if pd:
                ax.hist(pd, bins=delta_cols - 0.5, alpha=0.5, color=colors[k],
                        label=labels[k], edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("delta at peak |LoG|")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 1 col 2: specificity at delta=0 — best viable or center_blob
    ax = axes[1, 2]
    show_k = viable[0] if viable else "center_blob"
    bg_dict = bg_blob if "blob" in show_k else bg_rim_sigma
    bubble_vals = curves[show_k][:, delta0_idx]
    bubble_vals = bubble_vals[~np.isnan(bubble_vals)]
    bg_vals = np.array(bg_dict[0])
    ax.hist(bg_vals, bins=40, density=True, alpha=0.6, color="salmon",
            label=f"Background ({len(bg_vals)} samples)")
    ax.hist(bubble_vals, bins=40, density=True, alpha=0.6, color=colors[show_k],
            label=f"Bubble ({len(bubble_vals)}, {labels[show_k]})")
    ax.set_xlabel("|LoG| at delta=0")
    ax.set_ylabel("Density")
    snr_d0 = bubble_vals.mean() / max(bg_vals.mean(), 1e-9)
    ax.set_title(f"Specificity at delta=0\n{labels[show_k]}  SNR={snr_d0:.2f}×")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"E11 Step 2 — Corrected LoG discriminability ({n_bubbles} bubbles, "
                 f"r≥{MIN_RADIUS}px)", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
