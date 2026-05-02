#!/usr/bin/env python3
"""
E11 Step 1 — Bubble morphology cross-section survey.

Determines whether bubbles are filled dark discs, dark-rim rings, or bright-rim
rings. This result dictates the correct sigma for E11 Step 2 (corrected LoG test):
  - Filled Gaussian disc → sigma = bubble.radius / sqrt(2)
  - Ring (rim response) → sigma = rim_width / sqrt(2), independent of ring radius

Outputs:
  - Radial intensity profiles (mean intensity vs r/bubble.radius, 0=center, 1=edge)
    stratified by size bin and photometric regime
  - Raw intensity cross-sections (horizontal + vertical) for a handful of bubbles

USAGE: python scripts/experiments/profile_morphology.py [data_dir]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.data import AnnotatedDataset, load_image
from bubble_histogram.data import parse_annotations


MAX_IMG_MEAN = 0.6      # float [0,1] threshold; excludes saturated/dead frames
N_RADIAL_BINS = 20      # bins from r=0 to r=2 (0=center, 1=bubble edge, 2=far bg)
RADIAL_MAX = 2.0        # profile extends to 2× bubble.radius
N_PER_SIZE_BIN = 5      # bubbles to sample per size bin per image (if available)


def radial_profile(img: np.ndarray, cy: float, cx: float, radius: float,
                   n_bins: int = N_RADIAL_BINS,
                   r_max: float = RADIAL_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity vs normalised radius r/bubble.radius, from 0 to r_max."""
    h, w = img.shape
    margin = int(radius * r_max) + 2
    y0 = max(0, int(cy) - margin)
    y1 = min(h, int(cy) + margin + 1)
    x0 = max(0, int(cx) - margin)
    x1 = min(w, int(cx) + margin + 1)

    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist_norm = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius

    patch = img[y0:y1, x0:x1]
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (dist_norm >= bin_edges[i]) & (dist_norm < bin_edges[i + 1])
        if mask.any():
            means[i] = patch[mask].mean()
    return bin_centers, means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out", type=Path,
                        default=Path("output/e11_morphology_profiles.png"))
    parser.add_argument("--n-bubbles", type=int, default=40,
                        help="Total bubbles to profile (sampled across all images)")
    args = parser.parse_args()

    all_paths = sorted((Path(args.data_dir) / "images").glob("*.png"))
    tractable = []
    for p in all_paths:
        img = load_image(p)
        if img.mean() < MAX_IMG_MEAN:
            tractable.append((p, img.mean()))
    print(f"Tractable images: {len(tractable)}/{len(all_paths)}")

    rng = np.random.default_rng(42)
    per_image = max(1, args.n_bubbles // len(tractable))

    profiles = []   # list of dicts: r, intensity, radius, regime_mean

    for img_path, img_mean in tractable:
        lbl_path = Path(args.data_dir) / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        img = load_image(img_path)
        bubbles = parse_annotations(lbl_path)
        if not bubbles:
            continue

        # Filter to bubbles well inside the image
        valid = [b for b in bubbles
                 if b.cy > b.radius * 2 and b.cx > b.radius * 2
                 and b.cy < img.shape[0] - b.radius * 2
                 and b.cx < img.shape[1] - b.radius * 2
                 and b.radius >= 4.0]
        if not valid:
            continue

        # Sample evenly across log-radius space
        log_r = np.array([np.log(b.radius) for b in valid])
        bins = np.linspace(log_r.min(), log_r.max(), per_image + 1)
        selected = []
        for i in range(per_image):
            in_bin = [b for b, lr in zip(valid, log_r)
                      if bins[i] <= lr < bins[i + 1]]
            if in_bin:
                selected.append(in_bin[len(in_bin) // 2])
        if not selected:
            selected = valid[:per_image]

        for b in selected:
            r_vals, intensity = radial_profile(img, b.cy, b.cx, b.radius)
            profiles.append({
                "r": r_vals,
                "intensity": intensity,
                "radius": b.radius,
                "img_mean": img_mean,
                "name": img_path.stem[-20:],
            })

    print(f"Total profiles: {len(profiles)}")

    if not profiles:
        print("No data collected.")
        return

    # ── Classify by size bin ──────────────────────────────────────────────────
    radii = np.array([p["radius"] for p in profiles])
    q33, q66 = np.percentile(radii, 33), np.percentile(radii, 66)
    size_bins = [
        (radii < q33,         f"small  r<{q33:.1f}px",  "steelblue"),
        ((radii >= q33) & (radii < q66), "medium",       "seagreen"),
        (radii >= q66,        f"large  r>{q66:.1f}px",   "tomato"),
    ]

    # ── Text summary ──────────────────────────────────────────────────────────
    r_vals_ref = profiles[0]["r"]
    center_bin = np.argmin(np.abs(r_vals_ref - 0.1))   # ~center
    rim_bin    = np.argmin(np.abs(r_vals_ref - 0.85))  # ~rim
    bg_bin     = np.argmin(np.abs(r_vals_ref - 1.5))   # outside

    print(f"\n{'Morphology summary':}")
    print(f"  Radial positions sampled: center≈r{r_vals_ref[center_bin]:.2f}, "
          f"rim≈r{r_vals_ref[rim_bin]:.2f}, bg≈r{r_vals_ref[bg_bin]:.2f}")
    print(f"\n  {'name':<22}  {'r_px':>6}  {'center':>8}  {'rim':>8}  {'bg':>8}  {'type'}")
    for p in profiles:
        c = p["intensity"][center_bin]
        ri = p["intensity"][rim_bin]
        bg = p["intensity"][bg_bin]
        if np.isnan(c) or np.isnan(ri) or np.isnan(bg):
            morph = "?"
        elif ri < c and ri < bg:
            morph = "dark-rim"
        elif ri > c and ri > bg:
            morph = "bright-rim"
        elif c < bg * 0.85:
            morph = "filled-dark"
        elif c > bg * 1.15:
            morph = "filled-bright"
        else:
            morph = "flat"
        print(f"  {p['name']:<22}  {p['radius']:>6.1f}  "
              f"{c:>8.3f}  {ri:>8.3f}  {bg:>8.3f}  {morph}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: mean radial profiles by size bin
    ax = axes[0]
    for mask, label, color in size_bins:
        subset = [p for p, m in zip(profiles, mask) if m]
        if not subset:
            continue
        stack = np.vstack([p["intensity"] for p in subset])
        mean = np.nanmean(stack, axis=0)
        std  = np.nanstd(stack, axis=0)
        r    = subset[0]["r"]
        ax.plot(r, mean, "-o", color=color, linewidth=2, markersize=4, label=label)
        ax.fill_between(r, mean - std, mean + std, alpha=0.15, color=color)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="bubble edge (r=1)")
    ax.set_xlabel("r / bubble.radius")
    ax.set_ylabel("Mean intensity (float)")
    ax.set_title(f"Radial intensity profile by size bin\n({len(profiles)} bubbles, {len(tractable)} images)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Middle: individual profiles coloured by img_mean (photometric regime)
    ax = axes[1]
    img_means = np.array([p["img_mean"] for p in profiles])
    vmin, vmax = img_means.min(), img_means.max()
    cmap = plt.cm.viridis
    for p in profiles:
        c = cmap((p["img_mean"] - vmin) / max(vmax - vmin, 1e-6))
        ax.plot(p["r"], p["intensity"], "-", color=c, linewidth=0.8, alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="image mean (float)")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("r / bubble.radius")
    ax.set_ylabel("Intensity (float)")
    ax.set_title("Individual profiles coloured by photometric regime")
    ax.grid(True, alpha=0.3)

    # Right: scatter — center intensity vs rim intensity
    ax = axes[2]
    centers = np.array([p["intensity"][center_bin] for p in profiles])
    rims    = np.array([p["intensity"][rim_bin]    for p in profiles])
    bgs     = np.array([p["intensity"][bg_bin]     for p in profiles])
    valid_mask = ~(np.isnan(centers) | np.isnan(rims) | np.isnan(bgs))
    sc = ax.scatter(centers[valid_mask], rims[valid_mask], c=bgs[valid_mask],
                    cmap="plasma", s=30, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="bg intensity (r≈1.5)")
    lims = [min(centers[valid_mask].min(), rims[valid_mask].min()) - 0.02,
            max(centers[valid_mask].max(), rims[valid_mask].max()) + 0.02]
    ax.plot(lims, lims, "k--", linewidth=1, label="center=rim")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Center intensity (r≈0.1)")
    ax.set_ylabel("Rim intensity (r≈0.85)")
    ax.set_title("Morphology scatter: above diagonal = bright rim,\nbelow = dark rim")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
