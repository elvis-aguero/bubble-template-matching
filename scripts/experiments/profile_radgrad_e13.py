#!/usr/bin/env python3
"""
E13 — Radial gradient SNR at the bubble rim (patch-centered discriminability test).

Hypothesis to falsify:
  "The inward radial gradient at the bubble rim annulus (r/R ∈ [0.85, 1.15]) is
   not significantly larger than the same metric at random background locations of
   the same radius. SNR < 2× → dark-rim edge provides no useful discriminative
   signal beyond background gradient noise."

Design:
  - For each GT bubble (radius ≥ 8px), in each tractable image (img.mean < 0.6):
      * Compute Scharr gradient magnitude and direction on the full image.
      * For each rim pixel (r/R ∈ [0.85, 1.15]): compute the dot product of the
        gradient vector with the inward unit radial vector. Positive → gradient
        points inward (dark-inside edge) or outward with negative sign.
      * Inward radial gradient score = mean of these dot products over the annulus.
  - Background: same metric evaluated at N_BG random centres per bubble that are
      > 3R from any annotated bubble and have the same R value (matched radius).
  - SNR = mean(bubble scores) / mean(background scores), where both are abs-valued.
  - Stratify by:
      * Photometric regime: img.mean() quartiles across the dataset (Q1–Q4)
      * Morphology proxy: bubble score > 0 → "dark-rim" (inward gradient); ≤ 0 → other
      * Size bin: small (r < 12px), medium (12–24px), large (>24px)

Pre-committed falsification criteria:
  1. SNR < 2× across ALL strata → gradient edge is not viable; handcrafted features
     exhausted; CNN-A / CNN-B are the only remaining paths.
  2. SNR ≥ 2× in dark-rim proxy stratum → viable for 54% of bubbles; design
     radial gradient pipeline (E14).
  3. SNR ≥ 2× in both dark-rim AND filled-dark strata → broadly viable.

Note: SNR is computed separately for dark-rim (score > 0) and non-dark-rim strata.
Background is always the absolute value of the radial gradient score (no sign assumption).

USAGE: python scripts/experiments/profile_radgrad_e13.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import scharr_h, scharr_v

from bubble_histogram.data import load_image, parse_annotations

MAX_IMG_MEAN = 0.6
MIN_RADIUS   = 8.0
N_BG         = 30    # background samples per bubble
RNG_SEED     = 42
RIM_INNER    = 0.85
RIM_OUTER    = 1.15
BG_MIN_DIST  = 3.0   # multiples of R for background exclusion zone


def scharr_gradient(img: np.ndarray):
    """Return (gx, gy) float64 gradient via Scharr operator."""
    img_f = img.astype(np.float64)
    gx = scharr_v(img_f).astype(np.float64)  # ∂I/∂x (column direction)
    gy = scharr_h(img_f).astype(np.float64)  # ∂I/∂y (row direction)
    return gx, gy


def inward_radial_score(gx: np.ndarray, gy: np.ndarray,
                        cx: float, cy: float, R: float) -> float:
    """
    Mean inward radial gradient in annulus r/R ∈ [RIM_INNER, RIM_OUTER].

    Inward direction at (px, py) = -(px-cx, py-cy) / r.
    Dot with gradient (gx, gy) → positive if gradient points inward.
    Returns NaN if annulus has no pixels.
    """
    H, W = gx.shape
    r_inner = RIM_INNER * R
    r_outer = RIM_OUTER * R

    # bounding box for the annulus
    r0 = max(0, int(np.floor(cy - r_outer)))
    r1 = min(H, int(np.ceil(cy + r_outer)) + 1)
    c0 = max(0, int(np.floor(cx - r_outer)))
    c1 = min(W, int(np.ceil(cx + r_outer)) + 1)

    py = np.arange(r0, r1, dtype=np.float64)
    px = np.arange(c0, c1, dtype=np.float64)
    PX, PY = np.meshgrid(px, py)

    dx = PX - cx
    dy = PY - cy
    dist = np.sqrt(dx**2 + dy**2)

    mask = (dist >= r_inner) & (dist < r_outer) & (dist > 0)
    if not mask.any():
        return np.nan

    # inward unit radial vector: -(dx, dy) / dist
    ix = -dx[mask] / dist[mask]
    iy = -dy[mask] / dist[mask]

    gx_rim = gx[r0:r1, c0:c1][mask]
    gy_rim = gy[r0:r1, c0:c1][mask]

    dot = gx_rim * ix + gy_rim * iy
    return float(np.mean(dot))


def sample_background(gx: np.ndarray, gy: np.ndarray,
                      bubbles, R: float, n: int, rng) -> list:
    """Sample n background inward-radial scores at matched radius R."""
    H, W = gx.shape
    margin = int(np.ceil(RIM_OUTER * R)) + 1
    scores = []
    attempts = 0
    while len(scores) < n and attempts < n * 200:
        attempts += 1
        cx = rng.uniform(margin, W - margin)
        cy = rng.uniform(margin, H - margin)
        # reject if too close to any GT bubble
        too_close = any(
            np.sqrt((b.cx - cx)**2 + (b.cy - cy)**2) < BG_MIN_DIST * b.radius
            for b in bubbles
        )
        if too_close:
            continue
        s = inward_radial_score(gx, gy, cx, cy, R)
        if not np.isnan(s):
            scores.append(s)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out", type=Path,
                        default=Path("output/e13_radgrad_snr.png"))
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    data_dir = args.data_dir

    img_means = []
    for img_path in sorted((data_dir / "images").glob("*.png")):
        lbl_path = data_dir / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        img = load_image(img_path)
        if img.mean() < MAX_IMG_MEAN:
            img_means.append(img.mean())

    q1, q3 = np.percentile(img_means, [25, 75])
    med = np.median(img_means)

    def regime(mean_val):
        if mean_val < q1:
            return "Q1 (dark)"
        elif mean_val < med:
            return "Q2"
        elif mean_val < q3:
            return "Q3"
        else:
            return "Q4 (bright)"

    def size_bin(r):
        if r < 12:
            return "small(<12)"
        elif r < 24:
            return "medium(12-24)"
        else:
            return "large(≥24)"

    bubble_scores = []
    bg_scores_all = []

    img_paths = sorted((data_dir / "images").glob("*.png"))
    n_processed = 0
    for img_path in img_paths:
        lbl_path = data_dir / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        img = load_image(img_path)
        if img.mean() >= MAX_IMG_MEAN:
            continue
        bubbles = parse_annotations(lbl_path)
        if not bubbles:
            continue

        n_processed += 1
        gx, gy = scharr_gradient(img)
        reg = regime(img.mean())

        valid_bubbles = [b for b in bubbles if b.radius >= MIN_RADIUS]

        for b in valid_bubbles:
            s = inward_radial_score(gx, gy, b.cx, b.cy, b.radius)
            if np.isnan(s):
                continue
            # score > 0 → net inward gradient at rim (zero-crossing tail, NOT E11 dark-rim)
            # score < 0 → net outward gradient at rim (dominant majority, 91%)
            morpho = "inward-marginal" if s > 0 else "outward-dominant"
            bubble_scores.append({
                "score": s,
                "abs_score": abs(s),
                "radius": b.radius,
                "regime": reg,
                "size_bin": size_bin(b.radius),
                "morpho": morpho,
                "stem": img_path.stem,
            })

        # background at distribution of valid bubble radii
        for b in valid_bubbles:
            bg = sample_background(gx, gy, bubbles, b.radius, N_BG, rng)
            bg_scores_all.extend(abs(x) for x in bg)

    print(f"\nProcessed {n_processed} images")
    print(f"GT bubbles (r≥8px) measured: {len(bubble_scores)}")
    print(f"Background samples: {len(bg_scores_all)}")

    if not bubble_scores:
        print("ERROR: no bubble scores computed — check data path")
        return

    all_bubble_abs = np.array([r["abs_score"] for r in bubble_scores])
    all_bg_abs     = np.array(bg_scores_all) if bg_scores_all else np.array([0.0])

    mean_bub = float(np.mean(all_bubble_abs))
    mean_bg  = float(np.mean(all_bg_abs))
    overall_snr = mean_bub / mean_bg if mean_bg > 0 else float("inf")

    print(f"\n{'='*70}")
    print("OVERALL SNR")
    print(f"  Mean |bubble radgrad|:     {mean_bub:.4f}")
    print(f"  Mean |background radgrad|: {mean_bg:.4f}")
    print(f"  SNR = {overall_snr:.2f}×  (criterion: ≥ 2×)")

    # ── Stratified SNR ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STRATIFIED BY MORPHOLOGY PROXY (score>0 = 'inward-marginal'; NOT E11 dark-rim)")
    print("  NOTE: proxy selects zero-crossing tail (weakest signal), not 54% dark-rim from E11.")
    for morph in ["inward-marginal", "outward-dominant"]:
        vals = [r["abs_score"] for r in bubble_scores if r["morpho"] == morph]
        n = len(vals)
        if n == 0:
            continue
        m = float(np.mean(vals))
        snr = m / mean_bg if mean_bg > 0 else float("inf")
        frac = n / len(bubble_scores)
        flag = "PASS ≥2×" if snr >= 2.0 else "FAIL <2×"
        print(f"  {morph:<12}  n={n:4d} ({frac:.0%})  mean={m:.4f}  SNR={snr:.2f}×  [{flag}]")

    print(f"\n{'='*70}")
    print("STRATIFIED BY SIZE BIN")
    for sb in ["small(<12)", "medium(12-24)", "large(≥24)"]:
        vals = [r["abs_score"] for r in bubble_scores if r["size_bin"] == sb]
        if not vals:
            continue
        m = float(np.mean(vals))
        snr = m / mean_bg if mean_bg > 0 else float("inf")
        flag = "PASS ≥2×" if snr >= 2.0 else "FAIL <2×"
        print(f"  {sb:<18}  n={len(vals):4d}  mean={m:.4f}  SNR={snr:.2f}×  [{flag}]")

    print(f"\n{'='*70}")
    print("STRATIFIED BY PHOTOMETRIC REGIME")
    for reg in sorted(set(r["regime"] for r in bubble_scores)):
        vals = [r["abs_score"] for r in bubble_scores if r["regime"] == reg]
        if not vals:
            continue
        m = float(np.mean(vals))
        snr = m / mean_bg if mean_bg > 0 else float("inf")
        flag = "PASS ≥2×" if snr >= 2.0 else "FAIL <2×"
        print(f"  {reg:<14}  n={len(vals):4d}  mean={m:.4f}  SNR={snr:.2f}×  [{flag}]")

    # ── Score distributions ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RAW SCORE DISTRIBUTION (bubble inward radgrad, signed):")
    signed = np.array([r["score"] for r in bubble_scores])
    print(f"  Dark-rim fraction (score>0): {(signed>0).mean():.1%}")
    print(f"  Mean:   {signed.mean():+.4f}  Median: {np.median(signed):+.4f}")
    print(f"  Std:    {signed.std():.4f}   IQR: {np.percentile(signed,75)-np.percentile(signed,25):.4f}")
    print(f"  p5:     {np.percentile(signed,5):+.4f}  p95: {np.percentile(signed,95):+.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT:")
    inward_snr = float("nan")
    dr_vals = [r["abs_score"] for r in bubble_scores if r["morpho"] == "inward-marginal"]
    if dr_vals and mean_bg > 0:
        inward_snr = np.mean(dr_vals) / mean_bg
    outward_vals = [r["abs_score"] for r in bubble_scores if r["morpho"] == "outward-dominant"]
    outward_snr = (np.mean(outward_vals) / mean_bg) if (outward_vals and mean_bg > 0) else float("nan")

    # Large-bubble stratum check (critical: apparatus proximity may inflate bg for large R)
    large_inward = [r["abs_score"] for r in bubble_scores
                    if r["morpho"] == "inward-marginal" and r["size_bin"] == "large(≥24)"]
    large_outward = [r["abs_score"] for r in bubble_scores
                     if r["morpho"] == "outward-dominant" and r["size_bin"] == "large(≥24)"]
    large_inward_snr = (np.mean(large_inward) / mean_bg) if (large_inward and mean_bg > 0) else float("nan")
    large_outward_snr = (np.mean(large_outward) / mean_bg) if (large_outward and mean_bg > 0) else float("nan")

    if overall_snr >= 2.0:
        print(f"  Overall SNR = {overall_snr:.2f}× ≥ 2×: gradient edge signal is PRESENT overall.")
    else:
        print(f"  Overall SNR = {overall_snr:.2f}× < 2×: gradient edge signal is MARGINAL overall.")

    print(f"\n  Morphology proxy (NOT equivalent to E11 dark-rim classification):")
    print(f"    inward-marginal (score>0)   n={len(dr_vals):4d} ({len(dr_vals)/len(bubble_scores):.0%})  SNR={inward_snr:.2f}×  "
          f"[{'PASS ≥2×' if inward_snr >= 2.0 else 'FAIL <2×'}]")
    print(f"    outward-dominant (score<0)  n={len(outward_vals):4d} ({len(outward_vals)/len(bubble_scores):.0%})  SNR={outward_snr:.2f}×  "
          f"[{'PASS ≥2×' if outward_snr >= 2.0 else 'FAIL <2×'}]")

    print(f"\n  Large-bubble (≥24px) sub-strata (explicit check per PAL review):")
    print(f"    large × inward-marginal:    n={len(large_inward):3d}  SNR={large_inward_snr:.2f}×  "
          f"[{'PASS ≥2×' if large_inward_snr >= 2.0 else 'FAIL <2× — HIDDEN FAILURE'}]")
    print(f"    large × outward-dominant:   n={len(large_outward):3d}  SNR={large_outward_snr:.2f}×  "
          f"[{'PASS ≥2×' if large_outward_snr >= 2.0 else 'FAIL <2× — HIDDEN FAILURE'}]")

    both_strata_pass = inward_snr >= 2.0 and outward_snr >= 2.0
    if both_strata_pass:
        print(f"\n  CRITERION 3 MET (broad viability): both morphology proxy strata ≥ 2×.")
        print(f"  Signal is driven by outward-dominant majority (91%, SNR {outward_snr:.2f}×).")
        print(f"  Inward-marginal stratum (8.8%) marginal at {inward_snr:.2f}× — large-bubble sub-stratum may fail.")
        print(f"  E14 should threshold on |radial gradient score|, NOT on score > 0.")
    elif inward_snr >= 2.0 or outward_snr >= 2.0:
        print(f"\n  CRITERION 2 MET (partial viability): one stratum ≥ 2×.")
    else:
        print("\n  FALSIFIED: SNR < 2× in all tested strata. Handcrafted gradient features not viable.")
        print("  Path forward: CNN-A or CNN-B.")

    if not (overall_snr >= 2.0):
        print("  FALSIFIED: Overall SNR < 2×. Handcrafted gradient features not viable.")
        print("  Path forward: CNN-A or CNN-B.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. |Score| histogram: bubbles vs background (both abs for fair comparison)
    ax = axes[0]
    ax.hist(np.array([r["abs_score"] for r in bubble_scores]),
            bins=50, alpha=0.7, color="steelblue", label=f"bubble |score| (n={len(bubble_scores)})", density=True)
    bg_abs = np.array(bg_scores_all) if bg_scores_all else np.array([0.0])
    ax.hist(bg_abs, bins=50, alpha=0.6, color="tomato",
            label=f"background |score| (n={len(bg_scores_all)})", density=True)
    ax.axvline(mean_bub, color="steelblue", linewidth=1.2, linestyle="--",
               label=f"bubble mean={mean_bub:.3f}")
    ax.axvline(mean_bg, color="tomato", linewidth=1.2, linestyle="--",
               label=f"bg mean={mean_bg:.3f}")
    ax.set_xlabel("|Inward radial gradient score|")
    ax.set_ylabel("Density")
    ax.set_title(f"|Score| distribution: bubble vs background\n(SNR={overall_snr:.2f}×)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. SNR by regime
    ax = axes[1]
    regimes = sorted(set(r["regime"] for r in bubble_scores))
    snrs = []
    for reg in regimes:
        vals = [r["abs_score"] for r in bubble_scores if r["regime"] == reg]
        snrs.append(np.mean(vals) / mean_bg if (vals and mean_bg > 0) else 0.0)
    colors = ["steelblue" if s >= 2.0 else "tomato" for s in snrs]
    ax.bar(range(len(regimes)), snrs, color=colors, alpha=0.8)
    ax.axhline(2.0, color="green", linestyle="--", linewidth=1.5, label="2× threshold")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("SNR")
    ax.set_title("SNR by photometric regime")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. SNR by morphology + size
    ax = axes[2]
    labels, vals_list = [], []
    for morph in ["inward-marginal", "outward-dominant"]:
        for sb in ["small(<12)", "medium(12-24)", "large(≥24)"]:
            v = [r["abs_score"] for r in bubble_scores
                 if r["morpho"] == morph and r["size_bin"] == sb]
            if v:
                labels.append(f"{morph}\n{sb}")
                vals_list.append(np.mean(v) / mean_bg if mean_bg > 0 else 0.0)
    bar_colors = ["steelblue" if s >= 2.0 else "tomato" for s in vals_list]
    ax.bar(range(len(labels)), vals_list, color=bar_colors, alpha=0.8)
    ax.axhline(2.0, color="green", linestyle="--", linewidth=1.5, label="2× threshold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("SNR")
    ax.set_title("SNR by morphology × size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"E13 — Radial gradient SNR  "
        f"(overall SNR={overall_snr:.2f}×, {len(bubble_scores)} bubbles, {n_processed} images)",
        fontsize=11
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
