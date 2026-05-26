#!/usr/bin/env python3
"""
E14 — Phase congruency + Loy-Zelinsky radial symmetry transform.

phasepack is unavailable; E13 confirmed Scharr gradient magnitude has SNR=6.86× at
bubble rims. Gradient magnitude is used as the FRST input directly — this is the
signal the transform needs, and we know it exists.

Algorithm (Loy & Zelinsky 2003 — symmetric mode):
  For each image pixel with gradient (gx, gy):
    - compute gradient direction: d = (gx, gy) / ||(gx, gy)||
    - for each candidate radius r:
        vote for a center at pixel ± r*d (both directions)
  Accumulate votes in an orientation-projection map O_r and magnitude-projection map M_r.
  Response F_r = (|O_r|/kn)^alpha * (M_r/kn), smoothed with Gaussian(sigma=r*0.5).
  Symmetric mode: BOTH positive and negative orientation points contribute to |O_r|,
  so the transform responds to any circular boundary regardless of polarity.

Pipeline:
  1. Scharr gradient → FRST over all bubble radii (8–50px, 1/0.9 spacing ~18 levels)
  2. Per-radius: find 2D local maxima above a threshold
  3. Cross-radius NMS: suppress detections within max(r1,r2) pixels of a higher-response
     detection at any other radius (avoids multi-scale duplicate problem)
  4. Build size histogram → relL1 vs GT

Evaluation:
  - Per-image oracle relL1: find threshold that minimises relL1 for each image
    individually (ceiling of what FRST can achieve with ideal thresholding)
  - LOSO relL1: threshold tuned on the training fold, applied to held-out image

Hypothesis to falsify:
  "FRST does not improve relL1 below the NCC pipeline (0.851) in LOSO, because
   non-circular apparatus structures dominate the response map."

USAGE: python scripts/experiments/detect_frst_e14.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.filters import scharr_h, scharr_v

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import load_image, parse_annotations

MAX_IMG_MEAN = 0.6
MIN_RADIUS   = 8.0
FRST_ALPHA   = 2       # exponent for orientation term; 2 = standard
BG_THRESH    = 1e-6    # ignore pixels below this gradient magnitude


def scharr_gradient(img: np.ndarray):
    img_f = img.astype(np.float64)
    gx = scharr_v(img_f).astype(np.float64)
    gy = scharr_h(img_f).astype(np.float64)
    return gx, gy


def make_radii(cfg: PipelineConfig) -> np.ndarray:
    """Log-spaced radii from MIN_RADIUS to cfg.max_radius, step = 1/scale_factor."""
    step = 1.0 / cfg.scale_factor  # ~1.111
    radii = []
    r = MIN_RADIUS
    while r <= cfg.max_radius * 1.001:
        radii.append(r)
        r *= step
    return np.array(radii)


def frst(gx: np.ndarray, gy: np.ndarray, radii: np.ndarray,
         alpha: int = FRST_ALPHA) -> np.ndarray:
    """
    Fast Radial Symmetry Transform (symmetric mode).
    Returns response[n_radii, H, W].
    """
    H, W = gx.shape
    mag = np.sqrt(gx**2 + gy**2)
    eps = 1e-8
    dx = gx / (mag + eps)
    dy = gy / (mag + eps)

    # Only pixels with meaningful gradient vote; flat regions produce random
    # direction noise that inflates FP density uniformly.
    active = mag > BG_THRESH

    ys, xs = np.mgrid[0:H, 0:W]
    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64)

    # Pre-filter to active pixel coordinates (faster np.add.at on sparse arrays)
    ay = ys[active].ravel()
    ax = xs[active].ravel()
    adx = dx[active].ravel()
    ady = dy[active].ravel()
    amag = mag[active].ravel()

    response = np.zeros((len(radii), H, W), dtype=np.float32)

    for ri, r in enumerate(radii):
        O = np.zeros((H, W), dtype=np.float64)
        M = np.zeros((H, W), dtype=np.float64)

        # positive orientation point: pixel votes for center at p + r*d
        px = np.round(ax + r * adx).astype(np.int32)
        py = np.round(ay + r * ady).astype(np.int32)
        vp = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        np.add.at(O, (py[vp], px[vp]), 1)
        np.add.at(M, (py[vp], px[vp]), amag[vp])

        # negative orientation point: pixel votes for center at p - r*d
        nx = np.round(ax - r * adx).astype(np.int32)
        ny = np.round(ay - r * ady).astype(np.int32)
        vn = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        np.add.at(O, (ny[vn], nx[vn]), -1)
        np.add.at(M, (ny[vn], nx[vn]), amag[vn])

        # Normalise by circumference so response is scale-invariant.
        # kn = floor(2πr): expected number of rim pixels at radius r.
        # Without this, large bubbles get r^3 higher response than small ones.
        kn = max(1, int(2 * np.pi * r))
        F = (np.abs(O) / kn) ** alpha * (M / kn)

        # smooth over ~r/2 pixels so the peak centres on the bubble centre
        sigma = max(1.0, r * 0.5)
        response[ri] = gaussian_filter(F, sigma=sigma).astype(np.float32)

    return response


def local_maxima_2d(arr: np.ndarray, min_dist: int) -> np.ndarray:
    """Return boolean mask of local maxima with neighbourhood radius min_dist."""
    footprint = np.ones((2 * min_dist + 1, 2 * min_dist + 1))
    return (arr == maximum_filter(arr, footprint=footprint)) & (arr > 0)


def all_candidates(response: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Precompute ALL local maxima across all radii (threshold=0).
    Returns float32 array of shape (N, 4): [cx, cy, r, score], sorted by score desc.
    Call once per image; filter by score at query time instead of re-running NMS.
    """
    raw = []
    for ri, r in enumerate(radii):
        layer = response[ri]
        min_dist = max(1, int(round(r * 0.5)))
        lm_mask = local_maxima_2d(layer, min_dist)
        ys, xs = np.where(lm_mask)
        for cy, cx in zip(ys, xs):
            raw.append((float(cx), float(cy), float(r), float(layer[cy, cx])))
    if not raw:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.array(raw, dtype=np.float32)
    return arr[arr[:, 3].argsort()[::-1]]  # sorted by score descending


def apply_nms(candidates: np.ndarray) -> list:
    """Cross-radius NMS on a pre-sorted candidate array."""
    kept = []
    suppressed = set()
    for i in range(len(candidates)):
        if i in suppressed:
            continue
        cx, cy, r, sc = candidates[i]
        kept.append((float(cx), float(cy), float(r), float(sc)))
        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue
            cx2, cy2, r2, _ = candidates[j]
            if np.sqrt((cx - cx2)**2 + (cy - cy2)**2) < max(r, r2):
                suppressed.add(j)
    return kept


def find_detections(candidates: np.ndarray, threshold: float) -> list:
    """Filter pre-computed candidates by threshold then apply cross-radius NMS."""
    above = candidates[candidates[:, 3] >= threshold]
    if len(above) == 0:
        return []
    return apply_nms(above)


def build_histogram(detections: list, radii: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(radii))
    for cx, cy, r, sc in detections:
        idx = int(np.argmin(np.abs(radii - r)))
        counts[idx] += 1
    return counts


def gt_histogram(bubbles, radii: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(radii))
    # Use Voronoi boundaries on the log-spaced grid so only bubbles that FRST
    # can actually detect are counted in the denominator.
    step = (radii[1] / radii[0]) if len(radii) > 1 else (1.0 / 0.9)
    lo = radii[0] / step ** 0.5
    hi = radii[-1] * step ** 0.5
    for b in bubbles:
        if b.radius < lo or b.radius > hi:
            continue
        idx = int(np.argmin(np.abs(np.log(radii) - np.log(b.radius))))
        counts[idx] += 1
    return counts


def rel_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    total = gt.sum()
    if total == 0:
        return np.nan
    return float(np.abs(pred - gt).sum() / total)


def oracle_threshold(candidates: np.ndarray, gt_hist: np.ndarray,
                     radii: np.ndarray, thresholds: np.ndarray) -> tuple:
    """Find threshold minimising relL1 against GT histogram."""
    best_rl1, best_thr, best_det = float("inf"), thresholds[0], []
    for thr in thresholds:
        det = find_detections(candidates, thr)
        pred = build_histogram(det, radii)
        rl1 = rel_l1(pred, gt_hist)
        if rl1 < best_rl1:
            best_rl1, best_thr, best_det = rl1, thr, det
    return best_thr, best_rl1, best_det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--n-vis", type=int, default=2,
                        help="Number of images to visualise FRST response map for")
    parser.add_argument("--single-radius", type=float, default=None,
                        help="Test only this radius (px) instead of the full sweep")
    args = parser.parse_args()

    cfg    = PipelineConfig()
    radii  = make_radii(cfg)
    if args.single_radius is not None:
        # snap to nearest grid point
        idx = int(np.argmin(np.abs(radii - args.single_radius)))
        radii = radii[idx:idx+1]
        print(f"Single-radius mode: r={radii[0]:.2f}px")
    else:
        print(f"Testing {len(radii)} radii from {radii[0]:.1f}px to {radii[-1]:.1f}px")

    # ── Load all tractable images ─────────────────────────────────────────────
    samples = []
    for img_path in sorted((args.data_dir / "images").glob("*.png")):
        lbl_path = args.data_dir / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        img = load_image(img_path)
        if img.mean() >= MAX_IMG_MEAN:
            continue
        bubbles = parse_annotations(lbl_path)
        samples.append({
            "path":    img_path,
            "stem":    img_path.stem,
            "img":     img,
            "bubbles": bubbles,
            "gt_hist": gt_histogram(bubbles, radii),
            "n_gt":    len(bubbles),
        })

    print(f"Loaded {len(samples)} images", flush=True)

    # ── Compute FRST + precompute candidates for every image ──────────────────
    # Storing candidates (sorted local maxima) instead of full response maps
    # cuts memory ~100× and makes threshold search O(N) instead of O(N*T).
    print("Computing FRST responses and extracting candidates...", flush=True)
    for s in samples:
        gx, gy = scharr_gradient(s["img"])
        resp = frst(gx, gy, radii)
        s["resp_max"] = float(resp.max())
        s["candidates"] = all_candidates(resp, radii)   # (N,4) sorted by score
        # keep response only for the first n_vis images (needed for heatmap plot)
        if samples.index(s) < args.n_vis:
            s["response"] = resp
        print(f"  {s['stem'][-40:]}  resp_max={s['resp_max']:.4f}"
              f"  candidates={len(s['candidates'])}", flush=True)

    # ── Visual diagnostic: response map overlay on first N images ─────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in samples[:args.n_vis]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        ax.imshow(s["img"], cmap="gray", vmin=0, vmax=1)
        for b in s["bubbles"]:
            circ = plt.Circle((b.cx, b.cy), b.radius, color="lime",
                               fill=False, linewidth=0.6, alpha=0.7)
            ax.add_patch(circ)
        ax.set_title(f"GT bubbles (n={s['n_gt']})")
        ax.axis("off")

        # max-projection of FRST across all radii
        resp_max = s["response"].max(axis=0)
        ax = axes[1]
        ax.imshow(s["img"], cmap="gray", vmin=0, vmax=1, alpha=0.6)
        im = ax.imshow(resp_max, cmap="hot", alpha=0.7,
                       vmin=0, vmax=np.percentile(resp_max, 99))
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title("FRST response (max over radii)")
        ax.axis("off")

        # detections at oracle threshold using pre-computed candidates
        thr_vals = np.linspace(0, float(s["resp_max"]), 80)[1:]
        best_thr, best_rl1, best_det = oracle_threshold(
            s["candidates"], s["gt_hist"], radii, thr_vals)
        ax = axes[2]
        ax.imshow(s["img"], cmap="gray", vmin=0, vmax=1)
        for b in s["bubbles"]:
            circ = plt.Circle((b.cx, b.cy), b.radius, color="lime",
                               fill=False, linewidth=0.6, alpha=0.5)
            ax.add_patch(circ)
        for cx, cy, r, sc in best_det:
            circ = plt.Circle((cx, cy), r, color="red",
                               fill=False, linewidth=0.8, alpha=0.7)
            ax.add_patch(circ)
        ax.set_title(f"Oracle detections (thr={best_thr:.4f})\n"
                     f"n_det={len(best_det)}  relL1={best_rl1:.3f}")
        ax.axis("off")

        fig.suptitle(s["stem"][-50:], fontsize=10)
        fig.tight_layout()
        out = args.out_dir / f"e14_vis_{s['stem']}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved visual to {out}")

    # ── Oracle relL1 per image ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-IMAGE ORACLE relL1 (upper bound — threshold tuned per image):")
    print(f"{'Image':<44}  {'n_gt':>5}  {'oracle_rl1':>10}  {'n_det':>6}")
    print("-" * 70)

    oracle_results = []
    for s in samples:
        if s["gt_hist"].sum() < 100:
            continue
        thr_vals = np.linspace(0, float(s["resp_max"]), 80)[1:]
        best_thr, best_rl1, best_det = oracle_threshold(
            s["candidates"], s["gt_hist"], radii, thr_vals)
        oracle_results.append({**s, "oracle_rl1": best_rl1,
                                "oracle_thr": best_thr, "oracle_det": best_det})
        print(f"  {s['stem'][-42:]:<42}  {s['n_gt']:>5}  {best_rl1:>10.3f}  "
              f"{len(best_det):>6}", flush=True)

    if oracle_results:
        med_oracle = float(np.median([r["oracle_rl1"] for r in oracle_results]))
        print(f"\n  Median oracle relL1 (FRST): {med_oracle:.3f}")
        print(f"  Cross-session GT-oracle:    0.657  (E0-A baseline)")
        print(f"  NCC pipeline LOSO:          0.851  (E0-B baseline)")

    # ── LOSO relL1 ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LOSO relL1 (threshold tuned on training fold, applied to held-out):")
    print(f"{'Image':<44}  {'n_gt':>5}  {'loso_rl1':>9}  {'thr':>8}  {'n_det':>6}")
    print("-" * 70)

    # Use in-range GT count (gt_hist.sum()) not total bubble count.
    # Total n_gt is misleading when most bubbles fall outside the radius sweep window.
    stable = [s for s in samples if s["gt_hist"].sum() >= 100]
    loso_results = []
    for i, s_test in enumerate(stable):
        train = [s for j, s in enumerate(stable) if j != i]
        if not train:
            continue

        # threshold grid from training fold only
        thr_vals = np.linspace(
            0, float(max(s["resp_max"] for s in train)), 80)[1:]

        def loso_score(thr):
            rl1s = []
            for s_tr in train:
                det = find_detections(s_tr["candidates"], thr)
                pred = build_histogram(det, radii)
                rl1 = rel_l1(pred, s_tr["gt_hist"])
                if not np.isnan(rl1):
                    rl1s.append(rl1)
            return float(np.median(rl1s)) if rl1s else float("inf")

        best_thr = min(thr_vals, key=loso_score)

        det = find_detections(s_test["candidates"], best_thr)
        pred = build_histogram(det, radii)
        rl1 = rel_l1(pred, s_test["gt_hist"])
        loso_results.append(rl1)
        print(f"  {s_test['stem'][-42:]:<42}  {s_test['n_gt']:>5}  {rl1:>9.3f}  "
              f"{best_thr:>8.5f}  {len(det):>6}", flush=True)

    if loso_results:
        loso_med = float(np.median(loso_results))
        n_radii = len(radii)
        print(f"\n  Median LOSO relL1 (FRST, {n_radii} radius/radii): {loso_med:.3f}")
        if n_radii > 1:
            # Multi-radius: 27-bin metric comparable to E0-A/E0-B
            print(f"  Cross-session GT-oracle (E0-A):   0.657  [27-bin, same metric]")
            print(f"  NCC pipeline LOSO (E0-B):         0.851  [27-bin, same metric]")
            if loso_med < 0.851:
                print(f"  HYPOTHESIS SURVIVES: FRST ({loso_med:.3f}) beats NCC pipeline (0.851)")
            else:
                print(f"  HYPOTHESIS FALSIFIED: FRST ({loso_med:.3f}) does not beat NCC (0.851)")
        else:
            # Single-radius: 1-bin count metric — NOT comparable to 27-bin E0-A/E0-B
            print(f"  NOTE: single-radius = 1-bin count metric only.")
            print(f"  Comparison to E0-A oracle (0.657) or E0-B pipeline (0.851) is INVALID.")
            print(f"  Run multi-radius for a valid hypothesis test.")

    # ── Summary plot ──────────────────────────────────────────────────────────
    if oracle_results and loso_results:
        names = [r["stem"][-20:] for r in oracle_results]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - 0.25, [r["oracle_rl1"] for r in oracle_results],
               width=0.25, color="steelblue", alpha=0.8, label="FRST oracle")
        ax.bar(x,        loso_results[:len(names)],
               width=0.25, color="darkorange", alpha=0.8, label="FRST LOSO")
        ax.axhline(0.657, color="green",  linestyle="--", linewidth=1.5,
                   label="GT oracle E0-A (0.657)")
        ax.axhline(0.851, color="tomato", linestyle="--", linewidth=1.5,
                   label="NCC pipeline E0-B (0.851)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("relL1")
        ax.set_title(f"E14 — FRST relL1  "
                     f"(oracle median={med_oracle:.3f}, LOSO median={loso_med:.3f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        out = args.out_dir / "e14_frst_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSummary plot saved to {out}")


if __name__ == "__main__":
    main()
