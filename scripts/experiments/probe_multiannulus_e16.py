#!/usr/bin/env python3
"""
E16 — Multi-annulus radial gradient profile + R/4 re-centering gate.

Hypothesis to falsify:
  "A 10-annulus radial gradient profile (0–1.5R) computed at R/4-re-centered
   TP candidates achieves logistic-score SNR ≥ 3× vs non-TP candidates,
   AND inter-bubble FPs score below the TP median."

Context: E15 showed single rim-annulus SNR = 1.22× (FAIL) and inter-bubble
FP overlap 78.4% (FAIL). E15's SNR failure is partially confounded (~40–60%)
by R/2 matching tolerance causing annulus misalignment. This experiment
corrects that: R/4 re-centering (candidate must be within R/4 of GT center
to be scored as TP), and replaces the single-ring score with a 10-annulus
profile that should structurally discriminate inter-bubble FPs (elevated
across multiple annuli) from true bubble centers (peaked at rim).

Pre-committed gates (E16 is the LAST detection gate — no E17):
  SNR ≥ 3×  →  P(redesigned B) revises to ~20–25%; proceed
  SNR 2–3×  →  MARGINAL; inspect per-morphology; do NOT accept post-hoc
  SNR < 2×  →  detection path CLOSED; pivot to regression

USAGE: python scripts/experiments/probe_multiannulus_e16.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage.feature import blob_log
from skimage.filters import scharr_h, scharr_v

from bubble_histogram.data import load_image, parse_annotations

# ── constants ───────────────────────────────────────────────────────────────
MIN_RADIUS     = 8.0
MAX_RADIUS     = 50.0
RECENTER_FRAC  = 0.25   # R/4 — tighter than E15's R/2
N_ANNULI       = 10     # profile from 0 to 1.5R
ANNULUS_MAX_R  = 1.5    # outer limit of profile in units of R
N_PROBE_IMGS   = 3
MAX_IMG_MEAN   = 0.6
LOG_MIN_SIGMA  = MIN_RADIUS / np.sqrt(2)
LOG_MAX_SIGMA  = MAX_RADIUS / np.sqrt(2)
LOG_NUM_SIGMA  = 30
LOG_THRESHOLD  = 0.005
OUT_DIR        = Path("output/e16_multiannulus")

# annulus boundaries as fraction of R: 10 rings from 0 to 1.5R
ANNULUS_EDGES = np.linspace(0.0, ANNULUS_MAX_R, N_ANNULI + 1)


# ── gradient helpers ────────────────────────────────────────────────────────
def scharr_gradient(img):
    img_f = img.astype(np.float64)
    return scharr_v(img_f), scharr_h(img_f)


def annulus_radial_score(gx, gy, cx, cy, R, r_inner_frac, r_outer_frac):
    """Mean |inward radial gradient| in one annulus band."""
    H, W = gx.shape
    r_inner = r_inner_frac * R
    r_outer = r_outer_frac * R
    if r_outer < 0.5:
        return 0.0
    r0 = max(0, int(np.floor(cy - r_outer)))
    r1 = min(H, int(np.ceil(cy + r_outer)) + 1)
    c0 = max(0, int(np.floor(cx - r_outer)))
    c1 = min(W, int(np.ceil(cx + r_outer)) + 1)
    py = np.arange(r0, r1, dtype=np.float64)
    px = np.arange(c0, c1, dtype=np.float64)
    PX, PY = np.meshgrid(px, py)
    dx, dy = PX - cx, PY - cy
    dist = np.sqrt(dx**2 + dy**2)
    mask = (dist >= r_inner) & (dist < r_outer) & (dist > 1e-9)
    if not mask.any():
        return 0.0
    ix = -dx[mask] / dist[mask]
    iy = -dy[mask] / dist[mask]
    dot = gx[r0:r1, c0:c1][mask] * ix + gy[r0:r1, c0:c1][mask] * iy
    return float(np.mean(np.abs(dot)))


def radial_profile(gx, gy, cx, cy, R):
    """Return N_ANNULI-length feature vector."""
    return np.array([
        annulus_radial_score(gx, gy, cx, cy, R, ANNULUS_EDGES[k], ANNULUS_EDGES[k+1])
        for k in range(N_ANNULI)
    ], dtype=np.float32)


# ── generator (same as E15) ─────────────────────────────────────────────────
def run_generator(img):
    blobs_pos = blob_log(img, min_sigma=LOG_MIN_SIGMA, max_sigma=LOG_MAX_SIGMA,
                         num_sigma=LOG_NUM_SIGMA, threshold=LOG_THRESHOLD, overlap=0.5)
    blobs_neg = blob_log(1.0 - img, min_sigma=LOG_MIN_SIGMA, max_sigma=LOG_MAX_SIGMA,
                         num_sigma=LOG_NUM_SIGMA, threshold=LOG_THRESHOLD, overlap=0.5)
    parts = []
    if len(blobs_pos): parts.append(blobs_pos)
    if len(blobs_neg): parts.append(blobs_neg)
    if not parts:
        return np.zeros((0, 3))
    blobs = np.vstack(parts)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)  # sigma → radius
    return blobs


# ── per-image analysis ───────────────────────────────────────────────────────
def analyse_image(img_key, img, bubbles, gx, gy):
    bubbles = [b for b in bubbles if MIN_RADIUS <= b.radius <= MAX_RADIUS]
    if len(bubbles) < 10:
        return None

    candidates = run_generator(img)
    print(f"  {img_key:50s}  n_gt={len(bubbles):4d}  n_cand={len(candidates):5d}",
          flush=True)

    # ── match candidates at R/4 tolerance ───────────────────────────────────
    tp_entries, fp_entries, inter_entries = [], [], []

    gt_cx = np.array([b.cx for b in bubbles])
    gt_cy = np.array([b.cy for b in bubbles])
    gt_r  = np.array([b.radius for b in bubbles])

    gt_matched = np.zeros(len(bubbles), dtype=bool)

    for cy, cx, r_approx in candidates:
        dists = np.sqrt((cx - gt_cx)**2 + (cy - gt_cy)**2)
        within_recenter = dists < RECENTER_FRAC * gt_r
        near_any        = dists < 2.0 * gt_r

        if within_recenter.any():
            # TP: use GT radius and GT center for profile
            best = int(np.argmin(dists))
            gt_matched[best] = True
            bgt = bubbles[best]
            feat = radial_profile(gx, gy, bgt.cx, bgt.cy, bgt.radius)
            tp_entries.append(feat)
        elif near_any.sum() >= 2:
            # inter-bubble FP: use LoG radius at candidate position
            feat = radial_profile(gx, gy, cx, cy, r_approx)
            inter_entries.append(feat)
        else:
            feat = radial_profile(gx, gy, cx, cy, r_approx)
            fp_entries.append(feat)

    tp_arr    = np.array(tp_entries,    dtype=np.float32) if tp_entries    else None
    fp_arr    = np.array(fp_entries,    dtype=np.float32) if fp_entries    else None
    inter_arr = np.array(inter_entries, dtype=np.float32) if inter_entries else None

    return {
        "img_key":    img_key,
        "n_gt":       len(bubbles),
        "n_cand":     len(candidates),
        "gt_matched": gt_matched,
        "tp_arr":     tp_arr,
        "fp_arr":     fp_arr,
        "inter_arr":  inter_arr,
    }


def compute_snr(results):
    """
    Train a logistic regression on pooled TP vs FP profiles across all images.
    Returns SNR = mean(TP logistic score) / mean(FP logistic score),
    and reports inter-bubble FP overlap.
    """
    X_tp  = np.vstack([r["tp_arr"]  for r in results if r["tp_arr"]  is not None])
    X_fp  = np.vstack([r["fp_arr"]  for r in results if r["fp_arr"]  is not None])
    X_all = np.vstack([r["inter_arr"] for r in results if r["inter_arr"] is not None]
                      ) if any(r["inter_arr"] is not None for r in results) else None

    y_tp = np.ones(len(X_tp),  dtype=int)
    y_fp = np.zeros(len(X_fp), dtype=int)
    X_train = np.vstack([X_tp, X_fp])
    y_train = np.concatenate([y_tp, y_fp])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_scaled, y_train)

    tp_scores   = clf.predict_proba(scaler.transform(X_tp))[:, 1]
    fp_scores   = clf.predict_proba(scaler.transform(X_fp))[:, 1]

    mean_tp = float(np.mean(tp_scores))
    mean_fp = float(np.mean(fp_scores))
    snr     = mean_tp / mean_fp if mean_fp > 0 else np.nan

    inter_above_tp_median = np.nan
    if X_all is not None and len(X_all) > 0:
        inter_scores = clf.predict_proba(scaler.transform(X_all))[:, 1]
        tp_med = float(np.median(tp_scores))
        inter_above_tp_median = float((inter_scores > tp_med).mean())
    else:
        inter_scores = np.array([])

    return {
        "snr": snr,
        "mean_tp": mean_tp,
        "mean_fp": mean_fp,
        "tp_scores": tp_scores,
        "fp_scores": fp_scores,
        "inter_scores": inter_scores,
        "inter_above_tp_median": inter_above_tp_median,
        "n_tp": len(X_tp),
        "n_fp": len(X_fp),
        "n_inter": len(X_all) if X_all is not None else 0,
        "clf": clf,
        "scaler": scaler,
    }


def save_plots(results, snr_result):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("E16 — Multi-annulus radial profile (10 annuli, 0–1.5R)", fontsize=11)

    # Score distributions
    ax = axes[0]
    bins = np.linspace(0, 1, 40)
    ax.hist(snr_result["tp_scores"], bins=bins, alpha=0.6, density=True,
            color="cyan", label=f"TP (n={snr_result['n_tp']})")
    ax.hist(snr_result["fp_scores"], bins=bins, alpha=0.6, density=True,
            color="red",  label=f"FP (n={snr_result['n_fp']})")
    if len(snr_result["inter_scores"]) > 0:
        ax.hist(snr_result["inter_scores"], bins=bins, alpha=0.5, density=True,
                color="orange", label=f"inter-FP (n={snr_result['n_inter']})")
    ax.axvline(snr_result["mean_tp"], color="cyan", ls="--", lw=1.5)
    ax.axvline(snr_result["mean_fp"], color="red",  ls="--", lw=1.5)
    ax.set_xlabel("Logistic P(bubble)")
    ax.set_title(f"SNR={snr_result['snr']:.2f}×  "
                 f"(mean_TP={snr_result['mean_tp']:.3f} / mean_FP={snr_result['mean_fp']:.3f})\n"
                 f"inter-FP above TP median: {snr_result['inter_above_tp_median']:.1%}")
    ax.legend(fontsize=8)

    # Mean profiles per class
    ax = axes[1]
    annulus_centers = (ANNULUS_EDGES[:-1] + ANNULUS_EDGES[1:]) / 2
    for r in results:
        stacks = {}
        for key, label, color in [("tp_arr", "TP", "cyan"),
                                   ("fp_arr", "FP", "red"),
                                   ("inter_arr", "inter-FP", "orange")]:
            arr = r[key]
            if arr is not None and len(arr) > 0:
                stacks.setdefault((label, color), []).append(arr)

    pooled = {}
    for r in results:
        for key, label, color in [("tp_arr", "TP", "cyan"),
                                   ("fp_arr", "FP", "red"),
                                   ("inter_arr", "inter-FP", "orange")]:
            if r[key] is not None and len(r[key]) > 0:
                pooled.setdefault((label, color), []).append(r[key])
    for (label, color), arrays in pooled.items():
        mat = np.vstack(arrays)
        mean_profile = mat.mean(axis=0)
        std_profile  = mat.std(axis=0)
        ax.plot(annulus_centers, mean_profile, color=color, lw=2, label=label)
        ax.fill_between(annulus_centers,
                        mean_profile - std_profile,
                        mean_profile + std_profile,
                        color=color, alpha=0.2)
    ax.axvline(0.85, color="gray", ls=":", lw=1, label="rim inner (0.85R)")
    ax.axvline(1.15, color="gray", ls=":",  lw=1, label="rim outer (1.15R)")
    ax.set_xlabel("Annulus radius / R")
    ax.set_ylabel("Mean |inward radial gradient|")
    ax.set_title("Mean radial profile per class")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "e16_multiannulus_summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="seed_v04")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ann_dir = data_dir / "labels"
    img_dir = data_dir / "images"

    # pick N_PROBE_IMGS densest images (same selection as E15)
    candidates_list = []
    for key in sorted(p.stem for p in ann_dir.glob("*.json")):
        img_path = img_dir / f"{key}.png"
        lbl_path = ann_dir / f"{key}.json"
        if not img_path.exists():
            continue
        img = load_image(img_path)
        if img is None or img.mean() > MAX_IMG_MEAN:
            continue
        bubbles = parse_annotations(lbl_path)
        n = sum(1 for b in bubbles if MIN_RADIUS <= b.radius <= MAX_RADIUS)
        candidates_list.append((n, key))
    candidates_list.sort(reverse=True)
    probe_keys = [k for _, k in candidates_list[:N_PROBE_IMGS]]

    print(f"Probe images (top {N_PROBE_IMGS} by in-window GT count):", flush=True)
    for n, k in candidates_list[:N_PROBE_IMGS]:
        print(f"  {k}  n_gt={n}", flush=True)
    print(flush=True)

    results = []
    for key in probe_keys:
        img     = load_image(img_dir / f"{key}.png")
        bubbles = parse_annotations(ann_dir / f"{key}.json")
        if img is None:
            continue
        gx, gy = scharr_gradient(img)
        r = analyse_image(key, img, bubbles, gx, gy)
        if r is None:
            continue
        results.append(r)

    if not results:
        print("ERROR: no usable results.", flush=True)
        return

    # ── train logistic classifier & compute SNR ──────────────────────────────
    print(flush=True)
    print("Training logistic regression on pooled TP/FP profiles...", flush=True)
    snr_result = compute_snr(results)

    print(flush=True)
    print("=" * 70, flush=True)
    print("E16 RESULTS — MULTI-ANNULUS LOGISTIC SNR", flush=True)
    print("=" * 70, flush=True)
    print(f"  N_TP    = {snr_result['n_tp']}", flush=True)
    print(f"  N_FP    = {snr_result['n_fp']}", flush=True)
    print(f"  N_inter = {snr_result['n_inter']}", flush=True)
    print(f"  mean_TP = {snr_result['mean_tp']:.4f}", flush=True)
    print(f"  mean_FP = {snr_result['mean_fp']:.4f}", flush=True)
    print(f"  SNR     = {snr_result['snr']:.2f}×", flush=True)
    print(f"  inter-FP above TP median: "
          f"{snr_result['inter_above_tp_median']:.1%}", flush=True)

    # recall at R/4 tolerance
    recalls = []
    for r in results:
        if r["gt_matched"] is not None:
            recalls.append(r["gt_matched"].mean())
    print(f"  Recall at R/4 tolerance: "
          f"{np.median(recalls):.3f} (median)", flush=True)

    snr = snr_result["snr"]
    print(flush=True)
    print("=" * 70, flush=True)
    print("VERDICT (pre-committed gate — no E17):", flush=True)
    if snr >= 3.0:
        print(f"  SNR={snr:.2f}× ≥ 3×  →  PASS", flush=True)
        print(f"  P(redesigned B reaches relL1≤0.20) revises to ~20–25%.", flush=True)
        print(f"  Proceed with redesigned Experiment B.", flush=True)
    elif snr >= 2.0:
        print(f"  SNR={snr:.2f}× in [2, 3)  →  MARGINAL", flush=True)
        print(f"  Pre-committed rule: do NOT accept marginal as pass post-hoc.", flush=True)
        print(f"  Detection path CLOSED. Pivot to image-feature regression.", flush=True)
    else:
        print(f"  SNR={snr:.2f}× < 2×  →  FAIL", flush=True)
        print(f"  Detection path CLOSED. No E17.", flush=True)
        print(f"  Pivot to image-feature ridge regression (rank 9).", flush=True)

    save_plots(results, snr_result)


if __name__ == "__main__":
    main()
