#!/usr/bin/env python3
"""
E15 — Generator recall probe (gates Experiment B).

Hypothesis to falsify:
  "A polarity-agnostic LoG/DoG detector places a candidate within R/2 of
   ≥75% of GT bubble centers in dense images, AND the radial gradient SNR
   on those actual TP candidates is ≥3×."

If this hypothesis survives → proceed to full Experiment B (LoG/DoG spatial
locator + radial gradient patch classifier + NMS).

If generator recall < 60% → instance detection is structurally infeasible for
this dataset at this density; pivot to image-feature regression.

Design:
  - Run blob_log (skimage) with both polarities over σ range covering radii 8–50px.
    Polarity agnostic: find both bright-on-dark blobs (negative LoG peak)
    and dark-on-bright blobs (positive LoG peak). Scale estimate is ignored.
  - For each GT bubble: TP if any candidate lands within R/2 of its center.
    Report recall overall and per morphology proxy (inward/outward dominant from E13).
  - For TP candidates and FP candidates (generator outputs not matched to GT):
    compute radial gradient score (E13 metric) at the candidate's (x, y) using
    the candidate's proposed radius (from LoG σ) and the GT radius (from annotation).
    Report SNR = mean|TP scores| / mean|FP scores|.
  - Check inter-bubble FP score distribution: candidates landing between two nearby
    bubbles (within 2R of TWO GT bubbles but not within R/2 of either).

Pre-committed criteria:
  1. Generator recall ≥ 75%  →  PASS
     Generator recall 60–75% →  MARGINAL (inspect per-morphology breakdown)
     Generator recall < 60%  →  FAIL → instance detection infeasible
  2. Actual-candidate SNR ≥ 3× → PASS   (E13 signal survives real distribution)
     Actual-candidate SNR < 2× → FAIL   (signal degraded; Experiment B not viable)
  3. Inter-bubble FP overlap: < 20% of inter-bubble FPs score above TP median → PASS

USAGE: python scripts/experiments/probe_generator_e15.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_log
from skimage.filters import scharr_h, scharr_v

from bubble_histogram.data import load_image, parse_annotations

# ── constants ──────────────────────────────────────────────────────────────
MIN_RADIUS    = 8.0     # px — ignore smaller GT bubbles
MAX_RADIUS    = 50.0    # px — ignore larger GT bubbles
MATCH_DIST    = 0.5     # fraction of R — TP if candidate within R*MATCH_DIST
RIM_INNER     = 0.85
RIM_OUTER     = 1.15
N_PROBE_IMGS  = 3       # number of images to run (densest 3 by n_gt)
MAX_IMG_MEAN  = 0.6     # skip photometrically dead frames

# LoG blob detection parameters
# sigma_ratio=sqrt(2) is the standard octave step; we use a finer step
LOG_MIN_SIGMA = MIN_RADIUS / np.sqrt(2)
LOG_MAX_SIGMA = MAX_RADIUS / np.sqrt(2)
LOG_NUM_SIGMA = 30
LOG_THRESHOLD = 0.005   # low — we want high recall; FP rate measured separately

OUT_DIR = Path("output/e15_probe")


# ── gradient helpers (from E13) ─────────────────────────────────────────────
def scharr_gradient(img: np.ndarray):
    img_f = img.astype(np.float64)
    gx = scharr_v(img_f)
    gy = scharr_h(img_f)
    return gx, gy


def inward_radial_score(gx, gy, cx, cy, R):
    """Mean inward radial gradient in annulus r/R ∈ [RIM_INNER, RIM_OUTER]."""
    H, W = gx.shape
    r_inner, r_outer = RIM_INNER * R, RIM_OUTER * R
    r0 = max(0, int(np.floor(cy - r_outer)))
    r1 = min(H, int(np.ceil(cy + r_outer)) + 1)
    c0 = max(0, int(np.floor(cx - r_outer)))
    c1 = min(W, int(np.ceil(cx + r_outer)) + 1)

    py = np.arange(r0, r1, dtype=np.float64)
    px = np.arange(c0, c1, dtype=np.float64)
    PX, PY = np.meshgrid(px, py)
    dx, dy = PX - cx, PY - cy
    dist = np.sqrt(dx**2 + dy**2)

    mask = (dist >= r_inner) & (dist < r_outer) & (dist > 0)
    if not mask.any():
        return np.nan

    ix = -dx[mask] / dist[mask]
    iy = -dy[mask] / dist[mask]
    dot = gx[r0:r1, c0:c1][mask] * ix + gy[r0:r1, c0:c1][mask] * iy
    return float(np.mean(dot))


# ── generator ───────────────────────────────────────────────────────────────
def run_generator(img: np.ndarray):
    """
    Run polarity-agnostic LoG blob detection.

    Returns array of shape (N, 3): [cy, cx, sigma].
    Both polarities: positive blobs (dark-on-bright) from img,
    negative blobs (bright-on-dark) from 1-img.
    Scale estimate (sigma) is kept for scoring but not used as the final
    radius — GT radius is used when available.
    """
    blobs_pos = blob_log(img, min_sigma=LOG_MIN_SIGMA, max_sigma=LOG_MAX_SIGMA,
                         num_sigma=LOG_NUM_SIGMA, threshold=LOG_THRESHOLD,
                         overlap=0.5)
    blobs_neg = blob_log(1.0 - img, min_sigma=LOG_MIN_SIGMA, max_sigma=LOG_MAX_SIGMA,
                         num_sigma=LOG_NUM_SIGMA, threshold=LOG_THRESHOLD,
                         overlap=0.5)

    if len(blobs_pos) == 0 and len(blobs_neg) == 0:
        return np.zeros((0, 3))

    parts = []
    if len(blobs_pos) > 0:
        parts.append(blobs_pos)
    if len(blobs_neg) > 0:
        parts.append(blobs_neg)
    blobs = np.vstack(parts)

    # convert sigma → approximate radius: r ≈ sigma * sqrt(2)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return blobs  # [cy, cx, r_approx]


# ── matching ────────────────────────────────────────────────────────────────
def match_candidates(bubbles, candidates):
    """
    Match candidates to GT bubbles.

    Returns:
      tp_mask:      bool (N_candidates,) — candidate is within R/2 of some GT bubble
      gt_matched:   bool (N_bubbles,)    — GT bubble has at least one TP candidate
      inter_mask:   bool (N_candidates,) — candidate is within 2R of ≥2 GT bubbles
                                           but not a TP (inter-bubble region)
    """
    N_cand = len(candidates)
    N_gt   = len(bubbles)

    tp_mask    = np.zeros(N_cand, dtype=bool)
    gt_matched = np.zeros(N_gt,   dtype=bool)
    inter_mask = np.zeros(N_cand, dtype=bool)

    for ci, (cy, cx, _) in enumerate(candidates):
        near_count = 0
        is_tp = False
        for gi, b in enumerate(bubbles):
            bx, by, br = b.cx, b.cy, b.radius
            dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
            if dist < MATCH_DIST * br:
                tp_mask[ci]    = True
                gt_matched[gi] = True
                is_tp = True
            if dist < 2.0 * br:
                near_count += 1
        if not is_tp and near_count >= 2:
            inter_mask[ci] = True

    return tp_mask, gt_matched, inter_mask


# ── per-image analysis ───────────────────────────────────────────────────────
def analyse_image(img_key, img, bubbles, gx, gy):
    # filter to tractable bubbles
    bubbles = [b for b in bubbles if MIN_RADIUS <= b.radius <= MAX_RADIUS]
    if len(bubbles) < 10:
        return None

    candidates = run_generator(img)
    print(f"  {img_key:50s}  n_gt={len(bubbles):4d}  n_cand={len(candidates):5d}", flush=True)

    if len(candidates) == 0:
        print(f"    WARNING: generator produced 0 candidates", flush=True)
        return None

    tp_mask, gt_matched, inter_mask = match_candidates(bubbles, candidates)

    recall = gt_matched.mean()
    n_tp   = tp_mask.sum()
    n_fp   = (~tp_mask).sum()

    # radial gradient scores — use GT radius for TP, LoG radius for FP
    tp_scores, fp_scores, inter_scores = [], [], []

    for ci, (cy, cx, r_approx) in enumerate(candidates):
        if tp_mask[ci]:
            # find the matched GT bubble's radius
            best_r = r_approx
            best_d = np.inf
            for b in bubbles:
                d = np.sqrt((cx - b.cx)**2 + (cy - b.cy)**2)
                if d < best_d:
                    best_d, best_r = d, b.radius
            s = inward_radial_score(gx, gy, cx, cy, best_r)
            if not np.isnan(s):
                tp_scores.append(abs(s))
        elif inter_mask[ci]:
            s = inward_radial_score(gx, gy, cx, cy, r_approx)
            if not np.isnan(s):
                inter_scores.append(abs(s))
        else:
            s = inward_radial_score(gx, gy, cx, cy, r_approx)
            if not np.isnan(s):
                fp_scores.append(abs(s))

    # per-morphology recall: inward-dominant proxy = bubble inward radial score > 0
    bubble_scores_signed = []
    for b in bubbles:
        s = inward_radial_score(gx, gy, b.cx, b.cy, b.radius)
        bubble_scores_signed.append(s if not np.isnan(s) else 0.0)
    inward_mask  = np.array(bubble_scores_signed) > 0
    outward_mask = ~inward_mask

    inward_recall  = gt_matched[inward_mask].mean()  if inward_mask.any()  else np.nan
    outward_recall = gt_matched[outward_mask].mean() if outward_mask.any() else np.nan

    tp_arr    = np.array(tp_scores)    if tp_scores    else np.array([np.nan])
    fp_arr    = np.array(fp_scores)    if fp_scores    else np.array([np.nan])
    inter_arr = np.array(inter_scores) if inter_scores else np.array([np.nan])

    mean_tp   = np.nanmean(tp_arr)
    mean_fp   = np.nanmean(fp_arr)
    snr       = mean_tp / mean_fp if mean_fp > 0 else np.nan

    tp_median = np.nanmedian(tp_arr)
    inter_above_tp = (inter_arr > tp_median).mean() if len(inter_scores) > 0 else np.nan

    return {
        "img_key":        img_key,
        "n_gt":           len(bubbles),
        "n_cand":         len(candidates),
        "recall":         recall,
        "inward_recall":  inward_recall,
        "outward_recall": outward_recall,
        "n_tp":           n_tp,
        "n_fp":           n_fp,
        "n_inter":        inter_mask.sum(),
        "mean_tp_score":  mean_tp,
        "mean_fp_score":  mean_fp,
        "snr":            snr,
        "inter_above_tp": inter_above_tp,
        "tp_scores":      tp_arr,
        "fp_scores":      fp_arr,
        "inter_scores":   inter_arr,
        "candidates":     candidates,
        "gt_matched":     gt_matched,
        "tp_mask":        tp_mask,
        "inter_mask":     inter_mask,
        "bubbles":        bubbles,
    }


def save_visual(img, result, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(result["img_key"], fontsize=9)

    ax = axes[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    # GT bubbles — green=matched, red=missed
    for gi, b in enumerate(result["bubbles"]):
        color = "lime" if result["gt_matched"][gi] else "red"
        circle = plt.Circle((b.cx, b.cy), b.radius, fill=False, color=color, lw=0.8)
        ax.add_patch(circle)
    # candidates — blue=TP, orange=inter-bubble FP, grey=other FP
    for ci, (cy, cx, r) in enumerate(result["candidates"]):
        if result["tp_mask"][ci]:
            c, a = "cyan", 0.6
        elif result["inter_mask"][ci]:
            c, a = "orange", 0.4
        else:
            c, a = "white", 0.15
        circle = plt.Circle((cx, cy), max(r, 3), fill=False, color=c, lw=0.5, alpha=a)
        ax.add_patch(circle)
    ax.set_title(f"GT (green=matched, red=missed)  recall={result['recall']:.2f}\n"
                 f"candidates: cyan=TP, orange=inter-FP, white=FP", fontsize=7)
    ax.axis("off")

    ax = axes[1]
    bins = np.linspace(0, max(
        np.nanpercentile(result["tp_scores"], 99) if len(result["tp_scores"]) > 1 else 0.1,
        np.nanpercentile(result["fp_scores"], 99) if len(result["fp_scores"]) > 1 else 0.1,
    ) * 1.1, 50)
    ax.hist(result["tp_scores"][np.isfinite(result["tp_scores"])],
            bins=bins, alpha=0.6, label=f"TP (n={result['n_tp']})", color="cyan", density=True)
    ax.hist(result["fp_scores"][np.isfinite(result["fp_scores"])],
            bins=bins, alpha=0.6, label=f"FP (n={result['n_fp']})", color="red", density=True)
    if result["n_inter"] > 0:
        ax.hist(result["inter_scores"][np.isfinite(result["inter_scores"])],
                bins=bins, alpha=0.6, label=f"inter-FP (n={result['n_inter']})",
                color="orange", density=True)
    ax.axvline(result["mean_tp_score"], color="cyan", lw=1.5, ls="--")
    ax.axvline(result["mean_fp_score"], color="red",  lw=1.5, ls="--")
    ax.set_xlabel("|inward radial gradient score|")
    ax.set_title(f"SNR={result['snr']:.2f}×  (mean_TP={result['mean_tp_score']:.4f} / "
                 f"mean_FP={result['mean_fp_score']:.4f})\n"
                 f"inter-FP above TP median: {result['inter_above_tp']:.1%}", fontsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="seed_v04")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load all images, pick N_PROBE_IMGS densest (by n_gt in radius window)
    ann_dir = data_dir / "labels"
    img_dir = data_dir / "images"
    all_keys = sorted([p.stem for p in ann_dir.glob("*.json")])

    candidates_list = []
    for key in all_keys:
        img_path = img_dir / f"{key}.png"
        lbl_path = ann_dir / f"{key}.json"
        if not img_path.exists() or not lbl_path.exists():
            continue
        img = load_image(img_path)
        if img is None or img.mean() > MAX_IMG_MEAN:
            continue
        bubbles = parse_annotations(lbl_path)
        n_in_window = sum(1 for b in bubbles if MIN_RADIUS <= b.radius <= MAX_RADIUS)
        candidates_list.append((n_in_window, key))

    candidates_list.sort(reverse=True)
    probe_keys = [k for _, k in candidates_list[:N_PROBE_IMGS]]
    print(f"Probe images (top {N_PROBE_IMGS} by in-window GT count):", flush=True)
    for n, k in candidates_list[:N_PROBE_IMGS]:
        print(f"  {k}  n_gt_in_window={n}", flush=True)
    print(flush=True)

    results = []
    for key in probe_keys:
        img     = load_image(img_dir / f"{key}.png")
        bubbles = parse_annotations(ann_dir / f"{key}.json")
        if img is None:
            continue
        gx, gy  = scharr_gradient(img)
        r = analyse_image(key, img, bubbles, gx, gy)
        if r is None:
            continue
        results.append(r)
        vis_path = OUT_DIR / f"e15_vis_{key}.png"
        save_visual(img, r, vis_path)
        print(f"  Saved visual to {vis_path}", flush=True)

    if not results:
        print("ERROR: no usable results", flush=True)
        return

    print(flush=True)
    print("=" * 70, flush=True)
    print("GENERATOR RECALL (fraction of GT bubbles with candidate within R/2):", flush=True)
    print(f"  {'Image':50s}  {'n_gt':>5}  {'recall':>7}  {'inward':>7}  {'outward':>8}  {'n_cand':>7}", flush=True)
    print("-" * 70, flush=True)
    recalls = []
    for r in results:
        print(f"  {r['img_key']:50s}  {r['n_gt']:5d}  {r['recall']:7.3f}  "
              f"{r['inward_recall']:7.3f}  {r['outward_recall']:8.3f}  {r['n_cand']:7d}", flush=True)
        recalls.append(r["recall"])

    med_recall = float(np.median(recalls))
    print(flush=True)
    print(f"  Median generator recall: {med_recall:.3f}", flush=True)

    if   med_recall >= 0.75: recall_verdict = "PASS  (≥75%)"
    elif med_recall >= 0.60: recall_verdict = "MARGINAL  (60–75%) — inspect morphology breakdown"
    else:                     recall_verdict = "FAIL  (<60%) — instance detection infeasible"
    print(f"  Recall criterion: {recall_verdict}", flush=True)

    print(flush=True)
    print("=" * 70, flush=True)
    print("ACTUAL-CANDIDATE RADIAL GRADIENT SNR:", flush=True)
    print(f"  {'Image':50s}  {'SNR':>6}  {'mean_TP':>8}  {'mean_FP':>8}  {'inter_above_TP':>15}", flush=True)
    print("-" * 70, flush=True)
    snrs = []
    for r in results:
        print(f"  {r['img_key']:50s}  {r['snr']:6.2f}  {r['mean_tp_score']:8.4f}  "
              f"{r['mean_fp_score']:8.4f}  {r['inter_above_tp']:15.1%}", flush=True)
        if not np.isnan(r["snr"]):
            snrs.append(r["snr"])

    med_snr = float(np.median(snrs)) if snrs else np.nan
    print(flush=True)
    print(f"  Median actual-candidate SNR: {med_snr:.2f}×", flush=True)

    if   med_snr >= 3.0: snr_verdict = "PASS  (≥3×)"
    elif med_snr >= 2.0: snr_verdict = "MARGINAL  (2–3×)"
    else:                 snr_verdict = "FAIL  (<2×) — E13 signal does not survive real distribution"
    print(f"  SNR criterion: {snr_verdict}", flush=True)

    inter_fracs = [r["inter_above_tp"] for r in results if not np.isnan(r["inter_above_tp"])]
    med_inter = float(np.median(inter_fracs)) if inter_fracs else np.nan
    if   med_inter < 0.20: inter_verdict = "PASS  (<20% of inter-FPs above TP median)"
    elif med_inter < 0.50: inter_verdict = "MARGINAL  (20–50%)"
    else:                   inter_verdict = "FAIL  (≥50%) — NMS cannot separate inter-bubble FPs"
    print(f"  Inter-bubble FP criterion: {inter_verdict}  (median={med_inter:.1%})", flush=True)

    print(flush=True)
    print("=" * 70, flush=True)
    all_pass = (med_recall >= 0.75) and (med_snr >= 3.0) and (med_inter < 0.20)
    if all_pass:
        print("ALL CRITERIA PASS — proceed to full Experiment B", flush=True)
    elif med_recall < 0.60:
        print("RECALL CRITERION FAILED — instance detection infeasible; pivot to regression", flush=True)
    else:
        print("ONE OR MORE CRITERIA MARGINAL/FAILED — inspect visuals before proceeding", flush=True)


if __name__ == "__main__":
    main()
