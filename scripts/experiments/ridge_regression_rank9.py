#!/usr/bin/env python3
"""
Rank 9 — Image-feature ridge regression for bubble size histogram estimation.

Hypothesis to falsify:
  "Global image statistics (photometric, edge, frequency) and radial-gradient-integral
   features are not predictive of histogram bin counts. LOSO ridge regression relL1
   > 0.657 (worse than the GT oracle)."

Pre-committed verdicts:
  relL1 < 0.65  → features add signal; image-level regression is viable
  relL1 ≈ 0.657 → marginal; features contain no more information than cross-image GT averaging
  relL1 > 0.657 → FALSIFIED; image-feature regression not viable at n=14

Feature set (37 total):
  [0–3]   Photometric:          img_mean, img_std, skewness, kurtosis
  [4]     Edge density:         fraction of pixels above Otsu-thresholded Sobel gradient
  [5–9]   FFT octave power:     power fraction in 5 spatial-frequency octave bands
  [10–36] Radial-gradient-integral (27):
            For each radius bin k: mean |LoG| response at sigma = r_k/sqrt(2), summed
            over the full image. Connects directly to E13's 6.86x SNR on the outward
            radial gradient. Higher values indicate more bubble-sized structures at that
            scale in the image.

USAGE: python scripts/experiments/ridge_regression_rank9.py [data_dir]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import load_image, parse_annotations, get_session_id

MAX_IMG_MEAN = 0.6
MIN_N_GT = 100
OUT_DIR = Path("output/exp_rank9")
ALPHA_GRID = np.logspace(-3, 5, 40)


# ── shared utilities ──────────────────────────────────────────────────────────

def build_radius_bins(cfg):
    radii, r = [], cfg.min_radius
    while r <= cfg.max_radius * 1.001:
        radii.append(r)
        r = r / cfg.scale_factor
    return np.array(radii)


def gt_histogram(bubbles, bin_radii):
    counts = np.zeros(len(bin_radii), dtype=float)
    lo, hi = bin_radii[0] * 0.5, bin_radii[-1] * 2
    for b in bubbles:
        if b.radius < lo or b.radius > hi:
            continue
        idx = int(np.argmin(np.abs(np.log(bin_radii) - np.log(b.radius))))
        counts[idx] += 1
    return counts


def rel_l1(pred, gt):
    total = gt.sum()
    return float(np.abs(pred - gt).sum() / total) if total > 0 else np.nan


def loo_oracle_median(hists, target_idx):
    mask = np.ones(len(hists), dtype=bool)
    mask[target_idx] = False
    if mask.sum() == 0:
        return np.full(hists.shape[1], np.nan)
    return np.median(hists[mask], axis=0)


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(img: np.ndarray, bin_radii: np.ndarray) -> np.ndarray:
    """
    Extract 37-dim feature vector from a float32 [0,1] grayscale image.
    img: (H, W) float32 array
    """
    feats = []

    # ── photometric (4) ──────────────────────────────────────────────────────
    flat = img.flatten().astype(np.float64)
    feats.append(float(np.mean(flat)))
    feats.append(float(np.std(flat)))
    feats.append(float(stats.skew(flat)))
    feats.append(float(stats.kurtosis(flat)))

    # ── edge density (1) — Otsu threshold on Sobel magnitude ─────────────────
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    gx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_u8 = grad_mag.clip(0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    feats.append(float((grad_mag > otsu_thr).mean()))

    # ── FFT octave power (5) — fraction of power in each spatial-frequency band ─
    fft2 = np.fft.fft2(img.astype(np.float64))
    power = np.abs(np.fft.fftshift(fft2)) ** 2
    H, W = img.shape
    freq_y = np.fft.fftshift(np.fft.fftfreq(H))
    freq_x = np.fft.fftshift(np.fft.fftfreq(W))
    FX, FY = np.meshgrid(freq_x, freq_y)
    freq_mag = np.sqrt(FX**2 + FY**2)
    total_power = power.sum() + 1e-12
    for lo, hi in [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.12), (0.12, 0.5)]:
        mask = (freq_mag >= lo) & (freq_mag < hi)
        feats.append(float(power[mask].sum() / total_power))

    # ── radial-gradient-integral (27) ─────────────────────────────────────────
    # For each radius bin k with radius r_k: apply scale-normalized LoG at
    # sigma = r_k/sqrt(2) and take the mean |response| across the image.
    # sigma^2 * gaussian_laplace gives scale-normalized LoG (response magnitude
    # independent of scale for ideal blobs). Connects to E13's 6.86x SNR:
    # images with many bubbles at radius r_k have higher LoG energy at that scale.
    img_f64 = img.astype(np.float64)
    sqrt2 = np.sqrt(2.0)
    for r_k in bin_radii:
        sigma = max(r_k / sqrt2, 1.0)
        log_response = ndimage.gaussian_laplace(img_f64, sigma=sigma)
        # Scale-normalize: multiply by sigma^2 so blob peaks are sigma-independent
        log_response_normalized = (sigma ** 2) * log_response
        feats.append(float(np.mean(np.abs(log_response_normalized))))

    return np.array(feats, dtype=np.float64)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04"), nargs="?")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    bin_radii = build_radius_bins(cfg)
    n_bins = len(bin_radii)
    print(f"Radius bins: {n_bins} levels, r={bin_radii[0]:.2f}–{bin_radii[-1]:.2f}px", flush=True)

    # ── load and featurize ────────────────────────────────────────────────────
    samples = []
    for img_path in sorted((args.data_dir / "images").glob("*.png")):
        lbl = args.data_dir / "labels" / (img_path.stem + ".json")
        if not lbl.exists():
            continue
        img = load_image(img_path)
        if img.mean() >= MAX_IMG_MEAN:
            continue
        bubbles = parse_annotations(lbl)
        hist = gt_histogram(bubbles, bin_radii)
        print(f"  Featurizing {img_path.name} (n_gt={len(bubbles)})...", flush=True)
        feats = extract_features(img, bin_radii)
        try:
            session = get_session_id(img_path.stem)
        except ValueError:
            session = "UNKNOWN"
        samples.append({
            "name":    img_path.stem,
            "n_gt":    len(bubbles),
            "hist":    hist,
            "feats":   feats,
            "session": session,
            "img_mean": float(img.mean()),
        })

    n_total = len(samples)
    print(f"\nLoaded {n_total} tractable images", flush=True)
    stable = [s for s in samples if s["n_gt"] >= MIN_N_GT]
    print(f"Stable images (n_gt≥{MIN_N_GT}): {len(stable)}", flush=True)

    # ── feature sanity check ─────────────────────────────────────────────────
    feat_dim = len(samples[0]["feats"])
    print(f"Feature dimension: {feat_dim}", flush=True)
    feat_names = (
        ["img_mean", "img_std", "skewness", "kurtosis", "edge_density"]
        + [f"fft_band{i}" for i in range(5)]
        + [f"log_r{i}" for i in range(n_bins)]
    )
    assert len(feat_names) == feat_dim, f"Name mismatch: {len(feat_names)} vs {feat_dim}"

    # ── global LOO oracle (E0-A baseline) ─────────────────────────────────────
    all_hists = np.stack([s["hist"] for s in samples])
    oracle_rl1 = []
    for i, s in enumerate(samples):
        if s["n_gt"] < MIN_N_GT:
            continue
        pred = loo_oracle_median(all_hists, i)
        oracle_rl1.append(rel_l1(pred, s["hist"]))
    oracle_median = float(np.median(oracle_rl1))
    print(f"\nGlobal LOO oracle median relL1: {oracle_median:.4f}  (E0-A baseline: 0.657)", flush=True)

    # ── LOSO ridge regression ─────────────────────────────────────────────────
    print("\nRunning LOSO ridge regression...", flush=True)
    loso_results = []
    for i, s in enumerate(samples):
        if s["n_gt"] < MIN_N_GT:
            continue

        train_idx = [j for j in range(n_total) if j != i]
        X_train = np.stack([samples[j]["feats"] for j in train_idx])
        Y_train = np.stack([samples[j]["hist"]  for j in train_idx])
        X_test  = s["feats"].reshape(1, -1)

        # Normalize: fit scaler on training set only
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # RidgeCV: cv=None → LOO-CV (GCV) for alpha selection on training set
        ridge = RidgeCV(alphas=ALPHA_GRID, fit_intercept=True, cv=None)
        ridge.fit(X_train_sc, Y_train)
        best_alpha = ridge.alpha_

        pred = ridge.predict(X_test_sc).flatten()
        pred = np.maximum(pred, 0.0)  # clip negative predictions

        rl1 = rel_l1(pred, s["hist"])
        oracle_rl1_i = rel_l1(loo_oracle_median(all_hists, i), s["hist"])

        loso_results.append({
            "name":        s["name"],
            "n_gt":        s["n_gt"],
            "session":     s["session"],
            "img_mean":    s["img_mean"],
            "rl1":         rl1,
            "oracle_rl1":  oracle_rl1_i,
            "delta":       rl1 - oracle_rl1_i,
            "best_alpha":  best_alpha,
            "pred_hist":   pred,
            "gt_hist":     s["hist"],
        })
        print(f"  {s['name'][-38:]:<40}  n_gt={s['n_gt']:4d}  "
              f"ridge_rl1={rl1:.3f}  oracle_rl1={oracle_rl1_i:.3f}  "
              f"Δ={rl1 - oracle_rl1_i:+.3f}  alpha={best_alpha:.2g}", flush=True)

    # ── summary statistics ────────────────────────────────────────────────────
    ridge_rl1s  = [r["rl1"]        for r in loso_results]
    oracle_rl1s = [r["oracle_rl1"] for r in loso_results]
    deltas      = [r["delta"]      for r in loso_results]

    ridge_median  = float(np.median(ridge_rl1s))
    oracle_median2 = float(np.median(oracle_rl1s))
    n_beats_oracle = sum(1 for d in deltas if d < 0)

    print(f"\n{'='*70}")
    print("RANK 9 — IMAGE-FEATURE RIDGE REGRESSION VERDICT")
    print(f"{'='*70}")
    print(f"  Ridge LOSO median relL1:   {ridge_median:.4f}")
    print(f"  Oracle LOO median relL1:   {oracle_median2:.4f}  (E0-A: 0.657)")
    print(f"  Images where ridge < oracle: {n_beats_oracle}/{len(loso_results)}")
    print(f"  Median Δ (ridge − oracle): {np.median(deltas):+.4f}")

    # ── verdict ───────────────────────────────────────────────────────────────
    print()
    if ridge_median < 0.65:
        print(f"  PASS: relL1={ridge_median:.4f} < 0.65 — image features add predictive signal.")
        print(f"  Image-level regression is viable. Proceed to CNN-B / FamNet.")
        verdict = "PASS"
    elif ridge_median <= 0.657:
        print(f"  MARGINAL: relL1={ridge_median:.4f} ≈ oracle ({oracle_median2:.4f}).")
        print(f"  Features provide no signal beyond cross-image GT averaging.")
        verdict = "MARGINAL"
    else:
        print(f"  FAIL: relL1={ridge_median:.4f} > 0.657 — regression WORSE than oracle.")
        print(f"  Image-feature regression is not viable at n={n_total}.")
        verdict = "FAIL"

    # ── feature importance (in-sample, for diagnostic purposes only) ──────────
    X_all = np.stack([s["feats"] for s in samples])
    Y_all = np.stack([s["hist"]  for s in samples])
    scaler_all = StandardScaler()
    X_all_sc = scaler_all.fit_transform(X_all)
    ridge_all = RidgeCV(alphas=ALPHA_GRID, fit_intercept=True, cv=None)
    ridge_all.fit(X_all_sc, Y_all)
    # Coefficient L2 norm across bins as importance proxy
    coef_importance = np.linalg.norm(ridge_all.coef_, axis=0)  # (n_features,)

    top_k = 10
    top_idx = np.argsort(coef_importance)[::-1][:top_k]
    print(f"\n  Top-{top_k} features by coefficient |w| (in-sample, diagnostic only):")
    for rank, fi in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feat_names[fi]:<20}  |w|={coef_importance[fi]:.4f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Rank 9 — Ridge regression LOSO  |  verdict: {verdict}", fontsize=12)

    # Plot 1: ridge vs oracle per image
    ax = axes[0]
    x = np.arange(len(loso_results))
    ax.bar(x - 0.2, ridge_rl1s,  0.35, label="Ridge LOSO", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, oracle_rl1s, 0.35, label="Oracle LOO", color="seagreen", alpha=0.8)
    ax.axhline(0.20, color="gold", ls="--", lw=1.5, label="Target 0.20")
    ax.axhline(0.65, color="tomato", ls=":", lw=1.2, label="Pass threshold 0.65")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"][-12:] for r in loso_results], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("relL1")
    ax.set_title("LOSO relL1: Ridge vs Oracle (per image)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: feature importance
    ax = axes[1]
    ax.bar(range(feat_dim), coef_importance, color="steelblue", alpha=0.7)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("|w| (L2 norm across bins)")
    ax.set_title("Feature importance (in-sample coefficients)")
    ax.axvline(10, color="gray", ls="--", lw=1, label="LoG features start")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: histogram prediction for best and worst LOSO image
    ax = axes[2]
    best_r = min(loso_results, key=lambda r: r["rl1"])
    gt_h = best_r["gt_hist"]
    pred_h = best_r["pred_hist"]
    ax.bar(range(n_bins), gt_h,   alpha=0.5, label=f"GT ({best_r['name'][-12:]})")
    ax.bar(range(n_bins), pred_h, alpha=0.5, label=f"Ridge pred (rl1={best_r['rl1']:.3f})")
    ax.set_xlabel("Radius bin")
    ax.set_ylabel("Count")
    ax.set_title(f"Best LOSO prediction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    out = OUT_DIR / "rank9_ridge_loso.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out}", flush=True)

    # Save per-image results
    import json
    results_out = OUT_DIR / "rank9_results.json"
    results_out.write_text(json.dumps({
        "ridge_loso_median":  ridge_median,
        "oracle_loo_median":  oracle_median2,
        "verdict":            verdict,
        "n_beats_oracle":     n_beats_oracle,
        "n_stable":           len(loso_results),
        "per_image": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items()}
            for r in loso_results
        ],
    }, indent=2))
    print(f"Results saved to {results_out}", flush=True)


if __name__ == "__main__":
    main()
