#!/usr/bin/env python3
"""
Experiment D — Regime-conditional oracle.

Hypothesis to falsify:
  "Within any photometric-regime partition, the LOO GT-histogram oracle
   achieves relL1 ≤ 0.35 — meaning cross-image histogram variance is
   dominated by regime identity, not per-image structure."

If any partition achieves oracle ≤ 0.35 → regime-conditioned regression
has a principled basis; gates image-feature ridge regression path.

If oracle ≈ 0.657 in all partitions → per-image content is required to
reach relL1 ≤ 0.20; regression on global image stats cannot reach target.

Design:
  - Partition images 3 ways: (a) img.mean quartile bins, (b) session identity,
    (c) n_gt terciles (sparse/medium/dense).
  - For each partition with ≥ 2 images: LOO oracle using only other images
    in the same partition.
  - Report median LOO relL1 per partition and across all stable images.
  - Pre-committed criterion: any partition median ≤ 0.35 → PASS (regime
    conditioning is load-bearing).

USAGE: python scripts/experiments/oracle_regime_d.py [data_dir]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import load_image, parse_annotations

MAX_IMG_MEAN = 0.6
MIN_N_GT     = 100
OUT_DIR      = Path("output/exp_d")


# ── shared utilities (from E0-A) ────────────────────────────────────────────
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
    """LOO median oracle: median of all hists except target_idx."""
    mask = np.ones(len(hists), dtype=bool)
    mask[target_idx] = False
    if mask.sum() == 0:
        return np.full(hists.shape[1], np.nan)
    return np.median(hists[mask], axis=0)


# ── partitioning strategies ──────────────────────────────────────────────────
def session_id(stem):
    m = re.search(r'C1S\d+', stem)
    return m.group(0) if m else "UNKNOWN"


def partition_by(samples, key):
    """Return dict {label: [indices]}."""
    groups = {}
    for i, s in enumerate(samples):
        k = s[key]
        groups.setdefault(k, []).append(i)
    return groups


def compute_partition_oracle(samples, groups, bin_radii):
    """
    For each partition, compute per-image LOO oracle using only within-partition
    images. Returns list of dicts with per-image results, None if partition < 2.
    """
    all_hists = np.stack([s["hist"] for s in samples])
    results = []
    for label, idxs in sorted(groups.items()):
        if len(idxs) < 2:
            results.append({"label": label, "idxs": idxs,
                            "per_image": [], "median_rl1": np.nan,
                            "n_stable": 0, "note": "singleton — LOO undefined"})
            continue
        part_hists = all_hists[idxs]
        per_image = []
        for local_i, global_i in enumerate(idxs):
            s = samples[global_i]
            pred = loo_oracle_median(part_hists, local_i)
            rl1 = rel_l1(pred, s["hist"])
            per_image.append({"name": s["name"], "n_gt": s["n_gt"],
                               "img_mean": s["img_mean"], "rl1": rl1,
                               "stable": s["n_gt"] >= MIN_N_GT})
        stable_rl1 = [r["rl1"] for r in per_image
                      if r["stable"] and not np.isnan(r["rl1"])]
        med = float(np.median(stable_rl1)) if stable_rl1 else np.nan
        results.append({"label": label, "idxs": idxs, "per_image": per_image,
                        "median_rl1": med, "n_stable": len(stable_rl1), "note": ""})
    return results


def print_partition_results(name, results):
    print(f"\n{'='*70}")
    print(f"PARTITION: {name}")
    print(f"{'='*70}")
    for g in results:
        note = g["note"] or f"median_rl1={g['median_rl1']:.3f}  n_stable={g['n_stable']}"
        print(f"  [{g['label']}]  {note}")
        for r in g["per_image"]:
            flag = "  ✓" if r["stable"] else "  (unstable)"
            rl1_str = f"{r['rl1']:.3f}" if not np.isnan(r['rl1']) else "  nan"
            print(f"    {r['name'][-38:]:<40}  n_gt={r['n_gt']:4d}  "
                  f"mean={r['img_mean']:.3f}  rl1={rl1_str}{flag}")
    best = min((g for g in results if not np.isnan(g["median_rl1"])),
               key=lambda g: g["median_rl1"], default=None)
    if best:
        verdict = "PASS" if best["median_rl1"] <= 0.35 else (
                  "MARGINAL" if best["median_rl1"] <= 0.45 else "FAIL")
        print(f"\n  Best partition: [{best['label']}]  "
              f"median_rl1={best['median_rl1']:.3f}  → {verdict}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04"), nargs="?")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    bin_radii = build_radius_bins(cfg)
    print(f"Radius bins: {len(bin_radii)} levels, "
          f"r={bin_radii[0]:.2f}–{bin_radii[-1]:.2f}px", flush=True)

    # ── load ────────────────────────────────────────────────────────────────
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
        samples.append({
            "name":     img_path.stem,
            "img_mean": float(img.mean()),
            "n_gt":     len(bubbles),
            "hist":     hist,
            "session":  session_id(img_path.stem),
        })

    print(f"Loaded {len(samples)} tractable images", flush=True)

    # ── global cross-session oracle (E0-A baseline) ──────────────────────────
    all_hists = np.stack([s["hist"] for s in samples])
    global_rl1 = []
    for i, s in enumerate(samples):
        if s["n_gt"] < MIN_N_GT:
            continue
        pred = loo_oracle_median(all_hists, i)
        global_rl1.append(rel_l1(pred, s["hist"]))
    global_median = float(np.median(global_rl1)) if global_rl1 else np.nan
    print(f"\nGlobal LOO oracle (all sessions): median relL1 = {global_median:.3f}  "
          f"(E0-A baseline: 0.657)", flush=True)

    # ── partition A: session ─────────────────────────────────────────────────
    groups_session = partition_by(samples, "session")
    res_session = compute_partition_oracle(samples, groups_session, bin_radii)
    best_session = print_partition_results("Session identity", res_session)

    # ── partition B: img.mean quartile ───────────────────────────────────────
    means = np.array([s["img_mean"] for s in samples])
    quartiles = np.percentile(means, [25, 50, 75])
    def mean_bin(m):
        if m <= quartiles[0]: return "Q1_dark"
        elif m <= quartiles[1]: return "Q2"
        elif m <= quartiles[2]: return "Q3"
        else: return "Q4_bright"
    for s in samples:
        s["mean_bin"] = mean_bin(s["img_mean"])
    groups_mean = partition_by(samples, "mean_bin")
    res_mean = compute_partition_oracle(samples, groups_mean, bin_radii)
    best_mean = print_partition_results("Image brightness (img.mean quartile)", res_mean)

    # ── partition C: n_gt tercile ─────────────────────────────────────────────
    ngt_vals = np.array([s["n_gt"] for s in samples])
    t33, t67 = np.percentile(ngt_vals, [33, 67])
    def ngt_bin(n):
        if n <= t33: return "sparse"
        elif n <= t67: return "medium"
        else: return "dense"
    for s in samples:
        s["ngt_bin"] = ngt_bin(s["n_gt"])
    groups_ngt = partition_by(samples, "ngt_bin")
    res_ngt = compute_partition_oracle(samples, groups_ngt, bin_radii)
    best_ngt = print_partition_results("Bubble density (n_gt tercile)", res_ngt)

    # ── overall verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXPERIMENT D VERDICT")
    print(f"{'='*70}")
    print(f"  Global oracle:           {global_median:.3f}  (E0-A baseline)")

    bests = [(b, name) for b, name in [
        (best_session, "session"), (best_mean, "brightness"), (best_ngt, "density")]
        if b is not None and not np.isnan(b["median_rl1"])]
    if not bests:
        print("  No valid partitions found.")
        return

    overall_best_val = min(b["median_rl1"] for b, _ in bests)
    best_b, best_name = min(bests, key=lambda x: x[0]["median_rl1"])

    print(f"  Best partition found:    [{best_b['label']}] "
          f"via {best_name}  →  median_rl1={overall_best_val:.3f}")

    if overall_best_val <= 0.35:
        print(f"\n  PASS (≤0.35): regime conditioning is load-bearing.")
        print(f"  Regime-conditioned regression has a principled basis.")
        print(f"  → Proceed to image-feature ridge regression within [{best_b['label']}].")
    elif overall_best_val <= 0.45:
        print(f"\n  MARGINAL (0.35–0.45): weak regime signal.")
        print(f"  Regression may help marginally; not conclusive.")
    else:
        print(f"\n  FAIL (>0.45): regime conditioning does not reduce variance.")
        print(f"  Cross-image histogram heterogeneity is NOT explained by photometric")
        print(f"  regime, session, or bubble density. Per-image detection is required.")
        print(f"  Image-feature regression cannot reach relL1 ≤ 0.20.")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment D — Regime-conditional LOO oracle", fontsize=12)

    def plot_partition(ax, results, title):
        stable = [(g["label"], g["median_rl1"]) for g in results
                  if g["n_stable"] >= 1 and not np.isnan(g["median_rl1"])]
        if not stable:
            ax.set_title(title + "\n(no stable partitions)")
            return
        labels, vals = zip(*stable)
        colors = ["seagreen" if v <= 0.35 else ("orange" if v <= 0.45 else "tomato")
                  for v in vals]
        ax.bar(range(len(labels)), vals, color=colors, alpha=0.8)
        ax.axhline(global_median, color="gray", ls="--", lw=1.5,
                   label=f"Global oracle {global_median:.3f}")
        ax.axhline(0.35, color="seagreen", ls=":", lw=1.5, label="PASS threshold 0.35")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("LOO oracle relL1 (median, stable images)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(max(vals) * 1.2, 0.8))

    plot_partition(axes[0], res_session, "Session identity")
    plot_partition(axes[1], res_mean,    "Brightness quartile")
    plot_partition(axes[2], res_ngt,     "Bubble density tercile")

    out = OUT_DIR / "exp_d_regime_oracle.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out}", flush=True)


if __name__ == "__main__":
    main()
