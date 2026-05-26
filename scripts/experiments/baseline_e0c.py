#!/usr/bin/env python3
"""
E0-C — Within-session LOO oracle.

Hypothesis to falsify:
  "Within-session LOO median relL1 ≤ 0.45 for at least one session,
   meaning images from the same physical run are similar enough that
   intra-session averaging achieves near-target accuracy. If confirmed,
   session-conditional lookup may be actionable without per-image detection."

Design:
  - For each session with ≥2 stable images (n_gt ≥ 100), compute the LOO
    median GT histogram using only images from the same session.
  - Compare per-image within-session relL1 against the cross-session oracle
    from E0-A (reproduced here for direct comparison).
  - Summary: within-session median relL1 per session; overall median;
    improvement over cross-session oracle.
  - Pre-committed falsification criterion:
      within-session median relL1 < 0.45 in ANY session → PASS (actionable)
      all sessions ≥ 0.45                               → FAIL (no benefit)

Caveats:
  - C1S0024 has only 1 image — within-session LOO impossible.
  - Sessions with exactly 2 stable images: LOO is a 1-sample predictor
    (predict image A from image B alone). Interpret with caution.
  - The user has indicated the sparse-bubble regime (C1S0004_005070,
    C1S0010_019655, oracle > 1.0) is out of scope. Results including
    those images are reported but not used for the primary verdict.

USAGE: python scripts/experiments/baseline_e0c.py [data_dir]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import load_image, parse_annotations, get_session_id

MAX_IMG_MEAN = 0.6
MIN_N_GT     = 100

# E0-A cross-session oracle values (reproduced for comparison)
E0A_ORACLE = {
    "C1S0014_img006001":        0.536,
    "C1S0014_img009542":        0.705,
    "C1S0014_img018008":        0.781,
    "C1S0014_img018351":        0.502,
    "C1S0019_img003593":        0.543,
    "C1S0019_img011890":        0.432,
    "C1S0024_img014500":        0.548,
    "C1S0004_IMG_S0001004509":  0.755,
    "C1S0004_IMG_S0001005070":  1.462,
    "C1S0004_IMG_S0001012062":  0.609,
    "C1S0010_IMG_S0001005432":  0.740,
    "C1S0010_IMG_S0001019655":  1.172,
}

# Images the user has explicitly scoped OUT (sparse-bubble regime)
OUT_OF_SCOPE = {"C1S0004_IMG_S0001005070", "C1S0010_IMG_S0001019655"}


def build_radius_bins(cfg: PipelineConfig) -> np.ndarray:
    radii = []
    r = cfg.min_radius
    while r <= cfg.max_radius * 1.001:
        radii.append(r)
        r = r / cfg.scale_factor
    return np.array(radii)


def assign_bin(bubble_r: float, bin_radii: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.log(bin_radii) - np.log(bubble_r))))


def gt_histogram(bubbles, bin_radii: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bin_radii))
    for b in bubbles:
        if b.radius < bin_radii[0] * 0.5 or b.radius > bin_radii[-1] * 2:
            continue
        counts[assign_bin(b.radius, bin_radii)] += 1
    return counts


def rel_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    total = gt.sum()
    if total == 0:
        return np.nan
    return float(np.abs(pred - gt).sum() / total)


def lookup_e0a(stem: str) -> float | None:
    for key, val in E0A_ORACLE.items():
        if key in stem:
            return val
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out", type=Path, default=Path("output/e0c_within_session_oracle.png"))
    args = parser.parse_args()

    cfg       = PipelineConfig()
    bin_radii = build_radius_bins(cfg)
    data_dir  = args.data_dir

    # ── Load all tractable annotated images ───────────────────────────────────
    samples = []
    for img_path in sorted((data_dir / "images").glob("*.png")):
        lbl_path = data_dir / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue
        img = load_image(img_path)
        if img.mean() >= MAX_IMG_MEAN:
            continue
        bubbles   = parse_annotations(lbl_path)
        hist      = gt_histogram(bubbles, bin_radii)
        session   = get_session_id(img_path.name)
        oos       = any(k in img_path.stem for k in OUT_OF_SCOPE)
        samples.append({
            "path":      img_path,
            "stem":      img_path.stem,
            "session":   session,
            "img_mean":  float(img.mean()),
            "n_bubbles": len(bubbles),
            "hist":      hist,
            "stable":    len(bubbles) >= MIN_N_GT,
            "oos":       oos,
        })

    sessions = sorted(set(s["session"] for s in samples))
    print(f"Loaded {len(samples)} tractable images across sessions: {sessions}")

    # ── Cross-session oracle (E0-A reproduced) ────────────────────────────────
    all_hists = np.stack([s["hist"] for s in samples])
    for i, s in enumerate(samples):
        mask = np.ones(len(samples), dtype=bool)
        mask[i] = False
        s["cross_loo"] = rel_l1(np.median(all_hists[mask], axis=0), s["hist"])
        s["e0a_oracle"] = lookup_e0a(s["stem"])

    # ── Within-session LOO ────────────────────────────────────────────────────
    print(f"\n{'Image':<44}  {'n_gt':>5}  {'within':>8}  {'cross':>8}  {'Δ':>7}  {'oos':>4}  session")
    print("-" * 100)

    session_results = {}
    for sess in sessions:
        sess_stable = [s for s in samples if s["session"] == sess and s["stable"]]
        n = len(sess_stable)

        for s in sess_stable:
            s["within_loo"] = float("nan")

        if n < 2:
            for s in sess_stable:
                flag = "*OOS*" if s["oos"] else ""
                print(f"  {s['stem'][-42:]:<42}  {s['n_bubbles']:>5}  "
                      f"{'N/A':>8}  {s['cross_loo']:>8.3f}  {'—':>7}  {flag:>4}  {sess}")
            continue

        sess_hists = np.stack([s["hist"] for s in sess_stable])
        within_vals, cross_vals = [], []
        for i, s in enumerate(sess_stable):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            loo_pred = np.median(sess_hists[mask], axis=0)
            s["within_loo"] = rel_l1(loo_pred, s["hist"])
            flag = "*OOS*" if s["oos"] else ""
            delta = s["within_loo"] - s["cross_loo"]
            print(f"  {s['stem'][-42:]:<42}  {s['n_bubbles']:>5}  "
                  f"{s['within_loo']:>8.3f}  {s['cross_loo']:>8.3f}  {delta:>+7.3f}  {flag:>4}  {sess}")
            if not s["oos"]:
                within_vals.append(s["within_loo"])
                cross_vals.append(s["cross_loo"])

        session_results[sess] = {
            "within": within_vals,
            "cross":  cross_vals,
            "n":      len(within_vals),
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SESSION SUMMARY (in-scope stable images, excluding *OOS*):")
    print(f"{'Session':<12}  {'n':>3}  {'within_med':>11}  {'cross_med':>10}  {'Δ':>8}  {'<0.45?':>8}")

    all_within, all_cross = [], []
    for sess in sessions:
        if sess not in session_results or session_results[sess]["n"] == 0:
            continue
        r = session_results[sess]
        wm = float(np.median(r["within"]))
        cm = float(np.median(r["cross"]))
        all_within.extend(r["within"])
        all_cross.extend(r["cross"])
        flag = "YES" if wm < 0.45 else "no"
        print(f"  {sess:<10}  {r['n']:>3}  {wm:>11.3f}  {cm:>10.3f}  {wm-cm:>+8.3f}  {flag:>8}")

    overall_within = float(np.median(all_within)) if all_within else float("nan")
    overall_cross  = float(np.median(all_cross))  if all_cross  else float("nan")
    print(f"\n  Overall median within-session relL1 (in-scope): {overall_within:.3f}")
    print(f"  Overall median cross-session relL1 (in-scope):  {overall_cross:.3f}")
    print(f"  Improvement Δ = cross − within:                 {overall_cross - overall_within:+.3f}")

    print(f"\n{'='*70}")
    print("VERDICT:")
    any_below_45 = any(
        np.median(r["within"]) < 0.45
        for r in session_results.values() if r["n"] > 0
    )
    if any_below_45:
        passing = [s for s, r in session_results.items()
                   if r["n"] > 0 and np.median(r["within"]) < 0.45]
        print(f"  HYPOTHESIS SURVIVES: session(s) {passing} achieve within-LOO median < 0.45.")
        print("  Session-conditional lookup may be actionable for these sessions.")
        print("  Within-session histogram consistency is higher than cross-session.")
    else:
        print(f"  HYPOTHESIS FALSIFIED: no session achieves within-session LOO median < 0.45.")
        print("  Within-session and cross-session oracle are similar.")
        print("  Histogram variability is not explained by session identity.")
        print("  Session-conditional lookup adds no value over the cross-session oracle.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-image within vs cross oracle (in-scope stable only)
    ax = axes[0]
    in_scope = [s for s in samples if s["stable"] and not s["oos"]
                and not np.isnan(s.get("within_loo", float("nan")))]
    in_scope.sort(key=lambda s: s["session"])
    names = [s["stem"][-22:] for s in in_scope]
    x = np.arange(len(in_scope))
    ax.bar(x - 0.2, [s["within_loo"] for s in in_scope],
           width=0.4, color="steelblue", alpha=0.8, label="Within-session LOO")
    ax.bar(x + 0.2, [s["cross_loo"] for s in in_scope],
           width=0.4, color="seagreen",  alpha=0.8, label="Cross-session LOO (E0-A)")
    ax.axhline(0.45, color="red", linestyle="--", linewidth=1.5,
               label="0.45 actionability threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("relL1")
    ax.set_title(f"Within-session vs cross-session oracle\n(in-scope stable images)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: scatter within vs cross
    ax = axes[1]
    w_vals = [s["within_loo"] for s in in_scope]
    c_vals = [s["cross_loo"]  for s in in_scope]
    ax.scatter(c_vals, w_vals, s=70, alpha=0.9, color="steelblue", edgecolors="none")
    lim = max(max(w_vals), max(c_vals)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="within = cross")
    ax.axhline(0.45, color="red", linestyle=":", linewidth=1.2,
               label="0.45 threshold")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Cross-session LOO relL1 (E0-A)")
    ax.set_ylabel("Within-session LOO relL1 (E0-C)")
    ax.set_title("Within vs cross oracle scatter")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for s in in_scope:
        ax.annotate(s["session"], (s["cross_loo"], s["within_loo"]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    fig.suptitle(f"E0-C — Within-session LOO oracle ({len(in_scope)} in-scope stable images)",
                 fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
