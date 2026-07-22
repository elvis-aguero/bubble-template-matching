#!/usr/bin/env python3
"""
Fig 2/5/6-style comparison + relL1 summary, ALL 14 seed_v04 images:
GT vs Hough vs Oscar classical deterministic vs Oscar FRST+SAM3 hybrid
("hybrid_current", the actual state-of-the-art method from the Oscar
Bubble-tracking repo, freshly rerun on a GPU node 2026-07-14 -- not the
standalone classical scripts) vs ours (LOSO per session).

Hybrid instances come from bubble_frst_sam3_mask.py's consolidated ("COMBINED")
output, which already reports radius_equiv_px per instance.

USAGE
-----
  python scripts/seed_v04_gt_comparison_all14_with_hybrid.py \\
      --oscar-json-dir /tmp/oscar_pull/detect_bubbles_out_14 \\
      --hybrid-json-dir /tmp/oscar_pull/hybrid_out_14
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, get_session_id
from bubble_histogram.pipeline import BubblePipeline

plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12

CANNY_HIGH = 50
DP = 1
MIN_DIST = 8
PARAM2 = 20
MIN_N_GT = 100


def img_to_uint8(img: np.ndarray) -> np.ndarray:
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def run_hough(img: np.ndarray, min_r: int, max_r: int) -> np.ndarray:
    img_u8 = img_to_uint8(img)
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_DIST,
        param1=CANNY_HIGH, param2=PARAM2, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return np.array([])
    return circles[0][:, 2]


def load_oscar_json(json_dir: Path, image_path: Path) -> np.ndarray:
    path = json_dir / f"detected_{image_path.stem}.json"
    data = json.loads(path.read_text())
    return np.array([2 * r for (_, _, r) in data])


def load_hybrid_json(json_dir: Path, image_path: Path) -> np.ndarray:
    path = json_dir / f"{image_path.stem}_analysis.json"
    data = json.loads(path.read_text())
    return np.array([2 * inst["radius_equiv_px"] for inst in data["instances"]])


def rel_l1(pred_hist: np.ndarray, gt_hist: np.ndarray) -> float:
    total = gt_hist.sum()
    if total == 0:
        return float("nan")
    return float(np.abs(pred_hist - gt_hist).sum() / total)


def bin_diams(diam_px: np.ndarray, bin_diam_edges: np.ndarray) -> np.ndarray:
    return np.histogram(diam_px, bins=bin_diam_edges)[0].astype(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-dir", type=Path, default=Path("seed_v04/"))
    parser.add_argument("--min-radius", type=float, default=3.0)
    parser.add_argument("--max-radius", type=float, default=50.0)
    parser.add_argument("--oscar-json-dir", type=Path, required=True)
    parser.add_argument("--hybrid-json-dir", type=Path, required=True)
    parser.add_argument("--heatmap-bins", type=int, default=25)
    parser.add_argument("--heatmap-max-diam", type=float, default=60.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/test17_zerog_opt3/"))
    args = parser.parse_args()

    cfg = PipelineConfig(min_radius=args.min_radius, max_radius=args.max_radius,
                         local_maxima_calibration=True)

    all_paths = sorted((args.seed_dir / "images").glob("*.png"))
    sessions = sorted(set(get_session_id(p.name) for p in all_paths))

    rows = []
    ours_bin_diam_px = None

    for val_session in sessions:
        print(f"\n=== LOSO fold: holding out {val_session} ===")
        dataset = AnnotatedDataset(args.seed_dir, val_session=val_session)
        pipeline = BubblePipeline(cfg)
        pipeline.train(dataset)

        for img_path in dataset.test_images:
            sample = dataset.load_sample(img_path)
            gt_diam = np.array([2 * b.radius for b in sample.bubbles])
            n_gt = len(gt_diam)

            result = pipeline.predict(sample.image)
            radius_px = np.array(result["radius_px"])
            ours_counts = np.array(result["expected_count"])
            if ours_bin_diam_px is None:
                ours_bin_diam_px = 2 * radius_px

            hough_diam = 2 * run_hough(sample.image, min_r=int(args.min_radius), max_r=int(args.max_radius))
            oscar_diam = load_oscar_json(args.oscar_json_dir, img_path)
            hybrid_diam = load_hybrid_json(args.hybrid_json_dir, img_path)

            rows.append(dict(
                stem=img_path.stem, session=val_session, n_gt=n_gt,
                gt_diam=gt_diam, hough_diam=hough_diam, oscar_diam=oscar_diam,
                hybrid_diam=hybrid_diam, ours_counts=ours_counts,
            ))
            print(f"  {img_path.stem[-28:]:<28}  n_gt={n_gt:4d}  ours={ours_counts.sum():7.2f}  "
                  f"hough={len(hough_diam):5d}  oscar={len(oscar_diam):5d}  hybrid={len(hybrid_diam):5d}")

    # ========================================================================
    # relL1 SUMMARY
    # ========================================================================
    log_r = np.log(ours_bin_diam_px / 2)
    half = (log_r[1] - log_r[0]) / 2
    edges = np.exp(np.concatenate([[log_r[0] - half], (log_r[:-1] + log_r[1:]) / 2, [log_r[-1] + half]])) * 2

    print(f"\n{'image':<28} {'n_gt':>5} {'relL1_hough':>12} {'relL1_oscar':>12} "
          f"{'relL1_hybrid':>13} {'relL1_ours':>11}")
    relL1 = {k: [] for k in ["hough", "oscar", "hybrid", "ours"]}
    for r in rows:
        gt_hist = bin_diams(r["gt_diam"], edges)
        hough_hist = bin_diams(r["hough_diam"], edges)
        oscar_hist = bin_diams(r["oscar_diam"], edges)
        hybrid_hist = bin_diams(r["hybrid_diam"], edges)
        ours_hist = r["ours_counts"]

        rl = dict(
            hough=rel_l1(hough_hist, gt_hist),
            oscar=rel_l1(oscar_hist, gt_hist),
            hybrid=rel_l1(hybrid_hist, gt_hist),
            ours=rel_l1(ours_hist, gt_hist),
        )
        stable = r["n_gt"] >= MIN_N_GT
        r.update({f"relL1_{k}": v for k, v in rl.items()})
        r["stable"] = stable
        if stable:
            for k in relL1:
                relL1[k].append(rl[k])
        flag = "" if stable else " *"
        print(f"{r['stem'][-28:]:<28} {r['n_gt']:>5} {rl['hough']:>12.3f} {rl['oscar']:>12.3f} "
              f"{rl['hybrid']:>13.3f} {rl['ours']:>11.3f}{flag}")

    print(f"\n{'='*76}")
    print("MEDIAN relL1 (stable images, n_gt>=100), vs established "
          "E0-A oracle=0.657 and E0-B pipeline=0.851:")
    print(f"  Hough:                  {np.median(relL1['hough']):.3f}")
    print(f"  Oscar deterministic:    {np.median(relL1['oscar']):.3f}")
    print(f"  Oscar FRST+SAM3 hybrid: {np.median(relL1['hybrid']):.3f}")
    print(f"  Ours (LOSO):            {np.median(relL1['ours']):.3f}")
    print("  * unstable (n_gt<100), excluded from medians")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # FIG 2 — Overall distribution, pooled across all 14
    # ========================================================================
    gt_all = np.concatenate([r["gt_diam"] for r in rows])
    hough_all = np.concatenate([r["hough_diam"] for r in rows])
    oscar_all = np.concatenate([r["oscar_diam"] for r in rows])
    hybrid_all = np.concatenate([r["hybrid_diam"] for r in rows])
    ours_total_per_bin = np.stack([r["ours_counts"] for r in rows]).sum(axis=0)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bins = np.linspace(0, args.heatmap_max_diam, 30)
    ax.hist(gt_all, bins=bins, color="black", histtype="step", linewidth=2.5,
            label=f"Ground truth ({len(gt_all)} bubbles, 14 images)")
    ax.hist(hough_all, bins=bins, color="tomato", alpha=0.35,
            label=f"Hough ({len(hough_all)})")
    ax.hist(oscar_all, bins=bins, color="mediumseagreen", alpha=0.35,
            label=f"Oscar deterministic ({len(oscar_all)})")
    ax.hist(hybrid_all, bins=bins, color="darkorchid", alpha=0.35,
            label=f"Oscar FRST+SAM3 hybrid ({len(hybrid_all)})")
    ax2 = ax.twinx()
    ax2.bar(ours_bin_diam_px, ours_total_per_bin,
            width=np.diff(ours_bin_diam_px).mean() if len(ours_bin_diam_px) > 1 else 1.0,
            color="steelblue", alpha=0.35, label="Ours (expected count, LOSO)")
    ax.set_xlabel("Bubble Diameter (px)")
    ax.set_ylabel("Count (GT, Hough, Oscar, hybrid)")
    ax2.set_ylabel("Expected count (ours)")
    ax.set_xlim(0, args.heatmap_max_diam)
    ax.set_title("Overall Bubble Size Distribution — all 14 seed_v04 images (real GT)")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_all14_hybrid_fig2.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIG 5 — Count per image (log scale)
    # ========================================================================
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["session"], rows[i]["stem"]))
    labels = [rows[i]["stem"][-16:] for i in order]
    gt_counts = [rows[i]["n_gt"] for i in order]
    hough_counts = [len(rows[i]["hough_diam"]) for i in order]
    oscar_counts = [len(rows[i]["oscar_diam"]) for i in order]
    hybrid_counts = [len(rows[i]["hybrid_diam"]) for i in order]
    ours_counts_tot = [rows[i]["ours_counts"].sum() for i in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.semilogy(x, gt_counts, "o-", color="black", label="Ground truth")
    ax.semilogy(x, hough_counts, "o-", color="tomato", label="Hough")
    ax.semilogy(x, oscar_counts, "o-", color="mediumseagreen", label="Oscar deterministic")
    ax.semilogy(x, hybrid_counts, "o-", color="darkorchid", label="Oscar FRST+SAM3 hybrid")
    ax.semilogy(x, ours_counts_tot, "o-", color="steelblue", label="Ours (expected, LOSO)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Number of Bubbles (log scale)")
    ax.set_title("Bubble Count per Image, log scale — all 14 seed_v04 images (real GT)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_all14_hybrid_fig5_log.png", dpi=150)
    plt.close(fig)

    # ========================================================================
    # FIG 6 — heatmap panels, all 14 images
    # ========================================================================
    bins = np.linspace(0, args.heatmap_max_diam, args.heatmap_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    def density_matrix(diam_lists):
        mat = np.zeros((len(bins) - 1, len(diam_lists)))
        for k, d in enumerate(diam_lists):
            if len(d) > 0:
                mat[:, k] = np.histogram(d, bins=bins, density=True)[0]
        return mat

    gt_mat = density_matrix([rows[i]["gt_diam"] for i in order])
    hough_mat = density_matrix([rows[i]["hough_diam"] for i in order])
    oscar_mat = density_matrix([rows[i]["oscar_diam"] for i in order])
    hybrid_mat = density_matrix([rows[i]["hybrid_diam"] for i in order])
    ours_mat = np.zeros((len(bins) - 1, len(order)))
    for col, i in enumerate(order):
        counts_k = rows[i]["ours_counts"]
        total_k = counts_k.sum()
        if total_k > 0:
            interp_counts = np.interp(bin_centers, ours_bin_diam_px, counts_k, left=0, right=0)
            ours_mat[:, col] = interp_counts / (total_k * bin_width)

    fig, axes = plt.subplots(1, 5, figsize=(26, 6), sharey=True)
    for ax, mat, title in zip(axes, [gt_mat, hough_mat, oscar_mat, hybrid_mat, ours_mat],
                               ["Ground Truth", "Hough", "Oscar deterministic",
                                "Oscar FRST+SAM3 hybrid", "Ours (LOSO)"]):
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="hot",
                        extent=[0, len(order) - 1, 0, args.heatmap_max_diam])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=75, fontsize=7)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, label="PDF", fraction=0.046)
    axes[0].set_ylabel("Bubble Diameter (px)")
    fig.suptitle("Bubble Size Distribution — all 14 seed_v04 images (real GT)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "gt_all14_hybrid_fig6.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved gt_all14_hybrid_fig2/5log/6 to {args.out_dir}/")


if __name__ == "__main__":
    main()
