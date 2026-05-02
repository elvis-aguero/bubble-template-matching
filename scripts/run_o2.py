#!/usr/bin/env python3
"""
O2 — Per-level independent NMS experiment.

Hypothesis: Removing cross-scale IoU suppression eliminates scale bias from NMS.
Each level runs its own spatial NMS independently; the calibrator is trained on
per-level raw LMs (not NMS survivors), breaking the circular dependency.

Config: nms_iou_threshold=0.0, local_maxima_calibration=True.

USAGE: python scripts/run_o2.py [data_dir] [--out output/pipeline_o2.pkl]
"""
import argparse
import time
from pathlib import Path

import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps
from bubble_histogram.pipeline import BubblePipeline


def rel_l1(result, bubbles):
    radii = np.array(result["radius_px"])
    pred = np.array(result["expected_count"])
    log_r = np.log(radii)
    half = (log_r[1] - log_r[0]) / 2 if len(log_r) > 1 else 0.1
    edges = np.exp(np.concatenate([
        [log_r[0] - half],
        (log_r[:-1] + log_r[1:]) / 2,
        [log_r[-1] + half],
    ]))
    gt = np.histogram([b.radius for b in bubbles], bins=edges)[0]
    return float(np.abs(pred - gt).sum() / max(gt.sum(), 1)), pred, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    parser.add_argument("--out", type=Path, default=Path("output/pipeline_o2.pkl"))
    args = parser.parse_args()

    cfg = PipelineConfig(
        nms_iou_threshold=0.0,       # per-level 2D NMS; no cross-scale suppression
        local_maxima_calibration=True,
    )

    print("O2 config:")
    print(f"  nms_iou_threshold       = {cfg.nms_iou_threshold}")
    print(f"  local_maxima_calibration = {cfg.local_maxima_calibration}")
    print(f"  neg_sample_ratio        = {cfg.neg_sample_ratio}")

    ds = AnnotatedDataset(args.data_dir, template_frac=0.30, calibration_frac=0.65, seed=42)
    print(f"\nDataset split — template: {len(list(ds.template_images))}, "
          f"calibration: {len(list(ds.calibration_images))}, "
          f"test: {len(list(ds.test_images))}")

    t0 = time.time()
    pipeline = BubblePipeline(cfg)
    pipeline.train(ds)
    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")

    # Print per-level calibrator priors
    if pipeline.calibrators:
        print("\nPer-level calibrator priors (positive count / total):")
        ncc0 = compute_ncc_maps(ds.load_sample(list(ds.calibration_images)[0]).image,
                                pipeline.templates, cfg)
        eff_radii = [er for er, _ in ncc0]
        n_pos_total = 0
        for lv, cal in sorted(pipeline.calibrators.items()):
            prior = float(cal.p_bubble_given_score.max()) if cal.p_bubble_given_score is not None else 0.0
            er = eff_radii[lv] if lv < len(eff_radii) else float("nan")
            n_pos_total += 1
            print(f"  Level {lv:2d}  eff_r={er:5.1f}  max_P(bubble)={prior:.3f}")

    # Evaluate on first calibration image (same as debug_pipeline.py)
    cal_images = list(ds.calibration_images)
    sample = ds.load_sample(cal_images[0])
    result = pipeline.predict(sample.image)
    rl1, pred, gt = rel_l1(result, sample.bubbles)

    print(f"\n{'=' * 55}")
    print(f"relL1 on first calibration image: {rl1:.3f}")
    print(f"  pred_total={pred.sum():.0f}  gt_total={gt.sum()}")
    print(f"{'=' * 55}")

    print(f"\n{'Level':>5}  {'eff_r':>6}  {'pred':>7}  {'gt':>6}  {'ratio':>6}")
    for i, (er, p, g) in enumerate(zip(result["radius_px"], pred, gt)):
        ratio = p / g if g > 0 else float("inf")
        flag = " <--" if abs(ratio - 1) > 1 and (p > 5 or g > 5) else ""
        print(f"  {i:>5}  {er:>6.1f}  {p:>7.1f}  {g:>6}  {ratio:>6.2f}{flag}")

    # Baseline comparison
    print("\nBaseline E1 (cross-scale NMS + NMS-survivor calibration): relL1 = 0.950")
    if rl1 < 0.950:
        print(f"O2 IMPROVES over E1: {rl1:.3f} vs 0.950 (delta = {rl1 - 0.950:+.3f})")
    else:
        print(f"O2 does NOT improve over E1: {rl1:.3f} vs 0.950 (delta = {rl1 - 0.950:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(args.out)
    print(f"\nPipeline saved to {args.out}")


if __name__ == "__main__":
    main()
