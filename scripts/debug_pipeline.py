#!/usr/bin/env python3
"""
Diagnostic script for relL1 analysis.

USAGE:  python scripts/debug_pipeline.py [pipeline.pkl] [data_dir]

Checks, in order:
  1. GT radius distribution vs pyramid level coverage
  2. Per-level prediction vs GT (are fine scales over-counted?)
  3. Detection rate: for each GT bubble, does a high-score NMS survivor
     land within bubble.radius at the correct scale level?
  4. Cross-scale contamination: how many NMS survivors at the wrong level
     land within a GT bubble's radius?
  5. Score distributions of correct-level vs wrong-level survivors near GT bubbles
"""
import argparse
from pathlib import Path

import numpy as np

from bubble_histogram.calibration import nms_3d, _local_maxima, _lm_min_dist
from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps
from bubble_histogram.pipeline import BubblePipeline


def bin_gt(result, bubbles):
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
    rel_l1 = np.abs(pred - gt).sum() / max(gt.sum(), 1)
    return radii, pred, gt, edges, rel_l1


def matching_level(bubble_radius, ncc_results):
    """Index of the pyramid level whose eff_radius best matches this bubble."""
    eff_radii = np.array([r for r, _ in ncc_results])
    return int(np.argmin(np.abs(eff_radii - bubble_radius)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=Path, default=Path("output/pipeline.pkl"), nargs="?")
    parser.add_argument("data_dir", type=Path, default=Path("seed_v04/"), nargs="?")
    args = parser.parse_args()

    pipeline = BubblePipeline.load(args.pipeline)
    cfg = pipeline.config
    ds = AnnotatedDataset(args.data_dir, template_frac=0.30, calibration_frac=0.65, seed=42)

    all_images = list(ds.calibration_images) + list(ds.test_images)

    # ------------------------------------------------------------------ #
    # 1. GT radius distribution vs pyramid eff_radius coverage
    # ------------------------------------------------------------------ #
    print("=" * 65)
    print("1. GT RADIUS DISTRIBUTION")
    print("=" * 65)
    all_radii = [b.radius for p in all_images for b in ds.load_sample(p).bubbles]
    r = np.array(all_radii)
    print(f"   n={len(r)}  min={r.min():.1f}  p25={np.percentile(r,25):.1f}  "
          f"median={np.median(r):.1f}  p75={np.percentile(r,75):.1f}  max={r.max():.1f}")
    sample0 = ds.load_sample(all_images[0])
    ncc_r0 = compute_ncc_maps(sample0.image, pipeline.templates, cfg)
    eff_radii = np.array([er for er, _ in ncc_r0])
    print(f"\n   Pyramid eff_radius range: {eff_radii.min():.1f} – {eff_radii.max():.1f} px  "
          f"({len(eff_radii)} levels)")
    print(f"   Fraction of GT bubbles within pyramid range: "
          f"{((r >= eff_radii.min()) & (r <= eff_radii.max())).mean():.1%}")

    # ------------------------------------------------------------------ #
    # 2. Per-level prediction vs GT (one image)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("2. PER-LEVEL PREDICTION vs GT  (first calibration image)")
    print("=" * 65)
    s = sample0
    result = pipeline.predict(s.image)
    radii, pred, gt, edges, rl1 = bin_gt(result, s.bubbles)
    print(f"   relL1={rl1:.3f}  pred_total={pred.sum():.0f}  gt_total={gt.sum()}")
    print(f"\n   {'Level':>5}  {'eff_r':>6}  {'pred':>7}  {'gt':>6}  {'ratio':>6}")
    for i, (er, p, g) in enumerate(zip(radii, pred, gt)):
        ratio = p / g if g > 0 else float("inf")
        flag = " <--" if abs(ratio - 1) > 1 and (p > 5 or g > 5) else ""
        print(f"   {i:>5}  {er:>6.1f}  {p:>7.1f}  {g:>6}  {ratio:>6.2f}{flag}")

    # ------------------------------------------------------------------ #
    # 3. Detection rate: does the right NMS survivor find each GT bubble?
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("3. DETECTION RATE PER GT BUBBLE  (first calibration image)")
    print("=" * 65)
    ncc_r = compute_ncc_maps(s.image, pipeline.templates, cfg)
    survivors = nms_3d(ncc_r, cfg)

    matched_correct = 0   # highest-score survivor within radius at matching level
    matched_wrong   = 0   # highest-score survivor within radius but at wrong level
    missed          = 0   # no survivor within radius at any level

    correct_scores  = []
    wrong_scores    = []

    for b in s.bubbles:
        best_level = matching_level(b.radius, ncc_r)
        best_correct = None   # (score, level) within radius at correct level
        best_wrong   = None   # (score, level) within radius at wrong level

        for score, lv, y, x, _ in survivors:
            dist = np.sqrt((y - b.cy)**2 + (x - b.cx)**2)
            if dist < b.radius:
                if lv == best_level:
                    if best_correct is None or score > best_correct[0]:
                        best_correct = (score, lv)
                else:
                    if best_wrong is None or score > best_wrong[0]:
                        best_wrong = (score, lv)

        if best_correct is not None:
            matched_correct += 1
            correct_scores.append(best_correct[0])
        elif best_wrong is not None:
            matched_wrong += 1
            wrong_scores.append(best_wrong[0])
        else:
            missed += 1

    total = len(s.bubbles)
    print(f"   GT bubbles: {total}")
    print(f"   Detected at CORRECT level:    {matched_correct} ({matched_correct/total:.1%})  "
          f"mean_score={np.mean(correct_scores):.3f}" if correct_scores else
          f"   Detected at CORRECT level:    {matched_correct} ({matched_correct/total:.1%})")
    print(f"   Detected at WRONG level only: {matched_wrong}  ({matched_wrong/total:.1%})  "
          f"mean_score={np.mean(wrong_scores):.3f}" if wrong_scores else
          f"   Detected at WRONG level only: {matched_wrong}  ({matched_wrong/total:.1%})")
    print(f"   Missed entirely:              {missed}  ({missed/total:.1%})")

    # ------------------------------------------------------------------ #
    # 4. Cross-scale contamination: how many survivors near each GT bubble
    #    are at the wrong level and score high?
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("4. CROSS-SCALE CONTAMINATION NEAR GT BUBBLES  (first cal image)")
    print("=" * 65)

    survivors_near = {b_idx: {"correct": [], "wrong": []}
                      for b_idx in range(len(s.bubbles))}

    for score, lv, y, x, _ in survivors:
        for b_idx, b in enumerate(s.bubbles):
            dist = np.sqrt((y - b.cy)**2 + (x - b.cx)**2)
            if dist < b.radius * 2:   # within 2× radius
                best_lv = matching_level(b.radius, ncc_r)
                if lv == best_lv:
                    survivors_near[b_idx]["correct"].append(score)
                else:
                    survivors_near[b_idx]["wrong"].append(score)

    total_correct_near = sum(len(v["correct"]) for v in survivors_near.values())
    total_wrong_near   = sum(len(v["wrong"])   for v in survivors_near.values())
    print(f"   Survivors within 2×radius of any GT bubble:")
    print(f"     At correct level: {total_correct_near}")
    print(f"     At wrong level:   {total_wrong_near}   "
          f"({total_wrong_near/(total_correct_near+total_wrong_near):.1%} of near-GT survivors)")

    all_wrong_scores = [s for v in survivors_near.values() for s in v["wrong"]]
    all_corr_scores  = [s for v in survivors_near.values() for s in v["correct"]]
    if all_wrong_scores:
        print(f"     Wrong-level scores:   mean={np.mean(all_wrong_scores):.3f}  "
              f"max={np.max(all_wrong_scores):.3f}")
    if all_corr_scores:
        print(f"     Correct-level scores: mean={np.mean(all_corr_scores):.3f}  "
              f"max={np.max(all_corr_scores):.3f}")

    # ------------------------------------------------------------------ #
    # 5. Per-level survivor count vs expected for the GT distribution
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("5. SURVIVOR COUNT vs EXPECTED (by level, first cal image)")
    print("=" * 65)
    level_survivors = {}
    for _, lv, *_ in survivors:
        level_survivors[lv] = level_survivors.get(lv, 0) + 1

    print(f"   {'Level':>5}  {'eff_r':>6}  {'survivors':>10}  {'gt_in_bin':>10}")
    for i, (er, g) in enumerate(zip(radii, gt)):
        ns = level_survivors.get(i, 0)
        flag = " <-- over" if ns > g * 10 and ns > 20 else ""
        print(f"   {i:>5}  {er:>6.1f}  {ns:>10}  {g:>10}{flag}")


if __name__ == "__main__":
    main()
