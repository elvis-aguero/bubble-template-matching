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

    # ------------------------------------------------------------------ #
    # 6. EXPERIMENT 1 — raw LMs vs NMS survivors at the correct level
    #
    # Falsifies which mechanism drives under-prediction:
    #   Mech A (NMS eviction):        correct-level LM exists in raw score map
    #                                 but is absent from nms_3d() survivors.
    #   Mech B (labeling inversion):  correct-level peak survives NMS but wrong-
    #                                 level survivors arrive first in score order
    #                                 and consume the GT annotation slot.
    #
    # Decision rule:
    #   raw_hits >> nms_hits  →  Mech A dominates  (NMS is evicting real peaks)
    #   raw_hits ≈ nms_hits   →  correct-level peaks absent even before NMS
    #                            → template / NCC quality is the root limit
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("6. EXPERIMENT 1 — raw LMs vs NMS survivors at correct level")
    print("=" * 65)

    min_d = _lm_min_dist(cfg)
    eff_radii_arr = np.array([er for er, _ in ncc_r])
    canonical_r = cfg.template_size / (2.0 * cfg.template_context_factor)

    raw_hits = 0      # GT bubbles with a correct-level raw LM within bubble.radius
    nms_hits = 0      # GT bubbles with a correct-level NMS survivor within bubble.radius
    raw_only = 0      # correct-level raw LM exists but NMS suppressed it
    neither  = 0      # no correct-level peak at all (even in raw score map)

    per_level_raw  = {}   # level → count of GT bubbles with correct-level raw hit
    per_level_nms  = {}

    for b in s.bubbles:
        best_lv = int(np.argmin(np.abs(eff_radii_arr - b.radius)))
        eff_r, score_map = ncc_r[best_lv]
        alpha = cfg.template_size / (2.0 * cfg.template_context_factor * eff_r)

        # raw: any LM in the correct-level score map within bubble.radius
        peaks = _local_maxima(score_map, min_d)
        ys = peaks[:, 0] / alpha
        xs = peaks[:, 1] / alpha
        dists = np.hypot(ys - b.cy, xs - b.cx)
        has_raw = bool((dists < b.radius).any())

        # nms: any correct-level NMS survivor within bubble.radius
        has_nms = any(
            lv == best_lv and np.hypot(y - b.cy, x - b.cx) < b.radius
            for _, lv, y, x, _ in survivors
        )

        if has_raw:
            raw_hits += 1
            per_level_raw[best_lv] = per_level_raw.get(best_lv, 0) + 1
        if has_nms:
            nms_hits += 1
            per_level_nms[best_lv] = per_level_nms.get(best_lv, 0) + 1

        if has_raw and not has_nms:
            raw_only += 1
        elif not has_raw:
            neither += 1

    total_b = len(s.bubbles)
    print(f"\n   GT bubbles: {total_b}")
    print(f"   Correct-level raw LM within radius:   {raw_hits:4d}  ({raw_hits/total_b:.1%})")
    print(f"   Correct-level NMS survivor within r:  {nms_hits:4d}  ({nms_hits/total_b:.1%})")
    print(f"   Raw LM existed but NMS evicted it:    {raw_only:4d}  ({raw_only/total_b:.1%})  ← Mech A signal")
    print(f"   No correct-level peak at all:         {neither:4d}  ({neither/total_b:.1%})  ← template quality limit")

    if raw_hits > 0:
        eviction_rate = raw_only / raw_hits
        print(f"\n   NMS eviction rate (of raw hits): {eviction_rate:.1%}")
        if eviction_rate > 0.5:
            print("   → Mechanism A (NMS eviction) is the PRIMARY offender")
        else:
            print("   → Mechanism A is minor; template/NCC quality dominates")

    print(f"\n   Per-level breakdown  (levels with GT bubbles only):")
    print(f"   {'Level':>5}  {'eff_r':>6}  {'gt':>5}  {'raw_hits':>9}  {'nms_hits':>9}  {'evicted':>8}")
    for i, (er, g) in enumerate(zip(radii, gt)):
        if g == 0:
            continue
        rh = per_level_raw.get(i, 0)
        nh = per_level_nms.get(i, 0)
        ev = rh - nh
        print(f"   {i:>5}  {er:>6.1f}  {g:>5}  {rh:>9}  {nh:>9}  {ev:>8}")

    # ------------------------------------------------------------------ #
    # 7. EXPERIMENT 2 — scale normalization falsification
    #
    # Hypothesis: multiplying NCC scores by (eff_r / canonical_r)^alpha
    # before cross-scale NMS would allow the correct-level peak to outscore
    # wrong-level competitors, fixing NMS eviction without retraining.
    #
    # alpha=0  → no normalization (current state)
    # alpha>0  → fine-scale responses penalised relative to coarse-scale
    #
    # For each GT bubble that has a correct-level raw LM:
    #   1. Record its best correct-level raw LM score.
    #   2. Find the highest-scoring raw LM within bubble.radius at each
    #      competing level in the IoU suppression zone (levels where
    #      IoU > nms_iou_threshold with the correct level).
    #   3. Apply normalisation to all scores and check whether the
    #      correct-level normalised score beats every competitor.
    #
    # Decision rule:
    #   rescue_rate >> 0  →  normalisation would fix NMS ordering for many
    #                        GT bubbles  →  hypothesis survives, worth implementing
    #   rescue_rate ≈ 0   →  correct-scale NCC response is intrinsically weaker
    #                        even proportionally  →  normalisation is not the fix
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("7. EXPERIMENT 2 — scale normalisation falsification")
    print("=" * 65)

    # Build IoU suppression zone for each level.
    # IoU of two centered squares with half-sides r_i, r_j:
    #   IoU = (min(r_i, r_j) / max(r_i, r_j))^2
    # Suppressed when IoU > threshold.
    iou_thr = cfg.nms_iou_threshold
    footprints = eff_radii_arr * cfg.template_context_factor   # half-side of NMS box
    suppression_zone: list[list[int]] = []
    for i in range(len(eff_radii_arr)):
        zone = []
        for j in range(len(eff_radii_arr)):
            if i == j:
                continue
            ratio = min(footprints[i], footprints[j]) / max(footprints[i], footprints[j])
            if ratio ** 2 > iou_thr:
                zone.append(j)
        suppression_zone.append(zone)

    alphas = [0.0, 0.5, 1.0, 2.0]
    # rescued[alpha] = GT bubbles where normalised correct-level score > all zone competitors
    rescued      = {a: 0 for a in alphas}
    has_competitor = 0   # GT bubbles with a correct-level raw LM AND at least one competitor

    # Pre-cache raw LMs and scores per level (avoids recomputing for each bubble)
    level_peaks_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for lv, (er, sm) in enumerate(ncc_r):
        alp = cfg.template_size / (2.0 * cfg.template_context_factor * er)
        pks = _local_maxima(sm, min_d)
        if len(pks):
            sc  = sm[pks[:, 0], pks[:, 1]]
            ys_ = pks[:, 0] / alp
            xs_ = pks[:, 1] / alp
        else:
            sc  = np.array([])
            ys_ = np.array([])
            xs_ = np.array([])
        level_peaks_cache.append((ys_, xs_, sc))

    for b in s.bubbles:
        best_lv = int(np.argmin(np.abs(eff_radii_arr - b.radius)))
        ys_, xs_, sc_ = level_peaks_cache[best_lv]
        if len(sc_) == 0:
            continue
        dists_ = np.hypot(ys_ - b.cy, xs_ - b.cx)
        in_r = dists_ < b.radius
        if not in_r.any():
            continue
        correct_score = float(sc_[in_r].max())

        # Find best competing score in each suppression-zone level
        competitor_scores: list[float] = []
        for comp_lv in suppression_zone[best_lv]:
            yc, xc, scc = level_peaks_cache[comp_lv]
            if len(scc) == 0:
                continue
            dc = np.hypot(yc - b.cy, xc - b.cx)
            in_rc = dc < b.radius
            if in_rc.any():
                competitor_scores.append(float(scc[in_rc].max()))

        if not competitor_scores:
            continue
        has_competitor += 1
        max_comp = max(competitor_scores)

        for a in alphas:
            norm_correct = correct_score * (eff_radii_arr[best_lv] / canonical_r) ** a
            norm_comp    = max_comp      * (eff_radii_arr[
                # competitor level with max_comp score — approximate: use all competitors
                # re-evaluate per-alpha by normalising each competitor individually
                0] / canonical_r) ** a   # placeholder; recomputed below
            # Re-evaluate properly: normalise each competitor at its own level
            best_norm_comp = -np.inf
            for comp_lv in suppression_zone[best_lv]:
                yc, xc, scc = level_peaks_cache[comp_lv]
                if len(scc) == 0:
                    continue
                dc = np.hypot(yc - b.cy, xc - b.cx)
                in_rc = dc < b.radius
                if not in_rc.any():
                    continue
                raw_comp = float(scc[in_rc].max())
                norm_c = raw_comp * (eff_radii_arr[comp_lv] / canonical_r) ** a
                if norm_c > best_norm_comp:
                    best_norm_comp = norm_c
            if best_norm_comp == -np.inf:
                continue
            norm_correct = correct_score * (eff_radii_arr[best_lv] / canonical_r) ** a
            if norm_correct > best_norm_comp:
                rescued[a] += 1

    print(f"\n   GT bubbles with correct-level raw LM AND zone competitor: {has_competitor}")
    print(f"\n   {'alpha':>6}  {'normalisation':>20}  {'rescued':>8}  {'rescue_rate':>12}  verdict")
    for a in alphas:
        rate = rescued[a] / max(has_competitor, 1)
        norm_label = {0.0: "none (current)", 0.5: "sqrt(eff_r/r0)",
                      1.0: "linear (eff_r/r0)", 2.0: "squared (eff_r/r0)²"}.get(a, f"^{a}")
        verdict = ("would fix most"   if rate > 0.7 else
                   "partial fix"      if rate > 0.3 else
                   "not sufficient")
        print(f"   {a:>6.1f}  {norm_label:>20}  {rescued[a]:>8}  {rate:>11.1%}  {verdict}")

    print(f"\n   Interpretation:")
    best_a = max(alphas, key=lambda a: rescued[a])
    best_rate = rescued[best_a] / max(has_competitor, 1)
    if best_rate > 0.7:
        print(f"   → Scale normalisation (alpha={best_a}) rescues {best_rate:.0%} of evicted GT bubbles.")
        print(f"     Hypothesis survives: worth implementing score *= (eff_r/canonical_r)^alpha before NMS.")
    elif best_rate > 0.3:
        print(f"   → Scale normalisation gives partial improvement (best alpha={best_a}, {best_rate:.0%} rescued).")
        print(f"     Correct-scale NCC responses are somewhat weaker; normalisation alone may not suffice.")
    else:
        print(f"   → Scale normalisation does not rescue GT bubbles (best {best_rate:.0%}).")
        print(f"     The correct-scale NCC response is intrinsically weaker; normalisation is not the fix.")


if __name__ == "__main__":
    main()
