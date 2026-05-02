# Bubble Pipeline — Experiment Log

## Problem

The pipeline predicts ~89 bubbles on a calibration image with 492 annotated bubbles. The pipeline uses NCC on a 27-level scale pyramid, cross-scale IoU NMS to select detections, and a per-level ScoreCalibrator to convert NCC scores to expected counts.

## KPI

**relL1** (relative L1 error) = `sum|predicted_bin − gt_bin| / sum(gt_bin)` across size bins, evaluated on the first calibration image. Lower is better; 0 = perfect.  
Current best: **0.950** (per-level calibration, commit `d4e5ad7`).

## Glossary

- **NCC** — Normalized Cross-Correlation. Score in [−1, 1] measuring how well the bubble template matches a patch of the image at a given position and scale.
- **LM** — Local Maximum. A pixel whose NCC score is higher than all neighbours within a fixed radius.
- **NMS** — Non-Maximum Suppression. Greedy algorithm that keeps the highest-scoring detection and suppresses any lower-scoring detection whose bounding box overlaps it (IoU > threshold). Here applied cross-scale across the pyramid.
- **delta** — Level offset from the correct pyramid level for a given GT bubble. `delta = competitor_level − correct_level`. Negative = finer (more zoomed-in) scale; positive = coarser.
- **eff_r** — Effective radius at a pyramid level (in original-image pixels). The image is scaled so that a bubble of this size appears at `canonical_r` pixels in the scaled image.
- **canonical_r** — Template half-size in score-map pixels = `template_size / (2 × context_factor)` = 5.0 px. The pyramid level where `eff_r ≈ bubble.radius` is the "correct" level for that bubble.
- **rescue rate** — Fraction of GT bubbles for which the correct-level LM score (after applying a normalisation transform) exceeds all zone-competitor scores.
- **PAL** — External AI agent (separate Claude instance) consulted for independent, unbiased diagnosis. PAL has access to the codebase but is given no conclusions in advance.
- **relL1** — see KPI above.

---

## Completed Experiments

### E1 · Per-level calibration with scale-aware labeling
**Why:** The original global ScoreCalibrator could not distinguish a fine-scale edge artifact (high NCC score) from a true bubble center at the correct scale. The hypothesis was that one calibrator per pyramid level, with positives restricted to NMS survivors at the correct scale level, would fix this.  
**What:** Replaced global ScoreCalibrator with one per level. Labeled NMS survivors positive only when at the correct pyramid level AND within `bubble.radius` of a GT annotation. Added per-level NMS cap (`nms_max_candidates_per_level=1000`) and optimized the NMS inner loop.  
**Result:** relL1 improved from 1.024 → 0.950, training time ~46s.  
**Conclusion:** Marginal improvement, but still 10× under-prediction. The per-level calibrators are trained on NMS survivors, and only 11.6% of GT bubbles have a correct-level NMS survivor — so per-level priors remain near zero. The fix is built on a circular foundation: the calibrator is trained on survivors of the same broken NMS it is meant to correct.

---

### E2 · PAL independent diagnosis #1
**Why:** Sanity-check the root-cause hypothesis before committing to a fix.  
**What:** Shared code and debug output with PAL (an independent AI agent with codebase access). Asked for diagnosis without stating any suspected root cause.  
**Result:** PAL confirmed NMS eviction as the primary failure: wrong-level peaks score ~32% higher on average (0.715 vs 0.542 mean), win the IoU competition within ±3 adjacent levels, and correct-level peaks rarely survive. PAL also identified that the scale-aware labeling is circular — positives come only from the rare cases NMS did not already evict.  
**Conclusion:** Agreed. The circular dependency between NMS survivorship and calibrator training is the structural failure to address.

---

### E3 · Raw LMs vs NMS survivors at the correct level
**Why:** Distinguish two candidate mechanisms for the under-prediction. Mechanism A (NMS eviction): the correct-level LM exists in the raw score map but NMS suppresses it before labeling can see it. Mechanism B (greedy labeling inversion): the correct-level LM survives NMS but a wrong-level survivor arrives first in score order and claims the GT annotation slot.  
**What:** For each of 492 GT bubbles on one calibration image, checked whether a correct-level LM existed within `bubble.radius` in the raw score map (before NMS) vs in the NMS survivor list.  
**Result:** 95.1% of GT bubbles had a correct-level raw LM within radius; only 11.6% had a correct-level NMS survivor. NMS eviction rate = 87.8%.  
**Conclusion:** Mechanism A dominates. The NCC template produces valid responses at the correct scale for almost all bubbles — NMS discards 88% of them. Mechanism B is secondary.

---

### E4 · Global calibrator with centroid positives and random-pixel negatives
**Why:** Test whether removing the NMS-survivorship bias from calibration training would fix the counts. If a calibrator trained on GT center scores (positives) and random background pixels (negatives) gives accurate counts, then the NMS bias is the bottleneck, not the NCC feature itself.  
**What:** Switched `local_maxima_calibration=False`. Training used annotated bubble center NCC scores as positives and randomly sampled pixel NCC scores as negatives. Prior = n_bubbles / n_pixels.  
**Result:** relL1 worsened to 1.128. Fine-scale edge artifact NMS survivors score ~0.7–0.9, above the training positive distribution (~0.4–0.6), so the calibrator assigned them high P(bubble). Level 0 predicted 51 vs 0 GT.  
**Conclusion:** No calibration rearrangement can fix what NMS selects. The calibrator receives wrong-level survivors and cannot distinguish them from true bubbles. The fix must happen upstream of calibration, at the NMS or feature level.

---

### E5 · Scale normalisation sweep
**Why:** Fine-scale NCC scores are higher than correct-scale scores. Multiplying each candidate score by `(eff_r / canonical_r)^alpha` before NMS penalises fine scales and might restore the correct ordering.  
**What:** For 468 GT bubbles that had both a correct-level raw LM and at least one competitor within the IoU suppression zone (±3 adjacent levels), checked whether the normalised correct-level score beat all normalised zone-competitor scores. `alpha=0` is the current baseline (no normalisation). Tested alpha ∈ {0.5, 1.0, 2.0}.  
**Result:** Rescue rate (fraction of GT bubbles where the correct-level normalised score beats all zone competitors) was 7.5% at alpha=0, reaching only 13% at alpha=2.0. The score ratio distribution between competitor and correct-level scores is heavy-tailed: median 1.53×, mean 2.14×, with many cases above 3×.  
**Conclusion:** Monotonic scale normalisation is falsified. The score gap is too large and too variable for any fixed exponent to overcome.

---

### E6 · Direction flip under normalisation
**Why:** Understand why the rescue rate plateaus at 13% even at alpha=2.  
**What:** For each GT bubble, identified which zone level (delta = competitor_level − correct_level) held the highest-scoring competitor before and after alpha=2 normalisation.  
**Result:** Before normalisation: 86% of hardest competitors are at finer scales (delta < 0), dominated by delta=−3 (46%). After alpha=2: 78% shift to coarser scales (delta > 0), dominated by delta=+3 (40%). There is no alpha value at which the correct level consistently wins.  
**Conclusion:** The normalisation transfers dominance from fine to coarse scales without passing through a sweet spot. The correct-level NCC peak never sits at the top of the scale-space response — the averaged template produces a broad ridge of moderate-to-high NCC responses spanning ±3–4 levels in both directions. Scale normalisation is not the fix.

---

### E7 · PAL independent diagnosis #2
**Why:** Share the full experiment results (E3–E6) for independent critical review before deciding next steps.  
**What:** Shared experiments E3–E6 with PAL without stating any conclusions.  
**Result:** PAL identified three independent root causes:
(1) **Template scale offset** — the single averaged template (`num_templates=1`) pools bubbles from 3–45px (15× range), likely encoding an effective canonical radius of ~3.5–4px rather than the assumed 5px. This causes NCC to systematically peak 2–3 levels finer than expected.
(2) **NCC is not scale-neutral** — fine scales score higher due to higher image resolution and INTER_AREA smoothing at coarser levels, making raw NCC score a biased proxy for bubble confidence.
(3) **Circular calibration** — 411 correct-level peaks are invisible to the trainer; if random sampling lands on them, the calibrator learns "score ~0.38 = background."
PAL also flagged that `_iou()` uses square boxes, making the delta=±3 suppression boundary sensitive to the threshold.  
**Conclusion:** All three root causes must be addressed. The highest-leverage intervention is to confirm the template scale offset first (diagnostic O1), then test per-level NMS without cross-scale competition (O2), then scale-specific templates (O3).

---

## Completed Experiments (continued)

### E8 · O1 — NCC response curve across scale levels per GT bubble
**Why:** PAL E7 predicted the NCC peak falls at delta=−2 to −3 (the template encodes an effective canonical radius of ~3.5–4px). If confirmed, adjusting `template_context_factor` to realign the scale could be a low-cost fix.  
**What:** For 39 GT bubbles sampled evenly across log-radius space, computed the maximum NCC score within `bubble.radius` at each of 27 pyramid levels. Recorded mean score per delta and the delta at which each bubble's curve peaks. Script: `scripts/profile_ncc_response.py`.  
**Result:** Mean NCC score is monotonically decreasing from fine to coarse: 0.592 at delta=−6, 0.469 at delta=0, 0.320 at delta=+6. There is no peak — the curve is a downward slope across the entire ±6 delta range (slope ≈ −0.023 per level). Median peak delta = −4.0 (14 of 39 bubbles peak at delta=−6, the finest level measured).  
**Conclusion:** PAL's "offset by 2–3 levels" hypothesis is superseded by a more severe finding: NCC has **no scale-selective peak at all**. Finer scales always score higher due to higher image resolution, regardless of bubble size or template alignment. Adjusting `context_factor` cannot fix a monotone slope — there is no scale to realign to. The cross-scale NMS approach with raw NCC scores is structurally broken: it will always select the finest viable scale, not the correct one. The fix requires a scale-selective feature (LoG/DoG) that produces a genuine peak at the correct scale.

---

---

### E9 · O2 — Per-level independent NMS (no cross-scale competition)
**Why:** If cross-scale NMS is the mechanism evicting correct-level peaks, removing it should allow all correct-level peaks to survive and be scored by their per-level calibrator.  
**What:** Set `nms_iou_threshold=0.0`. Retrained `_train_per_level()` on raw per-level LMs: for each pyramid level, raw LMs are labeled positive if within `bubble.radius` of a GT bubble whose correct level is that level; all other raw LMs become negatives. Prediction: each level runs independent 2D NMS, and each level's calibrator accumulates P(bubble) independently. Script: `scripts/run_o2.py`.  
**Result:** relL1 = 23.080. pred_total = 11,847 vs gt_total = 492 (24× over-prediction). Fine-scale levels each contribute 600–1,600 predicted bubbles against near-zero ground truth. The per-level calibrators at fine scales (levels 0–7, eff_r 3–7px) correctly assign high P(bubble) to high-NCC peaks — but there are ~1,000 such peaks per level because NCC gives high scores to edge artifacts at every scale regardless of whether a bubble is present.  
**Conclusion:** Hypothesis falsified. Per-level independent NMS makes the problem catastrophically worse. Cross-scale NMS was the only mechanism suppressing the thousands of fine-scale edge artifact peaks. Without it, each fine-level calibrator (which cannot distinguish edge artifacts from real small bubbles at the same NCC score) independently contributes massive over-counting. This reinforces E8: NCC has no scale selectivity, so removing cross-scale competition does not expose a latent correct-scale signal — it unleashes every scale's spurious detections. The pipeline requires a genuinely scale-selective feature before any counting approach can work.

---

## Open Experiments (pending)

### O3 · Scale-specific templates (`num_templates > 1`)
**Hypothesis:** A single averaged template blurs scale-specific appearance across a 15× size range. Separate templates per size bin should narrow the NCC ridge in scale space, making the correct-level peak more discriminative relative to adjacent-level competitors.  
**Falsification:** Train with `num_templates=4`, re-run E3. If correct-level raw LM scores do not rise relative to adjacent-level competitor scores, template diversity is not the lever.
