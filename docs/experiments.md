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

### O3 · Scale-specific templates (`num_templates > 1`) — CLOSED WITHOUT RUNNING
**Hypothesis:** A single averaged template blurs scale-specific appearance across a 15× size range. Separate templates per size bin should narrow the NCC ridge in scale space, making the correct-level peak more discriminative relative to adjacent-level competitors.  
**Why ruled out by E8 (consensus with PAL):** O3 requires that the cross-scale score gradient is caused by template-shape mismatch. E8 shows it is not: the −0.023/level monotone slope is a signal-density artifact — finer scales retain more high-frequency texture and ZNCC (a cosine similarity) scores higher on richer gradient fields regardless of template shape. E9 confirms: 600–1,600 spurious peaks per fine level, none real bubbles. These artifacts score high under any template because the image at fine scale has abundant local high-contrast structure for any small kernel to lock onto. Template diversity cannot create a scale-space peak where none exists in the physics of the feature. Scale normalisation (E5/E6) already showed a multiplicative score boost at δ=0 achieves only 13% rescue at α=2, with median competitor advantage 1.53×; template-induced perturbations at δ=0 would be far smaller. **O3 is logically falsified by the evidence already collected and should not be run.**

---

## Architecture Verdict

### NCC cannot solve this dataset as annotated

The E1–E9 chain, reviewed independently by PAL and confirmed by cross-experiment consistency, closes every classical NCC-based escape route:

| Intervention | Experiment | Result | Mechanism closed |
|---|---|---|---|
| Per-level calibration | E1 | relL1 0.950 | Circular: trained on NMS survivors |
| Global calibrator, GT positives | E4 | relL1 1.128 | Edge artifacts outscore GT centers |
| Scale normalization | E5/E6 | ≤13% rescue | Score gap median 1.53×, heavy-tailed |
| Per-level independent NMS | E9 | 24× over-prediction | No cross-scale suppression of edge artifacts |
| Scale-specific templates | O3 | Not run | Ruled out by E8 physics argument |

Root cause: **NCC is not a scale-selective feature**. Its scale-space response is monotonically biased toward fine scales by signal density, not by template fit. Cross-scale NMS over an NCC pyramid will always select the finest viable scale (evicting 88% of correct-level peaks); removing cross-scale NMS exposes thousands of fine-scale edge artifacts per level. There is no NMS regime and no calibrator design that resolves this contradiction.

Additional hard floors:
- **~50% photometrically dead frames** — any per-frame appearance detector averages in near-random predictions on half the dataset.
- **GT label contamination** (low-circularity objects) — metric noise from bad labels is uncorrectable without cleaning.
- **4 appearance regimes** — a single model averages across incompatible photometric conditions.

---

## Completed Experiments (continued)

### E10 · LoG scale-space response diagnostic (INCONCLUSIVE — redesign required)
**Hypothesis:** LoG (Laplacian of Gaussian) has a genuine scale-selective peak near delta=0 for GT bubbles AND that peak is significantly higher than LoG at non-bubble background locations at the same scale. Both conditions must hold for LoG to be a viable NCC replacement.  
**Design:** Sample GT bubbles randomly from all non-saturated images (img_mean < 150) to cover multiple photometric regimes and bubble morphologies. For each bubble: measure max scale-normalized LoG response within `bubble.radius` at each pyramid level (sensitivity curve). At the correct level, also measure LoG at 20 random background locations > 3×radius from any GT bubble (specificity sample). Script: `scripts/experiments/profile_log_response.py`.

**Known bugs in E10 design (identified in post-hoc review):**
1. **Saturation filter bug:** `load_image` returns float [0,1], so `img_mean < 150` always passes. All 14/14 images are included regardless of saturation (likely harmless since no truly saturated frames in this set, but the filter never actually ran).
2. **Asymmetric background sampling:** Bubble metric = `max(|LoG|)` over a disk of radius `bubble.radius`. Background metric = single-pixel `|LoG|` (line 88). A real detector takes the max in a region; single-pixel sampling underestimates the background score a detector would face and inflates SNR.
3. **Background only measured at delta=0:** No per-scale SNR profile. Whether fine-scale or coarse-scale is more discriminative is unknown.
4. **Bubble morphology never examined:** The sigma = canonical_r/√2 choice was made for filled-Gaussian-blob theory. Microscopy bubbles may be optically ring-like (bright ring, uniform/dark interior), in which case the center-LoG theory does not apply and sigma = ring_wall_width/√2 is the correct choice, independent of ring radius.

**Result:**
```
Non-saturated images: 14/14 (see bug 1)
Bubbles profiled: 55
Background samples: 1045

delta   mean_|LoG|   median   n_valid
  -6      0.1355     0.1219      45   ← HIGHEST
  -5      0.1329     0.1174      45
  -4      0.1300     0.1151      48
  -3      0.1246     0.1133      52
  -2      0.1199     0.1098      54
  -1      0.1152     0.1047      55
  +0      0.1112     0.0949      55   ← correct level
  +1      0.1054     0.0930      55
  +2      0.0988     0.0950      55
  +3      0.0932     0.0935      51
  +4      0.0873     0.0829      49
  +5      0.0809     0.0748      48
  +6      0.0746     0.0671      48

Peak delta per bubble: mean=-2.76, median=-5.0, p25=-6.0, p75=0.0

Bubble |LoG| at delta=0: mean=0.1112, median=0.0949
Background |LoG| at delta=0: mean=0.0193, median=0.0095  (single-pixel, inflated SNR)
SNR at delta=0: 5.77× (overestimate due to asymmetric sampling)
```
Auto-verdict: **FALSIFIED (sensitivity)** — median peak = -5.0, far from delta=0.

**Critical analysis (reviewed with PAL, consensus reached):**

The auto-verdict is **correct for what was measured but incorrect about what was measured.** The measurement conflates two hypotheses:
- **H1 (scale-selectivity via max-in-region):** Falsified. The monotone curve is a measurement artifact, not a property of LoG.
- **H2 (LoG discriminability at delta=0):** NOT tested by the auto-verdict. 5.77× SNR is a real signal, though inflated by asymmetric sampling.

**Mechanistic explanation of monotone curve:** `max(|LoG|)` inside a disk with FIXED sigma=3.54px produces a monotone response by construction. At delta=-6, bubble subtends 8.84px in the scaled image (sigma/r ≈ 0.40 — edge-detection regime): the LoG picks up the sharp circular bubble boundary ring, not the blob interior. At delta=+6, sigma >> bubble_r_in_scaled — the blob is over-smoothed and |LoG| collapses. There is no pyramid level where the measurement switches from edge-response to blob-response; the max-in-disk always returns the dominant local feature, which is the edge at fine scales. LoG blob-detection theory (center response peaks at sigma = r/√2, i.e., delta=0) was never actually measured.

**LoG blob theory is also contingent on bubble morphology.** If bubbles are ring-like (gas-liquid interface → bright ring, dark interior), the optimal sigma is ring_wall_width/√2, not ring_radius/√2. The center LoG would be near zero at all scales for a ring-shaped bubble. E11 with center sampling would also see a monotone decline in that case — not because LoG fails, but because the physical model is wrong.

**Conclusion:** E10 is inconclusive. It does not falsify LoG as a feature; it falsifies a specific (flawed) measurement protocol. The 5.77× SNR at delta=0 is a positive preliminary signal. Before designing E11, two prerequisite questions must be answered: (1) Are bubbles filled disks or ring-like in this imaging modality? (2) Does background LoG grow at finer scales (degrading SNR), or stay proportional?

**Path to E11:**
1. Plot intensity cross-sections across ~10 GT bubbles across sizes (20 min) — determines filled vs. ring morphology and correct LoG model.
2. Re-run with (a) center-pixel LoG curve vs. delta, (b) symmetric background measurement (`max(|LoG|)` in small region), (c) background measured at ALL delta levels to expose SNR(delta).
3. If bubble is ring-like, test sigma tuned to ring width rather than ring radius.
4. Only then declare LoG viable or falsified.

---

## Open Experiments (pending)

### E11 · Bubble morphology cross-section + corrected LoG discriminability test

**Step 1 — Morphology survey — COMPLETED (script: `scripts/experiments/profile_morphology.py`)**

Ran radial intensity profiles for 28 bubbles across 14 images. Three structural defects in the measurement identified in post-hoc review with PAL (see below), but the qualitative picture is consistent.

**Raw morphology counts (r ≥ 8px only; sub-pixel bins excluded for r < 8px):**
- dark-rim (bright interior, dark ring at boundary): 14/26 = 54%  
- filled-dark (dark center, lighter surround): 7/26 = 27%  
- bright-rim / filled-bright: 3/26 = 12%  
- flat / indeterminate: 2/26 = 8%  

**95% CI on dark-rim prevalence (Wilson): [34%, 72%]** — preliminary evidence only, not a strong prevalence claim.

**Three measurement defects confirmed by PAL consensus:**
1. Rim sampled at r/R=0.85 (interior side of dark band) — minimum is at or just outside r/R=1.0. Rim contrast is underestimated.
2. Sub-pixel bins for r < 8px (20 bins over [0, 2.0R], each bin = 0.1R; at r=4.6px that is 0.46px/bin). Morphology classifications for r < 8px are unreliable.
3. Rim width never extracted — profiles exist but weren't reduced to a rim FWHM estimate. This is needed to choose σ for the ring detector model.

**Key finding for Step 2 design (PAL consensus):** For dark-rim bubbles with bright interior, center-pixel LoG at σ ≈ R/√2 is **not near-zero**. The dark rim falls in the negative annular lobe of the LoG kernel, reinforcing the center response. The correct question for Step 2 is whether that response *forms a scale-space peak at δ=0* — which is unresolved and testable.

**Step 2 — Corrected LoG discriminability — COMPLETED**  
Script: `scripts/experiments/profile_log_e11.py`

Corrected protocol (relative to E10):
- **Two sigma values tested:** σ = R/√2 (blob model, scales with R) and σ = 2px constant (ring/rim model — rim width does not scale with bubble radius if it is set by diffraction/PSF, not bubble geometry)
- **Two measurement locations per bubble per sigma:** center pixel (r=0) and rim pixel (r≈R, specifically the bin nearest the annotated bubble boundary)
- **Symmetric background:** `max(|LoG|)` in a 3px-radius disk at 20 random locations > 3×R from any GT bubble
- **Background at ALL delta levels** — exposes SNR(delta) curve, not just SNR at δ=0
- **Exclude r < 8px bubbles** from scale-space peak claims  
- **Image filter:** `img.mean() < 0.6` (float scale)

**Raw results (52 GT bubbles, r≥8px, 14 images):**
```
delta    ctr_blob    rim_blob     ctr_rim     rim_rim    bg_blob     bg_rim
   -6      0.0721      0.1142      0.0655      0.1199     0.0402     0.0439
   -5      0.0747      0.1089      0.0690      0.1172     0.0411     0.0464
   -4      0.0797      0.1044      0.0728      0.1168     0.0454     0.0437
   -3      0.0870      0.0994      0.0642      0.1149     0.0426     0.0481
   -2      0.0892      0.0938      0.0708      0.1128     0.0443     0.0497
   -1      0.0886      0.0893      0.0709      0.1104     0.0441     0.0503
   +0      0.0876      0.0863      0.0740      0.1040     0.0474     0.0529
   +1      0.0834      0.0851      0.0818      0.0988     0.0473     0.0516
   +2      0.0754      0.0827      0.0844      0.0942     0.0460     0.0501
   +3      0.0688      0.0821      0.0838      0.0888     0.0453     0.0518
   +4      0.0619      0.0800      0.0808      0.0851     0.0469     0.0523
   +5      0.0544      0.0794      0.0804      0.0824     0.0463     0.0557
   +6      0.0506      0.0767      0.0773      0.0820     0.0482     0.0548

Peak delta statistics (per bubble):
  center_blob:      mean=-2.10, median=-2.0, p25=-6.0, p75=+0.2   IQR=6.2
  rim_blob:         mean=-3.46, median=-5.5, p25=-6.0, p75=-2.8   IQR=3.2
  center_rim_sigma: mean=+0.63, median=+1.0, p25=-2.0, p75=+4.0   IQR=6.0
  rim_rim_sigma:    mean=-2.48, median=-4.5, p25=-6.0, p75=+1.0   IQR=7.0

SNR at delta=0 (bubble single-pixel or rim-max / bg max-in-3px-disk):
  center_blob:      1.85×  (asymmetric comparison — see note below)
  rim_blob:         1.82×
  center_rim_sigma: 1.40×
  rim_rim_sigma:    1.97×
```

**Auto-verdicts (script):**
```
FALSIFIED(sensitivity): center_blob (median_peak=-2.0, outside ±1.5)
FALSIFIED(sensitivity): rim_blob (median_peak=-5.5)
FALSIFIED(specificity): center_rim_sigma (peak at +1.0 ✓, SNR=1.40×<2×)
FALSIFIED(sensitivity): rim_rim_sigma (median_peak=-4.5)
```

**Critical analysis (reviewed with PAL, consensus reached):**

The auto-verdicts are **correct in outcome but wrong in diagnosis for center_blob**, and miss the most damning finding entirely.

**Finding 1 — IQR=6+ levels is the fatal result, not SNR:**  
All four curves have per-bubble IQR spanning ≥6 pyramid levels (48–54% of the measured delta range). p25=−6.0 for center_blob means ≥25% of bubbles peak at the finest measured scale — these have no scale-space peak at all. SNR of population means is irrelevant when per-bubble scale-space peaks are this scattered. **IQR=6 is fatal for any fixed-parameter detector. Tuning sigma shifts the population mean; it cannot compress the variance.**

Mechanistic explanation: E11 Step 1 found four bubble morphologies (dark-rim 54%, filled-dark 27%, bright-rim 12%, flat 8%). No single (location, sigma) combination produces a compact scale-space peak for all morphologies simultaneously. The IQR is structural, not parametric.

**Finding 2 — Three-way aggregation asymmetry biases center curves against bubbles:**  
The measurement footprints are not comparable:
- center_log: 1 px² (single pixel)
- background_max: ~28 px² (π·3² disk)
- rim_log: ~47 px² at delta=0 (annulus 0.85–1.15R at r_s≈5px)

Since all three report a `max()`, larger footprint → larger expected maximum from image texture. This means:
- center_blob / center_rim_sigma SNRs are **pessimistically biased** — background max is inflated vs. single-pixel bubble value
- rim_blob / rim_rim_sigma SNRs are **optimistically biased** — rim max aggregates ~1.7× more area than background

Under symmetric single-pixel comparison, center_blob SNR at delta=0 would rise from 1.85× to approximately **2.4–2.7×** — passing the 2× threshold. **center_blob's specificity failure is a measurement artifact, not a genuine signal deficiency.** This does not change the verdict (IQR=6.2 is still fatal) but center_blob should not be recorded as "low SNR."

**Finding 3 — center_blob delta=−2 offset is a population-average artifact, not a sigma error:**  
The actual sigma mismatch at delta=−2 is ~9% (sigma=3.54px vs. optimal 3.89px for a 1.026× pyramid level scale). This cannot shift the peak by 2 full levels. The delta=−2 mean peak is a weighted average over bubbles that peak near 0 (dark-rim, correct model), bubbles that peak at −6 (fine-bias, same failure as NCC), and scattered others. Redefining "correct level" as delta=−2 would be circular — the IQR=6.2 makes it meaningless.

**Conclusion: LoG as a fixed-parameter, fixed-location feature is conclusively falsified for this dataset.**  
Three independent failure modes:
1. **Sensitivity** (rim curves): mean peaks far from delta=0
2. **Specificity** (center_rim_sigma): SNR=1.40× below threshold even at population mean
3. **Reliability** (all curves): per-bubble IQR ≥ 6 levels — morphological heterogeneity prevents any single-parameter LoG from being scale-selective across the bubble population

---

## Open Experiments (pending)

### E12 · Hough circle transform diagnostic

**Hypothesis to falsify:** "Hough circle transform on gradient magnitude produces a per-bubble radius estimate within ≤2 pyramid-levels of the GT correct level for ≥70% of bubbles (r≥8px), with a background false-positive rate of ≤5 per image at that threshold."

**Motivation:** LoG falsified because no single (location, sigma) spans four morphological classes. Hough operates on gradient magnitude — directly exploits the dark-rim edge structure (dominant morphology, 54%) without requiring a model of the bubble interior. Built-in scale selection via the circle-radius accumulator: the accumulator peak radius is the detection radius, eliminating the scale-pyramid bias problem entirely. Known failure modes (gradient threshold sensitivity, thin rings at fine scale) are tunable and diagnosable.

**Design:**
- For each GT bubble (r≥8px), run OpenCV `HoughCircles` (or manual Hough accumulator) on a patch centered at the bubble. Record the peak accumulator radius vs. GT radius.
- Convert radius error to pyramid-level offset using the pyramid scale formula.
- Background: count Hough detections in random patches > 3R from any GT bubble.
- Stratify results by morphology class (dark-rim vs. filled-dark) — Hough should work best for dark-rim and worst for filled-dark.
- Failure criterion: if ≥30% of GT bubbles have radius error > 2 levels OR background FP rate > 5 per image, Hough is falsified and morphology-conditioned detectors must be evaluated.
