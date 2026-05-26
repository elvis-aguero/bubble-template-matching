# Bubble Pipeline — Experiment Log

## Problem

The pipeline predicts ~89 bubbles on a calibration image with 492 annotated bubbles. The pipeline uses NCC on a 27-level scale pyramid, cross-scale IoU NMS to select detections, and a per-level ScoreCalibrator to convert NCC scores to expected counts.

## KPI

**relL1** (relative L1 error) = `sum|predicted_bin − gt_bin| / sum(gt_bin)` across size bins. Lower is better; 0 = perfect. **Target: 0.1** (deployment minimum).

| Benchmark | relL1 | Notes |
|---|---|---|
| Pipeline single-image (C1S0024) | 0.950 | commit `d4e5ad7`; n=1, selection-biased, NOT a valid estimate |
| Pipeline LOSO median (E0-B) | **0.851** | 12 stable images, cross-image LOSO evaluation |
| LOO oracle median (E0-A) | **0.657** | GT-access lower bound; requires annotations from other images |
| Target | **0.20** | Revised from 0.10 (2026-05-03) — relL1 ~0.2 accepted by user |

The pipeline currently loses to a GT-oracle lookup on 9/12 stable images (gap +0.194). Closing the oracle gap (0.657 → 0.1) requires ~85% error reduction from the oracle floor — a counting-grade detection problem, not a distributional regression problem.

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

## Completed Experiments (continued)

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

## Completed Experiments (continued)

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

## Completed Experiments (continued)

### E12 · Full-image Hough circle transform diagnostic — FALSIFIED (with scope caveat)

**Hypothesis:** Full-image HoughCircles on per-image contrast-stretched uint8 gradient finds per-bubble radius within ≤2 pyramid levels for ≥70% of GT bubbles (r≥8px), FP/image ≤ 5.

**Script:** `scripts/experiments/profile_hough_e12.py`

**Design (as implemented — deviates from spec):** Full-image HoughCircles with param2 sweep {10, 20, 30}, GaussianBlur(5,5), Canny param1=50, dp=1, minDist=8, radius 8–60px. Per-image min/max contrast stretch. GT match: nearest circle within 0.5×r_gt. FP: no GT within 3×r_detected.

**Note on spec deviation (flagged by PAL):** The original design spec said "patch-centered Hough." The implementation ran full-image. These are fundamentally different: full-image accumulates votes from all edges in a 1024×1024 image; patch-based limits vote scope to the local region. Patch-based Hough remains untested.

**Raw results (param2 sweep, 2349 GT bubbles r≥8px, 14 images):**
```
param2   DR_matched   DR_in_tol(≤2lv)   mean_FP/img
    10        0.802             0.114          2429.6
    20        0.732             0.097          1569.1
    30        0.568             0.093           771.1

Level offset (param2=10, sign-corrected*): mean=+7.63, median=+8.31
  → r_det ≈ 2.2× r_gt (Hough detects circles far too large)
  within ±2 levels: 14.2%  ±3 levels: 19.7%

*Script has sign bug: log(r)/log(sf) flips sign vs. correct formula log(r)/log(1/sf).
```

**FALSIFIED.** DR_in_tol = 11% vs 70% target (4.9× below); FP/image = 771–2430 vs 5 target (154–486× above). Both failures are unambiguous and robust across all param2 values.

**Critical analysis (reviewed with PAL, consensus reached):**

1. **Scope: full-image Hough is falsified, not Hough as a method.** The radius bias (+7.6 levels, r_det ≈ 2.2×r_gt) and catastrophic FP both originate from the full-image accumulator collecting votes from all image texture at arbitrary distances. This is an accumulator noise problem, not a morphological incompatibility. GaussianBlur(5,5) σ≈1.1px cannot produce 7.6-level radius inflation (it would shift radius by ≈0.7 levels for a 30px bubble). Patch-based Hough would not have this pathology and remains an open question.

2. **DR_matched=80% is inflated.** The match logic is non-injective: one Hough detection can "match" multiple GT bubbles. When det/GT=373× (img019655), proximity matches succeed by random chance. DR_in_tol=11% is the honest figure, and is also slightly inflated by the same mechanism.

3. **Brightness/FP correlation is real but noisy.** Dark images (mean<0.2) generally have more FP; bright images have fewer. But IMG_005070 (mean=0.242) has FP=1193 and IMG_000001 (mean=0.351, 58 GT bubbles) fires **zero detections at all param2 values** — unexplained. Per-image contrast stretch + fixed Canny threshold produces highly variable Canny edge density across the 4 photometric regimes.

4. **FP criterion slightly underreports.** `count_fp` uses all GT bubbles (including r<8px) as FP absorbers. True operational FP (against r≥8px bubbles only) is higher. This makes the verdict stronger.

**Path forward: E13 — radial gradient SNR (patch-centered, tests rim edge signal directly)**

---

## Completed Experiments (continued)

### E0-A · LOO mean/median GT histogram oracle (ran 2026-05-02)

**Script:** `scripts/experiments/baseline_e0a.py`  
**Context:** After 12 experiments across NCC, LoG, and Hough, the PAL synthesis consultation (2026-05-02) identified a critical gap: the pipeline's 0.950 relL1 had never been compared against a trivial constant-histogram predictor. Without this floor, no claims about pipeline value or research headroom were defensible.

**Design:**
- For each of 14 tractable annotated images, predict the leave-one-out (LOO) GT histogram (mean AND median of the other 13 images).
- The median is the L1-minimizing estimator; the mean is reported for comparison but is suboptimal.
- Exclude images with n_gt < 100: relL1 has pathological outlier behavior at low counts (n_gt=14 → relL1=26.25 for any reasonable predictor, rendering cross-image means meaningless).
- Summary metric: **median relL1 over stable images (n_gt ≥ 100)**.
- **Important caveat**: this is a GT oracle — it requires annotated examples from the same apparatus. A deployment system has no such oracle. Do not conflate this with the lower bound achievable by a deployable system.
- **Session confounding**: C1S0024 (1 image) is the only image from its session; the LOO mean is a cross-session extrapolation.

**Raw results (n=14, 12 stable):**

| Image (truncated) | n_gt | img_mean | relL1_mean | relL1_median | stable |
|---|---:|---:|---:|---:|---|
| C1S0014_img006001 | 321 | 0.339 | 0.599 | 0.536 | ✓ |
| C1S0014_img009542 | 492 | 0.314 | 0.696 | 0.705 | ✓ |
| C1S0014_img018008 | 350 | 0.141 | 0.647 | 0.781 | ✓ |
| C1S0014_img018351 | 494 | 0.174 | 0.418 | 0.502 | ✓ |
| C1S0019_img003593 | 383 | 0.351 | 0.575 | 0.543 | ✓ |
| C1S0019_img011890 | 431 | 0.069 | 0.389 | 0.432 | ✓ |
| **C1S0024_img014500** | 474 | 0.156 | **0.571** | **0.548** | ✓ ← pipeline dev image |
| C1S0004_000001 | 76 | 0.351 | 4.537 | 3.413 | (excluded) |
| C1S0004_004509 | 641 | 0.365 | 0.778 | 0.755 | ✓ |
| C1S0004_005070 | 175 | 0.242 | 1.429 | 1.462 | ✓ |
| C1S0004_012062 | 550 | 0.132 | 0.547 | 0.609 | ✓ |
| C1S0010_000002 | 14 | 0.281 | 33.987 | 26.250 | (excluded) |
| C1S0010_005432 | 581 | 0.365 | 0.744 | 0.740 | ✓ |
| C1S0010_019655 | 602 | 0.224 | 1.189 | 1.172 | ✓ |

**LOO MEDIAN oracle (stable images):** median relL1 = **0.657**, mean = 0.732, std = 0.288, range [0.432, 1.462]

**Direct comparison on development test image (C1S0024):**
- LOO median oracle: **0.548**
- LOO mean oracle: **0.571**
- Pipeline best (NCC): **0.950** (n=1, likely selection-biased)
- Gap (oracle − pipeline): **−0.402** (oracle better on this image)

**Verdict:**

Oracle median relL1 = 0.657 — MODERATE cross-image consistency. Cross-image distributions are not highly stationary (C1S0004_005070 and C1S0010_019655 both exceed 1.0 even for the oracle, meaning the LOO histogram systematically over-predicts). At the same time, the best-case oracle achieves 0.432, suggesting non-trivial signal exists in the dataset's histogram structure.

**Critical findings (reviewed with PAL, consensus reached):**

1. **The pipeline's 0.950 is not a valid performance estimate.** It was evaluated on C1S0024, the primary development image whose filename appears in 6 output files (`pipeline_optB_*`, `pipeline_1500_*`). This constitutes likely selection bias. The correct comparison requires cross-image pipeline evaluation (14 images minimum).

2. **The oracle is better than the pipeline on the one image where both can be compared.** LOO median oracle (0.548) beats the pipeline (0.950) by 0.40 on C1S0024. However, this comparison has zero statistical power (n=1) and cannot generalize.

3. **The oracle is not "trivial" in a deployment sense.** It requires GT labels from 13 images from the same physical apparatus. A deployed system has no oracle. The meaningful question is: can the pipeline *without GT access* match what the oracle achieves with GT access? Currently the pipeline is worse even than the oracle — a pipeline that required no image-reading features would be preferable if it could match the oracle.

4. **C1S0004_005070 (n_gt=175) oracle relL1 = 1.462 > 1.0.** This is worse than predicting zero. The LOO mean/median over-predicts for this image because the other 13 images are atypically bubble-dense. This signals that the 14-image dataset contains distinct "rare" images that no distributional prior can handle — the 30:1 clutter ratio and photometric heterogeneity are image-specific, not dataset-wide constants.

5. **Metric pathology confirmed.** n_gt=14 image: LOO median relL1=26.25. Zero predictor relL1=1.000. The mean predictor over-predicts by 26×, not because it's a bad predictor but because the metric amplifies absolute errors inversely with population size. The exclusion of n_gt < 100 from summary statistics is required; mean relL1 across all 14 images is meaningless.

**What can be concluded:**
- Cross-image histogram consistency: moderate (oracle median 0.657). Not high enough to justify a pure lookup-table approach; not low enough to rule out distribution regression.
- Pipeline value: cannot be assessed without cross-image evaluation. The 0.950 figure is single-image and selection-biased.
- Architecture comparison (detect-then-count vs. distribution regression): premature. No valid baseline for the pipeline exists yet.

**Required next steps before any architecture decision:**
1. Run pipeline on all 14 images → compute relL1 distribution (not just one number).
2. Compare: does the pipeline beat the LOO median oracle on a per-image basis?
3. If pipeline > oracle on most images: detect-then-count has no value over a distributional prior.
4. If pipeline < oracle on most images: architecture is working; question is how far below oracle it can get.

**Path forward:** E0-B (cross-image pipeline evaluation) before E13 or any further feature diagnostic.

---

## Completed Experiments (continued)

### E0-B · Cross-image pipeline evaluation (ran 2026-05-02, FALSIFIED)

**Script:** `scripts/experiments/eval_pipeline_e0b.py`

**Design:** 5 LOSO folds (sessions C1S0004, C1S0010, C1S0014, C1S0019, C1S0024). For each fold: train on all other sessions, predict on held-out session images. Compute relL1 per image vs E0-A oracle.

**Raw results:**

| Image | n_gt | Pipeline relL1 | Oracle relL1 | Δ | Session |
|---|---:|---:|---:|---:|---|
| C1S0004_000001 | 76 | 0.944 | n/a | — | C1S0004 (unstable) |
| C1S0004_004509 | 641 | 0.980 | 0.755 | +0.225 | C1S0004 |
| **C1S0004_005070** | 175 | **0.837** | **1.462** | **−0.625** | C1S0004 |
| C1S0004_012062 | 550 | 0.846 | 0.609 | +0.237 | C1S0004 |
| C1S0010_000002 | 14 | 2.388 | n/a | — | C1S0010 (unstable) |
| C1S0010_005432 | 581 | 0.937 | 0.740 | +0.197 | C1S0010 |
| **C1S0010_019655** | 602 | **0.844** | **1.172** | **−0.328** | C1S0010 |
| **C1S0014_018008** | 350 | **0.718** | **0.781** | **−0.063** | C1S0014 |
| C1S0014_006001 | 321 | 0.833 | 0.536 | +0.297 | C1S0014 |
| C1S0014_009542 | 492 | 0.943 | 0.705 | +0.238 | C1S0014 |
| C1S0014_018351 | 494 | 0.761 | 0.502 | +0.259 | C1S0014 |
| C1S0019_003593 | 383 | 0.921 | 0.543 | +0.378 | C1S0019 |
| C1S0019_011890 | 431 | 0.857 | 0.432 | +0.425 | C1S0019 |
| C1S0024_014500 | 474 | 0.888 | 0.548 | +0.340 | C1S0024 |

**Summary (stable images, n_gt ≥ 100):**
- Pipeline LOSO median relL1: **0.851** (mean=0.864, std=0.073)
- Oracle median relL1: **0.657**
- Gap: **+0.194** (pipeline worse)
- Per-image wins: pipeline **3/12**, oracle **9/12**

**VERDICT: FALSIFIED.** The detect-then-count architecture does not outperform a GT-oracle distributional lookup. Pipeline loses 9/12 stable images; gap is +0.194 (not marginal).

**Critical analysis (reviewed with PAL):**

1. **The 3 pipeline "wins" are not genuine.** They all occur when the oracle itself is failing (relL1 > 0.78 — oracle over-predicts because the LOO distribution is a bad match for that image). The pipeline's conservative underprediction looks better than an oracle that overshoots by 50–100%. This is NOT evidence the pipeline extracts per-image signal. The one marginal win (C1S0014_018008, Δ=−0.063) is within noise. The wins span 3 different sessions, which PAL noted raises P(CNN-A) slightly (see below) — but this interpretation is undermined by the nature of the wins (beating a broken oracle, not a working one).

2. **Detection is net subtractive.** On 9/12 images, replacing the pipeline with "use the population mean histogram" would improve performance. The NMS eviction problem (88% of correct-level peaks suppressed) combined with 30:1 clutter means the detection step adds noise, not signal.

3. **C1S0024 in LOSO (0.888) vs original eval (0.950).** The original 0.950 was on a model trained on C1S0024 data. LOSO gives 0.888. Both lose badly to oracle (0.548). No selection-bias artifact can explain this gap.

4. **C1S0004_005070 is a structural outlier.** Pipeline 0.837 vs oracle 1.462 — but BOTH are bad. The oracle fails because this session's histogram is sparse (n_gt=175 vs session mean ~500). This image represents a regime neither architecture handles. See E0-B outlier note: needs audit (sparse-bubble physics vs annotation artifact).

---

### CNN architecture probability estimate (2026-05-02, consensus with PAL)

After E0-A and E0-B established that the detect-then-count architecture provides no measurable value over a GT-oracle lookup, the question becomes: what is the probability a CNN-based alternative would beat the oracle?

**P(CNN-A achieves LOSO median relL1 < 0.657) = 15% (range 11–22%, revised 2026-05-02; prior 8% was wrong)**

- CNN-A: replace NCC scorer with a CNN patch scorer; keep same NMS + calibration pipeline.
- The 8% prior incorrectly applied the "14-image data wall" to patch-level scoring. The actual training set is ~5,000 labeled bubble patches (not 14 images), which is adequate for binary patch classification. CNN is not algebraically broken like NCC — it can learn scale-selective responses.
- Binding constraints after revision: (a) cross-scale NMS is unchanged — CNN-A requires 4–5× NMS survival improvement (11.6% → 55%+) to beat the oracle; (b) coarse-scale data starvation (~30–50 positive patches per coarse level at 0.9^27 scale step); (c) calibration starvation in LOSO (~3 training images/photometric regime).
- Lower bound: 150K hard negatives at wrong scales is correct scale-rejection training signal; 3/12 oracle-failure images (oracle relL1 > 0.78) are systematic CNN-A opportunities where any per-image detection signal wins.
- Expected outcome: relL1 ≈ 0.70–0.80 with small probability of breaching 0.657.
- **Single best next experiment**: CNN-A NMS survival probe on 3-image validation fold. Decision: <20% NMS survival → abandon; ≥40% → raise P to 22–28% and run LOSO.

**P(CNN-B achieves LOSO median relL1 < 0.657) = 12% (range 9–15%)**

- CNN-B: direct image → 27-bin histogram regression (no per-instance detection, no NMS).
- Bypasses all mechanical failures of detect-then-count. Competes directly against the oracle using pixel features instead of GT labels.
- Oracle is already the L1-optimal LOO estimator from GT data; CNN-B must beat it from pixels alone — a strictly harder task with the same information ceiling.
- Most likely outcome: CNN-B matches oracle (relL1 ≈ 0.65–0.75). Would require per-image visual predictors of histogram deviation that generalize across sessions — unlikely to exist at n=14.
- C1S0004_005070 (oracle 1.462) is a structural outlier that CNN-B will also fail on without training examples for that regime.
- CNN-B is the **marginally better bet**: its failure mode (matches oracle) is better than CNN-A's failure mode (same NMS bottleneck).
- P(CNN-B) with 10 more annotated images (n=24): **~24%** (PAL consensus; interpolation between n=14 at 12% and n=34 at 35%).

**What moves the needle most:**
1. **Data collection** (binding constraint): 20 additional annotated images (n=34) raises P(CNN-B) to 35%, P(CNN-A) to 22–28%. 40 images raises P(CNN-B) to 50%.
2. **Diagnostic experiments first** (see Open Experiments): within-session oracle, apparatus metadata, motion audit, regime-conditional oracle. These cost <3 hours and could reveal a cheaper path.
3. **E13 radial gradient SNR** (4 hours): if SNR ≥ 2× for dark-rim (54%), radial gradient scoring replaces NCC without CNN generalization risk.

**Consensus (PAL + Claude, 2026-05-02):** Neither CNN approach clears 20% probability of beating the oracle. The data wall is the binding constraint, not model architecture. If CNN is the chosen direction, data collection must precede implementation.

---

### RNN-CNN hybrid assessment (2026-05-02, REJECTED)

**Proposal:** Annotate 10 consecutive video frames (temporally correlated), train a RNN-CNN hybrid to exploit temporal coherence and use each frame's information more efficiently.

**Assessment (consensus with PAL):** P ≈ 3–5%. Not a viable path.

**Evidence against:**
1. **Effective sample size from 10 correlated frames**: `n_eff = n / (1 + 2r/(1−r))`. At r=0.95 inter-frame correlation: n_eff ≈ 0.25 independent frames. 10 temporally correlated annotations contribute <1 independent data point.
2. **Temporal coherence is the wrong discriminator**: static clutter (apparatus background, fixed edges) is MORE persistent than bubbles frame-to-frame. An RNN trained to predict "what persists" will learn to predict static background, not bubbles. The temporal coherence discriminator is inverted.
3. **RNN architecture starvation**: 14 sequences × 10 frames = 14 usable RNN training examples (1 sequence per session). Insufficient for sequence model generalization.
4. **Annotation overhead without payoff**: annotating 10 correlated frames costs as much as 10 independent frames but yields <1 effective independent observation.

**Correct use of temporal data (if available)**: temporal differencing (`|I_t - I_{t-1}|`) cancels static apparatus exactly; bubbles that move ≥ 1px appear as blob-shaped residuals. This is a 10-line preprocessing step, not an architecture. Gated on motion audit (see Open Experiments — Lead A).

---

---

## Completed Experiments (continued)

### EV1 · Temporal background subtraction + watershed segmentation (ran 2026-05-03, FALSIFIED for dense regime)

**Motivation:** Discovery of unlabeled high-speed video ZeroG_Test3_Opt3 (7,501 frames, 1,250 fps, FASTCAM Nova, 1024×1024 monochrome). Early frames (mean=83/255) are nearly bubble-free; later frames darken monotonically (mean=32/255) as bubbles accumulate. Visual inspection showed background subtraction could isolate bubble structures, motivating a direct detection-free counting approach.

**Data:** `/…/Bubble Tacking/New Images/ZeroG_Test3_Opt3/` — 7,501 BMP frames, sequential, no GT annotations.

**Pipeline:**
1. Background model: median of frames 1–20 (img_mean ≈ 83/255; Step 0 confirmed bubble-free)
2. Foreground: `fg = clip(bg − img, 0, 255)`, threshold at 25
3. Distance-transform watershed seeded by `peak_local_max(dist_transform(mask), min_distance=4)`
4. Bubble radius: `sqrt(area/π)`, filtered to r=[3, 60]px

**Raw results (watershed counts across density range):**

| Frame | img_mean | seeds | valid_r | med_r | max_r |
|---|---|---|---|---|---|
| 200 | 0.320 | 323 | 134 | 5.4 | 22.5 |
| 500 | 0.292 | 1118 | 600 | 5.5 | 43.4 |
| 1000 | 0.260 | 2037 | 1037 | 6.1 | 38.3 |
| 1500 | 0.227 | 2658 | 1559 | 5.8 | 45.5 |
| 2000 | 0.208 | 2536 | 1722 | 6.4 | 44.5 |
| 2500 | 0.173 | 1672 | 1378 | 8.3 | 40.6 |
| 3000 | 0.176 | 1859 | 1593 | 8.2 | 36.7 |
| 4000 | 0.147 | 1061 | 969 | 11.2 | 46.6 |
| 5000 | 0.146 | 1039 | 960 | 11.3 | 44.8 |
| 6000 | 0.132 | 685 | 616 | 13.7 | 58.3 |
| 7000 | 0.123 | 643 | 591 | 14.0 | 58.0 |

**Labeled-image validation (cross-session, Gaussian-blur background — different algorithm, informative only):**
Applied to 14 seed_v04 images using `bg = GaussianBlur(sigma=40)` as background proxy. Over-counted 2–10× GT (155–1345% error). The 98%-error case (img_mean=0.069, nearest regime to late video) implicates watershed over-segmentation independent of background quality.

**Four pre-committed diagnostic checks (consensus with PAL, 2026-05-03):**

| Step | Check | Pass criterion | Result |
|---|---|---|---|
| 0 | img_mean frames 1–20 | Confirm bubble-free (stable mean) | **PASS** — mean 81–83/255, stable ±1.5 |
| 1 | Pearson(img_mean, med_r) on 11 rows | \|r\| < 0.85 to proceed | **FAIL** — r=−0.878, \|r\|=0.878>0.85 |
| 2 | FG fraction in bubble-free corner (top-left 80×80px) | Absolute change < 0.05 | **PASS** — fraction=0.000 throughout |
| 3 | Total foreground area (void fraction) per frame | Characterize coalescence vs depletion | **DECISIVE** — see below |

**Step 3 findings (decisive):**
- Void fraction grows monotonically: 0.07% (frame 1) → 62.4% (frame 7251). Never decreases.
- Pearson(img_mean, void_fraction) = **−0.9924** — void fraction is effectively `1 − img_mean/bg_mean`. The foreground area contains no information beyond what img_mean already encodes.
- Watershed bubble count peaked at frame ~1500–2000 then collapsed while void fraction continued rising. This is the signature of foreground saturation: above ~40% coverage the threshold mask forms a continuous sheet; distance-transform seeding tiles it into Voronoi cells; the "median radius" measures average tile size, which grows as cells get larger at higher packing.
- `max_r` clustering at 58–58.3px near the 60px filter cap across late frames is an artifact signature: the filter ceiling masquerades as the physical maximum bubble size.

**Mechanism confirmed (consensus with PAL):**
The median radius trend (5px → 14px) is an artifact of five simultaneous confounds: (1) fixed threshold on a drifting img_mean applies a different foreground definition at each timepoint; (2) non-monotone seed count contradicts smooth coalescence; (3) watershed Voronoi cells at high packing return region radius, not bubble radius; (4) max_r capped by filter boundary; (5) coalescence and depletion are observationally indistinguishable without void-fraction tracking. Step 3 resolves (5): void fraction monotonically increases → no depletion; the scene is accumulating bubbles, but the watershed cannot resolve them individually in the dense regime.

**Background model validity:** Steps 0 and 2 confirm the background model is physically valid (bubble-free at frames 1–20, no apparatus FP in bubble-free region). The failure is in the segmentation step, not the background.

**VERDICT: FALSIFIED for the dense regime (void fraction > 30%).**
The background subtraction is valid; the watershed segmentation breaks down once void fraction exceeds ~30%. The pipeline produces a brightness-laundered distance-transform tile-size histogram rather than a physical bubble size distribution. In the sparse regime (frames 1–500, void fraction < 10%), the watershed may be useful, but no GT exists to validate it and Step 1 barely failed even at the full range.

**What this rules out:**
- Watershed on threshold mask as a general bubble-counting method for this apparatus
- The median radius trend as evidence of bubble coalescence
- Any size-distribution claim from this pipeline without GT validation on at least 3 frames from this video

**What this does NOT rule out:**
- Temporal background subtraction as a valid preprocessing step — it works (Steps 0, 2)
- Sparse-regime counting (frames 1–500): watershed finds ~130–600 objects with plausible radii but needs annotation validation
- Alternative segmentation on the background-subtracted image: marker-based watershed with GT-seeded markers, active contours, or learned segmentation trained on the sparse regime

**Path forward:** E0-C and E13 remain the highest-priority labeled-data experiments. For the video data specifically, the next step requires annotating 2–3 sparse-regime frames (img_mean ≈ 0.29–0.32) before any further algorithmic investment, using the pre-committed criterion: ws_n/gt_n < 1.5 AND median_r within 2px of GT.

---

## Open Experiments (pending)

Ordered by suitability given current evidence. Run diagnostics first — they gate the expensive experiments and could reveal a path that bypasses CNN entirely.

---

### E0-C · Within-session LOO oracle (ran 2026-05-03, EFFECTIVELY FALSIFIED)

**Script:** `scripts/experiments/baseline_e0c.py`

**Hypothesis to falsify:** "Within-session LOO median relL1 ≤ 0.45 for at least one session, meaning images from the same physical run are similar enough that intra-session averaging achieves near-target error." If confirmed, same-session images function as approximate repeats; the counting problem reduces to choosing the right session, not reading individual image features.

**Design:** For each session with ≥2 stable images (n_gt≥100), predict each image from the within-session LOO median histogram. Compare to cross-session oracle (E0-A median = 0.657). Pre-committed criterion: within-session median < 0.45 in ANY session → PASS.

**Raw results (in-scope only — OOS = sparse-bubble regime excluded per user scope):**

| Image | n_gt | within_LOO | cross_LOO | Δ |
|---|---:|---:|---:|---:|
| C1S0004_004509 | 641 | 1.026 | 0.755 | +0.271 |
| C1S0004_012062 | 550 | 0.655 | 0.609 | +0.047 |
| C1S0010_005432 | 581 | 1.698 | 0.740 | +0.959 |
| C1S0014_006001 | 321 | 0.570 | 0.536 | +0.034 |
| C1S0014_009542 | 492 | 0.913 | 0.705 | +0.208 |
| C1S0014_018008 | 350 | 0.628 | 0.781 | −0.153 |
| C1S0014_018351 | 494 | 0.538 | 0.502 | +0.036 |
| C1S0019_003593 | 383 | 0.467 | 0.543 | −0.076 |
| C1S0019_011890 | 431 | 0.418 | 0.432 | −0.014 |

OOS (excluded from verdict, shown for completeness):

| C1S0004_005070 | 175 | 2.396 | 1.462 | +0.934 |
| C1S0010_019655 | 602 | 1.841 | 1.172 | +0.669 |

**Session summary (in-scope):**

| Session | n | within_med | cross_med | Δ | pairwise_Δ_signs | <0.45? |
|---|---:|---:|---:|---:|---|---|
| C1S0004 | 2 | 0.841 | 0.682 | +0.159 | +,+ (2/2 wrong) | no |
| C1S0010 | 1 | 1.698 | 0.740 | +0.959 | contaminated (see below) | no |
| C1S0014 | 4 | 0.599 | 0.621 | −0.021 | +,+,−,+ (3/4 wrong direction) | no |
| C1S0019 | 2 | 0.443 | 0.488 | −0.045 | −,− (2/2 correct — n=2) | marginal |

Overall within-session median: **0.628** vs cross-session **0.609** → within is WORSE (+0.019).

**Script verdict (automated):** "HYPOTHESIS SURVIVES: C1S0019 achieves within-session median 0.443 < 0.45."

**Critical analysis (consensus with PAL, 2026-05-03 — REJECTS the automated verdict):**

Five independent failures undermine the "HYPOTHESIS SURVIVES" call:

1. **C1S0019 pass is below the noise floor.** At n=2, the within-session LOO median is the arithmetic mean of two values (0.418 + 0.467)/2 = 0.4425 — note a minor discrepancy with the reported 0.443, suggesting a script rounding artifact. Either way, a margin of 0.007–0.022 on a metric that spans 0.418–1.698 across the dataset is indistinguishable from chance. No variance estimate is possible at n=2.

2. **C1S0010 is structurally contaminated.** C1S0010_019655 was excluded from the verdict scope because it is an OOS outlier, but it was **not** excluded from the predictor pool. With n_session=2, the within-session LOO for C1S0010_005432 is predicted solely by the OOS outlier's histogram — a 1:1 lookup from the single worst predictor in the dataset. The 1.698 result is uninterpretable as evidence about within-session consistency and should be discarded entirely from the verdict.

3. **C1S0014's apparent −0.021 improvement is a median-of-marginals artifact.** The correct comparison is the median of paired differences, not median(within) − median(cross). Individual Δ values: +0.034, +0.208, −0.153, +0.036 → pairwise median Δ = +0.035 (within WORSE). Three of four images show the wrong direction; the single outlier (018008, Δ=−0.153) drags the marginal median down.

4. **The overall signal is in the wrong direction.** Overall within-session median (0.628) > cross-session median (0.609). A sign test on all 9 in-scope paired observations gives 6/9 in the wrong direction (p ≈ 0.25 one-sided, not significant). Session identity does not explain meaningful histogram variance.

5. **Criterion design weakness.** The OR threshold rule (any session < 0.45 → PASS) is epistemically sound only when a pass is meaningfully above the noise floor. Here it was met by exactly one session, with n=2, at a sub-noise margin. A rigorous criterion would have required either n≥4 in the passing session, or a signed-rank test on paired Δ values at p < 0.05. Neither condition is met.

**VERDICT: EFFECTIVELY FALSIFIED.** The null hypothesis — "within-session histogram variability is not lower than cross-session variability" — is consistent with all the data. Session-conditional lookup is not demonstrated to be actionable. The cross-session oracle floor (0.657) remains the binding lower bound; the relL1=0.1 target is not conditionally achievable through session-level conditioning alone.

**What E0-C does reveal (not a complete negative):** C1S0019 has the lowest absolute within-session errors in the dataset (0.418, 0.467), suggesting this session may have unusually self-similar images. This is not evidence for within-session consistency as a general strategy, but it motivates including session ID as a feature in any future image-feature regression (Experiment A).

---

### Metadata check · Apparatus parameters as histogram predictors (~10 min)

**Hypothesis to falsify:** "Session ID (or any apparatus parameter — flow rate, pressure, camera gain) is uncorrelated with histogram shape; knowing the session provides no additional predictive power beyond the cross-session oracle."

**Design:** (1) Check whether session IDs are associated with apparatus settings in any available metadata file. (2) If session IDs encode distinct physical conditions (flow rate, pressure), compute relL1 of session-conditional oracle (use all images from the same session; predict the left-out image from within-session LOO). If this substantially beats the cross-session oracle (0.657), metadata is a viable predictor. (3) If no structured metadata exists, check whether img_mean alone (already computed in E0-A) correlates with histogram bin ratios across the 14 images.

**Falsification criteria:**
- Session-conditional oracle relL1 < 0.45 → apparatus metadata is load-bearing; deploy a session-conditional lookup instead of per-image detection.
- Session-conditional oracle relL1 ≈ cross-session oracle → apparatus variation does not explain histogram variation; per-image feature estimation required.

**Why this matters:** If session IDs encode discrete physical states that determine the bubble size distribution, the entire detection problem is bypass-able. A 10-minute metadata check is the highest-leverage possible experiment. If the apparatus is reconfigured between sessions and the reconfiguration uniquely determines the histogram, the relL1=0.1 target is achievable with zero image processing.

---

### Motion audit · Inter-frame bubble displacement in unlabeled video (~30 min)

**Hypothesis to falsify:** "Bubbles move ≥ 1px between consecutive frames (inter-frame displacement > 0.5px). If true, temporal differencing (`|I_t − I_{t-1}|`) cancels static apparatus structure exactly and isolates bubble residuals."

**Design:** Take any unlabeled video clip from the same apparatus. Compute `|I_t − I_{t-1}|` for 20 consecutive frame pairs. Measure: (1) mean displacement of blob-shaped residuals (estimated by centroid tracking or optical flow in a small region), (2) static background cancellation quality (residual in apparatus-only regions), (3) whether bubble-shaped residuals are visually distinguishable in the difference image.

**Falsification criteria:**
- Displacement < 0.5px/frame → bubbles are effectively static; temporal differencing cannot isolate them from background; Lead A is not viable.
- Displacement ≥ 1px/frame AND clear residuals visible → Lead A (temporal differencing as weak supervision) is viable; gates the conditional path below.

**Why this gates Lead A:** Temporal differencing is a 10-line preprocessing step that, if motion is sufficient, provides automatic weak supervision for bubble locations without human annotation. If the motion audit fails, this entire path closes.

---

### Regime-conditional oracle (~1 hr)

**Hypothesis to falsify:** "Conditioning the oracle on photometric regime (img_mean quartile) significantly reduces cross-image histogram variance and lowers relL1 below 0.50."

**Design:** Partition the 14 tractable images into photometric regimes by img_mean quartile. For each image, compute the LOO oracle using only images from the same quartile. Compare regime-conditional vs. cross-regime oracle relL1.

**Falsification criteria:**
- Regime-conditional oracle median relL1 < 0.50 → photometric regime is a meaningful partition; histogram heterogeneity is partly explained by appearance; regime-conditional models are worth building.
- Regime-conditional oracle relL1 ≈ cross-regime oracle → histogram heterogeneity is not driven by photometric regime; partitioning does not help.

**Why this matters:** If the oracle improves dramatically within regimes, then a two-stage pipeline — (1) classify regime from image, (2) apply regime-conditional lookup — could approach relL1=0.45 without any detection. This gates Lead B (parametric fitting).

---

### Image-feature ridge regression (~2 hrs)

**Hypothesis to falsify:** "Global image statistics (img_mean, img_std, edge density, low-frequency power ratio) are not predictive of any histogram bin count. Ridge regression from these features achieves relL1 > 0.657 (worse than the oracle)."

**Design:** Extract per-image features: img_mean, img_std, edge density (Canny, adaptive threshold), power in each octave band (FFT), histogram moments (skewness, kurtosis). Train LOSO ridge regression (one model per histogram bin, or single model predicting all 27 bins jointly). Report LOSO relL1 vs oracle.

**Falsification criteria:**
- relL1 < 0.65 → image features predict histogram better than a GT oracle; strong evidence for per-image feature extraction.
- relL1 ≈ 0.657 → image features contain the same information as cross-image GT averaging; the oracle is extracting image information by proxy.
- relL1 > 0.657 → image features add no predictive signal; distribution regression from raw features is not viable; the oracle's cross-image signal cannot be approximated from pixels alone at n=14.

**This directly tests Pillar B of the unachievability claim**: whether image-feature-conditioned estimators can exceed what GT averaging achieves.

---

### Lead A · Temporal differencing as bubble detector (CONDITIONAL on motion audit)

**Prerequisite:** Motion audit shows bubble displacement ≥ 1px/frame.

**Hypothesis to falsify:** "Temporal differencing (`|I_t − I_{t-1}|`) followed by blob detection produces per-bubble size estimates with relL1 < 0.657 on non-photometrically-dead frames, without any annotated training data."

**Design:** Apply differencing to consecutive frame pairs from unlabeled video. Threshold and run connected-component or blob analysis on the residuals to estimate bubble radii. Compute size histogram from radius estimates. Compare to GT-annotated frames if any video frames overlap with the labeled set.

**Expected outcome:** Weak supervision from motion will have high false-positive rate (moving background elements) but may achieve relL1 ≈ 0.50–0.70 on clear-frame images with active bubbles. Critical test: does it generalize across photometric regimes without per-regime tuning?

---

### Lead B · Parametric distribution fitting within regime (CONDITIONAL on regime-conditional oracle ≤ 0.30 AND within-regime R² > 0.90)

**Prerequisite:** Within-regime oracle achieves relL1 ≤ 0.30, AND log-normal fit to within-regime histograms achieves R² > 0.90.

**Hypothesis to falsify:** "A 2-parameter log-normal fit (μ, σ) to within-regime bubble size distributions does not generalize; LOSO relL1 from log-normal parameter regression exceeds 0.50."

**Design:** Fit log-normal (or 2-component mixture) to each image's GT histogram. Test whether (μ, σ) parameters are stable within regime and predictable from image features or apparatus metadata. Reduce 27-bin regression to 2–3 parameter estimation.

**Why conditional:** Parametric fitting collapses under two conditions: (1) within-regime histograms are not well-described by a log-normal (bimodal, heavy-tailed, or regime-misclassified images), or (2) the parameters vary as much within regime as across regimes. Both preconditions must be verified before investing implementation time.

---

### E13 · Radial gradient SNR at the bubble rim (ran 2026-05-03, BROADLY VIABLE — Criterion 3 met)

**Motivation:** Full-image Hough (E12) is falsified by accumulator noise, not proven to be morphologically incompatible. Before concluding handcrafted features are exhausted, test whether the bubble rim produces a detectable radial gradient signal independent of any detector architecture.

**Hypothesis to falsify:** "The inward radial gradient at the bubble rim annulus (r/R ∈ [0.85, 1.15]) is not significantly larger than the same metric at random background locations of the same radius. SNR < 2× → rim gradient provides no useful discriminative signal."

**Script:** `scripts/experiments/profile_radgrad_e13.py`

**Design:**
- Scharr gradient (skimage `scharr_v`/`scharr_h`) on full image (float [0,1]).
- For each GT bubble (r≥8px, 14 images, img.mean<0.6): compute mean dot product of gradient with inward unit radial vector over rim annulus r/R ∈ [0.85, 1.15]. This is the **inward radial gradient score**.
- Background: N_BG=30 random centres per bubble, >3R from any GT bubble, matched radius R.
- SNR = mean(|bubble scores|) / mean(|background scores|). Both sides abs().
- Stratified by size bin, photometric regime (img_mean quartiles), and morphology proxy (sign of score).

**Key results (2349 GT bubbles, 14 images, 70318 background samples):**

| Stratum | n | SNR | Pass (≥2×)? |
|---|---|---|---|
| Overall | 2349 | **6.86×** | YES |
| outward-dominant (score<0, 91%) | 2141 | 7.32× | YES |
| inward-marginal (score>0, 9%) | 208 | 2.14× | marginal |
| large × outward-dominant (r≥24px) | 123 | 4.25× | YES |
| **large × inward-marginal (r≥24px)** | **13** | **0.52×** | **FAIL** |
| small(<12px) | 1401 | 7.37× | YES |
| medium(12–24px) | 812 | 6.47× | YES |
| large(≥24px) | 136 | 3.89× | YES |
| Q1 dark (img.mean < Q1) | 721 | 5.85× | YES |
| Q3 | 500 | 10.45× | YES |
| Q4 bright (img.mean > Q3) | 958 | 5.49× | YES |

Signed score stats: mean=−0.047, median=−0.035, std=0.049, IQR=0.053. 91% of bubbles have **outward-dominant** gradient at the rim (score < 0), not inward.

**Critical finding (PAL + Claude consensus):** The morphology proxy "score > 0 → inward-marginal" does NOT correspond to E11's dark-rim class (54%). The score > 0 group selects the zero-crossing tail of the distribution — physically, bubbles where the outer rim-to-background boundary barely dominates the inner boundary. The proxy is incommensurable with E11's LoG-based classification; the comparison to 54% dark-rim is meaningless and should not be repeated.

The dominant signal is in the outward-dominant majority (91%). This is consistent with bubbles whose interior is brighter than the outer background: the rim-annulus gradient points inward on average, but the sign convention in this code uses (inward_dot < 0) when gradient magnitude decreases toward center — i.e., the bubble's bright interior creates an outward measured gradient at the rim. Regardless of polarity, the signal is real and strong.

**Hidden failure:** large × inward-marginal (n=13) SNR = 0.52× — fails the criterion. Aggregated large-bubble SNR (3.89×) passes because outward-dominant large bubbles dominate. E14 should be aware that a small subpopulation of large bubbles may have near-background radial gradient scores.

**IQR vs E11 comparison:** E13's within-class IQR is NOT the same failure mode as E11's IQR=6.2 (which encoded scale-localization failure). E13's IQR=0.053 on mean=0.050 (CV≈1.0) reflects per-bubble variability in edge contrast — relevant for per-instance threshold calibration in E14, but does not undermine the signal-presence conclusion.

**Verdict: Criterion 3 met (broad viability).** SNR ≥ 2× across all major strata. Signal is driven by the outward-dominant 91% majority, not specifically by the inward-marginal minority. Gradient edge signal is viable for E14 design.

**Implication for E14:** Score on `|radial gradient score|`, NOT on `score > 0`. No polarity assumption is justified. Reduce expectations for large inward-marginal bubbles (small n, SNR 0.52×).

---

## Brainstorm Record: Experiments A–D (2026-05-03, PAL + Claude consensus)

**Context:** After E1–E12 falsified NCC pyramid, LoG, and full-image Hough, four new directions (A–D) were proposed and critically evaluated. Key finding: E0-C (within-session LOO oracle, 30 min) was the critical missing experiment, already listed as Priority 1 above.

### Experiment A — Power spectrum texture correlation

**Proposed:** Radially-averaged image power spectrum correlated with GT histogram. Physics: bubble of radius r → Fourier contribution ~Bessel(1/r). Metric: R² > 0.7 per (frequency bin, histogram bin).

**Assessment: REDESIGNED — run as 1-hr LOO correlation check only.**

Four independent failure modes:
1. **Apparatus background dominates at the exact frequency range of interest** (k ≈ 0.008–0.17 px⁻¹, r=3–60px). E12 showed 771–2430 FP/image from fixed apparatus structure with higher contrast than bubbles. Their spectral contribution swamps the bubble signal.
2. **Photometric heterogeneity corrupts regression.** `P(k) ∝ contrast²`; 5× img_mean range creates 25× power range across images with identical bubble populations. Pre-normalizing by total power destroys bubble signal relative to apparatus background.
3. **R² > 0.7 threshold is a strawman.** In-sample R² > 0.7 on a 13-point regression with 27 frequency predictors is trivially guaranteed and meaningless. The correct metric is LOO Pearson r per bin pair. If no pair exceeds |r| > 0.6 in LOO, close the path.
4. **Bubble overlap breaks incoherent summation assumption.** At 300–600 bubbles in a ~1024² frame, mean inter-bubble distance ≈ median diameter. Interference terms generate spurious spectral contributions unrelated to individual radii.

**Redesigned protocol:** Compute LOO Pearson r per (frequency bin, histogram bin). Budget: 1 hour. Decision gate: any bin pair with |r| > 0.6 in LOO → proceed to regression. No bin pairs → close path. Do NOT build a regression model before seeing this correlation evidence.

**Priority:** Run after D and E0-C (cheap diagnostics that could change the framing entirely).

---

### Experiment B — Per-level patch classifier with per-level NMS (HOG → logistic regression)

**Proposed:** Logistic regression on HOG features of crops per pyramid level; positives = GT centers, negatives = hard NCC top-K non-GT; per-level 2D NMS only.

**Assessment: REDESIGNED — replace HOG with radial gradient profile; add E13 as prerequisite.**

Problems with B as stated:
1. **HOG does not mechanistically fix the FP problem from E9.** Apparatus boundary patches (generating ~1,000 peaks/level at fine scales) produce strong, structured gradient patterns. HOG will respond to these. Reducing FPs by 200× (from 1,000/level to ~5/image) requires the classifier to learn apparatus-specific rejection patterns that generalize across sessions with different apparatus configurations — infeasible at n=14.
2. **Coarse-scale training starvation is unchanged.** ~30–50 positive patches per level per training image at coarse scales; LOSO gives ~300–500 total positives. Class imbalance (millions of negatives) causes threshold collapse.
3. **HOG is not the wrong feature in principle** — it encodes the dark-rim ring's radially-symmetric orientation histogram. But radial gradient profile (1D vector, mean inward gradient vs. r/R, ~20 values) is lower-dimensional, directly motivated by E11 morphology findings, and interpretable. Phase congruency is contrast-invariant by construction (handles all 4 morphological regimes; HOG does not).

**Redesigned protocol:** Use radial gradient profile feature (E13's diagnostic metric) as the patch descriptor. Add E13 SNR ≥ 2× as an explicit prerequisite. If E13 fails, B is blocked.

**Priority:** Run only after E13 SNR ≥ 2× confirmed.

---

### Experiment C — Single-scale CNN + radius regression

**Proposed:** Single fixed-window CNN outputs (P(bubble), estimated_radius); 2D spatial NMS only.

**Assessment: FALSIFIED BY GEOMETRIC INCOMPATIBILITY. Do not run as described.**

- 3px bubble: optimal detection window ~10–15px. 60px bubble: optimal detection window ~120–160px. Ratio: 12–16×.
- A single-receptive-field CNN at 120px window sees a 3px bubble occupying 0.01% of its field. The radius regression head has no usable signal. This is why the scale pyramid exists — it is not an engineering choice that can be redesigned away at this scale range.
- The architecture required to handle 15× size range from a single input scale is FPN (Feature Pyramid Network) with a FCOS-style regression head and a pretrained backbone. That is a fundamentally different proposal with higher resource budget. P(FPN fine-tuned from COCO pretraining beats oracle at n=14) ≈ 12% per E0-B estimates.
- Note: explicit multi-resolution inference (run same network at 3 input scales) is equivalent to NCC's pyramid minus the problematic cross-scale NMS — it does not escape the original scale-selection problem.

**If CNN is the direction:** Propose as E15 (FPN + FCOS, pretrained backbone), with explicit acknowledgment that the 15× size range requires multi-scale architecture internally or externally.

---

### Experiment D — Regime-conditional oracle (img_mean quartile partition)

**Proposed:** Partition 14 images by img_mean quartile; LOO relL1 within partition; test if oracle drops below 0.50.

**Assessment: RUN FIRST (1 hr). Interpret conservatively.**

Correctly identifies regime conditioning as a cheap gate. Caveats:
- **n≈3–4 per quartile makes the 0.50 threshold statistically marginal.** Only clear signals (≤0.35) are interpretable; anything in 0.40–0.55 range is unresolvable at this n.
- **img_mean quartile ≠ photometric regime.** E11 showed morphological heterogeneity (4 types) that does not strictly align with brightness. A bright image may contain dark-rim AND bright-rim bubbles. The partition may conflate physical conditions.
- **Supplement with binary partition** (dark: img_mean < median vs. bright: img_mean ≥ median, n≈7 per group) to increase statistical stability.
- **Outlier contamination:** C1S0004_005070 (oracle relL1 = 1.462) will dominate any quartile it falls in. Report both with and without this outlier.

**Decision gates from D:**
- Oracle ≤ 0.35 in any partition → regime conditioning is load-bearing; image-feature ridge regression with regime conditioning is the priority next step.
- Oracle ≈ 0.657 in all partitions → img_mean does not explain histogram heterogeneity; detection path is not bypassable via regime lookup.

---

### New candidate: E14 — Phase congruency + Loy-Zelinsky radial symmetry transform

**Motivation:** All three falsified detectors (NCC, LoG, Hough) share a common weakness: either morphological polarity dependence or scale-algebraic bias. Phase congruency addresses both systematically. Not tested in E1–E12.

**Physical basis:**
- **Phase congruency** (Kovesi 1999): measures coherence of phase across spatial frequency channels at each pixel. Contrast-invariant by construction (threshold-independent). Polarity-agnostic — responds to boundaries regardless of whether they are dark-rim, bright-rim, or filled transitions. Well-suited to the 4 morphological regimes found in E11.
- **Loy-Zelinsky radial symmetry transform** (2003): votes for per-(x,y,r) radial symmetry by aggregating gradient contributions oriented toward/away from candidate centers. Produces a scale-selective response map by design — not via Gaussian derivative bias (LoG) and not via template inner product (NCC). Applied to cell detection and microscopy bubble detection in published literature.

**Combined pipeline:** Phase congruency map → radial symmetry vote accumulation → per-(x,y,r) response → 2D NMS at each r level → histogram.

**Hypothesis to falsify:** "Phase congruency + Loy-Zelinsky radial symmetry transform does not improve relL1 below E0-B pipeline (0.851) in LOSO. Residual FPs from non-circular apparatus structures dominate the response map."

**Key failure mode:** If bubble boundaries have low phase congruency (defocused, blurry, or low-gradient bubbles — the "flat" 8% and "filled-dark" 27% morphologies), the detector will miss them. Phase congruency requires a phase-aligned gradient edge. Test: inspect the phase congruency map on 3 sample images before building the full pipeline.

**Estimated cost:** 3–4 hours (phase congruency can be computed with existing Python libraries, e.g., `phasepack`; Loy-Zelinsky is a ~50-line implementation).

**Priority:** Run after E13 (E13 tests rim gradient SNR using a related but simpler metric; if E13 SNR < 2×, phase congruency at bubble boundaries is also likely weak).

---

### E14 Results (ran 2026-05-03, **FALSIFIED**)

**Script:** `scripts/experiments/detect_frst_e14.py`  
**Radii:** 18 log-spaced from 8–50px (step 1/0.9). Single-radius probe (r=10.97px) run first as a validity check.

**Full 18-radius sweep results (10 stable images, n_gt ≥ 100 within radius window):**

| Mode | Oracle median relL1 | LOSO median relL1 | Verdict vs NCC (0.851) |
|---|---|---|---|
| 18-radius sweep | **0.844** | **0.932** | **WORSE** |
| Single-radius (r=10.97px) | 0.089 | 0.594 | ⚠️ INVALID (see below) |

**NCC pipeline LOSO baseline: 0.851. GT oracle: 0.657.**

**Full-sweep failure mode:** Visual inspection of max-projected response maps (images C1S0014_img006001, C1S0014_img009542) confirms:
- Response is a **diffuse warm haze** across the entire bubble field with no distinguishable per-bubble peaks
- Oracle finds only 21/321 (6.5%) and 46/492 (9.4%) bubbles at the best threshold — structural recall failure
- **Pre-committed falsification criterion NOT triggered:** octagonal vessel apparatus is suppressed, not dominant
- Actual failure mode: in dense conditions (300–600 bubbles/image), rim pixels from all bubbles cast votes at radii that land near neighboring bubble centers. 18-radius accumulation raises the background vote density across the whole field to the point where individual bubble peaks cannot be distinguished
- Gaussian smoothing (σ=r×0.5≈5px) is a contributing factor; exact apportionment between cross-vote accumulation and over-smoothing was not diagnosed from max-projection alone (individual radius slice inspection not performed)

**Single-radius caveat (PAL consensus):** The 1-bin relL1 metric used in single-radius mode is explicitly incommensurable with the 27-bin NCC baseline. Oracle relL1=0.089 is a count-coincidence artifact — the oracle minimises |pred_count − gt_count| / gt_count, which succeeds whenever the response map produces a sufficiently dense population of local maxima at some threshold, regardless of spatial accuracy. LOSO=0.594 does not constitute a valid spatial detection result. The script explicitly flags this comparison as invalid.

**PAL consensus (2026-05-03):** E14 FALSIFIED unambiguously. Full-sweep FRST does not beat NCC LOSO and achieves ≤9.4% oracle recall. Single-radius result is non-comparable. Failure is not evidence that FRST is intrinsically incompatible — it is evidence that global vote accumulation in dense (300–600 bubble) fields generates a vote density incompatible with finding individual peaks. A single-radius per-candidate patch scorer using the radial gradient signal (E13) would be a different experiment (aligns with Experiment B redesigned), not a variant of E14.

**Additional structural concern (PAL):** Cross-radius NMS condition `distance < max(r1, r2)` is asymmetric: a 50px-radius false positive suppresses all smaller candidates within 50px, regardless of scale. In a dense bubble field with any large-radius spurious detections, this systematically eliminates correct small-radius candidates. Secondary concern for future FRST variants.

---

### Updated target (2026-05-03)

**relL1 ≤ 0.20** (revised from ≤ 0.10). The user confirmed that relL1 ~0.2 is acceptable. This changes the recall requirement from ~90% to ~80% and raises success probability from "essentially impossible" to "difficult but physically achievable." The oracle floor of 0.657 means a per-image detector must provide ~3.3× more information than a training-set histogram lookup — achievable because a detector measures *this image* directly, bypassing cross-image variance.

---

### Experiment B redesign and E15 generator probe (post-E14 consensus, 2026-05-03)

**What changed after E14:**

Prior analysis incorrectly claimed "14 images is too few data." That is only true for *global* models (whole image → histogram). A *local* patch classifier trains on ~5,000 annotated bubble instances across 14 images — enough to overdetermine a logistic regression by ~250×. The data wall does not apply.

**What Experiment B (redesigned) would be:**

1. A cheap generator proposes candidate (x, y, r) locations across the image
2. For each candidate: extract a small patch, compute the radial gradient score (E13 signal — dot product of Scharr gradient with inward unit radial vector over annulus r ∈ [0.85R, 1.15R])
3. A logistic regression trained on 5,000 annotated bubble patches classifies each candidate as bubble / not-bubble
4. NMS suppresses spatial duplicates; survivors form the detection list → histogram

This is structurally different from NCC and from FRST:
- NCC: global convolution score everywhere → scale-monotone bias evicts 88% of correct peaks before classification
- FRST: global vote accumulation → cross-bubble contamination in dense fields
- Experiment B: candidates proposed first, then scored in isolation — cross-bubble contamination is suppressed because neighboring bubbles are mostly outside the patch

**The dominant failure mode and what prior experiments say about it:**

The generator recall is the binding constraint — if a bubble is never proposed, no classifier can find it.

- Simple local intensity minima have a structural ceiling of ~27–38%: the most common morphology (dark-rim, 54% of bubbles) has its intensity minimum at the *rim*, not the center. Candidates land ~R off from the true center.
- LoG/DoG with both polarities is the natural alternative: dark-rim bubbles (bright interior) fire a negative LoG extremum near the center; filled-dark bubbles (dark interior) fire a positive extremum. Together, both polarities should cover ~81% of bubbles spatially.
- **Critical distinction:** E11 falsified LoG as a *scale estimator* — the response peak wanders ±6 scale levels (IQR=6.2), giving wrong radius estimates. E11 did NOT measure whether LoG extrema land near bubble centers spatially. The spatial localization performance of LoG/DoG is unknown from existing experiments. Using LoG/DoG to propose (x,y) while ignoring its scale estimate — and re-estimating radius from the radial gradient profile — is a hypothesis without direct supporting evidence, and also without falsifying evidence.

**Revised probability estimates (PAL + Claude consensus, 2026-05-03):**

| Generator | P(relL1 ≤ 0.2) |
|---|---|
| Local intensity minima | 10–15% |
| Polarity-agnostic LoG/DoG (spatial locator only) | 25–35% |

Three residual risks that survive the 5,000-example correction:
1. Session-level hard-negative distribution shift (apparatus boundaries vary per session; LOSO test session may have unseen hard negatives)
2. Calibration starvation: LOSO leaves ~3 images per photometric regime — threshold-to-count calibration is noisy
3. E13's SNR=6.86× was measured at confirmed bubble centers against clean background. In a dense field, a patch centered on a bubble will have neighboring bubble rims partially in frame — actual operating SNR is lower and unknown

---

### E15 — Generator recall probe (next experiment, 2026-05-03)

**Hypothesis to falsify:** "A polarity-agnostic LoG/DoG detector places a candidate within R/2 of ≥75% of GT bubble centers in dense images, and the radial gradient SNR on those actual TP candidates is ≥3×."

If this hypothesis survives, proceed to full Experiment B with polarity-agnostic LoG/DoG generator.  
If generator recall < 60%, instance detection is structurally infeasible for this dataset — pivot to regression.

**Design:**

- Run `skimage.feature.blob_log` (or equivalent) with both polarities (find both bright-on-dark and dark-on-bright blobs), σ sweep over radii 8–50px, on **3 labeled images** (one from each dominant session)
- For each GT bubble center: mark as TP if any candidate lands within R/2 of it. Report per-morphology recall (dark-rim vs filled-dark vs flat vs bright-rim)
- For TP candidates and sampled FP candidates (generator outputs not within R/2 of any GT bubble): compute radial gradient score. Report SNR = mean(TP scores) / mean(FP scores)
- Check inter-bubble FP score distribution: candidates that land between two touching bubbles — do their radial gradient scores overlap with the TP distribution? If yes, NMS cannot separate them

**Pre-committed criteria:**

| Criterion | Pass | Fail |
|---|---|---|
| Generator recall | ≥ 75% | < 60% → instance detection infeasible |
| Actual-candidate SNR | ≥ 3× | < 2× → E13 signal does not survive real distribution |
| Inter-bubble FP overlap | < 20% of FPs score above TP median | ≥ 50% overlap → NMS cannot separate |

If all three pass → proceed to full Experiment B.  
If generator recall is 60–75% → marginal; inspect which morphology is failing and decide.  
If generator recall < 60% → file as FALSIFIED and pivot to image-feature regression (Rank 8).

**Script written:** `scripts/experiments/probe_generator_e15.py`

---

### E15 Results (ran 2026-05-03, PARTIALLY FALSIFIED — current design closed; redesign gate open)

**Raw results (3 densest images: C1S0004, C1S0010, C1S0014):**

| Image | n_cand | recall | inward_r | outward_r | SNR | inter_above_TP |
|---|---:|---:|---:|---:|---:|---:|
| C1S0004 | 2834 | 0.898 | 0.825 | 0.907 | 1.22× | 82.3% |
| C1S0010 | 2670 | 0.902 | 0.826 | 0.907 | 1.54× | 78.4% |
| C1S0014 | 3175 | 0.815 | 0.875 | 0.806 | 1.13× | 74.0% |
| **Median** | | **0.898** | | | **1.22×** | **78.4%** |

**C1S0014 anomaly:** C1S0014 is the only image where inward_recall (0.875) > outward_recall (0.806), inverted relative to C1S0004 and C1S0010. In E13, 91% of bubbles are outward-dominant; LoG should preferentially recall the bright-interior (outward-dominant) population via the negative-polarity branch. The inversion in C1S0014 indicates a different morphological composition or photometric regime in that session. LOSO performance on C1S0014-like sessions may be worse than the median recalls suggest — a regime heterogeneity flag for E16 to investigate (recall stratified by session, not just by radius bin).

**Criterion verdicts:**
1. Generator recall ≥ 75%: **PASS** (median 0.898)
2. Actual-candidate SNR ≥ 3× (< 2× = FAIL): **FAIL** (median 1.22×)
3. Inter-bubble FP above TP median < 20% (≥ 50% = FAIL): **FAIL** (median 78.4%)

**Critical analysis (independent review by PAL + Claude, 2026-05-03, consensus reached):**

**Key structural finding:** E13 and E15 are not measuring the same thing despite sharing a metric name. E13 computed `score(GT_cx, GT_cy, GT_R)` vs. background >3R from any bubble. E15 computed `score(LoG_cx, LoG_cy, GT_R)` vs. LoG-fired candidates in a dense field. Both the TP pool and the FP pool changed between experiments. The 6.86× → 1.22× collapse cannot be attributed to a single cause.

**SNR criterion (1.22×) — confounded; not clean evidence of fundamental failure:**

The R/2 matching tolerance is appropriate for recall gating but inappropriate for the SNR probe. A TP candidate at offset δ ≈ R/3 (typical within R/2 disk) has its annulus [0.85R, 1.15R] half-misaligned: the near-side rim falls at ~0.67R (inside annulus), the far-side rim at ~1.33R (outside). This mechanistically guarantees a ~40–60% TP score drop from annulus misalignment alone, mapping E13's 6.86× to an expected **~2.3–3.4× from misalignment only**. The residual collapse to 1.22× requires FP elevation in addition.

FP elevation is real despite LoG scale underestimation (E11: peak delta = −2 to −5, meaning r_approx < true radius → annulus falls inside bubble interior → FP scores should be suppressed). That FP scores are still almost as high as TP scores despite this suppression means FP positions themselves are in genuinely high-gradient locations — confirmed by near-rim geometry in a dense field (almost all LoG FPs in a 600-bubble image land within 2R of at least one bubble). **This is genuine (structural) FP elevation, not a radius-estimation artifact.**

The LoG scale underestimate actually strengthens the fundamental-failure interpretation for FPs: it should have helped SNR, but didn't. The residual overlap is position-driven, not radius-driven.

**Inter-bubble FP criterion (78.4%) — structural and robust:**

An inter-bubble point (within 2R of ≥2 GT bubbles) is physically surrounded by rim gradients from multiple directions. Its inward radial score at any reasonable annulus radius crosses at least one neighboring bubble's rim and will be high. This is not explained by the TP misalignment confound. The 78% failure is genuine and directly falsifies score-based NMS: any score threshold that passes real bubbles passes 78% of inter-bubble candidates.

**What 78% does NOT falsify:** The feature entirely, or Experiment B under a redesigned suppression strategy. Specifically:
- A 10–15 annulus radial gradient profile has a structurally different signature for inter-bubble FPs (elevated at multiple radii) vs. true bubble centers (peaked at rim, low interior). A logistic regression on the full profile may discriminate these even when the single rim-annulus score overlaps.
- Geometric mutual exclusion (conflict graph: candidates within 1.5R cannot coexist) applied as max-weight independent set does not depend on score magnitude.

**Two risks the PAL/Claude consensus identified as underweighted in redesigned B:**

1. **Radius estimation dependency:** The multi-annulus profile classifier requires reliable radius estimates to define annulus boundaries. E11 showed LoG radius estimates are off by 2–5 scale levels. If the profile is computed at wrong radii, the "low interior / peak at rim / decay outside" discriminating shape doesn't appear. Redesigned B requires an explicit radius re-estimation step (annulus-fit optimization, not LoG σ) before the classifier can function — this is an implementation constraint that must be accounted for in E16 design.

2. **Missed-bubble distribution bias:** Generator recall = 0.898 means ~10% of GT bubbles are not proposed. If missed bubbles are non-uniform across the size histogram (e.g., systematically small bubbles at σ_min boundary, or one morphology type the LoG misses), the reconstructed histogram is biased at those bins regardless of classifier quality. This is an unchecked assumption — it must be verified in E16 by reporting recall stratified by radius bin and morphology type.

**Falsification statement:**

**Experiment B as designed** (single rim-annulus score + score-based NMS) is **definitively falsified** by the 78.4% inter-bubble FP failure alone, independent of SNR confounds.

**Experiment B as redesigned** (multi-annulus radial profile [0–1.5R, 10–15 annuli] + re-centering at R/4 tolerance + geometric constraint NMS) is **not definitively falsified**. P(relL1 ≤ 0.20 via redesigned B) drops from 25–35% to **~10–18%** (PAL + Claude consensus).

**The unresolved question keeping redesigned B open:**

"Does the multi-annulus radial gradient profile maintain SNR > 2× on re-centered TP candidates (R/4 tolerance) vs. the full non-TP pool in the same dense probe images?"

This is testable on the existing E15 candidate pool without new data collection (~2 hours).

**Decision gate (pre-committed):**
- Multi-annulus logistic score SNR < 2× on re-centered TPs → **close Experiment B, pivot to image-feature ridge regression (Rank 9)**
- Multi-annulus SNR ≥ 3× → **P(B) revises to ~20–25%, proceed with redesigned B**

**Methodological confounds in E15 (in order of severity):**
1. **MAJOR:** R/2 tolerance for SNR probe — wrong tolerance; conflates recall-gate with SNR probe. Explains estimated 2–3× of the SNR drop via annulus misalignment.
2. **MAJOR:** Densest-3 image selection makes FP pool pathological — almost no genuine clean background FPs exist; pool consists of near-rim positions. SNR is worst-case, not representative of full LOSO distribution.
3. **MODERATE:** Cross-polarity LoG stacking without post-stack NMS — near-duplicate blobs from both polarities can both match the same GT bubble, inflating n_TP and degrading mean_TP_score.
4. **MODERATE:** FP r_approx is systematically too small (E11 scale underestimate) → should suppress FP scores but doesn't → strengthens the structural FP elevation conclusion.
5. **MINOR:** Nearest-GT-bubble radius for TP scoring can assign the wrong bubble in dense fields with varying radii.

---

### Experiment D Results (ran 2026-05-03, **FAIL — regime conditioning not load-bearing**)

**Script:** `scripts/experiments/oracle_regime_d.py`

**Raw results:**

| Partition | Best label | Median relL1 | n stable | Note |
|---|---|---|---|---|
| Global LOO (all 14) | — | 0.657 | 14 | E0-A baseline confirmed |
| Session identity | C1S0019 | 0.443 | 2 | Same images as E0-C; not independent |
| Brightness quartile | Q4_bright | 0.437 | 3 | One image per session |
| Density tercile | medium | 0.523 | — | — |

**Pre-committed criterion:** any partition median ≤ 0.35 → PASS. Best found = 0.437. Script auto-verdict: MARGINAL.

**Final verdict after independent critical review (PAL + Claude consensus, 2026-05-03): FAIL**

Three grounds, in order of severity:

1. **The MARGINAL zone (0.35–0.45) does not exist in the pre-committed design.** The hypothesis in the script header and in the pre-committed documentation reads: *"any partition median ≤ 0.35 → PASS."* The MARGINAL label was introduced at `oracle_regime_d.py:138–141` when the execution script was written — post-hoc. The correct binary verdict on 0.437 is: **0.437 > 0.35 → FAIL.**

2. **Q4_bright = 0.437 (n=3) is a two-image coincidence, not a generalizable regime signal.** Decomposing the improvement: C1S0019_003593 improves by ~0.003 vs. its global LOO (statistically zero). The entire Q4_bright improvement is carried by C1S0004_004509 and C1S0010_005432 being mutually predictive in the bright partition — two bright-field industrial-session images that likely share apparatus-driven histogram shape, not bubble physics. Remove either from the predictor pool and the partition median collapses toward 0.54+. n=3 with one inert member is not distinguishable from a coincident pair.

3. **The C1S0019 session partition result (0.443) is not independent evidence.** The same two images (C1S0019_003593, C1S0019_011890) produced a 0.443 result in session partitioning AND contribute to Q4_bright. E0-C already rejected the C1S0019 pass at n=2 on five grounds (no variance estimate possible; margin 0.007 below noise floor; sign test 6/9 wrong direction, p≈0.25). The "new" 0.443 in Experiment D is E0-C restated under a different partition label — it adds no new degrees of freedom.

**Conclusion:** Session identity does not explain meaningful histogram variance (E0-C); brightness quartile does not explain it either (D). Cross-image histogram heterogeneity is dominated by per-image content, not regime. The 0.657 global oracle floor stands. **Image-feature ridge regression (Rank 9) cannot reliably reach relL1 ≤ 0.20 without features that predict histogram shape, which are not established.** P(image-feature regression reaches 0.20) drops to ~2–3%.

---

### E16 Results (ran 2026-05-03, **MARGINAL → Detection CLOSED**)

**Script:** `scripts/experiments/probe_multiannulus_e16.py`  
**Design:** 10-annulus radial gradient profile (0–1.5R), R/4 re-centering gate, logistic classifier trained on 3 densest probe images, same image selection as E15.

**Raw results:**

| Metric | Value | Gate |
|---|---|---|
| N_TP | 567 | — |
| N_FP | 6478 | — |
| N_inter (inter-bubble FP) | 1634 | — |
| mean_TP | 0.2083 | — |
| mean_FP | 0.0694 | — |
| **SNR** | **3.00×** | ≥ 3× → PASS gate |
| Inter-bubble FP above TP median | 31.6% | was 78.4% in E15 |
| Recall at R/4 tolerance | 0.527 | — |

Script auto-verdict: PASS (SNR ≥ 3×).

**Final verdict after independent critical review (PAL + Claude consensus, 2026-05-03): MARGINAL → Detection CLOSED. No E17.**

Two independent grounds:

**Ground 1 — SNR = 3.0014× with two systematic upward biases:**

The raw quotient is 0.2083 / 0.0694 = **3.0014×**, clearing the gate by 0.0014. Two independent measurement biases are each plausibly larger than this margin:

- **In-sample evaluation** (`compute_snr`, lines 166–216): the logistic classifier is trained and evaluated on the same 567 TPs and 6478 FPs. No cross-validation or held-out fold. In-sample soft probabilities are optimistically inflated. Bias direction: always upward.
- **Oracle TP profiles** (line 141: `radial_profile(gx, gy, bgt.cx, bgt.cy, bgt.radius)`): TP profiles are extracted at GT center with GT radius. In a deployed pipeline, both come from LoG estimates, which E11 showed peak at δ=-2 to -5 scale levels below correct (underestimate by 15–40%). The classifier discriminates oracle-quality positives from real noisy negatives — a condition that does not exist operationally. mean_TP is inflated relative to any real pipeline scenario.

The underlying operational SNR is plausibly in [2×, 3×). The pre-committed rule states: *"SNR 2–3× → MARGINAL → do NOT accept post-hoc → detection CLOSED."*

**Ground 2 — Recall ceiling: 47.3% of GT bubbles are structurally unproposable at R/4 (independent of SNR):**

E15 recall at R/2 = 0.898; E16 recall at R/4 = 0.527. The gap is 37.1% of GT bubbles that have a LoG candidate nearby (R/2) but not precisely (R/4). Under the R/4 re-centering gate:
- 47.3% of GT bubbles have no candidate within R/4 and are structurally unproposable
- With a perfect classifier on the remaining 52.7%, the reconstructed histogram is still missing ~47% of all bubbles
- Structural relL1 floor from missing detections alone: **≈0.47** (2.4× above the 0.20 target)
- The SNR measurement was computed *only on the easiest 52.7%* (oracle-centered profiles); the 37.1% in [R/4, R/2) with misaligned profiles were never scored — **E16 measured the wrong bottleneck**

This is the more decisive finding: the R/4 re-centering requirement that was necessary to make the SNR measurement credible simultaneously makes the recall insufficient to reach the KPI. The two requirements are in tension and cannot both be met with the LoG generator as designed.

**On the inter-bubble FP improvement (78.4% → 31.6%):**
The improvement is likely genuine — the 10-annulus profile structurally discriminates "elevated across scattered annuli" (inter-bubble) from "peaked at 0.85–1.15R" (bubble center), and the FP profiling methodology was unchanged from E15. Note that R/4 re-centering should have *inflated* inter-FP-above-TP-median (near-TP candidates from [R/4, R/2) now in FP pool), so 31.6% represents (a) overcoming (b). However: the result is entirely in-sample with oracle TP baselines and cannot be acted upon without a held-out evaluation. It does not reverse the closed verdict.

**Conclusion:** Detection path is closed. No E17. The E13 signal (radial gradient SNR 6.86×) is real but cannot be operationalized into a deployable pipeline within this dataset under any NMS strategy: (1) dense-field recall at R/4 is structurally insufficient; (2) the classifier measurement was optimistic by construction. **Proceed to image-feature ridge regression (Rank 9) as the final handcrafted path.**

---

### Consensus priority queue (updated 2026-05-03)

| Rank | Experiment | Gating condition | Expected outcome |
|---|---|---|---|
| ~~1~~ | ~~**E0-C** within-session LOO~~ | ~~None~~ | **DONE — EFFECTIVELY FALSIFIED (2026-05-03).** |
| ~~2~~ | ~~**D** regime-conditional oracle~~ | ~~None~~ | **DONE — FAIL (2026-05-03). Best partition Q4_bright=0.437 > 0.35; MARGINAL zone is post-hoc code artifact; improvement is a two-image (C1S0004/C1S0010) coincidence. Regime conditioning not load-bearing.** |
| 3 | Metadata check | None | If session ↔ apparatus params → bypass detection entirely |
| ~~4~~ | ~~**E13** radial gradient SNR~~ | ~~After cheap diagnostics~~ | **DONE — CRITERION 3 MET (2026-05-03). SNR=6.86×.** |
| ~~5~~ | ~~**E15** generator recall probe~~ | ~~E13 passed~~ | **DONE — PARTIALLY FALSIFIED (2026-05-03). Recall PASS (0.898); SNR FAIL (1.22×, confounded); inter-bubble FAIL (78.4%). Experiment B as designed closed. Redesigned B gate open pending E16.** |
| 6 | **A** (redesigned) power spectrum LOO r check | — | LOO \|r\| > 0.6 in any bin pair → signal exists (low priority; regression path weakened by D) |
| ~~7~~ | ~~**E14** FRST (Loy-Zelinsky)~~ | ~~After E13~~ | **DONE — FALSIFIED (2026-05-03). Oracle=0.844, LOSO=0.932.** |
| ~~5.5 / E16~~ | ~~**Multi-annulus profile + re-centering gate**~~ | ~~E15 done~~ | **DONE — MARGINAL → CLOSED (2026-05-03). SNR=3.0014× with in-sample + oracle-TP upward biases; recall at R/4=0.527 imposes structural relL1 floor ≈0.47. Detection path closed. No E17.** |
| ~~8~~ | ~~**B** (redesigned) multi-annulus + geometric-NMS~~ | ~~E16 SNR ≥ 3×~~ | **CLOSED by E16 verdict.** |
| ~~**9**~~ | ~~**Image-feature ridge regression**~~ | ~~**NOW**~~ | **DONE — FAIL (2026-05-03). Ridge median 0.6807 > oracle 0.6569. Wilcoxon p=0.589. LoG global features degrade to regime-identity proxies. P(≤0.20) revised to ~0%. Path CLOSED.** |
| 10 | **C** (rearchitected) FPN + FCOS | After Rank 9 | P ≈ 8–14% beats oracle |

---

## Meta-assessment consensus (2026-05-03) — PAL + Claude independent synthesis

**Context:** After E15 partial falsification, a full meta-assessment was conducted to evaluate whether to abandon the problem entirely. PAL (independent) and Claude assessed all 5 questions; full consensus reached.

### Critical correction — r=-0.9924 is a definitional tautology

`void_fraction = 1 − img.mean / bg_mean`. With stable background (EV1 Step 0 confirmed), `r → −1` by algebra alone. **This correlation carries zero predictive information about bubble size or histogram shape.** It must be removed from all future experiment proposals and achievability arguments. Every prior co-citation with `r = -0.878` (img.mean vs median_r, which IS a real signal) artificially doubled the apparent correlation evidence. That framing stops here.

The actionable correlation is `r = -0.878` only.

### What r=-0.878 implies for the full histogram

- In LOO at n=13: R² for median prediction ≈ 0.55–0.65 (in-sample R² ≈ 0.77 degrades under LOO variance)
- Histogram width (σ_log): NOT predicted by img.mean — 4 morphological regimes produce different widths independently of brightness
- Histogram shape / skewness / multi-modality: NOT predicted
- **Best-case LOO relL1 from img.mean → histogram: ≈ 0.35–0.50 (assumes stable σ_log, no multi-modality)**
- **Realistic LOO relL1: ≈ 0.50–0.65** — marginal vs. oracle

The path from r=-0.878 to relL1 ≤ 0.20 requires additional features that predict σ_log and histogram shape. No current evidence establishes those features exist.

### P(relL1 ≤ 0.20) by path

| Path | P(reaches 0.20) | Status |
|---|---|---|
| ~~E16 → redesigned B~~ | ~~5–9%~~ | **CLOSED — E16 MARGINAL→CLOSED (2026-05-03)** |
| FPN+FCOS CNN (pretrained backbone) | 8–14% | Open |
| Image-feature ridge regression | **2–3%** (revised down from 4–5% after D failure) | Open — next to run |
| Power spectrum LOO r → regression | 2–4% | Open (low priority) |
| ~~Regime oracle alone~~ | ~~1–3%~~ | **CLOSED — D FAIL (2026-05-03)** |

Failures are positively correlated. **P(at least one approach reaches relL1 ≤ 0.20): 12–22%, central ~15–17%.**  
For the original target of 0.10: ~5–8%.

### Detection abandonment criterion (pre-committed)

E16 (multi-annulus + re-centering at R/4 tolerance) returns SNR < 2×: **detection is closed. No E17.**  
The E13 signal (6.86×) exists but cannot be operationalized in dense fields under any NMS strategy.

**Do not accept 1.8× as "marginal" post-hoc.**

### Sequencing error corrected

Experiment D (regime-conditional oracle, ~1 hr) must run **in parallel with E16**, not sequentially after it. D is independent of E16 and gates the only viable non-detection path. Deferring it further is a sequencing error.

### Revised priority queue (post meta-assessment)

| Action | Timing | Pre-committed gate | Outcome |
|---|---|---|---|
| ~~**Exp. D**~~ | ~~Now~~ | ~~D oracle ≤ 0.35 in any partition → regime load-bearing~~ | **FAIL (2026-05-03). 0.437 > 0.35. MARGINAL zone is post-hoc.** |
| ~~**E16**~~ | ~~Now~~ | ~~SNR ≥ 3× → proceed; 2–3× → CLOSED~~ | **MARGINAL → CLOSED (2026-05-03). SNR=3.0014× with upward biases; recall floor ≈0.47.** |
| **Image-feature ridge regression (Rank 9)** | **NOW** | LOSO relL1 vs oracle | P(0.20) ≈ 2–3%; final handcrafted path |
| **FPN+FCOS CNN (Rank 10)** | After Rank 9 | Beats oracle | P ≈ 8–14%; data collection binding constraint |
| Remove r=-0.9924 from all future documentation | Immediately | — | Completed |

---

## Literature Survey — Dense Object Counting / Histogram Estimation (2026-05-03)

Motivated by the closure of all detection paths and prior to committing to Rank 9, searched arxiv for approaches to analogous problems. Four search threads executed; results synthesized below and reviewed by PAL.

### Directly relevant papers

| Paper | Title | Relevance |
|---|---|---|
| 2203.15691 | Improved Counting and Localization from Density Maps (Microscopy) | Counting + localization of dense overlapping objects with Gaussian density map regression; validated on 2D/3D microscopy. Detection bypass. |
| 2511.19351 | CellFMCount / SAM-Counter | Dot annotations → density maps → SAM backbone; state-of-the-art on dense overlapping fluorescent cells. Exact supervision type (dot = center location). |
| 2012.15685 | Survey: Deep Learning-based Crowd Counting (2020) | CSRNet-type architectures, density map estimation as the standard approach for hundreds of overlapping instances; dot-annotated supervision. |
| 1705.10118 | Beyond Counting: Density Maps for Detection and Tracking | Density map quality metrics; original-resolution density maps for spatial localization. |
| **2106.02051** | **Earth Mover's Pinball Loss: Quantiles for Histogram-Valued Regression** | **Directly predicts histogram-valued outputs from images/features using EMD loss; validated on astrophysical CV. Most directly analogous to this problem's output type.** |
| 2211.14638 | Cross-domain Cell Counting by Disentangled Transfer Learning | Disentangles domain-specific / domain-agnostic knowledge; works with few annotated target-domain images. Relevant to limited-annotation regime. |

### Architecture proposal: scale-conditioned multi-channel density map — REJECTED (PAL + independent consensus)

Proposed: predict K density maps simultaneously (one per radius bin); each trained with 2D Gaussian blobs placed at GT bubble centers for that bin; integrating channel k gives count in bin k; the histogram is the integral vector across K channels. Bypasses NMS entirely.

**Consensus verdict: REJECTED. Five independent failure modes identified:**

| Failure | Evidence |
|---|---|
| 14-image wall still binds | 5,000 examples are 350/image × 14 — not independent scenes; inter-image generalization variance (E0-C, D) is still the bottleneck |
| Scale discrimination burden transfers | Backbone must implicitly learn which channel a bubble belongs to from appearance alone. E10–E16 showed appearance is unreliable for scale discrimination — the burden is harder here, not easier |
| Dense-regime blob merging (worse than CNN-B) | CSRNet canonical failure: adjacent Gaussians from K channels merge in dense fields. 27 channels × 300–600 bubbles is structurally worse than single-channel global pooling in CNN-B |
| Oracle floor 0.432 is 2.16× above target | Oracle best-case (GT access, n=2 images) = 0.432. Any model without GT access needs to exceed this. Target 0.20 requires 2.16× improvement beyond what a GT lookup achieves |
| P(≤ 0.20) ≈ 2–4% | Does not exceed CNN-B |

**One PAL error corrected:** PAL cited an "inference channel-assignment gap" claiming the model cannot construct Gaussians at inference without knowing radius. This is incorrect — density map channels are predicted end-to-end; Gaussians are only GT training targets. The real concern is implicit scale discrimination from appearance, which is real but correctly framed.

### Critical P-value correction (PAL consensus)

**P(CNN-B) = 12% (previously recorded) is for P(relL1 < 0.657, i.e., beats the oracle), NOT P(relL1 ≤ 0.20).** These are categorically different targets. At the 0.20 threshold, all architecture probabilities are in the 2–6% range. All prior references to P(CNN-B)=12% must be read against the oracle-beat target, not the 0.20 deployment target.

| Path | P(beats oracle 0.657) | P(relL1 ≤ 0.20) |
|---|---|---|
| Image-feature ridge regression | — | **2–3%** |
| Scale-conditioned density map | — | **2–4%** (not worth pursuing) |
| FPN+FCOS CNN (CNN-B equivalent) | ~12–15% | **3–6%** |
| Data collection (20 more images, n=34) | — | raises CNN-B P(≤0.20) to **~20–30%** |

### Best-fit DL architecture from literature — FamNet / DAVE (exemplar conditioning)

**FamNet** (Ranjan, ICCV 2021) and **DAVE** (Pelhan, ECCV 2023) are exemplar-conditioned counting models: the estimator is conditioned on example crops drawn from the **test image itself**, not a fixed global model. Directly addresses the 4-regime photometric heterogeneity problem by construction — each test frame provides its own conditioning signal, so cross-regime generalization is not required.

**Structural fit:** Regime heterogeneity (4 appearance regimes, LOSO regime starvation) is the dominant failure mode for all proposed CNN approaches. An exemplar-conditioned model bypasses this without requiring domain labels or session IDs.

**Caveat:** In the dense regime (void fraction >30%), exemplar crops are themselves contaminated by overlapping bubbles. Whether this is fatal requires a quick probe (extract 5 exemplar crops from a dense frame, visually inspect overlap). If exemplars are clean enough to characterize bubble appearance, FamNet/DAVE is the best-matched DL architecture for this dataset.

**Caveat 2:** Dense (300–600 bubbles) fields may produce noisy density estimates from exemplar-matched correlation. FamNet was validated primarily on crowd-counting (humans, more separable than touching bubbles). Must verify whether the dense-bubble regime is within the operational envelope of the method.

### Immediate next steps (updated)

1. **Run Rank 9 image-feature ridge regression** (already scheduled). **Add radial-gradient-integral features:** E13 measured 6.86× SNR for the outward radial gradient — globally pool across the image into a 27-bin feature vector (one per radius bin, summing inward dot-product at each candidate location across that level's response map). Near-zero implementation cost; this is the strongest confirmed physical signal in the dataset and is not yet in the Rank 9 feature set.

2. **Investigate FamNet/DAVE exemplar conditioning** as the best-fit DL architecture. Quick prior questions: (a) Are dense-frame exemplar crops contaminated? (b) Does FamNet handle 300-600-count density fields? If both pass, this is the highest-P CNN alternative.

3. **Data collection (20 more images → n=34) is the binding constraint** for any CNN path to reach P(≤0.20) > 10%. Architecture improvements without more data will not break through the oracle floor.

4. Density map regression (standard single-channel) was not formally tested but its P(≤0.20) is estimated at 2–4%, the same range as ridge regression, and should not be prioritized over Rank 9 or FamNet.

---

## Rank 9 — Image-feature ridge regression (ran 2026-05-03, **FAIL — image-feature regression CLOSED**)

**Script:** `scripts/experiments/ridge_regression_rank9.py`

**Design:** 37-feature vector per image — photometric (img_mean, img_std, skewness, kurtosis), edge density (Otsu-thresholded Sobel), FFT octave power (5 bands), radial-gradient-integral (27 features: mean scale-normalized |LoG| at sigma=r_k/√2 per radius bin, motivated by E13's 6.86× SNR). LOSO ridge regression (RidgeCV with LOO-CV alpha grid 1e-3 to 1e5) on 12 stable images (n_gt≥100).

**Raw per-image results:**

| Image | n_gt | Ridge relL1 | Oracle relL1 | Δ |
|---|---:|---:|---:|---:|
| C1S0014_006001 | 321 | 1.161 | 0.536 | +0.625 |
| C1S0014_009542 | 492 | 0.578 | 0.705 | −0.127 |
| C1S0014_018008 | 350 | 0.471 | 0.781 | −0.310 |
| C1S0014_018351 | 494 | 0.860 | 0.502 | +0.358 |
| C1S0019_003593 | 383 | 0.651 | 0.543 | +0.108 |
| C1S0019_011890 | 431 | 0.712 | 0.432 | +0.280 |
| C1S0024_014500 | 474 | 0.498 | 0.548 | −0.049 |
| C1S0004_004509 | 641 | 0.711 | 0.755 | −0.044 |
| C1S0004_005070 | 175 | 1.776 | 1.462 | +0.314 |
| C1S0004_012062 | 550 | 0.409 | 0.609 | −0.200 |
| C1S0010_005432 | 581 | 0.497 | 0.740 | −0.243 |
| C1S0010_019655 | 602 | 1.019 | 1.172 | −0.153 |

**Summary:**
- Ridge LOSO median relL1: **0.6807**
- Oracle LOO median relL1: **0.6569** (E0-A confirmed)
- Images where ridge < oracle: 7/12
- Median Δ (ridge − oracle): −0.047

Script auto-verdict: MARGINAL (script error: used ≤0.70 as MARGINAL boundary, corrected to ≤0.657 in post-hoc fix).

**Final verdict after critical review (PAL + Claude consensus, 2026-05-03): FAIL — image-feature regression CLOSED.**

**Pre-committed criterion:** relL1 > 0.657 → FAIL. Ridge median 0.6807 > 0.6569 oracle. Unambiguous.

**The four critical findings (PAL + independent, unanimous):**

**1. Criterion drift in the script output.** Script printed "MARGINAL" using ≤0.70 boundary — this zone was not in the pre-committed text criterion. Text criterion governs: >0.657 → FAIL. Script corrected post-hoc.

**2. The median-of-differences (−0.047) is not evidence of MARGINAL.** `median(ridge − oracle) ≠ median(ridge) − median(oracle)`. The divergence is caused by the outlier C1S0004_005070 (Δ=+0.314) pulling the marginal median up without affecting the sorted-differences median. The correct paired test is the Wilcoxon signed-rank:

| Test | Statistic | p-value |
|---|---|---|
| Binomial (k=7, n=12, H₁: p>0.5) | — | 0.387 |
| Wilcoxon signed-rank | W=44 | 0.589 |
| Paired t-test (two-sided) | t=0.557 | 0.589 |
| Cohen's d | +0.161 | (ridge *worse*) |

The 7/12 win rate is entirely consistent with noise (expected under null: 6.0 wins). No evidence of predictive signal.

**3. LoG feature failure is mechanistically explained — and validates E13.** E13's 6.86× SNR was measured *conditional* on oracle-known rim positions. The Rank 9 radial-gradient-integral features compute **global mean |LoG|** across the full image, which mixes bubble-rim signal with background structure, lighting gradients, wall reflections, and dense-overlap cancellations. The 6.86× figure requires localization conditioning to exist — it evaporates when integrated globally. Feature importance correctly shows kurtosis, img_mean, img_std dominating: these capture *photometric regime identity* (which session this is), not *histogram shape*. **E13's result is valid but motivates a localizer, not a global image statistic.**

**4. No post-hoc rescue is defensible.** Ridge + CV-alpha over 1e-3 to 1e5 is already optimal for this regime (n/p = 11/37 = 0.30). Feature selection, nonlinear SVR, or PCA compression applied to these same 14 images are all circular. The only defensible rescue would require a held-out session not used in any prior analysis — the dataset does not have one.

**P(relL1 ≤ 0.20) revised to ~0%** (from prior 2–3%). This is an informative negative, not a weak signal. The oracle floor (0.657) and the failed paired tests together mean the entire image-feature family provides no generalizable histogram signal at n=14.

**Binding constraint (consensus):** Data collection. ≥20 additional labeled images is the only path to a reliable regression estimator. Without it, **P(any currently open approach reaches relL1 ≤ 0.20) should be revised downward from 12–22% to ~5–10%** — the oracle floor governs all regression-family approaches.

---

### Final priority queue (post Rank 9, 2026-05-03)

All handcrafted and regression paths are closed. Open paths (by priority):

| Path | P(relL1 ≤ 0.20) | Status |
|---|---|---|
| **FamNet/DAVE exemplar conditioning** | 5–10%? | Not yet probed — highest remaining P given photometric regime handling by construction |
| **FPN+FCOS CNN with pretrained backbone** | 3–6% | n=14 bottleneck; requires ≥20 more images for >10% |
| **Data collection (20 more images → n=34)** | raises CNN path to ~20–30% | Binding constraint; architecture is NOT the bottleneck |

All remaining paths require either data collection or a qualitatively different model class (exemplar conditioning). Continuing to engineer features or add regularization variants is not defensible.
