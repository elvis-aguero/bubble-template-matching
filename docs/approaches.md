# Bubble Pipeline — Approaches Tried

A concise reference for every major approach attempted, why it was tried, and the decisive finding that ruled it in or out. For full experimental detail (raw numbers, scripts, PAL consensus notes) see `docs/experiments.md`.

This document covers **two sequential attempts** at the same scientific goal:

- **Attempt 1** (`Github/Bubble-tracking`, Jan–Mar 2026): Instance segmentation / detection framing. Goal was per-bubble instance masks; evaluation metric was AP@IoU. Abandoned when the KPI was clarified as relL1 on the size histogram, which instance segmentation cannot directly optimize.
- **Attempt 2** (`Github/template-matching`, Mar 2026–present): Histogram estimation framing. Directly targets relL1. Approaches 1–12 below all belong to this attempt.

---

## Attempt 1 — Instance Segmentation (Github/Bubble-tracking)

### Context

A research intern set up a full training pipeline (Jan–Mar 2026) targeting per-bubble instance segmentation on the same ZeroG FlightDay dataset. The pipeline ran on the Oscar HPC cluster (Brown CCV, project `dharri15`), managed via `manage_bubbly.py` with config-driven Slurm submission. Annotated data: same 14 full-frame images (1024×1024 uint16 PNG), labeled as instance-ID masks. Training outputs were saved to `~/scratch/bubble-models/trained/` (not retained locally).

### Literature Review

Nine papers (2016–2025) were surveyed, all covering CNN/classical bubble detection in Earth-gravity air-water systems. None validated on microgravity data.

| Paper | Method | Key result | Relevance |
|---|---|---|---|
| Fu & Liu 2016 | Classical geometry/optics/topology | Void fraction accurate up to 18% | Upper baseline; fails at high occlusion |
| Kim & Park 2021 (BubMask) | Mask R-CNN + size-weighted loss | AP50=98%, >95% recall on untrained swarms | Best published instance segmentation; GitHub public |
| Cerqueira & Paladino 2021 | CNN anchor-point + ellipsoidal candidates | Generalization across fluid systems | Shape reconstruction, not just detection |
| Hessenkemper et al. 2022 | StarDist (HZDR weights) + 3 CNN variants | AP@0.5 = 0.91 on RODARE data | StarDist weights directly usable; RODARE models public |
| Cui et al. 2022 | Mask R-CNN + FPN + ResNet-101 | mAP > 0.95 at up to 20% gas holdup | Handles dense overlapping bubbles |
| Yang et al. 2025 | YOLOv8 + omni-dimension conv + P2 FPN | Multi-scale overlapping + Botsort tracking | Speed-optimized; tracking not needed here |
| Maduabuchi 2024 | U-Net for boiling (HSV images) | Uncertainty quantification | Semantic segmentation only; irrelevant morph. |
| Nizovtseva et al. 2024 | CV + DL survey for multiphase flows | Multi-method comparison | Background reference |
| Hessenkemper et al. 2024 | 3D tracking of deformable bubble swarms | Deformable tracking pipeline | Out of scope (requires calibrated stereo setup) |

**Critical cross-cutting finding:** All high-AP results (AP@0.5 ≥ 0.91) are on Earth-gravity air-water data where buoyancy deforms bubbles into oblate shapes with characteristic cap boundaries. Microgravity bubbles (surface-tension dominant) are more spherical and have qualitatively different boundary contrast patterns. Published accuracy numbers are not transferable.

### Models Attempted

**MicroSAM ViT-B** (Plan A, executed): Fine-tuning of SAM encoder on `gold_seed_v04` (14 full-frame images, 3× augmentation → 42 training samples). Multiple Slurm jobs submitted (Jan 30 and Mar 6, 2026). This was the only model with confirmed GPU training runs. No evaluation results are retained locally — model weights were stored in `~/scratch/bubble-models/trained/`.

**FRST + SAM3 composite** (classical baseline, executed Feb 2026): Fast Radial Symmetry Transform detects bubble candidate points; SAM3 (Facebook's SAM 2.1) generates masks from those prompts. A second pass used adaptive thresholding (blackhat morphology). Union of all detectors followed by SAM3-NMS. On a single test image (`img19655`): FRST=621 instances, adaptive=947, SAM3 big-prompt=24, combined=636 instances after containment deduplication. Radius equivalents computed from mask area (√(area/π)).

**StarDist** (Plan B, scaffolded): Config JSON present, pipeline integrated; no training runs logged in `diary.log`.

**BubMask / Mask R-CNN** (Plan C, scaffolded): Config JSON present; no training runs logged.

### Evaluation Metric

Primary: **AP@IoU[0.5:0.95]** (COCO-style mean Average Precision). Secondary: touching-bubble F1 (Hungarian-matched pairs). The downstream goal — bubble size histogram relL1 — was never computed. Output JSON files recorded per-instance `radius_equiv_px = sqrt(area_px / π)`, but no code aggregated these into a histogram or scored it against GT. The data to compute relL1 existed; the evaluation step was simply never implemented.

### Why Attempt 1 Was Abandoned

Three factors converged, all recorded in `claude/hypotheses.md` decision log (2026-03-02):

1. **Metric misalignment recognized (decisive)**: AP@0.5 optimization does not align with relL1 — a model can achieve AP@0.5 = 0.91 while systematically misestimating bubble radii by 1–3px, placing instances in the wrong histogram bin. relL1 was never computed: output JSONs contained per-instance `radius_equiv_px = sqrt(area_px/π)`, but no code aggregated these into a histogram and scored it against GT.

2. **Dense-field occlusion structural ceiling**: At void fractions >10% (dataset range 0.3%–16.3%; several images exceed this threshold), touching bubbles merge in instance segmentation. Each architecture fails for a distinct structural reason: StarDist's star-convex boundary prior cannot separate touching instances where the boundary is non-convex; Mask R-CNN's bounding-box anchors merge nearby bubbles before the mask head; MicroSAM's SAM point-prompt generator produces unstable masks when bubbles overlap. This directly corrupts the size histogram for the high-count regime where accuracy matters most.

3. **Domain gap from published baselines**: The AP@0.5 ≥ 0.91 numbers from RODARE-trained StarDist and BubMask apply to Earth-gravity air-water data. Zero-G bubble morphology (near-spherical, 4 distinct photometric types, background-driven polarity variation) is qualitatively different. Zero-shot transfer was not evaluated, but no published number could be taken as a prior for how the model would perform.

### What Transferred to Attempt 2

- **The 14 annotated full-frame images** (`gold_seed_v04`) — same dataset, re-annotated with (cx, cy, radius) rather than instance masks.
- **FRST as a candidate point generator** — appeared in Attempt 2 as Experiment E14; falsified at the detection level (oracle recall ≤ 9.4%, LOSO relL1 = 0.932) due to cross-vote contamination in dense 300–600 bubble fields. The accumulator fills with a diffuse warm haze; individual peaks become indistinguishable. Localization itself is structurally broken at this density — the pipeline never reaches histogram evaluation.
- **The radius-from-area formula** (`radius_equiv_px = sqrt(area_px / π)`) — used in Attempt 2 output formats.
- **The core negative finding**: Instance-level accuracy metrics cannot substitute for histogram-level relL1. Any Attempt 2 approach must be evaluated end-to-end on relL1, not on detection precision/recall.

---

## Problem

Estimate the bubble **size histogram** (27 log-spaced radius bins, r=3–46px) from single dense underwater video frames containing 300–600 overlapping bubbles. Dataset: 14 annotated images, ~5,000 GT bubble instances (center + radius).

**Metric:** `relL1 = sum|pred_bin − gt_bin| / sum(gt_bin)`. Lower is better; 0 = perfect. **Target: ≤ 0.20.**

**Key benchmarks:**

| Baseline | relL1 | What it means |
|---|---|---|
| NCC pipeline LOSO | 0.851 | Current best detector; evaluated cross-image |
| GT oracle LOO | 0.657 | Lower bound achievable with GT histograms from other images |
| Target | **0.20** | Deployment minimum |

---

## Bubble Morphology (relevant to all approaches)

Four morphological types confirmed by radial intensity cross-sections (E11 Step 1):
- **Dark-rim** (bright interior, dark ring at boundary): 54%
- **Filled-dark** (dark center, lighter surround): 27%
- **Bright-rim / filled-bright**: 12%
- **Flat / indeterminate**: 8%

Any detector that depends on a single polarity, a single spatial scale, or a single intensity model will structurally miss 12–46% of the population.

---

## Photometric Context

- ~50% of all video frames are "photometrically dead" — too bright (bubbles blend into background) to contain usable signal.
- 4 distinct photometric regimes across the 14 annotated images (dark, bright, dense, sparse).
- img.mean correlates with median bubble radius (r=−0.878), but **not** with histogram shape or width.
- Cross-image histogram variance is high (oracle median 0.657) and is NOT explained by session identity, brightness, or bubble density (Experiments D, E0-C).

---

## Approaches Tried

---

### 1 · Normalized Cross-Correlation (NCC) pyramid + NMS

**What it is:** 27-level image pyramid scaled by factor 0.9/level. At each level, cross-correlate the image with an averaged bubble template. A local-maximum score at level k means "there may be a bubble of radius eff_r_k here." Cross-scale Non-Maximum Suppression (NMS) keeps the highest-scoring candidate when two scales compete for the same location.

**Why tried:** NCC is the standard template-matching baseline. The 27-level pyramid covers radii 3–46px. NMS prevents double-counting across scales.

**What was found:** NCC is **not scale-selective**. Its scale-space response is a monotone downward slope — finer pyramid levels always score higher because finer resolution has richer high-frequency texture for the kernel to correlate with. This has nothing to do with whether a bubble is actually present at that scale.

**Consequence:** Cross-scale NMS always picks the finest viable scale (often wrong by 4–6 levels), evicting the correct-scale peak. Measured eviction rate: **87.8%** of GT bubbles have a correct-scale raw peak that NMS suppresses.

**Fixes attempted — all failed:**

| Fix | Experiment | Result | Why it fails |
|---|---|---|---|
| Per-level calibrators | E1 | relL1 0.950 | Circular: trained on NMS survivors; only 11.6% of GT peaks survive |
| GT-centroid calibrator | E4 | relL1 1.128 | Edge artifacts score above GT distribution; classifier can't separate them |
| Score normalization `(r/r₀)^α` | E5/E6 | Max 13% rescue at α=2 | Competitor advantage is heavy-tailed (median 1.53×); just shifts which wrong scale wins |
| Per-level independent NMS | E9 | 24× over-prediction | Each fine level has ~1,000 spurious edge peaks; without cross-scale suppression, every level floods |
| Scale-specific templates | O3 | Ruled out by physics | The slope is signal-density artifact, not template-shape artifact; templates can't create a peak where none exists |

**Verdict: Closed.** NCC is algebraically broken for scale discrimination in this dataset. Experiments E1–E9 + O3 exhaustively falsified every escape route.

---

### 2 · Laplacian of Gaussian (LoG) / Difference of Gaussians (DoG)

**What it is:** Apply a Gaussian Laplacian at sigma=r/√2; for ideal filled circular blobs, the response peaks at the correct scale. DoG approximates LoG via difference of two Gaussians. Standard blob-detection method.

**Why tried:** LoG is the theoretical blob-detector for circular objects. Unlike NCC, LoG is derived from first principles — it should produce a genuine scale-space peak at the correct bubble size, addressing NCC's algebraic monotone failure.

**What was found:** LoG has **no reliable scale-space peak** for this bubble population (E11). IQR of per-bubble peak-delta across 52 GT bubbles = **6.2 pyramid levels** — the lower quartile peaks at the finest measured scale (floor-censored; true IQR worse). This is structural, not parametric: the 4 morphological types respond to different sigmas, so no single σ produces a compact peak for the whole population.

Tested 4 configurations: center-pixel blob-sigma, rim-pixel blob-sigma, center-pixel ring-sigma, rim-pixel ring-sigma. All falsified — either sensitivity (median peak ≠ 0) or specificity (SNR < 2×) or both.

**However:** LoG/DoG as a *spatial locator* (ignoring its scale estimate) achieves **89.8% recall** at R/2 matching tolerance (E15). The spatial position of LoG extrema lands near bubble centers reliably even when the scale estimate is wrong. This partially motivated E15/E16.

**Verdict: Closed as a scale estimator.** Open (and confirmed useful) as a spatial locator only — but the detection path was closed by E16 (see §5 below).

---

### 3 · Full-image Hough Circle Transform

**What it is:** OpenCV `HoughCircles` — every edge pixel votes for circles it could lie on; accumulate votes across (x, y, r) space; peaks are circle candidates.

**Why tried:** Hough is the canonical circle detector and is explicitly designed for arbitrary radii. Should be robust to morphology variation because it uses gradient orientation votes, not a scalar score.

**What was found:** Full-image Hough **FALSIFIED** (E12). DR_in_tol = 11% (target 70%), FP/image = 771–2430 (target ≤5). Root cause: with 300–600 bubbles and apparatus structure all simultaneously voting, the accumulator is noise-dominated — detected radius averages 2.2× GT radius from spurious background votes.

**Scope caveat:** Patch-based Hough (limit vote integration to a local image region around a candidate center) was never tested. The failure is specific to full-image accumulation. Patch-based Hough remains technically open but low-priority given the detection-path closure.

**Verdict: Full-image Hough closed.** Patch-based Hough not attempted.

---

### 4 · Radial Gradient at Bubble Rim

**What it is:** Compute the Scharr gradient at each pixel, dot with inward unit radial vectors over annulus r/R ∈ [0.85, 1.15] centered on a candidate location. Measures whether there is a consistent inward-pointing gradient ring — the signature of a bubble rim.

**Why tried:** After NCC, LoG, and Hough all failed, tested whether the physical bubble rim produces *any* detectable gradient signal, independent of detector architecture. E13 is a pure signal-characterization experiment.

**What was found:** Signal is **real and strong at known locations** — overall SNR = 6.86× (E13, 2349 GT bubbles, 14 images). 91% of bubbles are outward-dominant (the rim gradient points outward, as expected for a bright-interior bubble). Signal holds across all size bins, photometric regimes, and morphology types except large × inward-marginal (n=13, SNR = 0.52×).

**Key limitation discovered:** The 6.86× figure was measured at *oracle-known* bubble centers and radii. When evaluated on actual LoG-proposed candidates in a dense field (E15, E16), the signal collapses due to: (a) candidate position offset from true center misaligns the annulus; (b) inter-bubble FPs land between touching bubbles where multiple rims contribute high gradient. This signal requires knowing where to look — it cannot be used as a standalone detector.

**Verdict: Signal is real; not independently deployable.** The E13 result motivated E15/E16 (see §5).

---

### 5 · Per-candidate Patch Scorer (LoG generator + radial gradient classifier)

**What it is:** Two-stage pipeline — (1) LoG/DoG generates candidate (x, y) locations ignoring scale; (2) for each candidate, extract a radial gradient profile (10 annuli from 0–1.5R) and classify as bubble / not-bubble using logistic regression trained on 5,000 annotated bubble instances.

**Why tried:** Addresses the structural failure of global vote accumulators (NCC, FRST, Hough) by scoring candidates in isolation. Each candidate's patch is evaluated independently — neighboring bubbles are outside the patch window and don't contaminate the score. LoG spatial recall (89.8%) is good enough to propose most bubbles.

**E15 — generator recall probe:** Tested LoG/DoG on 3 densest images.
- Generator recall at R/2: **89.8%** (PASS — sufficient proposals)
- Radial gradient SNR on actual candidates: **1.22×** (FAIL — too low for reliable classification)
- Inter-bubble FPs above TP median: **78.4%** (FAIL — candidates between touching bubbles score as high as real bubbles)

**E16 — multi-annulus profile probe:** 10-annulus profile at R/4 re-centering to fix SNR confound.
- Logistic regression SNR: **3.0014×** (barely clears 3× gate — but in-sample with oracle GT centers/radii; operational SNR likely 2–3×)
- Recall at R/4: **52.7%** — 47.3% of GT bubbles are structurally unproposable at R/4 precision
- Structural relL1 floor from missing detections: **≈ 0.47** (2.4× above target)

**Decisive failure:** The R/4 re-centering required to make the SNR measurement credible simultaneously makes recall insufficient. These two requirements are in fundamental tension and cannot both be satisfied with the LoG generator.

**Verdict: Detection path formally closed after E16.** No E17.

---

### 6 · Fast Radial Symmetry Transform (FRST / Loy-Zelinsky)

**What it is:** Each gradient-magnitude pixel votes for radial symmetry centers at distance r in the gradient direction. Accumulate votes across all image pixels → response map per r value → NMS.

**Why tried:** Unlike NCC and Hough, FRST is specifically designed for scale-selective radial symmetry detection. Published for cell/bubble microscopy detection. Theoretically robust to morphological variation.

**What was found (E14):** FRST FALSIFIED — oracle recall ≤ 9.4%, LOSO relL1 = 0.932 (worse than NCC baseline 0.851). Failure mode: in a dense 300–600 bubble field, rim pixels from all bubbles cross-vote at radii landing near neighboring bubble centers. At 18 radii simultaneously, the background vote density rises uniformly until individual bubble peaks are indistinguishable. Visual inspection confirms a diffuse warm haze, not discrete peaks.

**Note:** Pre-committed criterion (apparatus domination) was NOT triggered — the vessel walls were suppressed correctly. The actual failure is dense-field cross-vote contamination, same family as Hough.

**Verdict: Closed for full-image accumulation.** Per-candidate patch scoring using FRST as a feature (vs. as a detector) was tested in E15/E16 and also closed (see §5).

---

### 7 · Temporal Background Subtraction + Watershed

**What it is:** Compute a per-pixel background model from the first 20 (bubble-free) frames. Subtract background from each subsequent frame. Threshold the residual and apply distance-transform watershed segmentation to separate individual bubbles and estimate their radii.

**Why tried:** Unlabeled video ZeroG_Test3_Opt3 (7,501 frames) was available. Early frames are confirmed bubble-free (median of first 20 frames is a clean background). Temporal differencing cancels static apparatus structure exactly. Required no additional annotations.

**What was found (EV1):** FALSIFIED for the dense bubbly regime (which is the target). Watershed median radius monotonically increases as the frame darkens (5px at frame 200 → 14px at frame 4000). The correlation Pearson(img_mean, med_r) = −0.878 appears strong, but is a **Voronoi-tiling artifact**: as more bubbles pack in, watershed region boundaries are shared — each bubble gets a smaller allocated area, and sqrt(area/π) underestimates the true radius. The "coalescence" trend is spurious.

**Additional context:** RNN-CNN hybrid using temporally correlated frames was assessed and rejected. 10 correlated frames at r=0.95 inter-frame correlation contribute < 0.25 effective independent training examples.

**Verdict: Closed for dense regime.** Sparse regime (void fraction <10%) was not tested.

---

### 8 · Regime-conditional Oracle (session / brightness / density partitions)

**What it is:** Partition the 14 images into groups by session identity, brightness quartile, or bubble density tercile. Within each group, use LOO oracle — predict each image from the within-group histogram median. If within-group oracle beats the cross-group oracle (0.657), then regime identity explains histogram variance and a lookup table approach becomes viable.

**Why tried:** If same-session or same-brightness images have statistically similar histograms, the detection problem is bypassable — just identify the regime and return its representative histogram.

**What was found:**
- Within-session (E0-C): Only C1S0019 technically passes (0.443 < 0.45 threshold), at n=2 images with 0.007 margin. Wilcoxon sign test on 9 in-scope pairs: 6/9 in wrong direction, p≈0.25. EFFECTIVELY FALSIFIED.
- Brightness quartile + density tercile (Experiment D): Best partition = Q4_bright, median relL1 = 0.437. Pre-committed criterion: ≤ 0.35 → PASS. 0.437 > 0.35 → **FAIL**. The "MARGINAL" label printed by the script was a post-hoc code artifact, not a pre-committed category.

**Verdict: Closed.** Cross-image histogram heterogeneity is not explained by session, brightness, or bubble density. The 0.657 oracle floor stands as the binding lower bound for any regression-family approach.

---

### 9 · Image-feature Ridge Regression (Rank 9)

**What it is:** Extract a 37-dim feature vector per image (photometric statistics, Sobel edge density, FFT octave powers, scale-normalized |LoG| per radius bin). Train ridge regression (RidgeCV, alpha selected by leave-one-out cross-validation) to predict the 27-bin histogram. LOSO evaluation on 12 stable images.

**Why tried:** The last handcrafted path. If global image statistics — including the E13-motivated radial gradient features — contain any signal about histogram shape, ridge regression should find it. The scale-normalized |LoG| feature (one value per radius bin, globally pooled) was specifically added to connect E13's 6.86× SNR to a regressor.

**What was found (Rank 9, 2026-05-03):** LOSO median relL1 = **0.6807**, oracle = **0.6569**. Ridge regression is *worse* than the oracle. Wilcoxon signed-rank p = 0.589; binomial test on 7/12 oracle wins: p = 0.387. No signal at any significance level.

**Decisive mechanistic explanation:** E13's 6.86× SNR was measured *conditional on oracle-known rim positions*. Global image |LoG| pooling mixes the bubble-rim signal with background structure, lighting gradients, and dense-overlap cancellations. The 6.86× figure **requires localization** — it evaporates when pooled globally. The dominant features (kurtosis, img_mean) index *which photometric session this is*, not *what the histogram looks like*. Experiment D already showed session identity is not load-bearing.

No post-hoc rescue is defensible at n=14: ridge + CV-alpha over 5 decades is already optimal for n/p = 11/37 = 0.30. Feature selection or nonlinear regression on the same 14 images is circular.

**Verdict: Closed.** P(image-feature regression reaches relL1 ≤ 0.20) revised to ~0%.

---

### 10 · Scale-conditioned Multi-channel Density Map (proposed, rejected without running)

**What it is:** Predict K=27 density maps simultaneously (one per radius bin), each trained with 2D Gaussian blobs at GT bubble centers for that bin. Integrate channel k to get per-bin count → histogram. Bypasses NMS entirely.

**Why considered:** Density map regression (CSRNet, DM-Count) is the established approach for dense overlapping objects in crowd counting and microscopy. Dot annotations — exactly what we have — are the standard supervision type.

**Why rejected (PAL + independent consensus, 2026-05-03):**
1. **14-image wall still binds.** 5,000 examples are 350/image × 14 scenes — the backbone must generalize across images, not just across instances within one image. Inter-image variance (oracle 0.657) is the bottleneck, same as all other regression approaches.
2. **Scale discrimination burden.** The backbone must implicitly assign each bubble to the correct channel from appearance alone. E10–E16 established that appearance is not reliably discriminative of scale.
3. **Dense-field blob merging is structurally worse.** CSRNet's canonical failure (adjacent Gaussians merge in dense fields) is multiplied by 27 channels.
4. **Oracle floor.** Best-case oracle relL1 = 0.432 (GT access, n=2 images). Target 0.20 requires 2.16× improvement beyond what a GT lookup achieves. No mechanism that accomplishes this has been established.
5. **P(≤ 0.20) ≈ 2–4%** — same as CNN-B, no improvement over already-explored paths.

**Verdict: Rejected.** Not worth implementing at n=14.

---

### 11 · Phase Congruency + Radial Symmetry (proposed, assessed, not run)

**What it is:** Phase congruency (Kovesi 1999) — measures coherence of phase across spatial frequency channels; contrast-invariant and polarity-agnostic. Apply as a preprocessing step before a radial symmetry vote accumulator.

**Why considered:** NCC, LoG, and Hough all fail partly due to polarity/morphology sensitivity. Phase congruency responds to boundaries regardless of whether they are dark-to-bright or bright-to-dark, addressing the 4-regime morphology problem.

**Why not run:** Assessed as low-priority after E16 closed the detection path. Even with a better feature, any full-image vote accumulator shares the cross-bubble contamination failure of FRST and Hough in dense fields. Per-candidate patch scoring using phase congruency as the feature is technically open, but the generator recall at R/4 (52.7%, E16) creates a structural relL1 floor ≈ 0.47 independent of the scoring feature.

**Verdict: Not run. Low priority given detection-path closure.**

---

### 12 · RNN-CNN hybrid on temporally correlated frames

**What it is:** Annotate 10 consecutive video frames; train an RNN-CNN to exploit inter-frame temporal coherence and use each frame's temporal context.

**Why considered:** Consecutive frames should contain correlated bubble populations. A temporal model could smooth per-frame noise and potentially improve histogram estimates.

**Why rejected:** At inter-frame correlation r=0.95, effective sample size from 10 correlated frames = n_eff ≈ 0.25 independent frames. Annotating 10 correlated frames costs the same as 10 independent frames but yields < 1 effective training observation. Additionally, static apparatus structure is MORE temporally persistent than bubbles — a temporal feature would learn to predict the apparatus, not the bubbles. Direction is inverted.

**Verdict: Rejected without running.**

---

## What Remains Open

| Approach | P(relL1 ≤ 0.20) | Status |
|---|---|---|
| **FamNet / DAVE exemplar conditioning** (Ranjan ICCV 2021, Pelhan ECCV 2023) | 5–10% (unprobed) | Not yet attempted. Conditions on crops from the *test image itself* — bypasses cross-regime generalization by construction. Best remaining architectural fit for the 4-regime heterogeneity problem. |
| **FPN + FCOS with pretrained backbone** | 3–6% at n=14 | Not yet attempted. Data is the bottleneck, not architecture. |
| **Data collection (≥ 20 more annotated images → n=34)** | Raises CNN to ~20–30% | The binding constraint for all remaining CNN paths. |

---

## Why the Oracle Floor (0.657) Is the Central Obstacle

The GT oracle — predict each image's histogram as the LOO median of the other 13 images' GT histograms — achieves median relL1 = 0.657. This is NOT a trivially achievable result: it requires GT annotations from other images, which are not available at deployment. But it represents the ceiling of what any *image-invariant* predictor can achieve.

Reaching relL1 ≤ 0.20 requires a model that extracts **per-image discriminative signal** about the bubble size distribution — not just a good prior. Every tested approach that did not read the current image's bubble structure (regression, regime lookup, density maps) is bounded by this floor. Every approach that did try to read per-image bubble structure (NCC, LoG, Hough, FRST, patch scorer) was falsified by one of three dense-field failure modes: scale discrimination failure, vote accumulation contamination, or generator recall ceiling.

The oracle floor itself would need to drop (more labeled images → lower variance → lower cross-image LOO error) for any regression-family approach to have a realistic shot at 0.20.
