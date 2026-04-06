# Counting Method Comparison

Research log tracking all counting approaches tried for the bubble size histogram pipeline. Each entry records what was tried, the metric results, and what was learned. Read this before implementing a new counting strategy.

**Context**: 14 annotated images from a zero-gravity flight experiment. Goal: estimate the per-frame bubble size histogram without annotating every frame. Training data covers 5 sessions (C1S0004, C1S0010, C1S0014, C1S0019, C1S0024), with 100–500 bubbles per image, 87% under 10 px radius.

---

## Metrics

- **Ratio**: predicted_total / true_total  (1.0 = perfect total count)
- **MARE**: mean |ratio − 1| across seeds  (0 = perfect, symmetric penalty)
- **relL1**: mean Σ_l |N_pred(l) − N_true(l)| / N_true_total across seeds
  — L1 distance between predicted and true per-bin histograms, normalised by true total.
  Penalises both wrong total count AND wrong size distribution.
  relL1 ≥ MARE always; the gap measures distribution shape error beyond count error.
  A model that predicts all zeros gets relL1 = 1.0.

N_true(l) computed by assigning each annotated bubble to the pyramid level whose effective radius is closest to the bubble's annotated radius.

**Evaluation protocol**: 8 random seeds, each with a fresh 30/65/5% image-level split (template/calibration/test). Seeds whose test image has fewer than 50 annotated bubbles (the C1S0010 image with 14 annotations) are excluded. All reported metrics are averages across valid seeds.

---

## Results

### Single-seed exploratory (earlier, MARE only, no L1)

| Approach | Description | Ratio range (1 seed) | Notes |
|---|---|---|---|
| Dense (baseline) | Σ P over all pixels, all levels, pixel prior | 4.1–19.2× | Severe spatial + scale overcounting |
| Stride s=10 | Evaluate at every 10th pixel per level | 0.06× | 21% of bubbles missed by grid; prior not adjusted |
| Argmax-scale | Per pixel, assign to argmax-level only | 0.64× | Upsampling artifacts; undercounts |
| Stride + argmax | Both combined | 0.01× | Essentially zero — two undercounts compounded |
| Dense / calib\_bias | Dense ÷ empirical overcounting factor from calibration images | 0.17–0.94× | Bias varies by image; not transferable |
| LM (pixel prior, single seed) | Σ P at spatial local maxima, pixel prior | 0.65× | Single seed only |
| LM + LM prior (broken) | LM counting but prior = n\_bubbles/n\_lm; f⁻ still from random pixels | 6.5–29× | f⁻ mismatch: background peaks appear as bubbles |
| SS-LM + SS prior (broken) | Grid-suppress LM across levels; prior = n\_bubbles/n\_ss | 5.1–20× | Same f⁻ mismatch |

### Multi-seed systematic (8 seeds, MARE + relL1)

| Approach | mean ratio | MARE | relL1 | std(relL1) | ratio range |
|---|---|---|---|---|---|
| Dense (pixel prior) | 7.08× | 6.08 | 6.08 | 4.50 | [3.88×, 18.47×] |
| **LM (pixel calib, pixel prior)** | **0.79×** | **0.44** | **0.78** | **0.38** | [0.41×, 1.91×] |
| LM (LM calib, LM prior) | 1.53× | 0.60 | 1.09 | 0.82 | [0.80×, 3.82×] |

### Counterfactual comparison (8 seeds, relL1 primary)

Testing three hypotheses simultaneously: (CF-1) scale-space NMS before summing,
(CF-2) multi-template (3 templates, log-spaced by radius), (CF-3) trivial null baseline
that ignores the test image entirely and returns the mean annotated histogram from
calibration images.

| Approach | mean relL1 | std(relL1) | mean ratio | ratio range | Notes |
|---|---|---|---|---|---|
| LM 1-template (current best) | 0.778 | 0.376 | 0.791× | [0.41×, 1.91×] | Baseline for comparison |
| CF-1: SS-argmax LM | 0.748 | 0.105 | 0.371× | [0.18×, 0.83×] | Lowest variance; systematic undercounting |
| CF-2: LM 3-templates | 0.769 | 0.339 | 0.717× | [0.33×, 1.74×] | Marginal improvement over 1-template |
| **CF-3: Constant histogram** | **0.648** | **0.317** | **1.013×** | [0.63×, 2.41×] | **Beats all model approaches** |

Interpretation: the constant histogram predicts the mean calibration histogram regardless of the test image content. That it outperforms every model-based approach (relL1 0.648 vs best model 0.748) means the current models are not learning image-discriminative features — they reproduce the training distribution without adding useful signal.

---

## Key findings

1. **Spatial overcounting dominates.** Each bubble generates elevated NCC scores
   over a ≈ s×s = 100-pixel neighbourhood. Dense summation counts each bubble
   ~5–18× depending on image density. The ratio is NOT stable across images
   (std = 4.5×), so a fixed empirical correction is unreliable.

2. **Scale-space spreading is real but secondary.** NCC scores are elevated at
   2–4 adjacent pyramid levels per bubble. This contributes a factor of ~3–5×
   on top of spatial overcounting in the dense sum.

3. **Local maxima (pixel prior) is the best current approach.** Restricting the
   sum to spatial local maxima (min\_distance = template\_size/2) eliminates most
   spatial duplication with no change to calibration or prior.
   - mean relL1 = 0.78 (vs 6.08 for dense)
   - Still undercounts (mean ratio 0.79×) — some bubbles have no nearby peak or
     their peak falls in a low-P score bin
   - relL1 > MARE (0.78 vs 0.44): size distribution is also distorted,
     not just total count

4. **Changing the prior without fixing f⁻ makes things worse.** Using
   n\_bubbles/n\_lm as the prior while keeping f⁻ from random pixels inflates P
   at background local maxima (which score higher than random background pixels).

5. **LM calibration (f⁻ from local maxima) does not consistently improve over
   pixel calibration.** Mean relL1 rises from 0.78 → 1.09. Likely causes:
   - Level mismatch: positives sampled at best-match level, negatives at level 0
   - f⁻ at level-0 local maxima may not represent the distribution at higher-level
     local maxima where most detections happen

---

## Open questions / next directions

### Critical finding: null baseline wins

The constant histogram (CF-3) achieving relL1=0.648 vs best model 0.748 is the
most diagnostic result to date. Two explanations:

1. **Data insufficiency.** With 14 annotated images across a handful of sessions,
   the calibration split contains 4–5 images. The mean of 4–5 histograms is a
   low-variance estimator of the training distribution; it accidentally tracks
   the test distribution because all images come from the same experiment.

2. **NCC discriminability floor.** A single template computed by averaging all
   bubbles (regardless of size) may lack the discriminative power to distinguish
   size bins. The NCC score at a given scale level may not be meaningfully higher
   for on-scale bubbles than off-scale ones.

### Per-bin relL1 decomposition (8 seeds, best model: LM pixel calib)

| Size range | true_frac | rel contribution | ratio err/true |
|---|---|---|---|
| 5.0 px | 6% | 0.046 | 1.05 |
| 5.6–6.9 px | 27% | 0.111 | 0.60–0.68 |
| 7.6–12.9 px | 40% | 0.172 | 0.57–0.70 |
| 14–20 px | 12% | 0.056 | 0.62–0.75 |
| >20 px | 4% | 0.020 | 0.68–1.7 |

**Key finding**: error is distributed proportionally across all size bins, tracking the true distribution shape. There is no single size range that is particularly miscalibrated. The ratio `|err|/true` is approximately constant (~0.6×) across all mid-range bins.

This means the problem is **not** a size-specific failure (wrong template for small vs large bubbles). It is a **global scale error**: the model cannot reliably estimate how many bubbles are in the image; it recovers the approximate shape of the histogram but gets the overall magnitude wrong per image. The constant histogram baseline wins because it predicts the correct shape (training distribution) and the correct average magnitude, at the cost of zero per-image sensitivity.

### Recommended next experiments (in priority order)

1. **Per-bin relL1 decomposition**: which size bins are most wrong? If error
   concentrates in a few bins, a targeted fix (larger template, separate template
   per bin) may be sufficient.

2. **CNN patch classifier**: replace the NCC scorer with a small (e.g. 3-layer)
   CNN trained with binary cross-entropy. If the CNN beats the null baseline, the
   failure is in NCC discriminability not data volume.

3. **More annotated data**: the constant baseline exploits the fact that all
   images come from the same distribution. If we can access frames from different
   conditions (different bubble populations), the null baseline would fail and the
   model would have to generalise.

4. **Matched LM calibration**: positives AND negatives sampled at local maxima,
   at the bubble's best-match level (not level 0). This would fix the f⁻ level
   mismatch identified in the multi-seed experiments.

5. **SS-argmax with recalibrated prior**: CF-1 consistently undercounts
   (mean 0.37×) but has the lowest variance. If the prior can be adjusted to
   compensate, this may become the most stable approach.
