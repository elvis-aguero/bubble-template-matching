# Bubble Size Histogram Pipeline — Operator Guide

This document explains what the pipeline does, how to run it, and how to tune it. After reading this you should be able to train a new pipeline, run it on video frames, and understand why the numbers look the way they do.

---

## What it does

We have zero-gravity flight video in which hundreds of bubbles float around per frame. We want a **size histogram per frame** — how many bubbles of each radius are present — without hand-labeling every frame.

The pipeline learns from 14 manually annotated images (`seed_v04/`) and then estimates bubble counts on any new image.

The core idea: **counting is easier than detection.** Instead of drawing a box around each bubble (detection), we ask "how many bubbles of each size are in this frame?" (counting). We count by summing probabilities — never by thresholding. Consider 10 locations each with 70% probability of containing a bubble: if you threshold to get a hard yes/no, you end up with either 0 or 10 — both wrong. Sum the probabilities instead and you get 7, which is correct on average.

### The three-step logic

```
14 annotated images
        │
        ▼
1. Build a template      — average appearance of a bubble at canonical size
   (template.py)
        │
        ▼
2. NCC score maps        — how well does each patch in the image match the template?
   (ncc.py)                Produces a score C[x,y] at every pixel location.
   (one per scale level)   Each scale level = one size bin in the histogram
        │
        ▼
3. Bayesian calibration  — convert NCC score → P(bubble) using training statistics
   (calibration.py)
        │
        ▼
   expected_count[size_bin] = Σ P(bubble | score[x,y])   over all pixels
```

**Why NCC?** NCC is scale- and contrast-invariant within a window. A score of +1 means the patch looks exactly like the template; 0 means uncorrelated; −1 means the inverse pattern. The image patch W is divided by its norm to make the score invariant to overall brightness and to turn the dot product into something like an angle between two directions. `skimage.match_template` does all of this for us.

**Why Bayesian calibration?** NCC scores are not probabilities. The calibrator learns the empirical distribution of scores at bubble vs. non-bubble locations and converts them to `P(bubble|score)` via Bayes' rule. This is a non-parametric approach: histogram the scores at bubble and non-bubble locations from the training data, then invert using Bayes — no curve-fitting involved. Expected-value counting (`Σ P`) gives a soft count that degrades gracefully when the detector is uncertain.

---

## How big bubbles are handled: the image pyramid

The key idea: to count big bubbles, shrink the image until they look small, then apply the same small-bubble template.

The template is always `template_size × template_size` pixels (default 10×10). Instead of resizing the template to match each bubble size, we **shrink the image** by a factor of 0.9 at each step. At each scale level, a bubble that was originally large now looks small — exactly the size the template was built for.

Each shrink step is one histogram bin. Using a multiplicative shrink factor (like 0.9 per step) keeps the number of levels manageable, and since the images get smaller at each level, the computation is fast — most of the work happens at the first few levels where the image is still large. With defaults, this gives ~21 bins covering radii from 1 px to 50 px.

This sequence of shrunk images is called the **image pyramid** — wider at the base (original image, small bubbles) and narrower at the top (very small image, large bubbles).

---

## Dataset layout

```
seed_v04/
├── images/     ← 14 PNG files (grayscale, 8-bit or 16-bit)
├── labels/     ← 14 LabelImg JSON files, one per image
└── manifest.csv
```

Sessions (used for train/val splits):

| Session | Files |
|---------|-------|
| C1S0004 | 4 images |
| C1S0010 | 2 images |
| C1S0014 | 3 images |
| C1S0019 | 2 images |
| C1S0024 | 3 images |

Each JSON contains circle and/or polygon annotations labeled `"bubble"`. Circles are stored as `[center, edge_point]`; polygons are converted to their minimum enclosing circle internally.

---

## Installation

```bash
pip install -e ".[dev]"   # installs the package + pytest
```

Dependencies: `numpy`, `scikit-image`, `Pillow`, `matplotlib`.

---

## Typical workflow

### Step 1 — Train

```bash
python scripts/train.py seed_v04/ output/pipeline.pkl
```

With a held-out validation session (leave-one-session-out):

```bash
python scripts/train.py seed_v04/ output/pipeline.pkl --val-session C1S0010
```

This produces `pipeline.pkl` containing the template, calibrator, and config. Training on all 14 images takes a few seconds.

During training the pipeline does two things:
1. **Builds the template** — extracts a patch around each annotated bubble, resizes it to 10×10, normalizes to sum=1, averages them all, then divides by the norm to make it a unit vector (`T = T / ||T||`).
2. **Fits the calibrator** — runs the template across a separate set of images, collects the NCC scores at annotated bubble centers (positives) and at random non-bubble locations (negatives), histograms both, and applies Bayes' rule to build the `P(bubble|score)` lookup table.

**Important:** The 14 annotated images are split into three non-overlapping sets before training begins (default: 30% template / 65% calibration / 5% test). Template construction and score calibration each see different images. This prevents data leakage: an image that contributed to the template would have inflated NCC scores at bubble locations, which would bias the calibrator.

**Artifacts written alongside `pipeline.pkl`** (all in the same directory, same filename stem):

| File | Contents |
|------|----------|
| `*_templates.png` | The learned template(s) — check for visible structure |
| `*_score_histograms.png` | Overlapping density histograms of NCC scores at bubble (blue) and non-bubble (red) locations from the calibration set |
| `*_ncc_<name>.png` | Original image alongside the NCC score map at the most active scale level (written for up to 2 **calibration** images) |
| `*_ncc_TEST_<name>.png` | Same NCC diagnostic, but for each **test** image — clearly labelled to avoid confusion with the calibration ones |
| `*_split.json` | Exact list of which images went to template / calibration / test sets, for reproducibility |
| `*_size_hist_<name>.png` | Predicted bubble size histogram for each test image, with annotated ground-truth counts overlaid in red hatching |

### Step 2 — Predict on new images

```bash
python scripts/predict.py pipeline.pkl path/to/frame_*.png --output histograms.csv
```

`histograms.csv` has one row per (image × size bin):

```
image,radius_px,expected_count
frame_001.png,5.0,342.1
frame_001.png,5.56,289.4
...
```

`radius_px` is the effective bubble radius in the **original image's pixel coordinates** that corresponds to that histogram bin. `expected_count` is the expected number of bubbles of that size — the sum of P(bubble) over all pixel locations at that scale level.

### Step 3 — Visualize

```bash
python scripts/visualize.py pipeline.pkl \
    --image seed_v04/images/ZeroG_FlightDay_Test_C1S0014_img006001.png \
    --output-dir plots/
```

Saves figures to `plots/`:

| File | What to look for |
|------|-----------------|
| `templates.png` | Should show a **coherent, spatially structured pattern** — whatever the average bubble looks like in your imaging setup. The important thing is that it has structure: a clear center, edge, or ring. A uniform grey blob means the averaging cancelled out (too much size or appearance variation). |
| `calibration.png` | `P(bubble|score)` should be higher for scores > 0 and near 0 for very negative scores. |
| `histogram.png` | Most count should be concentrated at small radii (2–10 px), matching the annotation statistics. |

If `templates.png` looks like uniform grey with no structure, that is a sign the averaging is washing out the pattern — try reducing `num_templates` variation or narrowing the radius range.

The `*_score_histograms.png` produced during training (not by `visualize.py`) shows the score distributions used to fit the calibrator. The blue (bubble) and red (non-bubble) curves should be separated — bubbles should score higher on average. If the two distributions overlap heavily, the template has poor discrimination and the expected counts will be unreliable.

The `*_size_hist_<name>.png` produced during training shows the predicted bubble size distribution for the held-out test image, with the annotated ground-truth counts overlaid as red hatching. This is the first honest look at generalization — the test image was never used for template construction or calibration. The two bars should follow the same shape; systematic over- or under-prediction in a particular size range points to a template or calibration issue at that scale.

---

## Hyperparameters — the knobs

All parameters live in `PipelineConfig` (`bubble_histogram/config.py`). Pass them as CLI flags to `train.py`.

### Size range

| Parameter | Default | CLI flag | Effect |
|-----------|---------|----------|--------|
| `min_radius` | 1.0 px | `--min-radius` | Smallest bubble radius the pipeline looks for |
| `max_radius` | 50.0 px | `--max-radius` | Largest bubble radius |

The annotated bubbles span roughly 1–31 px radius (median ~4.4 px). The default `max_radius=50` is intentionally wider than needed.

**When to change:** If you know bubbles never exceed 20 px, set `--max-radius 20` to reduce the number of pyramid levels and speed things up. If you see very large bubbles in a new experiment, increase `--max-radius`.

### Scale resolution

| Parameter | Default | CLI flag | Effect |
|-----------|---------|----------|--------|
| `scale_factor` | 0.9 | `--scale-factor` | Each pyramid level downscales the image by this factor. Controls histogram bin width in log-radius space. |

This is the multiplicative shrink factor applied at each pyramid level. The number of histogram bins is computed automatically:

```
n_levels = ceil( log(max_radius / (template_size/2)) / log(1/scale_factor) )
```

With defaults: `ceil(log(10) / log(1/0.9)) ≈ 21` levels.

**When to change:** Smaller `scale_factor` (e.g. 0.8) → fewer, wider bins → faster but coarser histogram. Larger `scale_factor` (e.g. 0.95) → more, narrower bins → finer size resolution but slower. Do not set `scale_factor >= 1.0`.

### Template

| Parameter | Default | CLI flag | Effect |
|-----------|---------|----------|--------|
| `template_size` | 10 px | `--template-size` | Canonical template side length in pixels. Trying both 10×10 and 5×5 is worthwhile. |
| `num_templates` | 1 | `--num-templates` | Number of distinct appearance templates (one per size bin) |

With `num_templates=1` (default), all annotated bubbles are pooled into a single average template. This is appropriate when bubble appearance does not change significantly with size.

**When to change `template_size`:** Larger templates capture more surrounding context, potentially increasing discrimination, but they also require more image to be present at each scale level. With small bubbles (< 5 px radius), keep `template_size` small (5–10). With large bubbles, consider increasing to 15–20. Trying both 5×5 and 10×10 is worth doing if the results look off.

**When to change `num_templates`:** Only if you believe small and large bubbles look qualitatively different (e.g. different internal structure). With 14 training images this is unlikely to help — you have very few large-bubble examples. Leave at 1 unless you have strong visual evidence of appearance variation.

### Calibration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_score_bins` | 50 | Number of bins in the P(score\|bubble) and P(score\|not-bubble) histograms used for non-parametric calibration. |
| `neg_sample_ratio` | 10 | How many non-bubble locations to sample per bubble during calibration |
| `min_neg_dist` | 10 px | Minimum pixel distance from any annotated bubble center before a location can be used as a negative sample |

**When to change `n_score_bins`:** With very few training images the calibration curves will be noisy. Reducing to 20–30 bins smooths the curve but loses resolution. The default 50 is reasonable.

**When to change `neg_sample_ratio`:** Higher ratio gives the calibrator more negative evidence. Leave at 10 unless the calibration curve `P(bubble|score)` looks poorly constrained.

**When to change `min_neg_dist`:** Must be at least as large as the typical bubble radius so you do not accidentally include bubble pixels as negatives. The default 10 px matches `template_size`. Increase if you have large bubbles.

---

## Understanding the output

### Per-frame histogram

```python
result = pipeline.predict(image)
# result["radius_px"]      — list of N floats, one per histogram bin
# result["expected_count"] — list of N floats, expected bubble count per bin
```

`expected_count[i]` is the sum of P(bubble|score) over all pixel locations at pyramid level `i`. It is not an integer count — it is the expected value of a sum of Bernoulli random variables (each pixel either contains a bubble center or not, with some probability). A value of 300 means: if you sampled many frames from the same distribution, you would see on average 300 bubbles of that size.

**Sanity check:** On a training image, sum `expected_count` across all bins and compare to the annotated bubble count. They will not match exactly (the calibration is approximate and scores at many locations contribute), but the order of magnitude should agree.

### Multi-frame aggregation

```python
from bubble_histogram.histogram import aggregate_histograms

results = [pipeline.predict(img) for img in frames]
total = aggregate_histograms(results)
# total["expected_count"] — summed across all frames
```

---

## Leave-one-session-out validation

To evaluate how well the pipeline generalizes, train on all sessions except one and compare predicted counts to annotations on the held-out session:

```bash
python scripts/train.py seed_v04/ pipeline_no_C1S0010.pkl --val-session C1S0010
python scripts/predict.py pipeline_no_C1S0010.pkl seed_v04/images/ZeroG_FlightDay_Test__C1S0010_*.png \
    --output val_predictions.csv
```

Compare `expected_count` summed across radius bins to the annotated bubble count in the corresponding JSON files.

Available sessions for `--val-session`: `C1S0004`, `C1S0010`, `C1S0014`, `C1S0019`, `C1S0024`.

---

## Empirical findings and current status

This section summarises what has been tried, the landmark evaluation numbers, and what we learned. The full comparison table with per-approach details is in `output/counting_methods.md`.

### Evaluation protocol

All results use **8 random seeds**, each with a fresh 30/65/5% image-level split (template / calibration / test). Seeds whose test image has fewer than 50 annotated bubbles are excluded (the C1S0010 image with 14 annotations). The primary metric is **relL1** — the L1 distance between the predicted and true per-bin histograms, normalised by the true total count:

```
relL1 = Σ_l |N_pred(l) − N_true(l)| / N_true_total
```

A value of 0 is perfect; 1.0 is as bad as predicting all zeros. A model that gets the correct total but the wrong shape will have relL1 > 0 even if MARE = 0.

### Key results

| Approach | mean relL1 | std | mean ratio | Notes |
|---|---|---|---|---|
| Dense sum (all pixels) | 6.08 | 4.50 | 7.08× | Severe overcounting |
| **LM prediction, pixel calibration** | **0.78** | **0.38** | **0.79×** | **Best model approach** |
| LM calibration + LM prior | 1.09 | 0.82 | 1.53× | Worse than pixel calibration |
| Scale-space argmax LM | 0.75 | 0.11 | 0.37× | Low variance but systematic undercount |
| 3-template LM | 0.77 | 0.34 | 0.72× | Marginal vs 1-template |
| **Constant histogram (null baseline)** | **0.65** | **0.32** | **1.01×** | **Beats all model approaches** |

### What was learned

**1. Dense summation overcounts by 5–18×.**
Each bubble produces an elevated NCC response over a ~s×s ≈ 100-pixel halo in the score map (because the template overlaps many nearby pixels). Summing P(bubble) over all pixels counts each bubble ~100× (tempered by a small prior to ~5–7× in practice). The variance is high (std 4.5×), so no fixed correction factor transfers across images.

**2. Summing over local maxima is the right fix.**
Restricting the prediction sum to spatial local maxima (`peak_local_max`, `min_distance = template_size // 2`) collapses each bubble's halo to at most one candidate point, reducing relL1 from 6.08 to 0.78. This is controlled by `predict_local_maxima = True` in `PipelineConfig`. The calibration can stay on all pixels (`local_maxima_calibration = False`); mixing LM prediction with pixel calibration is deliberate and empirically optimal.

**3. The null baseline (constant histogram) beats the model.**
Predicting the mean annotated histogram from the calibration images — ignoring the test image entirely — achieves relL1 = 0.65 vs. the best model's 0.78. This means the pipeline is not currently learning anything image-specific. It recovers roughly the correct shape (because all images come from the same experiment and have similar size distributions) but cannot adapt the total count to the individual frame.

**4. The error is not size-specific.**
A per-bin breakdown shows that the absolute prediction error is distributed proportionally across all size bins, tracking the true distribution shape. There is no particular size range that is especially miscalibrated. The per-bin ratio `|error|/true_count` is approximately constant (~0.6×) across all mid-range bins (5–20 px). The problem is a **global scale error** (wrong total count per image), not a shape error.

### Why the model cannot beat the null baseline

The Bayesian calibrator estimates P(bubble|score) from training statistics. The prior π₀ = n_bubbles / n_pixels is fixed at training time and is the same for every test image. If a test image has more or fewer bubbles than the training average, the prior is simply wrong — the NCC scores carry no information about absolute bubble density (only relative information within one image). This is a structural limitation of the current counting formula, not a tuning problem.

### What to try next

1. **Add a density normaliser.** After predicting the histogram shape, rescale the total count using an image-level density proxy (e.g. background fluorescence intensity or a fast blob count). The current model gets the shape roughly right; it just needs the right scale.

2. **CNN patch classifier.** Replace the NCC scorer with a small CNN trained with binary cross-entropy. If the CNN's discrimination is sharper than NCC, the P(bubble|score) posterior becomes more peaked, reducing the halo problem and potentially improving per-image count estimates. If even the CNN does not beat the null baseline, the bottleneck is data volume (14 images from one experiment) rather than the scorer.

3. **More annotated images from varied conditions.** The null baseline works because all 14 images come from the same experiment with a similar bubble population. If images from different experiments (different bubble densities) are included, the null baseline degrades and the model has an incentive to generalise.

---

## What if NCC is not good enough?

Based on empirical testing, the NCC pipeline with dense summation overcounts by 5–18×. Switching to local-maxima prediction (`predict_local_maxima = True`) brings the mean relL1 to 0.78, but the trivial null baseline (constant histogram from training data) achieves 0.65. This means NCC-based counting does not currently add per-image discriminative value.

If you want per-image sensitivity, the most promising next step is a small CNN patch classifier (see "What to try next" above), which would replace the NCC scoring step while keeping the image pyramid and expected-value counting unchanged.

---

## Running the tests

```bash
pytest tests/ -v
```

All 32 tests should pass. The test suite covers the data layer, template construction, NCC, calibration, pipeline integration, and histogram plotting.

---

## Module map

```
bubble_histogram/
├── config.py        PipelineConfig — all hyperparameters in one place
├── data.py          Load images, parse JSON annotations, train/val split
├── template.py      Average annotated bubble patches → unit-norm template T
├── ncc.py           Image pyramid + NCC score maps (C[x,y] = dot(W/||W||, T))
├── calibration.py   Build P(score|bubble) and P(score|not-bubble); apply Bayes
├── pipeline.py      BubblePipeline: .train(), .predict(), .save(), .load()
└── histogram.py     plot_histogram(), aggregate_histograms()

scripts/
├── train.py         CLI: fit pipeline, save .pkl
├── predict.py       CLI: run on images, write CSV
└── visualize.py     CLI: save template/calibration/histogram PNGs
```
