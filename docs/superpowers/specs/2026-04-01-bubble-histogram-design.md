# Bubble Size Histogram Pipeline — Design Spec

**Date:** 2026-04-01  
**Domain:** Zero-gravity flight experiment image analysis  
**Goal:** Per-frame histogram of bubble sizes for unannotated video frames

---

## Problem Statement

Videos from zero-gravity flight experiments contain hundreds of frames with dense bubble populations (100–500 bubbles per 1024×1024 grayscale frame). 14 frames have been annotated by humans using LabelImg (circle and polygon shapes). The pipeline must use these annotations to train a model that estimates a bubble size histogram for every unannotated video frame.

**Bubble size distribution (from training annotations):**
- Q1: 2.7px radius, Q2: 4.4px, Q3: 7.3px, max: 31px
- 87% of bubbles have radius < 10px
- Bubbles are dark features on a lighter background

---

## Architecture: Multi-Template NCC + Bayesian Calibration

### Core Design Decision

Approach B with a configurable `num_templates` parameter:
- **`num_templates=1` (default):** Exactly equivalent to Approach A (image pyramid + single template). One average template pooled from all annotated bubbles, applied identically at every pyramid level.
- **`num_templates=k > 1`:** Bubbles partitioned into k log-spaced size bins; each bin gets its own appearance template applied only at pyramid levels whose effective radius falls within that bin's range.

The histogram dimension always comes from the image pyramid (scale levels). `num_templates` only controls how many distinct appearance templates are maintained.

**Alternative (documented, not implemented):** Replace both NCC scorer and Bayesian calibrator with an end-to-end CNN patch classifier trained with binary cross-entropy. Revisit if NCC calibration proves insufficiently expressive.

---

## Pipeline Stages

### Stage 1: Data Layer (`bubble_histogram/data.py`)
- Parse JSON annotations: circles → (center, radius) directly; polygons → minimum enclosing circle (centroid + max vertex distance from centroid)
- Load images: handle 8-bit and 16-bit PNG, normalize to float [0, 1]
- Train/val split: leave-one-session-out by session ID from filename (C1S0004, C1S0010, C1S0014, C1S0019, C1S0024)
- Expose: `AnnotatedDataset(root_dir, split)`

### Stage 2: Template Construction (`bubble_histogram/template.py`)
- Partition annotated bubbles into `num_templates` size bins (log-spaced by radius)
- For each bin:
  1. Extract `2r × 2r` patch centered on each annotated bubble
  2. Resize to `template_size × template_size` (bilinear interpolation)
  3. Normalize each patch to sum=1
  4. Average across all patches in bin → raw template T
  5. L2-normalize: `T = T / ||T||`
- With `num_templates=1`: all bubbles pooled into one bin

### Stage 3: Image Pyramid (`bubble_histogram/ncc.py`)
- Number of levels: `ceil(log(max_radius / (template_size/2)) / log(1/scale_factor))`
- At level `l`: effective bubble radius = `(template_size/2) / scale_factor^l` in original image coordinates
- Each level is a downscaled copy; compute is dominated by first few levels (images shrink)
- Template assignment per level: bin whose center radius is closest to effective radius at that level

### Stage 4: NCC Computation (`bubble_histogram/ncc.py`)
- `skimage.feature.match_template` per pyramid level (handles boundary padding, FFT-based)
- Score map C[x,y] ∈ [-1, 1]: normalized dot product between local patch and template
- Output: list of score maps, one per level

### Stage 5: Calibration (`bubble_histogram/calibration.py`)
Built once from training split:
- **Positive samples:** NCC score at each annotated bubble center, at the pyramid level whose effective radius best matches the annotated radius
- **Negative samples:** Random (x,y) with minimum distance `min_neg_dist` from any annotated center; `neg_sample_ratio` × as many negatives as positives
- Build empirical histograms of P(score|bubble) and P(score|not-bubble) with `n_score_bins` bins
- Bayesian inversion: `P(bubble|score) = P(score|bubble) * P(bubble) / P(score)`
  - Prior P(bubble) = total annotated bubbles / total valid locations across training images
- Store as lookup table (score → probability)

### Stage 6: Per-Frame Counting (`bubble_histogram/pipeline.py`)
For each frame:
1. Build image pyramid
2. Compute NCC score map at each level
3. Look up P(bubble|score) for every location
4. Expected count at level `l` = Σ P(bubble|score_l[x,y]) over all (x,y)
5. Map level index → effective bubble radius in original image pixels

Output: `{"radius_px": [...], "expected_count": [...]}` — one entry per pyramid level

### Stage 7: Output & Visualization (`bubble_histogram/histogram.py`)
- Per-frame histogram: `(radius_bin_center, expected_count)` pairs
- Aggregate: sum counts across frames
- Visualization: log-scale radius axis, linear count axis
- Optional temporal smoothing: causal exponentially-weighted moving average on count vector. Apply only after empirically verifying autocorrelation of per-frame outputs.

---

## Hyperparameters (`bubble_histogram/config.py`)

All hyperparameters live in a single `PipelineConfig` dataclass.

| Parameter | Default | Notes |
|---|---|---|
| `num_templates` | 1 | Appearance templates / size bins |
| `template_size` | 10 | Canonical template width/height (px) |
| `scale_factor` | 0.9 | Pyramid downscale factor per level |
| `min_radius` | 1.0 | Smallest detectable bubble radius (px) |
| `max_radius` | 50.0 | Largest detectable bubble radius (px) |
| `n_score_bins` | 50 | Bins for calibration histograms |
| `neg_sample_ratio` | 10 | Negatives per positive for calibration |
| `min_neg_dist` | 10 | Min distance from bubble for negative sample (px) |

---

## Project Structure

```
template-matching/
├── seed_v04/                          (existing annotated data — do not modify)
├── bubble_histogram/
│   ├── __init__.py
│   ├── config.py                      (PipelineConfig dataclass)
│   ├── data.py                        (annotation parsing, dataset, splits)
│   ├── template.py                    (template construction per size bin)
│   ├── ncc.py                         (image pyramid + NCC score maps)
│   ├── calibration.py                 (score → probability via Bayes)
│   ├── pipeline.py                    (train + predict entry points)
│   └── histogram.py                   (output, aggregation, visualization)
└── scripts/
    ├── train.py                       (fit on annotated data, save artifacts)
    ├── predict.py                     (run on video frames, output CSV)
    └── visualize.py                   (templates, calibration curves, histograms)
```

---

## Verification

1. **Template sanity check:** Visualize T — should show a dark spot on a light background (bubbles are dark features)
2. **Calibration separability:** P(score|bubble) and P(score|not-bubble) histograms should be separable; P(bubble|score) should increase monotonically with score
3. **Score map alignment:** Overlay NCC score map on a training image at a given scale level — peaks should align with annotated bubble centers
4. **Count validation:** Run on held-out annotated images; compare Σ expected_count across levels to actual annotated bubble count
5. **Temporal smoothing decision:** Compute autocorrelation of per-frame counts before applying smoothing
