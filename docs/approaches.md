# Bubble size distribution: approaches tried

A record of every approach tried to estimate the bubble size distribution from our bubbly flow images: what the method does, why we thought it would work, and what ended the effort. Written for a reader familiar with fluid experiments but not image processing.

Two attempts are covered. The first (Jan–Mar 2026) tried to find and outline each bubble individually. The second (Mar 2026–present) tries to estimate the size histogram without per-bubble detection. Approaches 1–12 below all belong to the second.

---

## Attempt 1: Detecting individual bubbles

The obvious first step toward a bubble size distribution is: find every bubble, measure its radius, bin the results. The computer vision literature has methods that do this well in Earth-gravity air-water bubbly flows, so we started there.

### Literature

We reviewed nine papers (2016–2025).

| Paper | Method | Result |
|---|---|---|
| Fu & Liu 2016 | Classical edge geometry | Void fraction accurate to 18% gas |
| Kim & Park 2021 | Neural network (Mask R-CNN) | 98% detection in held-out tests |
| Cerqueira & Paladino 2021 | Neural net + ellipse fitting | Transfer across different fluids |
| Hessenkemper et al. 2022 | StarDist (lab-trained weights) | 91% detection |
| Cui et al. 2022 | Feature-pyramid detection network | Robust to 20% gas content |
| Yang et al. 2025 | YOLO-based detector | Fast, multi-scale |
| Maduabuchi 2024 | U-Net for boiling surfaces | Uncertainty quantification |
| Nizovtseva et al. 2024 | Classical and DL survey | General review |
| Hessenkemper et al. 2024 | 3D tracking of deformable swarms | Requires stereo setup |

Every paper reporting over 90% detection accuracy used Earth-gravity experiments. Under buoyancy, bubbles deform into oblate, cap-shaped objects with characteristic boundary contrast. Our microgravity bubbles are near-spherical (surface tension dominates) and look different. Published accuracy numbers do not transfer.

### What we ran

We fine-tuned MicroSAM (based on Meta's Segment Anything Model) on 14 annotated frames from our dataset. It takes an image and outputs a pixel outline around each bubble; radius comes from outline area via r = √(area/π).

We also ran a non-learned method: the Fast Radial Symmetry Transform (FRST) votes for image locations with circular symmetry at a range of radii, proposing candidate centers, and then SAM draws outlines around each. No training required.

Two more methods, StarDist and Mask R-CNN, were set up but not fully evaluated before the pivot.

### Why we stopped

We were measuring the wrong thing. We tracked detection accuracy (the fraction of annotated bubbles the algorithm found and correctly outlined). That is the standard metric in object detection, but it says nothing about whether the radius estimates are right. A model with 91% detection accuracy can still have a 1–3 pixel radius error on every bubble it finds. In our image pyramid, one pixel of radius error moves a bubble into the wrong size bin. The data to compute histogram accuracy was in the output files all along; we just never computed it.

The second problem is touching bubbles. Our images have up to 641 bubbles in frame with gas fractions reaching 16%. When two bubbles touch, any boundary-based method has to decide where one ends and the next begins, and they all fail by merging the pair into one larger object. Two 10-pixel bubbles in contact become one apparent 14-pixel bubble: wrong bin, wrong count. This failure is worst in our densest images, which are exactly the most data-rich. Different architectures fail at this for different geometric reasons (star-convex boundary constraints, anchor-box merging, overlapping masks), but the outcome is the same.

Third, the domain gap is real. No published model was trained on microgravity data. An oblate Earth-gravity bubble looks nothing like a near-spherical microgravity bubble at the boundary. Published numbers are not a useful prior.

The 14 annotated images carried over into Attempt 2, with annotations changed from pixel outlines to (center, radius) triples. The main lesson, that detection accuracy does not predict histogram accuracy, shaped every subsequent experiment.

---

## Problem

Given a single video frame with 300–600 overlapping microgravity bubbles, estimate the fraction of bubbles in each of 27 log-spaced size bins (radius 3–46 px). The accuracy metric is relL1: total absolute mismatch between predicted and true bin counts, divided by total bubble count. Zero is perfect; the target is ≤ 0.20.

Reference points:

| Method | relL1 |
|---|---|
| Best detection pipeline (NCC) | 0.851 |
| Oracle: LOO median of other images' true histograms | 0.657 |
| Target | 0.20 |

The oracle number matters most. It says: if you had ground truth annotations from the other 13 images and simply averaged their histograms, you'd still be off by 0.657, more than three times the target. The size distributions vary enormously across images, and that variance is not explained by session, lighting, or bubble density (§8). Getting to 0.20 requires reading something in the current image itself.

---

## Why these images are hard

Radial intensity cross-sections through 52 manually inspected bubbles found four visually distinct types:

- Dark-rim (bright center, dark ring at boundary): 54%
- Filled-dark (uniformly dark): 27%
- Bright-rim / filled-bright (no dark feature): 12%
- Flat (indistinguishable from background): 8%

Any method that relies on a specific contrast polarity or gradient direction misses 12–46% of bubbles regardless of how well it performs on the type it was designed for. The 14 annotated images also span four distinct lighting conditions (dark field, bright field, high density, low density) that do not correspond to session identity.

---

## Approaches in Attempt 2

---

### 1 · Template matching at multiple scales

We built an image pyramid with 27 levels (scale factor 0.9 per level, covering radii 3–46 px). At each level, we cross-correlated the image against an averaged bubble template. Peaks in the cross-correlation map are bubble candidates; when two size levels competed for the same location, we kept the better-matching one.

The problem is that cross-correlation scores increase monotonically as scale gets finer, regardless of whether any bubble is actually there. Finer scales have more high-frequency texture for the template to correlate against. The size-competition step therefore almost always picks the finest viable scale, often wrong by 4–6 levels. Measured eviction rate: 87.8% of correctly-sized matches are overridden by a finer-scale competitor.

We tried five fixes:

| Fix | Why it fails |
|---|---|
| Per-level score recalibration | Trains only on the biased survivors from the competition step |
| Classifier at known bubble centers | Edge artifacts from vessel walls score above real bubbles |
| Score scaling by (r/r₀)^α | Competitor score advantage is too large and heavy-tailed (median 1.53×) to overcome by scaling |
| Independent competition per level | ~1,000 spurious edge detections per level flood the results without cross-scale suppression |
| Size-specific templates | The problem is in the image texture, not the template shape; a different template cannot create a score peak where the signal is absent |

The monotone scale response is a property of cross-correlation in textured images. There is no parametric fix.

---

### 2 · Laplacian-of-Gaussian blob filter

A Laplacian-of-Gaussian (LoG) filter is theoretically the right tool for circular blobs. For an ideal filled dark circle against a bright background, the filter response peaks exactly at the circle's radius. Unlike template matching, this peak is a real property of the filter's mathematics, not an artifact of image texture.

It fails here because our bubbles are not ideal. Measuring the LoG response peak across 52 manually verified bubbles, the inter-quartile range of "which size level gave the largest response" spans 6.2 scale levels, roughly a factor of two in radius. The four appearance types respond to different filter scales: filled-dark bubbles have a strong center response, dark-rim bubbles respond at the rim, bright-rim bubbles give an inverted response. There is no single filter scale that consistently peaks at the true radius across all types.

One useful result: though LoG cannot estimate bubble size, it locates bubble centers well: 89.8% of true centers have a LoG extremum within half a bubble radius. That motivated the two-stage approach in §5.

---

### 3 · Hough circle transform

Every edge pixel votes for every circle it could lie on, across all possible radii and center locations. Peaks in the three-dimensional (x, y, radius) vote space are circle candidates. Hough is explicitly designed for arbitrary radii and uses gradient direction rather than intensity, so it is less sensitive to bubble appearance type than LoG or template matching.

With 300–600 bubbles and vessel walls all voting simultaneously, the vote space fills uniformly with noise. Individual bubble peaks become indistinguishable. Only 11% of true bubbles were detected (target: 70%), with 771–2,430 false detections per image (target: ≤5) and detected radii averaging 2.2× the true value from spurious vote clusters near vessel walls. The mechanism is the same as FRST (§6): global vote accumulation breaks down in a dense field.

---

### 4 · Radial gradient at the bubble rim

At a bubble boundary, intensity changes from inside to outside, creating an inward- or outward-pointing gradient. Measuring this gradient at pixels forming a ring at the expected bubble radius, and checking whether the vectors point consistently toward or away from the center, tests for a bubble rim without assuming any particular contrast polarity.

The signal is real. At manually annotated bubble centers and radii, the rim gradient is 6.86× stronger than at random non-bubble locations (2,349 bubbles, 14 images). 91% of bubbles show outward-dominant gradient, as expected for bright-interior bubbles. The signal holds across all four appearance types and all lighting conditions, with one exception: very large bubbles with weak inward gradient (n=13, SNR ≈ 0.5×).

The limitation is that this number was measured at oracle-known positions. When we applied the same measurement to LoG-proposed candidate positions (a few pixels off from the true center), the signal collapsed. A small position error misaligns the measurement annulus with the actual rim, and candidate positions between touching bubbles see gradient from both rims at once, making a non-bubble location score as high as a real center. The signal exists, but it requires knowing where to look.

---

### 5 · Two-stage pipeline: LoG locator + gradient classifier

Given that LoG finds positions (not sizes) and the radial gradient is informative at known positions, the natural next step is: use LoG to propose candidate (x, y) locations, then classify each by its radial gradient profile.

Two experiments tested this.

First, we checked whether LoG proposed enough candidates. It found positions within half a bubble-radius of 89.8% of true centers, which is enough. But the signal-to-noise on actual LoG-proposed positions was 1.22×, not the 6.86× from oracle positions. The reason is that 78% of false-candidate positions (between touching bubbles) scored as high as real centers. The contact zone between two bubbles looks like a rim.

Second, we required candidates to be within one-quarter bubble-radius of the true center before measuring the gradient profile. Signal-to-noise improved to 3.0×, which is marginally workable. But at this precision, only 52.7% of true bubbles had a LoG candidate close enough to be proposed at all. The 47% that are unproposable create a relL1 floor of ≈0.47 from missing detections alone, 2.4× above the target before the classifier even runs.

Tighter spatial precision makes the gradient measurement more reliable but reduces coverage below what the target requires. There is no operating point where both conditions are satisfied with this generator.

---

### 6 · Fast Radial Symmetry Transform (FRST)

FRST is a filter where each gradient pixel votes for radial symmetry centers at a specific distance in the gradient direction. It was developed for scale-selective radial symmetry detection in microscopy and has published results in bubble and cell detection at moderate densities.

In our images, the dense bubble field defeats it the same way it defeats Hough. Rim pixels from all 300–600 bubbles cross-vote at radii landing near neighboring bubble centers. At 27 radius values simultaneously, background vote density rises until there are no distinguishable peaks, just a diffuse warm response across the full image. Only 9.4% of true bubble centers were found; LOSO relL1 = 0.932. The vessel walls were correctly suppressed; the failure is from inter-bubble voting in the dense field.

---

### 7 · Background subtraction + watershed

The first 20 frames of each recording contain no bubbles. Averaging them gives a clean per-pixel background model. Subtracting it from each subsequent frame should isolate bubble signal, since the static vessel structure cancels exactly. We then applied watershed segmentation, which treats the image as a terrain, floods from local intensity minima, and assigns each basin to one object.

This approach works in sparse frames and fails in dense ones. As gas fraction increases, watershed assigns each bubble the region nearest to its center, a Voronoi tiling of the image. In a dense field, these regions shrink as more bubbles pack in. Radius estimated from region area (r = √(area/π)) therefore decreases monotonically with packing density, not because bubbles are actually smaller but because each bubble's allocated region is smaller. The correlation between image brightness and estimated median radius (r = −0.878) looks physically meaningful but is a geometric artifact of the tiling.

---

### 8 · Regime lookup

If images from the same session, or with similar brightness or bubble density, consistently have similar size distributions, then detection is unnecessary: identify the regime, return its characteristic histogram.

The within-group distributions are not meaningfully more similar than across-group ones. For session partitioning, the within-session oracle beats the global oracle in 6 of 9 eligible image pairs (chance would give 4.5). For brightness or density partitions, the best group (brightest images) achieves relL1 = 0.437 against a criterion of ≤0.35. Session identity, brightness, and bubble density collectively explain none of the between-image histogram variance.

---

### 9 · Image-level regression

Extract summary statistics from each image (overall brightness, texture roughness, edge density, frequency content, and scale-specific gradient measurements motivated by §4) and train a regression to predict the 27-bin histogram. Evaluate cross-image: train on 13, predict the 14th, rotate.

LOSO median relL1 = 0.6807, vs. oracle 0.6569. The regression is worse than the oracle. Wilcoxon signed-rank test: p = 0.59, with no evidence of any predictive signal. The dominant features were overall brightness and intensity kurtosis, both of which identify which recording session the image is from. Since §8 already showed session identity has no predictive value, those features are providing no useful information.

The scale-specific gradient features, despite the genuine 6.86× physical signal from §4, produced nothing when pooled globally. Averaging over the full image mixes the bubble rim signal with background, vessel walls, and the contact zones between touching bubbles. The 6.86× figure only holds at known bubble positions.

---

### 10 · Density map networks (rejected without running)

Density map regression, used in crowd counting and cell density estimation, predicts a spatial heat map of bubble presence for each size bin, then integrates to get per-bin counts. This bypasses bubble boundaries entirely, which looked attractive after the detection failures.

We rejected it. With 14 images spanning four photometric regimes, any network must generalize across regimes in the same way the regression in §9 must, and §9 already showed no image-level features predict the histogram. The network must also assign each bubble to the correct size channel from appearance alone, which is the scale-discrimination problem that closed template matching and LoG. The estimated probability of reaching relL1 ≤ 0.20 at n=14 is ~2–4%.

---

### 11 · Contrast-invariant boundary detection (not run)

Phase congruency (Kovesi 1999) measures whether image structure is coherent across spatial frequencies, independently of whether a boundary is bright-to-dark or dark-to-bright. It was considered as a preprocessing step to address the four-appearance-type problem.

By the time it was assessed, §5 had already shown that better boundary detection does not help: any global vote accumulator inherits the cross-bubble contamination of §3 and §6, and any patch classifier still runs into the 52.7% recall ceiling from §5. Better boundary detection does not change the geometry of touching bubbles.

---

### 12 · Consecutive-frame recurrence (rejected without running)

Consecutive frames from the same recording are correlated at r = 0.95. Ten consecutive frames are statistically equivalent to about 0.25 independent observations; annotating them costs the same as 10 independent frames but contributes almost nothing to effective training set size. Additionally, the static vessel structure is more temporally stable than the bubbles themselves, so a model with temporal context would tend to learn apparatus features rather than bubble population dynamics.

---

## What remains open

All handcrafted and regression approaches are closed. The remaining options require neural networks and more data.

| Approach | P(relL1 ≤ 0.20) | Notes |
|---|---|---|
| Exemplar-conditioned counting (FamNet / DAVE) | 5–10% | Conditions on crops from the test image itself; sidesteps cross-image generalization |
| Detection network with pretrained backbone (FPN + FCOS) | 3–6% at n=14 | Data is the bottleneck, not architecture |
| Collecting ≥20 more annotated images | Raises CNN success to ~20–30% | Binding constraint for all CNN paths |

---

## The oracle floor

The GT oracle (predict each image's histogram as the LOO median of the other 13 images' true histograms) achieves relL1 = 0.657, with access to ground truth it would not have at deployment. That is 3.3× above our target, and it quantifies how much the size distributions vary across the 14 images.

Getting below 0.20 requires a method that reads the current image's bubble population specifically. Every approach that did not do that (§§8–10) is bounded by this 0.657 floor. Every approach that tried to (§§1–7) ran into one of three problems: the scale discrimination failure common to template matching and LoG, the vote contamination in dense fields that defeats Hough and FRST, or the geometric recall ceiling that closes the patch-scorer path. The floor drops with more data: more annotated images reduce between-image variance, which is why data collection is the first item in the open paths table.
