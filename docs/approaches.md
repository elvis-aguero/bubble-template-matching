# Bubble size distribution — approaches tried

This document summarizes every major approach we tried to estimate the bubble size distribution from bubbly flow images, why each seemed promising, and what the decisive finding was. It is written for a reader with experimental fluids background but no specific computational or image-processing background.

This document covers two sequential attempts:

- **Attempt 1** (Jan–Mar 2026): Detect and outline each individual bubble, measure its radius from the outline, aggregate into a size histogram. Abandoned because the accuracy metric we used does not measure histogram quality, and because neural networks fail systematically when bubbles touch at high packing fractions.
- **Attempt 2** (Mar 2026–present): Estimate the size histogram directly, without per-bubble detection. All approaches below belong to this attempt.

---

## Attempt 1 — Detecting Individual Bubbles

### The idea

The natural first approach to getting a bubble size distribution is: find every bubble, measure its radius, histogram the radii. The computer vision literature has many methods that do this well for dense particle images, cell microscopy, and — crucially — bubbly flows in Earth-gravity air-water systems. We started here.

### Literature review

We reviewed nine papers (2016–2025) covering classical image processing, neural network detectors, and hybrid approaches, all applied to bubble or cell detection in dense images.

| Paper | Approach | Key result |
|---|---|---|
| Fu & Liu 2016 | Classical edge geometry and topology | Accurate void fraction up to 18% gas content |
| Kim & Park 2021 (BubMask) | Neural network fine-tuned to outline bubbles | 98% of bubbles correctly detected in held-out tests |
| Cerqueira & Paladino 2021 | Neural network + ellipse fitting | Demonstrated transfer across different fluid systems |
| Hessenkemper et al. 2022 | StarDist neural network (trained on lab bubble images) | 91% detection rate on their data |
| Cui et al. 2022 | Detection network with feature pyramid | Robust up to 20% gas content |
| Yang et al. 2025 | YOLO detector with tracking | Fast and multi-scale |
| Maduabuchi 2024 | U-Net segmentation for boiling | Works on strongly illuminated surfaces |
| Nizovtseva et al. 2024 | Review of classical and deep learning methods | General survey |
| Hessenkemper et al. 2024 | 3D tracking of deformable bubble swarms | Requires stereo camera setup |

**The critical problem with the literature:** Every paper with high reported accuracy used Earth-gravity experiments, where buoyancy deforms bubbles into oblate, mushroom-cap shapes with characteristic dark rims and bright interior gradients. Our microgravity bubbles are near-spherical — surface tension dominates, not buoyancy — and have qualitatively different image appearance. Their accuracy numbers cannot be assumed to apply here.

### What we tried

**MicroSAM (neural network segmentation):** We took a state-of-the-art general-purpose segmentation model (based on Meta's Segment Anything Model) and fine-tuned it on 14 annotated frames from our dataset. This model takes an image and outputs an outline around each object it identifies as a bubble. Each bubble outline gives us an area, and radius follows from area = πr².

**FRST + SAM (symmetry detection + segmentation):** We also tested a classical, non-learned approach. A symmetry-detection filter (Fast Radial Symmetry Transform) scans the image for locations with circular symmetry at different radii and proposes candidate bubble centers. We then used SAM to draw outlines around each proposed candidate. This approach requires no training — it works from first principles of circular symmetry.

**StarDist and Mask R-CNN:** Two additional well-established methods were set up but did not reach full evaluation before we decided to pivot.

### How we measured performance

We measured **detection accuracy**: what fraction of annotated bubbles did the algorithm correctly locate and outline, with sufficient boundary overlap? This is the standard metric for object detection in the computer vision community.

### Why we abandoned this approach

**1. We were measuring the wrong thing.** Detection accuracy — the fraction of bubbles correctly found — does not measure whether the radius estimates are right. A model can correctly find 91% of bubbles while estimating each radius with a 1–3 pixel error. In our image pyramid, a 1-pixel radius error puts the bubble in the wrong size bin. We realized mid-project that a detector with state-of-the-art detection accuracy can still produce a completely wrong size histogram. The data to compute histogram accuracy existed, but we never computed it during Attempt 1.

**2. Touching bubbles merge, and this problem is worst exactly where we need accuracy most.** Our images have up to 641 bubbles visible simultaneously, with gas fractions reaching 16%. When bubbles touch, any boundary-based method must decide where one ends and the next begins. Neural networks systematically fail by merging touching bubbles into a single large apparent bubble. A pair of 10-pixel bubbles touching becomes one 14-pixel bubble — wrong size bin, wrong count. This is a well-documented failure mode of all instance segmentation approaches in dense packing regimes. Critically, the problem is worst in our most densely packed images — precisely the images that contain the most bubbles and should be the most informative.

Each architecture fails at touching bubbles for a different structural reason: StarDist uses a star-shaped boundary model that cannot represent non-convex contact zones between touching circles; bounding-box detectors merge nearby bubbles before even reaching the boundary step; general-purpose segmentation models produce unstable, fluctuating outlines in near-overlap configurations.

**3. Our bubbles look different from any training data.** All published models — even those with 98% accuracy — were trained on Earth-gravity bubbles. The boundary of an oblate, buoyancy-deformed bubble looks nothing like the boundary of a near-spherical microgravity bubble. We cannot use published accuracy numbers as priors for how a model will behave on our images.

### What carried forward into Attempt 2

The same 14 annotated images were used in Attempt 2, with annotations changed from pixel outlines to (center x, center y, radius) per bubble — more compact and directly useful for the histogram. The core lesson — that detection accuracy is not a proxy for histogram accuracy — governed every subsequent evaluation criterion.

---

## Problem definition for Attempt 2

**The question:** Given a single video frame showing 300–600 overlapping microgravity bubbles, estimate the fraction of bubbles in each of 27 size bins (radii from 3 to 46 pixels, log-spaced). This is a 27-number summary called the size histogram.

**The accuracy metric:** We use relative L1 error (relL1) — the total absolute mismatch between predicted and true bin counts, divided by the total number of bubbles. A value of 0 means perfect. A value of 1 means the error is as large as the bubble count itself. **Target: ≤ 0.20.** This is strict: it means the size distribution is off by less than 20% in aggregate.

**Key reference points:**

| Method | relL1 | What it means |
|---|---|---|
| Current best detector (NCC pipeline) | 0.851 | Our best detection-based estimate, tested across images |
| "Oracle" using other images' true histograms | 0.657 | If we had GT annotations from other images and just averaged them, this is how wrong we'd be |
| Target | **0.20** | Minimum accuracy for deployment |

The oracle gap — 0.657 for a method that has access to true histograms from other images — establishes how variable our bubble size distributions are across images. Even "looking up the answer" from other annotated images gets you only to 0.657. Beating that floor requires extracting per-image information about the bubble size distribution.

---

## Why these images are hard

**Four bubble appearance types.** Detailed cross-section measurements (radial intensity profiles through 52 manually inspected bubbles) confirmed four visually distinct types:
- **Dark-rim** (bright center, dark ring at boundary): 54% of bubbles
- **Filled-dark** (uniformly dark bubble): 27%
- **Bright-rim / filled-bright** (no dark feature): 12%
- **Flat** (indistinguishable from background): 8%

Any method that relies on a specific contrast pattern — a dark ring, a bright center, a specific gradient direction — will structurally miss 12–46% of the population, regardless of how well it works on the type it was designed for.

**Four photometric regimes.** The 14 annotated images span four distinct lighting conditions — dark field, bright field, high density, low density — that are not predictable from session identity alone. Across-image histogram variance is high: even the oracle using true GT from other sessions achieves only 0.657 relL1, and partitioning by session, brightness, or bubble density does not reduce this variance.

---

## Approaches tried in Attempt 2

---

### 1 · Template matching at multiple scales

**The idea:** Take an averaged image of what a typical bubble looks like. Slide this template across the image at 27 different sizes (covering radii 3–46px). At each location and size, measure how closely the image matches the template. A strong match at size k means "there is probably a bubble of radius r_k here." When two different sizes claim the same location, keep only the best-matching one (this is called non-maximum suppression, or NMS).

**Why it seemed promising:** Template matching is the standard baseline for object finding in images. It is simple, interpretable, and well understood.

**What happened:** Template matching is **not scale-selective in our images**. The match score increases monotonically as you go to finer (smaller) scale levels — not because smaller bubbles are actually there, but because finer scales have more high-frequency texture for the template to correlate with. This is a fundamental property of the method, not a tunable parameter.

The consequence: whenever two sizes compete for the same location, the finer scale almost always wins, even when the true bubble is large. We measured that 87.8% of correctly-sized template matches are overridden in favor of wrong (too-small) candidates. We tried five distinct fixes:

- Recalibrating scores per size level (circular: it trains on the already-biased survivors)
- Training a classifier at known bubble centers (failed: edge artifacts from apparatus walls score above real bubbles)
- Multiplying scores by a size-dependent factor to boost larger scales (failed: the advantage of finer scales is too large and heavy-tailed — we measured a median 1.53× score advantage — to overcome by any fixed multiplier)
- Running each size level independently without cross-scale competition (failed: each fine level generates ~1,000 spurious detections per image from texture noise)
- Using multiple templates, one per size (ruled out: the problem is in the image texture, not the template shape; templates cannot create a score peak where none exists)

**Conclusion:** Template matching cannot discriminate bubble size in this dataset. The scale-space response is algebraically monotone.

---

### 2 · Laplacian-of-Gaussian blob filter

**The idea:** A Laplacian-of-Gaussian (LoG) filter is a mathematical operation that "lights up" — produces a strong response — at circular blobs matching a specific size. For a perfectly filled, circular, dark blob against a bright background, the filter response is maximized at the correct size. Scanning across sizes gives a scale-space response whose peak identifies the bubble radius. This is the theoretically grounded approach for circular blob detection.

**Why it seemed promising:** Unlike template matching, LoG is derived from first principles and should produce a genuine peak at the correct scale for circular objects — not a monotone slope.

**What happened:** LoG has **no reliable scale-space peak** for our bubble population. Measuring the peak response across 52 manually verified bubbles, the inter-quartile range of "which size level gave the peak response" spans 6.2 scale levels (roughly a factor of 1.9× in radius). The lower quartile peaks at the finest scale we measured — the true spread is worse.

The reason is structural: the four bubble appearance types respond to the LoG filter at different sizes. Dark-filled bubbles have a strong center response; dark-rim bubbles have a rim response; bright-rim bubbles produce an inverted response. No single filter size reliably peaks at the true bubble size for all four types. We tested four variants of the filter targeting different features (center vs. rim, filled vs. ring response). All failed — either the peak was shifted from the true size, or the contrast between "at the right size" and "at the wrong size" was too small to use reliably.

**One positive finding:** Although LoG cannot reliably estimate bubble size, it does reliably locate bubble *positions* — the spatial coordinates where bubbles are, ignoring their size. This motivated the next approach.

**Conclusion:** LoG cannot measure bubble size. It can locate bubble centers, which we tried to exploit separately (Approach 5).

---

### 3 · Hough circle transform

**The idea:** Every edge pixel (a location where brightness changes sharply) votes for all circles it could lie on, at all possible radii and center positions. Across the full image, we accumulate votes in a three-dimensional space (x, y, radius). Peaks in this vote space are candidate circles — locations and sizes where many edge pixels "agree" that a circle exists.

**Why it seemed promising:** Hough is explicitly designed to detect circles at arbitrary radii. Unlike LoG, it uses gradient direction rather than overall brightness, making it less sensitive to whether a bubble is dark or bright inside. It appeared to address both the scale-discrimination failure of template matching and the appearance-type failure of LoG.

**What happened:** With 300–600 bubbles and the vessel walls all voting simultaneously, the vote space is overwhelmed with noise. The background vote density rises until individual bubble peaks are indistinguishable from the noise floor. We found: only 11% of true bubbles were detected within the allowed radius tolerance (against a 70% target); false detections ran at 771–2,430 per image (against a target of ≤5); and detected radii averaged 2.2× the true radius because spurious vote clusters from wall structure dominated.

This is the same fundamental failure as FRST (Approach 6) — dense fields create a cross-talk problem for any global vote accumulator.

**Conclusion:** Full-image Hough transform fails in dense bubbly fields. A localized version (voting only within a small patch around a candidate center) was not tested.

---

### 4 · Radial gradient at the bubble rim

**The idea:** At a bubble boundary, brightness changes from inside (bright or dark depending on type) to outside. This creates an inward- or outward-pointing gradient (brightness gradient vector). If we measure the gradient at pixels forming a ring at the expected bubble radius, and those vectors all point consistently toward or away from the bubble center, that is strong evidence of a bubble rim. This approach measures a specific physical signal — the ring of consistent gradient at the bubble boundary.

**Why it seemed promising:** After three methods failed, we asked a more basic question: *is there any measurable signal at bubble boundaries at all?* If there is, and it is strong enough, a different detection strategy might exploit it. This experiment was purely diagnostic — to characterize whether the physical signal exists, independent of detection architecture.

**What was found:** The signal is real and strong — but only at *known positions*. When measured at manually annotated bubble centers and radii, the inward-gradient signal is 6.86 times stronger at true bubble rims than at randomly chosen non-bubble locations. The signal holds across all size ranges, all four bubble appearance types, and all photometric regimes (with one small exception: very large bubbles with weak inward gradient, n=13, where signal-to-noise is ~0.5×).

**The critical limitation:** This 6.86× signal-to-noise ratio was measured assuming we already know exactly where the bubble center is and what its radius is. In practice, we do not know either. When we applied this measurement to candidate positions proposed by the LoG locator — which are approximately right but not exact — the signal collapsed. Two mechanisms:
1. A few pixels of offset between the proposed center and the true center misaligns the measurement annulus with the actual rim.
2. Between two touching bubbles, candidate positions from the LoG locator land in the contact zone, where the gradient from *both* rims contributes — making a non-bubble location score as high as a real bubble center.

**Conclusion:** The physical signal is there, but it requires knowing where to look. It cannot serve as a standalone detector.

---

### 5 · Two-stage pipeline: locate then classify

**The idea:** Split the problem in two. First, use LoG to propose candidate bubble locations (ignoring the size estimate, since LoG gets positions right even when sizes are wrong). Second, for each candidate location, extract a detailed gradient profile (measuring the gradient at 10 concentric rings from 0 to 1.5× the candidate radius) and use this profile to classify: is this a bubble or not? If the classification is good enough, estimate the radius from the ring where the gradient signal peaks.

**Why it seemed promising:** All previous failures (template matching, LoG, Hough, FRST) were *global* methods — they accumulate information across the full image, so bubbles contaminate each other's votes. Scoring each candidate in its own small patch evaluates it in isolation, where neighboring bubbles are outside the measurement window.

**What happened (two experiments):**

*First test — Can LoG propose enough candidate positions?* We tested LoG on the three most densely packed images. LoG proposed candidate positions that fell within half a bubble-radius of 89.8% of true bubble centers. That is good enough — missing 10% of bubbles is acceptable if the radius estimates are accurate. But when we measured the gradient signal at these LoG-proposed positions (rather than at oracle-known positions), the signal-to-noise dropped from 6.86× to 1.22× — far too low to distinguish real bubbles from false candidates. The reason: 78% of false-candidate positions (between touching bubbles) scored as high as real bubble positions. Touching-bubble contact zones look like bubble rims.

*Second test — Can tighter spatial precision fix this?* We tried requiring candidate positions to be within one-quarter bubble-radius of the true center before measuring the gradient profile. The signal-to-noise improved to 3.0×, which is marginally usable. But at this tighter tolerance, only 52.7% of true bubbles have a LoG candidate close enough — we cannot even propose 47% of real bubbles. Those missing bubbles create a size-histogram error of ≈0.47 regardless of how well the classifier works on the ones we can find. We are 2.4× above our target of 0.20 from missing detections alone.

**The fundamental tension:** Tighter spatial precision makes the gradient measurement more reliable — but also makes it impossible to propose enough candidate positions. These two requirements cannot both be satisfied with LoG as the locator.

**Conclusion:** The detection-then-classify path is formally closed. No locator we tested can simultaneously provide sufficient spatial precision and sufficient recall.

---

### 6 · Symmetry-detection filter (FRST)

**The idea:** The Fast Radial Symmetry Transform is a filter where each gradient pixel votes for radial symmetry centers at a specific distance in the gradient direction. Accumulate votes across all pixels and all radii. Peaks indicate locations of radial symmetry — i.e., bubble-like circular structures.

**Why it seemed promising:** Unlike template matching, FRST is specifically designed for scale-selective radial symmetry, which is precisely what a spherical bubble boundary produces. It was originally developed for microscopy cell detection (a similar problem) and published results are strong in low-to-moderate density fields.

**What happened:** FRST fails in our dense fields by the same mechanism as Hough (Approach 3). Rim pixels from each of the 300–600 bubbles simultaneously vote at radii that land near neighboring bubble centers. At 27 different radius values applied simultaneously, the background vote density rises uniformly until there are no distinguishable peaks — just a diffuse warm response across the entire image. Only 9.4% of true bubble centers were recovered. The apparatus walls (vessel edges) were not the problem — these were correctly suppressed — but the dense bubble field itself creates overwhelming mutual contamination.

**Conclusion:** FRST fails for the same structural reason as Hough: any method that accumulates global votes across a dense field is defeated by inter-bubble cross-talk.

---

### 7 · Background subtraction + watershed segmentation

**The idea:** Early frames in each recording contain no bubbles (only the clean fluid). Average these early frames to get a background model. Subtract the background from each subsequent frame to isolate the bubble signal. Apply watershed segmentation — a method that treats the image as a landscape, flooding from local minima to find individual basins, each corresponding to one bubble.

**Why it seemed promising:** Subtracting the static background eliminates the vessel walls and apparatus structure entirely — no votes, no artifacts, no contamination from static features. This is a clean physical separation of the bubble signal. And we have access to clean background frames.

**What happened:** This approach works on sparse frames but fails systematically in dense frames — exactly where the measurement matters most. As more bubbles pack in and gas fraction increases, watershed divides the image based on local intensity minima. In a densely packed field, the boundaries between bubble regions are determined by inter-bubble proximity, not true bubble boundaries. Each bubble gets allocated an image region roughly equal to its Voronoi cell — the "territory" closest to it. As density increases, Voronoi cells shrink, making each bubble look smaller. Measured: the apparent median radius increases monotonically with image brightness in a specific sequence, but this is a tiling artifact, not a physical trend. A bubble that is 10px in a sparse frame appears to shrink as more bubbles appear around it.

Additionally, using consecutive video frames for more information was considered but rejected: consecutive frames from the same video are extremely correlated (r=0.95 inter-frame). Using 10 consecutive frames is equivalent, statistically, to having 0.25 independent observations — the effective information gain is negligible relative to annotation cost.

**Conclusion:** Background subtraction correctly isolates bubble signal, but watershed segmentation fails in dense packing because it measures Voronoi territory, not bubble area.

---

### 8 · Looking up the histogram from similar images

**The idea:** Rather than detecting bubbles at all, exploit the fact that different images might have similar size distributions. Partition our 14 annotated images into groups by recording session, by image brightness, or by bubble density. Within each group, use the average true histogram as the prediction for any new image from the same group. If within-group images are more similar to each other than to the overall average, this approach beats the naive baseline.

**Why it seemed promising:** If session identity or lighting conditions determine bubble size distribution — as might be expected if bubble injection rate or camera settings control the regime — then knowing the regime gives you the distribution for free, without any detection.

**What happened:** The within-group histograms are not significantly more similar than across-group histograms. For session partitioning, the within-session oracle beats the cross-image oracle in only 6 out of 9 eligible image pairs — the wrong direction by chance would give 4.5, so this is not significant. For brightness or density partitioning, the best group (bright images) achieves relL1 = 0.437, against our target criterion of 0.35 — a clear failure. Across-image histogram heterogeneity is not explained by any regime variable we could measure.

**Conclusion:** The bubble size distribution is not predictable from session identity, lighting, or bubble density. Each image has a distinct histogram that must be estimated from its own content.

---

### 9 · Statistical regression from image-level features

**The idea:** Extract a set of summary statistics from each image — overall brightness, texture roughness, edge density, frequency content, and a set of scale-specific gradient measurements connected to the radial gradient signal from Approach 4. Feed these features into a standard regression model trained to predict the 27-bin histogram. Evaluate across images: train on 13 images, test on the 14th, and rotate.

**Why it seemed promising:** This is the last handcrafted approach. Even if we cannot detect individual bubbles, perhaps the overall image appearance — how bright, how textured, how edge-rich — carries enough information about the size distribution to predict it. The scale-specific gradient features were directly motivated by the strong (6.86×) signal-to-noise found in Approach 4.

**What happened:** The regression performs *worse* than the oracle (predicted relL1 = 0.68 vs. oracle relL1 = 0.66). Statistical tests find no evidence of any predictive signal (p = 0.59). The dominant features driving predictions are overall image brightness and the kurtosis of the intensity distribution — both of which identify which recording session the image came from. But we already know (from Approach 8) that session identity does not predict the histogram. The features are functioning as session fingerprints, not as bubble-size measurements.

The scale-specific gradient features — despite the strong signal at known positions — evaporated when pooled globally. Global averaging mixes bubble rim signal with background, apparatus walls, lighting gradients, and inter-bubble cancellations. The 6.86× signal-to-noise requires knowing where the bubble is. This experiment confirms that finding.

**Conclusion:** No image-level statistic — including gradient features motivated by genuine physical signal — predicts the size histogram. The regression achieves no better than the oracle at n=14.

---

### 10 · Density map regression (considered and rejected without running)

**The idea:** Instead of predicting the histogram directly, predict a "heat map" for each size bin — a map over the image where the intensity at each pixel represents the probability density of a bubble of that size being centered there. Summing up the heat map gives the count for that bin. This is a neural network approach used successfully in crowd counting and cell density estimation.

**Why it was considered:** Density map methods bypass per-bubble detection entirely — they do not require finding individual bubble boundaries, so the touching-bubble problem disappears. They use the same dot annotations we have (bubble center + radius) as training supervision.

**Why we rejected it without running:** The core constraint is the number of training images (14), not the number of individual bubble annotations (5,000). A density map network must learn to generalize across the four photometric regimes and across the high between-image variance in histograms. That generalization problem is identical to what closed all regression approaches above — the network must explain why image A has mostly small bubbles and image B has mostly large bubbles, from their raw pixel content. Nothing in our experiments suggests image content carries that information at the image level. Additionally, density maps for 27 simultaneous size bins multiply the challenge: the network must also assign each bubble to the correct size channel from appearance alone, which is exactly the scale-discrimination problem that closed template matching and LoG. The expected probability of reaching our target is ~2–4%.

**Conclusion:** Rejected. The data constraint and scale-discrimination challenge make this not worth implementing.

---

### 11 · Contrast-invariant boundary detection (considered, not run)

**The idea:** Most boundary detectors — including LoG and template matching — are sensitive to whether the bubble interior is brighter or darker than the surroundings. Phase congruency (Kovesi 1999) measures coherence of image structure across spatial frequencies, and is contrast-polarity agnostic: it responds to boundaries whether they are bright-to-dark or dark-to-bright. It could address the four-appearance-type problem.

**Why we didn't run it:** By the time this was considered, the detection path was already formally closed by Approach 5. Even with a contrast-invariant boundary detector, any full-image voting accumulator shares the inter-bubble cross-talk failure of Hough and FRST. And any patch-based classifier using this feature still runs into the recall ceiling (52.7%) identified in Approach 5. The boundary quality does not change the geometry of touching bubbles.

**Conclusion:** Not run. Low value given detection-path closure.

---

### 12 · Temporal recurrence across consecutive frames (rejected without running)

**The idea:** Train a neural network using multiple consecutive video frames as input, exploiting the fact that bubble positions change slowly between frames. This temporal context might allow the network to estimate bubble sizes more reliably than from a single frame.

**Why we rejected it:** Consecutive frames from the same recording are extremely correlated — measuring the frame-to-frame correlation gives r = 0.95. Using 10 consecutive frames provides the statistical equivalent of approximately 0.25 independent observations. Annotating 10 consecutive frames costs the same as annotating 10 independent frames, but contributes almost nothing to the effective training set size. More subtly: the apparatus structure (vessel walls, connectors) is more temporally stable than the bubbles — a temporal network would disproportionately learn to predict apparatus features, not bubble population dynamics.

**Conclusion:** Rejected. The statistical return on annotation investment is negligible.

---

## What remains open

All handcrafted detection approaches are closed. The remaining options require neural networks and more data.

| Approach | Estimated P(reaching target) | Status |
|---|---|---|
| **Exemplar-conditioned counting** (FamNet / DAVE) | 5–10% | Not tried. These models estimate bubble counts by comparing the image to a few labeled example crops from *the same test image* — they bypass the cross-image generalization problem by construction. Best remaining fit for our 4-regime heterogeneity. |
| **Detection network with pretrained backbone** (FPN + FCOS) | 3–6% at n=14 | Not tried. Data is the bottleneck, not architecture. |
| **Collecting more data (≥ 20 additional annotated images)** | Raises CNN success rate to ~20–30% | The binding constraint for all CNN paths. |

---

## Why the oracle floor (relL1 = 0.657) is the central obstacle

The oracle — predict each image's histogram as the average of the true histograms from the other 13 images — achieves relL1 = 0.657. This already has access to ground-truth annotations from other images, which are not available at deployment. Yet it still fails at a level 3.3× above our target.

This number quantifies how variable the bubble size distributions are across our 14 images. Reaching relL1 ≤ 0.20 requires a method that extracts per-image information specific to that image's bubble population — not a regression, not a regime lookup, not a fixed prior. Every approach tested that did not read the current image's individual bubble structure (regression, regime lookup, density maps) is bounded by this 0.657 floor. Every approach that did attempt to read per-image bubble structure (template matching, LoG, Hough, FRST, patch scorer) was falsified by one of three failure modes: scale discrimination failure, vote-accumulation cross-talk in dense fields, or a geometric ceiling on candidate recall.

The floor drops if we collect more annotated images — more data reduces between-image histogram variance and makes regression approaches more viable. At n=34 images, the oracle floor falls enough that CNN-based approaches have a realistic (~20–30%) chance of reaching the target.
