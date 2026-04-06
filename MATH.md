# Mathematical Description of the Bubble Size Histogram Pipeline

---

## 1. Setup and Notation

Let $\mathbb{Z}_+ = \{0, 1, 2, \ldots\}$. A grayscale image is a function $I : \Omega \to [0,1]$ where $\Omega = \{0,\ldots,H-1\} \times \{0,\ldots,W-1\} \subset \mathbb{Z}^2$ is the pixel lattice. Images are stored as $H \times W$ arrays of type `float32`.

A **bubble annotation** is a triple $(c, r) \in \mathbb{R}^2 \times \mathbb{R}_{>0}$ specifying a center $c = (c_x, c_y)$ and radius $r$. The annotated training corpus is

$$\mathcal{D} = \bigl\{(I_k,\, \mathcal{B}_k)\bigr\}_{k=1}^{N}, \qquad \mathcal{B}_k = \bigl\{(c_{k,j},\, r_{k,j})\bigr\}_{j=1}^{M_k},$$

where $N = 14$ images and $\sum_k M_k = 5{,}584$ annotated bubbles (radii ranging from $0.81$ to $263.6$ px; quartiles $Q_1 = 4.2$, $Q_2 = 7.0$, $Q_3 = 10.3$ px).

**Polygon annotations.** LabelImg encodes some bubbles as polygons $\{v_i\}_{i=1}^n \subset \mathbb{R}^2$. These are converted to the $(c, r)$ form via the minimum enclosing circle heuristic:

$$c = \frac{1}{n}\sum_{i=1}^n v_i, \qquad r = \max_{i} \|v_i - c\|_2.$$

**Hyperparameters.** All hyperparameters are collected in a single configuration object:

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $s$ | `template_size` | 10 |
| $\sigma$ | `scale_factor` | 0.9 |
| $r_{\min}$, $r_{\max}$ | `min_radius`, `max_radius` | 1.0, 50.0 |
| $K$ | `num_templates` | 1 |
| $n_b$ | `n_score_bins` | 50 |
| $\eta$ | `neg_sample_ratio` | 10 |
| $d$ | `min_neg_dist` | 10 |

---

## 2. Data Split

The $N$ images are partitioned into three disjoint index sets $\mathcal{I}_T,\, \mathcal{I}_C,\, \mathcal{I}_{\mathrm{test}}$ by applying a seeded random permutation $\pi$ of $\{1,\ldots,N\}$:

$$n_{\mathrm{test}} = \max\!\bigl(1,\, \mathrm{round}(N(1 - f_T - f_C))\bigr), \quad n_T = \mathrm{round}(N f_T),$$

$$\mathcal{I}_{\mathrm{test}} = \{\pi(1),\ldots,\pi(n_{\mathrm{test}})\}, \quad \mathcal{I}_T = \{\pi(n_{\mathrm{test}}+1),\ldots,\pi(n_{\mathrm{test}}+n_T)\},$$
$$\mathcal{I}_C = \{\pi(n_{\mathrm{test}}+n_T+1),\ldots,\pi(N)\}.$$

Default fractions: $f_T = 0.30$, $f_C = 0.65$, giving $|\mathcal{I}_T| = 4$, $|\mathcal{I}_C| = 9$, $|\mathcal{I}_{\mathrm{test}}| = 1$ for $N = 14$.

Images in $\mathcal{I}_T$ are used exclusively for template construction; images in $\mathcal{I}_C$ exclusively for calibration. The sets are never mixed. Images in $\mathcal{I}_{\mathrm{test}}$ are held out entirely.

---

## 3. Template Construction

### 3.1 Patch Extraction

For each annotated bubble $(c, r)$ in image $I_k$ with $k \in \mathcal{I}_T$, let $\hat{r} = \max(1, \mathrm{round}(r))$. Extract the square patch

$$P = I_k\bigl[\lfloor c_y \rfloor - \hat{r} \;:\; \lfloor c_y \rfloor + \hat{r},\; \lfloor c_x \rfloor - \hat{r} \;:\; \lfloor c_x \rfloor + \hat{r}\bigr] \in [0,1]^{2\hat{r} \times 2\hat{r}},$$

discarding any bubble whose bounding box extends outside $\Omega$.

### 3.2 Normalisation and Resizing

Apply a bilinear resize with anti-aliasing to obtain $\tilde{P} \in \mathbb{R}^{s \times s}$. Normalise to unit $\ell^1$ mass:

$$\hat{P} = \frac{\tilde{P}}{\|\tilde{P}\|_1},$$

so that every patch contributes equally regardless of absolute brightness. Patches with $\|\tilde{P}\|_1 = 0$ are discarded.

### 3.3 Size Bins and Averaging

The radius range $[r_{\min}, r_{\max}]$ is partitioned into $K$ log-spaced bins with edges

$$e_k = r_{\min} \left(\frac{r_{\max}}{r_{\min}}\right)^{k/K}, \quad k = 0, \ldots, K,$$

and geometric-mean centres $\mu_k = \sqrt{e_k\, e_{k+1}}$. Each bubble is assigned to bin $\kappa(r) = \min\bigl(\lfloor\!\log_{r_{\max}/r_{\min}}(r/r_{\min}) \cdot K\rfloor, K-1\bigr)$.

Let $\mathcal{P}_k$ denote the set of normalised patches assigned to bin $k$. The raw template for bin $k$ is the arithmetic mean:

$$\bar{T}_k = \frac{1}{|\mathcal{P}_k|} \sum_{\hat{P} \in \mathcal{P}_k} \hat{P} \in \mathbb{R}^{s \times s}.$$

The final template is $\ell^2$-normalised:

$$T_k = \frac{\bar{T}_k}{\|\bar{T}_k\|_2} \in \mathbb{R}^{s \times s}, \quad \|T_k\|_2 = 1.$$

The $\ell^2$ normalisation is theoretically motivated by Pedro's formulation (making $T_k$ a unit vector so the NCC equals a cosine similarity), but is redundant in practice since `skimage.match_template` normalises the template internally.

With $K = 1$ (default) all patches pool into a single template $T_0$.

---

## 4. Image Pyramid

### 4.1 Levels and Effective Radii

Define the number of pyramid levels

$$L = \left\lceil \frac{\log\bigl(r_{\max} / (s/2)\bigr)}{\log(1/\sigma)} \right\rceil.$$

With defaults $(s = 10,\, \sigma = 0.9,\, r_{\max} = 50)$: $L = 22$.

At level $\ell \in \{0, \ldots, L-1\}$, the image is downscaled by

$$\alpha_\ell = \sigma^\ell,$$

yielding $I_\ell = \mathrm{rescale}(I,\, \alpha_\ell) \in [0,1]^{H_\ell \times W_\ell}$ with $H_\ell = \lfloor \alpha_\ell H \rfloor$, $W_\ell = \lfloor \alpha_\ell W \rfloor$. Level 0 is the original image ($\alpha_0 = 1$, $I_0 = I$).

The **effective radius** at level $\ell$ is the radius (in pixels of the original image $I$) of a bubble that appears with radius $s/2$ pixels in $I_\ell$:

$$\rho_\ell = \frac{s/2}{\alpha_\ell} = \frac{s}{2}\,\sigma^{-\ell}.$$

The sequence $(\rho_\ell)$ is strictly increasing: $\rho_0 = s/2 = 5$ px through $\rho_{21} \approx 45.7$ px with defaults. Each pyramid level corresponds to one bin of the output size histogram.

### 4.2 Template Assignment

The template used at level $\ell$ is

$$T^{(\ell)} = T_{k^*(\ell)}, \qquad k^*(\ell) = \operatorname*{arg\,min}_{k \in \{0,\ldots,K-1\}} |\rho_\ell - \mu_k|.$$

With $K = 1$: $T^{(\ell)} = T_0$ for all $\ell$.

---

## 5. Normalised Cross-Correlation Score Maps

For each pyramid level $\ell$, `skimage.match_template` computes the score map $C_\ell : \Omega_\ell \to [-1, 1]$ defined at each position $(x, y) \in \Omega_\ell$ by

$$C_\ell(x, y) = \frac{\langle W_{x,y},\, T^{(\ell)} \rangle}{\|W_{x,y}\|_2\, \|T^{(\ell)}\|_2},$$

where $W_{x,y} \in \mathbb{R}^{s \times s}$ is the patch of $I_\ell$ centred at $(x, y)$ (with reflected boundary padding), and $\langle\cdot,\cdot\rangle$ denotes the Frobenius inner product on $\mathbb{R}^{s \times s}$.

Since $\|T^{(\ell)}\|_2 = 1$, this simplifies to $C_\ell(x,y) = \langle W_{x,y}/\|W_{x,y}\|_2,\, T^{(\ell)} \rangle$, i.e. the cosine similarity between the vectorised normalised patch and the template. $C_\ell(x,y) \in [-1, 1]$ with $C_\ell(x,y) = 1$ iff $W_{x,y} \propto T^{(\ell)}$ (same direction in $\mathbb{R}^{s^2}$).

---

## 6. Bayesian Calibration

### 6.1 Positive Score Sampling

For each calibration image $I_k$ with $k \in \mathcal{I}_C$, and each annotated bubble $(c, r) \in \mathcal{B}_k$:

1. Select the level best matching the annotated radius: $\ell^*(c, r) = \operatorname*{arg\,min}_\ell |\rho_\ell - r|$.
2. Map the bubble centre to scaled coordinates: $\tilde{c} = \alpha_{\ell^*} \cdot c \in \mathbb{R}^2$.
3. Record the positive score: $s^+_{k,j} = C_{\ell^*}(\lfloor\tilde{c}\rfloor)$, provided $\lfloor\tilde{c}\rfloor \in \Omega_{\ell^*}$.

The positive score set is $\mathcal{S}^+ = \{s^+_{k,j}\}$.

### 6.2 Negative Score Sampling

Negatives are sampled from the level-0 score map $C_0$ (full resolution). For each calibration image $I_k$, define the exclusion set

$$\mathcal{E}_k = \bigl\{(x,y) \in \Omega_0 : \exists\, j,\; \|{(x,y)} - c_{k,j}\|_\infty \leq d\bigr\},$$

i.e. a square neighbourhood of half-side $d$ around every annotated bubble centre. Candidate negative positions are $\mathcal{N}_k = \Omega_0 \setminus \mathcal{E}_k$. A random subset of size $\min(\eta \cdot |\mathcal{S}^+|,\, |\mathcal{N}_k|)$ is drawn uniformly without replacement from $\mathcal{N}_k$, yielding scores $\mathcal{S}^-$.

### 6.3 Empirical Likelihood Estimation

Partition $[-1, 1]$ into $n_b$ equal-width bins $B_1, \ldots, B_{n_b}$ of width $h = 2/n_b$. Estimate the score densities at bubble and non-bubble locations by normalised histograms:

$$\hat{f}^+(s) = \frac{|\{s_i^+ \in B_m : s \in B_m\}|}{|\mathcal{S}^+| \cdot h}, \qquad \hat{f}^-(s) = \frac{|\{s_i^- \in B_m : s \in B_m\}|}{|\mathcal{S}^-| \cdot h}.$$

### 6.4 Prior Probability

The prior probability that a randomly chosen pixel location in a training image contains a bubble centre is estimated by

$$\pi_0 = \frac{\sum_{k \in \mathcal{I}_C} M_k}{\sum_{k \in \mathcal{I}_C} H_k W_k}.$$

### 6.5 Posterior via Bayes' Rule

By Bayes' theorem, the posterior probability that location $(x,y)$ contains a bubble given NCC score $s$ is

$$p(s) \triangleq P\!\left(\text{bubble} \mid \text{score} = s\right) = \frac{\hat{f}^+(s)\,\pi_0}{\hat{f}^+(s)\,\pi_0 + \hat{f}^-(s)\,(1-\pi_0)},$$

with $p(s) := 0$ wherever the denominator vanishes. The function $p : [-1,1] \to [0,1]$ is stored as a piecewise-constant lookup table (one value per bin).

---

## 7. Per-Frame Histogram Estimation

Given a new image $I$, the estimated expected bubble count at scale level $\ell$ is

$$\hat{N}_\ell = \sum_{(x,y)\,\in\,\Omega_\ell} p\!\left(C_\ell(x,y)\right).$$

The output histogram is the sequence $\bigl\{(\rho_\ell,\, \hat{N}_\ell)\bigr\}_{\ell=0}^{L-1}$.

**Probabilistic interpretation.** Define binary random variables $B_{x,y}^{(\ell)} \in \{0,1\}$ indicating whether pixel $(x,y)$ in $I_\ell$ is a bubble centre, with $P(B_{x,y}^{(\ell)} = 1) = p(C_\ell(x,y))$. By linearity of expectation,

$$E\!\left[\sum_{(x,y)} B_{x,y}^{(\ell)}\right] = \sum_{(x,y)} p\!\left(C_\ell(x,y)\right) = \hat{N}_\ell,$$

regardless of the joint distribution of the $B_{x,y}^{(\ell)}$. The formula is thus an unbiased estimator of the expected number of bubble-centre pixels at scale $\ell$, under the model that each pixel independently contains a bubble centre with probability $p(C_\ell(x,y))$.

---

## 8. Remarks and Limitations

### 8.1 Dense Summation Overcounts by 5–18× (empirically confirmed)

The NCC response $C_\ell(\cdot,\cdot)$ is not a delta function at bubble centres — it has spatial extent of order $s$ pixels. A single bubble with centre $(c_x, c_y)$ at scale $\ell$ produces elevated scores across a neighbourhood of radius $\approx s/2$ pixels in $I_\ell$, i.e. over an area of order $\pi(s/2)^2 \approx 78$ pixels.

Under the dense-sum formula $\hat{N}_\ell = \sum_{(x,y)} p(C_\ell(x,y))$, if $p \approx \bar{p}$ over this neighbourhood, a single bubble contributes $\approx 78\bar{p}$ to $\hat{N}_\ell$ instead of 1. **Empirically: mean ratio = 7.08×, std = 4.50×, range [3.88×, 18.47×] across 8 seeds.** The variance is too large for a fixed correction factor to be reliable.

**Fix: local-maxima summation.** Restricting the sum to spatial local maxima of $C_\ell$ with `min_distance` $= \lfloor s/2 \rfloor$ collapses each bubble's halo to at most one candidate:

$$\hat{N}_\ell^{\mathrm{LM}} = \sum_{(x,y)\,\in\,\mathcal{M}_\ell} p\!\left(C_\ell(x,y)\right), \qquad \mathcal{M}_\ell = \mathrm{peak\_local\_max}(C_\ell,\, d_{\min} = \lfloor s/2 \rfloor).$$

This reduces mean relL1 from 6.08 to 0.78 (8-seed average). The pixel-prior calibration (Section 6.4) is retained unchanged; only the prediction formula changes. (See `PipelineConfig.predict_local_maxima`.)

### 8.2 The Prior is Fixed — Per-Image Density Cannot Be Estimated

The prior $\pi_0$ is estimated once from the calibration images and held fixed at prediction time. If a test image has bubble density $\lambda_{\mathrm{test}} \neq \lambda_{\mathrm{train}}$, then:

$$E[\hat{N}_\ell] \approx \lambda_{\mathrm{test}} / \lambda_{\mathrm{train}} \cdot N_\ell^{\mathrm{true}}.$$

This is the principal failure mode in practice. **Empirically: a constant histogram (mean calibration histogram, ignoring the test image entirely) achieves relL1 = 0.65 vs the model's 0.78** — meaning the current pipeline adds no per-image information beyond the training distribution. The NCC scores carry shape information (which size bins contain bubbles) but not density information (how many bubbles total). Any improvement requires either a density normaliser applied post-hoc, or replacing $\pi_0$ with a per-image estimate.

### 8.3 Per-Bin Error is Proportional to True Distribution Shape

A per-bin decomposition of the prediction error reveals no size-specific failure: the absolute prediction error $|\hat{N}_\ell - N_\ell^{\mathrm{true}}|$ is approximately proportional to $N_\ell^{\mathrm{true}}$ with a constant ratio of $\approx 0.6$ across all size bins from 5–20 px. There is no bin range where NCC is qualitatively worse. This rules out template quality at a specific scale as the bottleneck and confirms the problem is a global scale (total count) error.

### 8.4 Non-Overlapping Assumption

The probabilistic interpretation requires that bubble centres are sufficiently separated that at most one centre occupies any pixel position in $I_\ell$. This is approximately satisfied for non-overlapping bubbles at the resolution of $I_\ell$.

### 8.5 Negative Sampling Level Mismatch

Negatives are sampled only from $C_0$ (level 0), yet positives are sampled from the best-matching level $\ell^*$ per bubble. This means $\hat{f}^-$ reflects the score distribution at the smallest scale only. For a fully consistent calibration, negatives should be stratified across all levels. The empirical consequence: switching to LM-based negative sampling (`local_maxima_calibration = True`) increases relL1 from 0.78 to 1.09, likely because level-0 LM scores do not represent the background distribution at higher pyramid levels.

### 8.6 Template Averaging Across Scales

The template $T_k$ is an average of patches resized to $s \times s$, drawn from bubbles of all radii assigned to bin $k$. Patches from small bubbles ($r < s/2$) are upsampled; patches from large bubbles are downsampled. Empirically, using 3 templates ($K=3$, each covering one third of the radius range) yields relL1 = 0.77 vs 0.78 for $K=1$ — a negligible difference. With 14 training images, the benefit of finer template bins is outweighed by fewer training patches per template.

### 8.7 Score-Based Calibration is Monotone Only in Expectation

The Bayes posterior $p(s)$ need not be monotone in $s$ — the empirical histograms $\hat{f}^+$ and $\hat{f}^-$ can cross multiple times with limited data. Isotonic regression could be applied post hoc to enforce monotonicity if desired.
