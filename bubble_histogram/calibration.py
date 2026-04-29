import numpy as np
from skimage.feature import peak_local_max

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps


def _lm_min_dist(config: PipelineConfig) -> int:
    # Small fixed minimum to suppress sub-pixel noise while keeping tightly-packed
    # bubbles as separate peaks. The NCC response is smooth enough that a single
    # bubble produces a single maximum without a large exclusion radius.
    return 2


def sample_scores(
    dataset: AnnotatedDataset,
    templates: np.ndarray,
    config: PipelineConfig,
    image_paths: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract NCC scores at annotated bubble centers (positives) and
    non-bubble locations (negatives).

    When config.local_maxima_calibration is False (default):
      - Positives: score at the exact bubble-centre pixel at the matching level
      - Negatives: random non-bubble pixels at level 0

    When config.local_maxima_calibration is True:
      - Positives: score at the local maximum nearest to the bubble centre
        (within template_size/2 px) at the matching level
      - Negatives: all local maxima at level 0 that are outside the
        per-bubble exclusion zone
    """
    paths = image_paths if image_paths is not None else dataset.train_images
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    rng = np.random.default_rng(seed=42)    # fixed seed so calibration is reproducible across runs
    lm_mode = config.local_maxima_calibration
    min_d = _lm_min_dist(config)

    for image_path in paths:
        sample = dataset.load_sample(image_path)
        ncc_results = compute_ncc_maps(sample.image, templates, config)

        if not ncc_results:
            continue

        eff_radii = np.array([r for r, _ in ncc_results])  # effective radius at each pyramid level

        # pre-compute local maxima once per level (only needed in lm_mode, saves recomputing per bubble)
        level_peaks: dict[int, np.ndarray] = {}
        if lm_mode:
            for li, (_, sm) in enumerate(ncc_results):
                level_peaks[li] = peak_local_max(sm, min_distance=min_d,
                                                  exclude_border=False)

        for bubble in sample.bubbles:
            cx, cy, r = bubble.cx, bubble.cy, bubble.radius
            # find the pyramid level whose effective_radius is closest to this bubble's actual radius
            level_idx = int(np.argmin(np.abs(eff_radii - r)))
            eff_r, score_map = ncc_results[level_idx]

            # img_scale maps original image coordinates to score map coordinates at this level
            # canonical_radius / eff_r = scale factor applied to reach this pyramid level
            img_scale = (config.template_size / (2 * config.template_context_factor)) / eff_r
            sx = int(round(cx * img_scale))  # bubble centre x in score map coordinates
            sy = int(round(cy * img_scale))  # bubble centre y in score map coordinates
            h, w = score_map.shape

            if not (0 <= sx < w and 0 <= sy < h):
                continue    # bubble centre maps outside the score map; skip

            if lm_mode:
                # use the nearest local maximum to the annotated centre as the positive score
                peaks = level_peaks[level_idx]
                if len(peaks) == 0:
                    continue
                dists = np.linalg.norm(peaks - np.array([[sy, sx]]), axis=1)
                nearest_idx = int(np.argmin(dists))
                if dists[nearest_idx] <= min_d:
                    py, px = peaks[nearest_idx]
                    pos_scores.append(float(score_map[py, px]))
            else:
                # use the score exactly at the annotated centre pixel
                pos_scores.append(float(score_map[sy, sx]))

        # build exclusion mask at level 0: pixels within one bubble radius of any annotation
        # are marked so they are not used as negative samples (they may contain bubble signal)
        _, score_map_0 = ncc_results[0]
        h0, w0 = score_map_0.shape
        img_scale_0 = (config.template_size / (2 * config.template_context_factor)) / eff_radii[0]
        excl = np.zeros((h0, w0), dtype=bool)
        for bubble in sample.bubbles:
            sx0 = int(round(bubble.cx * img_scale_0))
            sy0 = int(round(bubble.cy * img_scale_0))
            d = max(config.min_neg_dist, int(np.ceil(bubble.radius * img_scale_0)))    # exclusion radius = at least min_neg_dist
            excl[max(0, sy0 - d):min(h0, sy0 + d),
                 max(0, sx0 - d):min(w0, sx0 + d)] = True

        if lm_mode:
            # negatives: local maxima at level 0 that fall entirely outside all bubble zones
            peaks_0 = peak_local_max(score_map_0, min_distance=min_d,
                                     exclude_border=False)
            for py, px in peaks_0:
                if not excl[py, px]:
                    neg_scores.append(float(score_map_0[py, px]))
        else:
            # negatives: randomly sampled pixels outside the exclusion mask
            candidates = np.argwhere(~excl)
            n_neg = min(len(pos_scores) * config.neg_sample_ratio, len(candidates))  # cap at neg_sample_ratio × n_positives
            if n_neg > 0:
                chosen = rng.choice(len(candidates), size=n_neg, replace=False)
                for idx in chosen:
                    y, x = candidates[idx]
                    neg_scores.append(float(score_map_0[y, x]))

    return np.array(pos_scores, dtype=np.float32), np.array(neg_scores, dtype=np.float32)


def nms_3d(
    ncc_results: list[tuple[float, np.ndarray]],
    config: PipelineConfig,
) -> list[tuple[float, int, float, float]]:
    """
    3D NMS across spatial locations and scale levels.

    Collects 2D local-maxima peaks at each pyramid level, maps them to original-image
    coordinates, then greedily suppresses any peak that has a higher-scoring neighbour
    within ``template_size / 2`` pixels AND within 1 adjacent scale level.

    Returns
    -------
    List of (score, level, y_orig, x_orig) sorted by score descending.
    """
    min_d = _lm_min_dist(config)
    D_orig = config.template_size / 2  # suppression radius in original-image pixels

    # Coordinate identity: alpha_l = template_size / (2 * eff_radius)
    # so (y_l, x_l) → (y_l / alpha_l, x_l / alpha_l) = (y_l * 2*eff_r / ts, ...)
    candidates: list[tuple[float, int, float, float]] = []
    for level, (eff_radius, score_map) in enumerate(ncc_results):
        alpha = config.template_size / (2.0 * config.template_context_factor * eff_radius)
        peaks = peak_local_max(score_map, min_distance=min_d, exclude_border=False)
        for y_l, x_l in peaks:
            candidates.append((
                float(score_map[y_l, x_l]),
                level,
                y_l / alpha,    # map score map coordinates back to original image coordinates
                x_l / alpha,
            ))

    candidates.sort(key=lambda c: c[0], reverse=True)  # process highest-scoring peaks first
    suppressed = [False] * len(candidates)
    kept: list[tuple[float, int, float, float]] = []

    for i, (si, li, yi, xi) in enumerate(candidates):
        if suppressed[i]:
            continue
        kept.append(candidates[i])
        # suppress any lower-scoring peak within D_orig pixels and at an adjacent scale level
        for j in range(i + 1, len(candidates)):
            if suppressed[j]:
                continue
            _, lj, yj, xj = candidates[j]
            if abs(li - lj) <= 1 and abs(yi - yj) < D_orig and abs(xi - xj) < D_orig:
                suppressed[j] = True

    return kept


def count_local_maxima(
    ncc_results: list[tuple[float, np.ndarray]],
    config: PipelineConfig,
) -> int:
    """Count total spatial local maxima across all pyramid levels."""
    min_d = _lm_min_dist(config)
    return sum(
        len(peak_local_max(sm, min_distance=min_d, exclude_border=False))
        for _, sm in ncc_results
    )


class ScoreCalibrator:
    """Maps NCC scores → P(bubble|score) via Bayesian non-parametric calibration."""

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins
        self.bin_edges: np.ndarray | None = None
        self.p_bubble_given_score: np.ndarray | None = None

    def fit(self, pos_scores: np.ndarray, neg_scores: np.ndarray, prior: float) -> None:
        """
        Build calibration table from empirical score distributions.

        Parameters
        ----------
        pos_scores : NCC scores at annotated bubble locations
        neg_scores : NCC scores at sampled non-bubble locations
        prior : P(bubble) — fraction of locations containing a bubble
        """
        self.bin_edges = np.linspace(-1.0, 1.0, self.n_bins + 1)  # NCC scores are bounded in [-1, 1]

        # estimate P(score | bubble) and P(score | not-bubble) as density histograms
        p_score_given_bubble, _ = np.histogram(pos_scores, bins=self.bin_edges, density=True)
        p_score_given_not_bubble, _ = np.histogram(neg_scores, bins=self.bin_edges, density=True)

        p_not_bubble = 1.0 - prior
        numerator = p_score_given_bubble * prior    # Bayes numerator: P(score|bubble) * P(bubble)
        denominator = numerator + p_score_given_not_bubble * p_not_bubble   # total probability of this score

        with np.errstate(divide="ignore", invalid="ignore"):
            # where denominator is zero (no training data in that bin), P(bubble|score) = 0
            self.p_bubble_given_score = np.where(
                denominator > 0, numerator / denominator, 0.0
            ).astype(np.float32)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Look up P(bubble|score) for an array of NCC scores."""
        if self.bin_edges is None or self.p_bubble_given_score is None:
            raise RuntimeError("ScoreCalibrator must be fit before calling predict.")
        bin_idxs = np.digitize(scores, self.bin_edges) - 1         # find which calibration bin each score falls into
        bin_idxs = np.clip(bin_idxs, 0, self.n_bins - 1)           # clamp boundary scores to the last valid bin
        return self.p_bubble_given_score[bin_idxs]                  # table lookup: score → P(bubble)
