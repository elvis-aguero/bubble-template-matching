import numpy as np
from skimage.feature import peak_local_max

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps


def _lm_min_dist(config: PipelineConfig) -> int:
    return max(1, config.template_size // 2)


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
    rng = np.random.default_rng(seed=42)
    lm_mode = config.local_maxima_calibration
    min_d = _lm_min_dist(config)

    for image_path in paths:
        sample = dataset.load_sample(image_path)
        ncc_results = compute_ncc_maps(sample.image, templates, config)

        if not ncc_results:
            continue

        eff_radii = np.array([r for r, _ in ncc_results])

        # Pre-compute peaks once per level (only needed in lm_mode)
        level_peaks: dict[int, np.ndarray] = {}
        if lm_mode:
            for li, (_, sm) in enumerate(ncc_results):
                level_peaks[li] = peak_local_max(sm, min_distance=min_d,
                                                  exclude_border=False)

        for bubble in sample.bubbles:
            cx, cy, r = bubble.cx, bubble.cy, bubble.radius
            level_idx = int(np.argmin(np.abs(eff_radii - r)))
            eff_r, score_map = ncc_results[level_idx]

            img_scale = (config.template_size / 2) / eff_r
            sx = int(round(cx * img_scale))
            sy = int(round(cy * img_scale))
            h, w = score_map.shape

            if not (0 <= sx < w and 0 <= sy < h):
                continue

            if lm_mode:
                peaks = level_peaks[level_idx]
                if len(peaks) == 0:
                    continue
                dists = np.linalg.norm(peaks - np.array([[sy, sx]]), axis=1)
                nearest_idx = int(np.argmin(dists))
                if dists[nearest_idx] <= min_d:
                    py, px = peaks[nearest_idx]
                    pos_scores.append(float(score_map[py, px]))
            else:
                pos_scores.append(float(score_map[sy, sx]))

        # Build exclusion mask (level-0 scale)
        _, score_map_0 = ncc_results[0]
        h0, w0 = score_map_0.shape
        img_scale_0 = (config.template_size / 2) / eff_radii[0]
        excl = np.zeros((h0, w0), dtype=bool)
        for bubble in sample.bubbles:
            sx0 = int(round(bubble.cx * img_scale_0))
            sy0 = int(round(bubble.cy * img_scale_0))
            d = max(config.min_neg_dist, int(np.ceil(bubble.radius * img_scale_0)))
            excl[max(0, sy0 - d):min(h0, sy0 + d),
                 max(0, sx0 - d):min(w0, sx0 + d)] = True

        if lm_mode:
            # Negatives: local maxima outside the exclusion zone at level 0
            peaks_0 = peak_local_max(score_map_0, min_distance=min_d,
                                     exclude_border=False)
            for py, px in peaks_0:
                if not excl[py, px]:
                    neg_scores.append(float(score_map_0[py, px]))
        else:
            candidates = np.argwhere(~excl)
            n_neg = min(len(pos_scores) * config.neg_sample_ratio, len(candidates))
            if n_neg > 0:
                chosen = rng.choice(len(candidates), size=n_neg, replace=False)
                for idx in chosen:
                    y, x = candidates[idx]
                    neg_scores.append(float(score_map_0[y, x]))

    return np.array(pos_scores, dtype=np.float32), np.array(neg_scores, dtype=np.float32)


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
        self.bin_edges = np.linspace(-1.0, 1.0, self.n_bins + 1)

        p_score_given_bubble, _ = np.histogram(pos_scores, bins=self.bin_edges, density=True)
        p_score_given_not_bubble, _ = np.histogram(neg_scores, bins=self.bin_edges, density=True)

        p_not_bubble = 1.0 - prior
        numerator = p_score_given_bubble * prior
        denominator = numerator + p_score_given_not_bubble * p_not_bubble

        with np.errstate(divide="ignore", invalid="ignore"):
            self.p_bubble_given_score = np.where(
                denominator > 0, numerator / denominator, 0.0
            ).astype(np.float32)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Look up P(bubble|score) for an array of NCC scores."""
        if self.bin_edges is None or self.p_bubble_given_score is None:
            raise RuntimeError("ScoreCalibrator must be fit before calling predict.")
        bin_idxs = np.digitize(scores, self.bin_edges) - 1
        bin_idxs = np.clip(bin_idxs, 0, self.n_bins - 1)
        return self.p_bubble_given_score[bin_idxs]
