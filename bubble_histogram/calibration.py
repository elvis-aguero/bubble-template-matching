import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps


def sample_scores(
    dataset: AnnotatedDataset,
    templates: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract NCC scores at annotated bubble centers (positives) and random
    non-bubble locations (negatives) from all training images.

    Each bubble is sampled at the pyramid level whose effective radius best
    matches the annotated bubble radius.
    """
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    rng = np.random.default_rng(seed=42)

    for image_path in dataset.train_images:
        sample = dataset.load_sample(image_path)
        ncc_results = compute_ncc_maps(sample.image, templates, config)

        if not ncc_results:
            continue

        eff_radii = np.array([r for r, _ in ncc_results])

        for bubble in sample.bubbles:
            cx, cy, r = bubble.cx, bubble.cy, bubble.radius
            # Pick level whose effective radius is closest to bubble radius
            level_idx = int(np.argmin(np.abs(eff_radii - r)))
            eff_r, score_map = ncc_results[level_idx]

            # Convert bubble center to scaled coordinates
            # effective_radius = (template_size/2) / scale_factor^l
            # scale_factor^l = (template_size/2) / effective_radius
            img_scale = (config.template_size / 2) / eff_r
            sx = int(round(cx * img_scale))
            sy = int(round(cy * img_scale))

            h, w = score_map.shape
            if 0 <= sx < w and 0 <= sy < h:
                pos_scores.append(float(score_map[sy, sx]))

        # Sample negatives from the first (full-res) level
        _, score_map = ncc_results[0]
        h, w = score_map.shape

        # Build exclusion mask around all bubbles (at level-0 scale)
        img_scale_0 = (config.template_size / 2) / eff_radii[0]
        excl = np.zeros((h, w), dtype=bool)
        d = config.min_neg_dist
        for bubble in sample.bubbles:
            sx = int(round(bubble.cx * img_scale_0))
            sy = int(round(bubble.cy * img_scale_0))
            excl[max(0, sy - d):min(h, sy + d), max(0, sx - d):min(w, sx + d)] = True

        candidates = np.argwhere(~excl)
        n_neg = min(len(pos_scores) * config.neg_sample_ratio, len(candidates))
        if n_neg > 0:
            chosen = rng.choice(len(candidates), size=n_neg, replace=False)
            for idx in chosen:
                y, x = candidates[idx]
                neg_scores.append(float(score_map[y, x]))

    return np.array(pos_scores, dtype=np.float32), np.array(neg_scores, dtype=np.float32)


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
