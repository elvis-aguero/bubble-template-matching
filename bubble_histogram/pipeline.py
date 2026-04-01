import pickle
from pathlib import Path

import numpy as np

from bubble_histogram.calibration import ScoreCalibrator, sample_scores
from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps
from bubble_histogram.template import build_templates


class BubblePipeline:
    """
    End-to-end bubble size histogram pipeline.

    Usage
    -----
    pipeline = BubblePipeline(config)
    pipeline.train(dataset)          # fit templates + calibrator
    result = pipeline.predict(img)   # dict with radius_px + expected_count
    pipeline.save(path)
    pipeline = BubblePipeline.load(path)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.templates: np.ndarray | None = None
        self.calibrator: ScoreCalibrator | None = None

    def train(self, dataset: AnnotatedDataset) -> None:
        self.templates = build_templates(dataset, self.config)

        pos_scores, neg_scores = sample_scores(dataset, self.templates, self.config)

        # Estimate prior P(bubble) = total bubbles / total pixel locations
        total_bubbles = 0
        total_locs = 0
        for p in dataset.train_images:
            sample = dataset.load_sample(p)
            total_bubbles += len(sample.bubbles)
            total_locs += int(np.prod(sample.image.shape))

        prior = total_bubbles / max(total_locs, 1)

        self.calibrator = ScoreCalibrator(n_bins=self.config.n_score_bins)
        self.calibrator.fit(pos_scores, neg_scores, prior)

    def predict(self, image: np.ndarray) -> dict:
        """
        Estimate bubble size histogram for a single image.

        Returns
        -------
        dict with keys:
          "radius_px"      : list[float] — effective bubble radius per level (original image px)
          "expected_count" : list[float] — expected bubble count per level
        """
        if self.templates is None or self.calibrator is None:
            raise RuntimeError("Pipeline must be trained before calling predict.")

        ncc_results = compute_ncc_maps(image, self.templates, self.config)

        radius_px = []
        expected_counts = []

        for eff_radius, score_map in ncc_results:
            probs = self.calibrator.predict(score_map.ravel())
            expected_counts.append(float(probs.sum()))
            radius_px.append(eff_radius)

        return {"radius_px": radius_px, "expected_count": expected_counts}

    def save(self, path: Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "templates": self.templates,
                    "calibrator": self.calibrator,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BubblePipeline":
        with open(path, "rb") as f:
            data = pickle.load(f)
        pipeline = cls(data["config"])
        pipeline.templates = data["templates"]
        pipeline.calibrator = data["calibrator"]
        return pipeline
