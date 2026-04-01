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
        self.templates = build_templates(dataset, self.config,
                                         image_paths=dataset.template_images)

        pos_scores, neg_scores = sample_scores(dataset, self.templates, self.config,
                                               image_paths=dataset.calibration_images)

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

    def predict(self, image: np.ndarray) -> dict[str, list[float]]:
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

    def save(
        self,
        path: Path,
        ncc_images: list[np.ndarray] | None = None,
        ncc_names: list[str] | None = None,
    ) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({"config": self.config, "templates": self.templates,
                         "calibrator": self.calibrator}, f)

        # Always save templates PNG
        self._save_templates_png(path.with_name(path.stem + "_templates.png"))

        # Optionally save NCC score maps
        if ncc_images:
            names = ncc_names or [f"sample_{i}" for i in range(len(ncc_images))]
            for img, name in zip(ncc_images, names):
                out = path.with_name(f"{path.stem}_ncc_{name}.png")
                self._save_ncc_png(out, img)

    def _save_templates_png(self, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        templates = self.templates
        n = len(templates)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for i, (ax, T) in enumerate(zip(axes, templates)):
            ax.imshow(T, cmap="gray")
            ax.set_title(f"Template {i}")
            ax.axis("off")
        fig.suptitle("Learned templates (dark = low intensity = bubble)")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_ncc_png(self, path: Path, image: np.ndarray) -> None:
        """Save original image alongside NCC score map at the most populated scale level."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncc_results = compute_ncc_maps(image, self.templates, self.config)
        if not ncc_results:
            return

        # Pick the level with the highest total score magnitude (most signal)
        best_idx = int(np.argmax([np.abs(sm).sum() for _, sm in ncc_results]))
        eff_radius, score_map = ncc_results[best_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image, cmap="gray")
        ax1.set_title("Original")
        ax1.axis("off")
        im = ax2.imshow(score_map, cmap="hot", vmin=-1, vmax=1)
        ax2.set_title(f"NCC score map  (eff. radius \u2248 {eff_radius:.1f} px)")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def load(cls, path: Path) -> "BubblePipeline":
        with open(path, "rb") as f:
            data = pickle.load(f)
        pipeline = cls(data["config"])
        pipeline.templates = data["templates"]
        pipeline.calibrator = data["calibrator"]
        return pipeline
