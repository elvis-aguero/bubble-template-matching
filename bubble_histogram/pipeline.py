import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from bubble_histogram.calibration import ScoreCalibrator, _local_maxima, _lm_min_dist, nms_3d, sample_scores
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
        self.templates: np.ndarray | None = None        # shape (n_bins, template_size, template_size)
        self.calibrator: ScoreCalibrator | None = None  # maps NCC score → P(bubble); used when local_maxima_calibration=False
        self.calibrators: dict[int, ScoreCalibrator] | None = None  # per-level calibrators; used when local_maxima_calibration=True
        self._split_info: dict | None = None            # saved for the *_split.json artifact
        self._pos_scores: np.ndarray | None = None      # saved for the *_score_histograms.png artifact
        self._neg_scores: np.ndarray | None = None

    def train(self, dataset: AnnotatedDataset) -> None:
        self._split_info = getattr(dataset, "split_info", None)
        # build templates from the template split only (not calibration images — avoids data leakage)
        self.templates = build_templates(dataset, self.config,
                                         image_paths=dataset.template_images)

        if self.config.local_maxima_calibration:
            self._train_per_level(dataset)
        else:
            self._train_global(dataset)

    def _train_per_level(self, dataset: AnnotatedDataset) -> None:
        """
        Per-level calibration.  Training population depends on nms_iou_threshold:

        > 0 (cross-scale NMS mode, E1): train on NMS survivors; label positive only
            when at the correct pyramid level AND within bubble.radius of a GT
            annotation.  Scale bias in NMS means only ~12% of GT bubbles contribute
            positives.

        == 0 (per-level NMS mode, O2): train on raw per-level local maxima.  A raw
            LM at level lv is positive iff within bubble.radius of a GT bubble whose
            correct level is lv; all other LMs at lv are negative.  Fine-scale
            aliases of larger bubbles land as negatives at fine levels, teaching the
            calibrator to suppress them.  Negatives are capped at
            neg_sample_ratio × n_pos to avoid overwhelming the calibrator.
        """
        level_pos: dict[int, list[float]] = defaultdict(list)
        level_neg: dict[int, list[float]] = defaultdict(list)
        n_levels = 0
        use_raw_lm = (self.config.nms_iou_threshold == 0.0)
        rng = np.random.default_rng(seed=42) if use_raw_lm else None
        min_d = _lm_min_dist(self.config)

        for image_path in dataset.calibration_images:
            sample = dataset.load_sample(image_path)
            ncc_results = compute_ncc_maps(sample.image, self.templates, self.config)
            if not ncc_results:
                continue
            n_levels = max(n_levels, len(ncc_results))
            eff_radii = np.array([er for er, _ in ncc_results])

            gt = [(b.cy, b.cx, b.radius) for b in sample.bubbles]
            gt_levels = [int(np.argmin(np.abs(eff_radii - r))) for _, _, r in gt]

            if use_raw_lm:
                # O2: iterate raw LMs per level, label against correct-level GT only
                gt_by_level: dict[int, list] = defaultdict(list)
                for i, (cy, cx, r) in enumerate(gt):
                    gt_by_level[gt_levels[i]].append((cy, cx, r))

                for lv, (eff_r, score_map) in enumerate(ncc_results):
                    alpha = self.config.template_size / (2.0 * self.config.template_context_factor * eff_r)
                    peaks = _local_maxima(score_map, min_d)
                    if len(peaks) == 0:
                        continue
                    scores_lv = score_map[peaks[:, 0], peaks[:, 1]]
                    ys = peaks[:, 0] / alpha
                    xs = peaks[:, 1] / alpha

                    level_gt = gt_by_level.get(lv, [])
                    matched_lv: set[int] = set()
                    lv_pos: list[float] = []
                    lv_neg: list[float] = []

                    for k in np.argsort(scores_lv)[::-1]:
                        y, x, sc = float(ys[k]), float(xs[k]), float(scores_lv[k])
                        best_i, best_d = -1, float("inf")
                        for i, (cy, cx, r) in enumerate(level_gt):
                            if i in matched_lv:
                                continue
                            d = math.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                            if d < r and d < best_d:
                                best_d, best_i = d, i
                        if best_i >= 0:
                            matched_lv.add(best_i)
                            lv_pos.append(sc)
                        else:
                            lv_neg.append(sc)

                    level_pos[lv].extend(lv_pos)
                    max_neg = max(len(lv_pos) * self.config.neg_sample_ratio, 50)
                    if len(lv_neg) > max_neg:
                        idx = rng.choice(len(lv_neg), size=max_neg, replace=False)
                        lv_neg = [lv_neg[j] for j in idx]
                    level_neg[lv].extend(lv_neg)

            else:
                # E1: NMS survivors with scale-aware positive labeling
                survivors = nms_3d(ncc_results, self.config)
                matched_gt: set[int] = set()

                for score, level_idx, y, x, _ in survivors:
                    best_i, best_d = -1, float("inf")
                    for i, (cy, cx, r) in enumerate(gt):
                        if i in matched_gt:
                            continue
                        d = math.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                        if d < r and level_idx == gt_levels[i] and d < best_d:
                            best_d, best_i = d, i
                    if best_i >= 0:
                        matched_gt.add(best_i)
                        level_pos[level_idx].append(score)
                    else:
                        level_neg[level_idx].append(score)

        # Flatten for the score-histogram artifact
        all_pos = np.array([s for ss in level_pos.values() for s in ss], dtype=np.float32)
        all_neg = np.array([s for ss in level_neg.values() for s in ss], dtype=np.float32)
        self._pos_scores, self._neg_scores = all_pos, all_neg

        # Fit one calibrator per level.  Use fewer bins than the global calibrator
        # because per-level positive counts can be very small (O(tens) per level).
        bins_per_level = max(10, self.config.n_score_bins // 3)
        self.calibrators = {}
        for lv in range(n_levels):
            p = np.array(level_pos.get(lv, []), dtype=np.float32)
            n = np.array(level_neg.get(lv, []), dtype=np.float32)
            prior = len(p) / max(len(p) + len(n), 1)
            cal = ScoreCalibrator(n_bins=bins_per_level)
            cal.fit(p, n, prior)
            self.calibrators[lv] = cal

        # Keep a global calibrator fitted for backwards compat (test assertions, score histogram)
        global_prior = len(all_pos) / max(len(all_pos) + len(all_neg), 1)
        self.calibrator = ScoreCalibrator(n_bins=self.config.n_score_bins)
        self.calibrator.fit(all_pos, all_neg, global_prior)

    def _train_global(self, dataset: AnnotatedDataset) -> None:
        """Global calibration (local_maxima_calibration=False): centroid-based positives + random pixel negatives."""
        pos_scores, neg_scores = sample_scores(dataset, self.templates, self.config,
                                               image_paths=dataset.calibration_images)
        self._pos_scores = pos_scores
        self._neg_scores = neg_scores

        total_bubbles = sum(len(dataset.load_sample(p).bubbles)
                            for p in dataset.calibration_images)
        total_pixels = sum(int(np.prod(dataset.load_sample(p).image.shape))
                           for p in dataset.calibration_images)
        prior = total_bubbles / max(total_pixels, 1)

        self.calibrator = ScoreCalibrator(n_bins=self.config.n_score_bins)
        self.calibrator.fit(pos_scores, neg_scores, prior)
        self.calibrators = None

    def predict(self, image: np.ndarray, ncc_results: list | None = None) -> dict[str, list[float]]:
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

        if ncc_results is None:
            ncc_results = compute_ncc_maps(image, self.templates, self.config)

        radius_px = []
        expected_counts = []

        from bubble_histogram.calibration import _local_maxima
        min_d = 2   # minimum pixel distance between local maxima; small because NCC response is smooth

        use_lm = self.config.predict_local_maxima or self.config.local_maxima_calibration

        if use_lm and self.config.nms_iou_threshold > 0.0:
            # Cross-scale IoU NMS: collect all peaks across all levels, suppress overlapping
            # detections globally (not per-level), then accumulate P(bubble) per level.
            # This prevents the same bubble from being counted at multiple scale levels.
            survivors = nms_3d(ncc_results, self.config)
            level_probs: dict[int, float] = {i: 0.0 for i in range(len(ncc_results))}
            use_per_level = self.calibrators is not None
            for score, level_idx, _y, _x, _r in survivors:
                if use_per_level:
                    cal = self.calibrators.get(level_idx)
                    prob = float(cal.predict(np.array([score], dtype=np.float32))[0]) if cal is not None else 0.0
                else:
                    prob = float(self.calibrator.predict(np.array([score], dtype=np.float32))[0])
                level_probs[level_idx] += prob
            for i, (eff_radius, _) in enumerate(ncc_results):
                expected_counts.append(level_probs[i])
                radius_px.append(eff_radius)
        else:
            # Per-level 2D NMS: peaks found independently at each scale (no cross-scale suppression)
            use_per_level_cal = self.calibrators is not None
            for i, (eff_radius, score_map) in enumerate(ncc_results):
                if use_lm:
                    peaks = _local_maxima(score_map, min_d)
                    scores = score_map[peaks[:, 0], peaks[:, 1]] if len(peaks) else np.array([])
                else:
                    scores = score_map.ravel()  # dense: every pixel contributes (not recommended)
                if len(scores):
                    if use_per_level_cal:
                        cal = self.calibrators.get(i)
                        probs = cal.predict(scores) if cal is not None else np.zeros(len(scores), dtype=np.float32)
                    else:
                        probs = self.calibrator.predict(scores)
                else:
                    probs = np.array([])
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
                         "calibrator": self.calibrator,
                         "calibrators": self.calibrators}, f)

        # Always save templates PNG
        self._save_templates_png(path.with_name(path.stem + "_templates.png"))

        # Always save split manifest
        if self._split_info is not None:
            import json
            split_path = path.with_name(path.stem + "_split.json")
            split_path.write_text(json.dumps(self._split_info, indent=2))

        # Always save score histograms if available
        if self._pos_scores is not None and self._neg_scores is not None:
            self._save_score_histograms_png(
                path.with_name(path.stem + "_score_histograms.png")
            )

        # Optionally save NCC score maps
        if ncc_images:
            names = ncc_names or [f"sample_{i}" for i in range(len(ncc_images))]
            for img, name in zip(ncc_images, names):
                out = path.with_name(f"{path.stem}_ncc_{name}.png")
                self._save_ncc_png(out, img)

    def _save_templates_png(self, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe to call without a display
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

    def _save_score_histograms_png(self, path: Path) -> None:
        """Save overlapping histograms of positive and negative NCC scores."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(-1.0, 1.0, self.config.n_score_bins + 1)
        # good calibration: blue (bubble) distribution peaks at high scores, red (background) peaks near 0
        ax.hist(self._pos_scores, bins=bins, density=True, alpha=0.6,
                color="steelblue", label=f"Bubble ({len(self._pos_scores)} samples)")
        ax.hist(self._neg_scores, bins=bins, density=True, alpha=0.6,
                color="salmon", label=f"Non-bubble ({len(self._neg_scores)} samples)")
        ax.set_xlabel("NCC score")
        ax.set_ylabel("Density")
        ax.set_title("NCC score distributions (calibration set)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_top_matches_png(
        self,
        path: Path,
        image: np.ndarray,
        top_n: int = 100,
        ncc_results: list | None = None,
    ) -> None:
        """
        Save original image annotated with the top-N highest-scoring peaks
        after 3D NMS across scale levels.

        Each box is 2×eff_radius wide/tall in original image pixels (= the template
        footprint at that scale). Colour encodes NCC score (plasma, warm = high).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from bubble_histogram.calibration import nms_3d

        if ncc_results is None:
            ncc_results = compute_ncc_maps(image, self.templates, self.config)
        if not ncc_results:
            return

        all_peaks = nms_3d(ncc_results, self.config)    # 3D NMS: suppresses cross-scale duplicates
        peaks = all_peaks[:top_n]                       # keep only the top-N highest-scoring survivors
        if not peaks:
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image, cmap="gray")
        cmap = plt.cm.plasma    # warm colours = high NCC score
        scores = [p[0] for p in peaks]
        s_min, s_max = min(scores), max(scores)

        for score, level, y_orig, x_orig, eff_radius in peaks:
            colour = cmap((score - s_min) / max(s_max - s_min, 1e-6))
            rect = mpatches.Rectangle(
                (x_orig - eff_radius, y_orig - eff_radius),
                2 * eff_radius, 2 * eff_radius,
                linewidth=0.8, edgecolor=colour, facecolor="none", alpha=0.85,
            )
            ax.add_patch(rect)
            ax.text(
                x_orig + eff_radius - 1, y_orig - eff_radius + 1,
                f"{score:.2f}",
                color=colour, fontsize=4, ha="right", va="top", alpha=0.9,
            )

        ax.set_title(
            f"Top {len(peaks)} of {len(all_peaks)} NCC matches after 3D NMS  "
            f"(colour = score, warm = high)"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_pr_curve_png(
        self,
        path: Path,
        samples: list,
        precomputed_ncc: list | None = None,
    ) -> None:
        """
        Save a precision-recall curve PNG evaluated on the given samples.

        Detections come from nms_3d (ranked by score).  A detection is a true
        positive if its centre lands within the annotated bubble's radius of any
        unmatched annotation (greedy, highest-score first).  The curve is pooled
        across all samples; AP is computed as the area under the interpolated curve.

        precomputed_ncc : optional list of ncc_results (one per sample) to avoid
                          recomputing NCC maps when they were already computed by the caller.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from bubble_histogram.calibration import nms_3d

        all_detections: list[tuple[float, bool]] = []   # (score, is_tp)
        total_gt = 0

        for idx, sample in enumerate(samples):
            ncc_results = (precomputed_ncc[idx]
                           if precomputed_ncc is not None
                           else compute_ncc_maps(sample.image, self.templates, self.config))
            if not ncc_results:
                continue

            peaks = nms_3d(ncc_results, self.config)    # sorted by score desc
            gt = [(b.cy, b.cx, b.radius) for b in sample.bubbles]
            total_gt += len(gt)
            matched_gt: set[int] = set()

            for score, _level, y_orig, x_orig, _eff_r in peaks:
                best_dist = float("inf")
                best_i = -1
                for i, (cy, cx, r) in enumerate(gt):
                    if i in matched_gt:
                        continue
                    dist = float(np.sqrt((y_orig - cy) ** 2 + (x_orig - cx) ** 2))
                    if dist < r and dist < best_dist:
                        best_dist = dist
                        best_i = i
                if best_i >= 0:
                    matched_gt.add(best_i)
                    all_detections.append((score, True))
                else:
                    all_detections.append((score, False))

        if not all_detections or total_gt == 0:
            return

        all_detections.sort(key=lambda d: d[0], reverse=True)

        tp = 0
        fp = 0
        precisions: list[float] = []
        recalls: list[float] = []
        for _, is_tp in all_detections:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / total_gt)

        # Interpolated AP: for each point, use the max precision at that recall or higher
        prec = np.array([1.0] + precisions + [0.0], dtype=np.float64)
        rec  = np.array([0.0] + recalls   + [recalls[-1]], dtype=np.float64)
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        ap = float(np.trapezoid(prec, rec))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recalls, precisions, color="steelblue", linewidth=1.5)
        ax.fill_between(recalls, precisions, alpha=0.15, color="steelblue")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_title(
            f"Precision–Recall Curve  (AP = {ap:.3f})\n"
            f"{total_gt} GT bubbles · {len(all_detections)} detections · "
            f"{len(samples)} image(s)"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_ncc_png(self, path: Path, image: np.ndarray, ncc_results: list | None = None) -> None:
        """Public wrapper — save NCC score map for a given image."""
        self._save_ncc_png(path, image, ncc_results=ncc_results)

    def _save_ncc_png(self, path: Path, image: np.ndarray, ncc_results: list | None = None) -> None:
        """Save original image alongside NCC score map at the most populated scale level."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if ncc_results is None:
            ncc_results = compute_ncc_maps(image, self.templates, self.config)
        if not ncc_results:
            return

        # pick the level where the NCC signal has the most total energy — usually the scale matching the most bubbles
        best_idx = int(np.argmax([np.abs(sm).sum() for _, sm in ncc_results]))
        eff_radius, score_map = ncc_results[best_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image, cmap="gray")
        ax1.set_title("Original")
        ax1.axis("off")
        im = ax2.imshow(score_map, cmap="hot", vmin=-1, vmax=1)  # hot colormap: bright = high NCC score
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
        pipeline = cls(data["config"])      # reconstruct with the config that was used during training
        pipeline.templates = data["templates"]
        pipeline.calibrator = data["calibrator"]
        pipeline.calibrators = data.get("calibrators")  # None for pipelines saved before per-level calibration
        return pipeline
