import math
import numpy as np
from skimage.transform import resize as sk_resize

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset


def build_templates(
    dataset: AnnotatedDataset,
    config: PipelineConfig,
    image_paths: list | None = None,
) -> np.ndarray:
    """
    Build appearance templates from annotated training bubbles.

    Returns
    -------
    np.ndarray of shape (n_bins, template_size, template_size)
        Each template is L2-normalized. n_bins <= num_templates (empty bins are skipped).
    """
    paths = image_paths if image_paths is not None else dataset.train_images

    bin_edges = np.logspace(
        math.log10(config.min_radius),
        math.log10(config.max_radius),
        config.num_templates + 1,
    )

    bin_patches: list[list[np.ndarray]] = [[] for _ in range(config.num_templates)]

    for image_path in paths:
        sample = dataset.load_sample(image_path)
        img = sample.image
        h, w = img.shape

        for bubble in sample.bubbles:
            cx, cy, r = bubble.cx, bubble.cy, bubble.radius

            if config.num_templates == 1:
                bin_idx = 0
            else:
                bin_idx = int(np.searchsorted(bin_edges[1:], r))
                bin_idx = min(bin_idx, config.num_templates - 1)

            r_int = max(1, int(round(r)))
            x0, x1 = int(cx) - r_int, int(cx) + r_int
            y0, y1 = int(cy) - r_int, int(cy) + r_int

            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                continue

            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            resized = sk_resize(
                patch,
                (config.template_size, config.template_size),
                anti_aliasing=True,
            ).astype(np.float32)

            s = resized.sum()
            if s > 0:
                resized /= s

            bin_patches[bin_idx].append(resized)

    templates = []
    for patches in bin_patches:
        if not patches:
            continue
        T = np.mean(patches, axis=0)
        norm = np.linalg.norm(T)
        if norm > 0:
            T /= norm
        templates.append(T)

    if not templates:
        raise ValueError("No valid patches found for any size bin.")

    return np.stack(templates)
