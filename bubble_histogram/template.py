import math
import cv2
import numpy as np

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

    # bin_edges divides [min_radius, max_radius] into num_templates logarithmically equal intervals
    # log spacing is used because bubble size variation is multiplicative, not additive
    bin_edges = np.logspace(
        math.log10(config.min_radius),
        math.log10(config.max_radius),
        config.num_templates + 1,
    )

    bin_patches: list[list[np.ndarray]] = [[] for _ in range(config.num_templates)]  # collect patches per size bin

    for image_path in paths:
        sample = dataset.load_sample(image_path)
        img = sample.image
        h, w = img.shape

        for bubble in sample.bubbles:
            cx, cy, r = bubble.cx, bubble.cy, bubble.radius

            if config.num_templates == 1:
                bin_idx = 0  # all bubbles go into one template regardless of size
            else:
                bin_idx = int(np.searchsorted(bin_edges[1:], r))        # find which size bin this bubble belongs to
                bin_idx = min(bin_idx, config.num_templates - 1)         # clamp to last bin if radius > max_radius

            # context_factor > 1.0 makes the crop larger than the bubble bounding box,
            # including a ring of background that makes the template more discriminative
            r_crop = max(1, int(round(r * config.template_context_factor)))
            x0, x1 = int(cx) - r_crop, int(cx) + r_crop
            y0, y1 = int(cy) - r_crop, int(cy) + r_crop

            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                continue    # skip bubbles too close to the image border to crop cleanly

            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            # resize every patch to the same template_size × template_size shape
            # so that patches from different-sized bubbles can be averaged together
            # cv2.resize takes (width, height) — note the transposed order vs numpy shape
            # INTER_AREA for downscaling (correct area average); INTER_LINEAR for upscaling
            # (patches from small bubbles are smaller than template_size and need upscaling)
            ts = config.template_size
            interp = cv2.INTER_AREA if patch.shape[0] >= ts else cv2.INTER_LINEAR
            resized = cv2.resize(patch, (ts, ts), interpolation=interp).astype(np.float32)

            # Zero-mean + L2-normalise each patch before averaging.
            # This preserves the spatial contrast structure (dark bubble interior vs
            # bright surround) so the averaged template retains discriminative features.
            # Sum-normalisation was used previously but collapses every patch to ~1/169
            # per pixel, washing out the dark-disc pattern and making the template flat.
            resized -= resized.mean()
            norm_p = np.linalg.norm(resized)
            if norm_p > 0:
                resized /= norm_p

            bin_patches[bin_idx].append(resized)

    templates = []
    for patches in bin_patches:
        if not patches:
            continue                        # skip bins with no annotated bubbles
        T = np.mean(patches, axis=0)        # average all patches in this bin → mean appearance
        norm = np.linalg.norm(T)
        if norm > 0:
            T /= norm                       # L2-normalise so dot(T, W/||W||) = cos(angle) in NCC
        templates.append(T)

    if not templates:
        raise ValueError("No valid patches found for any size bin.")

    return np.stack(templates)  # shape: (n_non_empty_bins, template_size, template_size)
