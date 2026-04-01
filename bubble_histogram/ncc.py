import math
import numpy as np
from skimage.transform import rescale
from skimage.feature import match_template

from bubble_histogram.config import PipelineConfig


def build_pyramid(image: np.ndarray, config: PipelineConfig) -> list[tuple]:
    """
    Build multi-scale image pyramid.

    Returns
    -------
    List of (level_index, scaled_image, effective_radius_px) tuples.
    Level 0 = original scale; effective_radius increases with level.
    """
    if config.scale_factor >= 1.0:
        raise ValueError(f"scale_factor must be < 1.0, got {config.scale_factor}")

    n_levels = math.ceil(
        math.log(config.max_radius / (config.template_size / 2))
        / math.log(1 / config.scale_factor)
    )

    levels = []
    for l in range(n_levels):
        scale = config.scale_factor ** l
        effective_radius = (config.template_size / 2) / scale
        if effective_radius > config.max_radius:
            break

        if scale == 1.0:
            scaled = image
        else:
            scaled = rescale(image, scale, anti_aliasing=True, channel_axis=None).astype(np.float32)

        levels.append((l, scaled, effective_radius))

    return levels


def _assign_template(eff_radius: float, bin_centers: np.ndarray) -> int:
    """Return index of closest bin center to effective_radius."""
    return int(np.argmin(np.abs(bin_centers - eff_radius)))


def compute_ncc_maps(
    image: np.ndarray,
    templates: np.ndarray,
    config: PipelineConfig,
) -> list[tuple[float, np.ndarray]]:
    """
    Compute NCC score maps at each pyramid level.

    Parameters
    ----------
    image : (H, W) float32 image
    templates : (n_bins, template_size, template_size) array
    config : PipelineConfig

    Returns
    -------
    List of (effective_radius_px, score_map) pairs.
    """
    n_bins = len(templates)
    bin_edges = np.logspace(
        math.log10(config.min_radius),
        math.log10(config.max_radius),
        n_bins + 1,
    )
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    pyramid = build_pyramid(image, config)
    results = []

    for _l, scaled_img, eff_radius in pyramid:
        h, w = scaled_img.shape
        ts = config.template_size

        if h < ts or w < ts:
            continue

        tmpl_idx = 0 if n_bins == 1 else _assign_template(eff_radius, bin_centers)
        T = templates[tmpl_idx]

        score_map = match_template(scaled_img, T, pad_input=True, mode="reflect")
        results.append((eff_radius, score_map.astype(np.float32)))

    return results
