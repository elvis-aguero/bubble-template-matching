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

    # canonical_radius: the bubble radius (in original px) that the template represents at level 0.
    # With context_factor > 1 the crop is wider than the bubble, so the bubble only fills
    # template_size / context_factor pixels of the template side.
    # Example: template_size=13, context_factor=1.3 → canonical_radius=5px.
    # A 5px-radius bubble at level 0 fills exactly the template's inner region.
    canonical_radius = config.template_size / (2 * config.template_context_factor)

    # number of pyramid levels needed to cover [canonical_radius, max_radius]
    # derived from: canonical_radius / scale_factor^n_levels = max_radius
    n_levels = math.ceil(
        math.log(config.max_radius / canonical_radius)
        / math.log(1 / config.scale_factor)
    )

    levels = []
    for lvl in range(n_levels):
        scale = config.scale_factor ** lvl          # cumulative scale at this level
        effective_radius = canonical_radius / scale  # bubble radius in original px that matches the template at this level
        if effective_radius > config.max_radius:
            break

        if lvl == 0:
            scaled = image  # level 0: use the original image unchanged
        else:
            # shrink the image so that a larger bubble now occupies canonical_radius pixels
            scaled = rescale(image, scale, anti_aliasing=True, channel_axis=None).astype(np.float32)

        levels.append((lvl, scaled, effective_radius))

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
    # bin_centers uses config.num_templates (not len(templates)) so the radius ranges
    # match exactly what was used when templates were built — even if some bins were empty
    bin_edges = np.logspace(
        math.log10(config.min_radius),
        math.log10(config.max_radius),
        config.num_templates + 1,
    )
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean of each bin's edges

    pyramid = build_pyramid(image, config)
    results = []

    for _l, scaled_img, eff_radius in pyramid:
        h, w = scaled_img.shape
        ts = config.template_size

        if h < ts or w < ts:
            continue    # skip levels where the image has been shrunk smaller than the template

        # select which template to use for this pyramid level (only matters if num_templates > 1)
        tmpl_idx = 0 if n_bins == 1 else _assign_template(eff_radius, bin_centers)
        T = templates[tmpl_idx]

        # match_template computes C[y,x] = dot(W[y,x] / ||W[y,x]||, T) at every location
        # pad_input=True keeps the output the same size as the input (otherwise it shrinks by template_size-1)
        # mode="reflect" pads the border by mirroring pixel values — avoids edge artifacts
        score_map = match_template(scaled_img, T, pad_input=True, mode="reflect")
        results.append((eff_radius, score_map.astype(np.float32)))

    return results  # one (radius, score_map) pair per pyramid level = one per histogram bin
