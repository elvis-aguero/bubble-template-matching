import math
import numpy as np
import pytest
from bubble_histogram.config import PipelineConfig
from bubble_histogram.ncc import build_pyramid, compute_ncc_maps


def test_pyramid_levels():
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=1.0, max_radius=50.0,
                         template_context_factor=1.0)
    img = np.random.rand(100, 100).astype(np.float32)
    levels = build_pyramid(img, cfg)
    canonical_radius = cfg.template_size / (2 * cfg.template_context_factor)
    expected_n = math.ceil(math.log(cfg.max_radius / canonical_radius) / math.log(1 / cfg.scale_factor))
    assert len(levels) == expected_n


def test_pyramid_shapes_shrink():
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=1.0, max_radius=50.0)
    img = np.random.rand(200, 200).astype(np.float32)
    levels = build_pyramid(img, cfg)
    shapes = [lvl[1].shape for lvl in levels]
    for i in range(1, len(shapes)):
        assert shapes[i][0] <= shapes[i - 1][0]


def test_pyramid_effective_radii_increase():
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=1.0, max_radius=50.0)
    img = np.random.rand(200, 200).astype(np.float32)
    levels = build_pyramid(img, cfg)
    radii = [lvl[2] for lvl in levels]
    for i in range(1, len(radii)):
        assert radii[i] > radii[i - 1]


def test_ncc_score_range():
    cfg = PipelineConfig(template_size=5, scale_factor=0.9, min_radius=1.0, max_radius=20.0)
    img = np.random.rand(100, 100).astype(np.float32)
    template = np.random.rand(5, 5).astype(np.float32)
    template /= np.linalg.norm(template)
    templates = template[np.newaxis]
    results = compute_ncc_maps(img, templates, cfg)
    for eff_radius, score_map in results:
        assert score_map.min() >= -1.0 - 1e-5
        assert score_map.max() <= 1.0 + 1e-5


def test_ncc_perfect_match():
    """Score should peak near the location of the template pattern."""
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=4.9, max_radius=5.1)
    T = np.zeros((10, 10), dtype=np.float32)
    T[4:6, 4:6] = 1.0
    T /= np.linalg.norm(T)

    img = np.zeros((50, 50), dtype=np.float32)
    img[0:10, 0:10] = T

    templates = T[np.newaxis]
    results = compute_ncc_maps(img, templates, cfg)
    assert len(results) >= 1
    _, score_map = results[0]
    peak_yx = np.unravel_index(score_map.argmax(), score_map.shape)
    assert peak_yx[0] < 10
    assert peak_yx[1] < 10
