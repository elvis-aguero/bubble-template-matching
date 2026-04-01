import numpy as np
import pytest
from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.template import build_templates


def test_template_shape(tmp_dataset):
    cfg = PipelineConfig(num_templates=1, template_size=10)
    ds = AnnotatedDataset(tmp_dataset)
    templates = build_templates(ds, cfg)
    assert templates.shape == (1, 10, 10)


def test_template_unit_norm(tmp_dataset):
    cfg = PipelineConfig(num_templates=1, template_size=10)
    ds = AnnotatedDataset(tmp_dataset)
    templates = build_templates(ds, cfg)
    norms = np.linalg.norm(templates.reshape(1, -1), axis=1)
    assert norms[0] == pytest.approx(1.0, abs=1e-5)


def test_template_dark_center(tmp_dataset):
    """Bubbles are dark on light background → center of template should be darker."""
    cfg = PipelineConfig(num_templates=1, template_size=10)
    ds = AnnotatedDataset(tmp_dataset)
    templates = build_templates(ds, cfg)
    T = templates[0]
    center = T[4:6, 4:6].mean()
    border = np.concatenate([T[0, :], T[-1, :], T[:, 0], T[:, -1]]).mean()
    assert center < border, "Template center should be darker than border"


def test_template_multi_bin_shape(tmp_dataset):
    cfg = PipelineConfig(num_templates=2, template_size=8, min_radius=1.0, max_radius=20.0)
    ds = AnnotatedDataset(tmp_dataset)
    # With only 1 bubble (r=5), only 1 bin has patches — skip empty bins
    templates = build_templates(ds, cfg)
    assert templates.shape[0] >= 1
    assert templates.shape[1] == 8
    assert templates.shape[2] == 8
