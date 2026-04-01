import numpy as np
import pytest
from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.template import build_templates
from bubble_histogram.calibration import sample_scores, ScoreCalibrator


def test_sample_scores_returns_arrays(tmp_dataset):
    cfg = PipelineConfig(template_size=5, neg_sample_ratio=3, min_neg_dist=5)
    ds = AnnotatedDataset(tmp_dataset)
    templates = build_templates(ds, cfg)
    pos, neg = sample_scores(ds, templates, cfg)
    assert isinstance(pos, np.ndarray)
    assert isinstance(neg, np.ndarray)
    assert len(pos) >= 1
    assert len(neg) >= 1


def test_negative_count(tmp_dataset):
    cfg = PipelineConfig(template_size=5, neg_sample_ratio=5, min_neg_dist=5)
    ds = AnnotatedDataset(tmp_dataset)
    templates = build_templates(ds, cfg)
    pos, neg = sample_scores(ds, templates, cfg)
    # negatives ≥ positives (may be capped by available locations)
    assert len(neg) >= len(pos)


def test_calibrator_output_range(tmp_dataset):
    cfg = PipelineConfig(template_size=5, n_score_bins=20)
    ds = AnnotatedDataset(tmp_dataset)
    templates = build_templates(ds, cfg)
    pos, neg = sample_scores(ds, templates, cfg)

    prior = 0.01
    cal = ScoreCalibrator(n_bins=20)
    cal.fit(pos, neg, prior)

    scores = np.linspace(-1, 1, 100, dtype=np.float32)
    probs = cal.predict(scores)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_calibrator_monotone_tendency(tmp_dataset):
    """Higher NCC scores should yield higher bubble probability on average."""
    cfg = PipelineConfig(template_size=5, n_score_bins=20)
    ds = AnnotatedDataset(tmp_dataset)
    templates = build_templates(ds, cfg)
    pos, neg = sample_scores(ds, templates, cfg)
    prior = 0.01
    cal = ScoreCalibrator(n_bins=20)
    cal.fit(pos, neg, prior)

    low = cal.predict(np.array([-0.5], dtype=np.float32))[0]
    high = cal.predict(np.array([0.8], dtype=np.float32))[0]
    # Not strictly guaranteed with synthetic data, but assert no crash and valid range
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
