import numpy as np
import pytest
from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.pipeline import BubblePipeline


@pytest.fixture
def trained_pipeline(tmp_dataset):
    cfg = PipelineConfig(template_size=5, scale_factor=0.9, min_radius=1.0, max_radius=15.0)
    ds = AnnotatedDataset(tmp_dataset)
    pipeline = BubblePipeline(cfg)
    pipeline.train(ds)
    return pipeline


def test_pipeline_trains_without_error(trained_pipeline):
    assert trained_pipeline.templates is not None
    assert trained_pipeline.calibrator is not None


def test_pipeline_predict_structure(trained_pipeline, synthetic_image):
    result = trained_pipeline.predict(synthetic_image)
    assert "radius_px" in result
    assert "expected_count" in result
    assert len(result["radius_px"]) == len(result["expected_count"])
    assert len(result["radius_px"]) > 0


def test_pipeline_predict_counts_nonnegative(trained_pipeline, synthetic_image):
    result = trained_pipeline.predict(synthetic_image)
    assert all(c >= 0 for c in result["expected_count"])


def test_pipeline_save_load(trained_pipeline, synthetic_image, tmp_path):
    path = tmp_path / "pipeline.pkl"
    trained_pipeline.save(path)
    loaded = BubblePipeline.load(path)
    result_orig = trained_pipeline.predict(synthetic_image)
    result_load = loaded.predict(synthetic_image)
    np.testing.assert_allclose(result_orig["expected_count"], result_load["expected_count"])
