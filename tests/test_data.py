import json
import re
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import parse_annotations, load_image, get_session_id, AnnotatedDataset


# ---------------------------------------------------------------------------
# Task 2: PipelineConfig
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = PipelineConfig()
    assert cfg.num_templates == 1
    assert cfg.template_size == 10
    assert cfg.scale_factor == 0.9
    assert cfg.min_radius == 1.0
    assert cfg.max_radius == 50.0
    assert cfg.n_score_bins == 50
    assert cfg.neg_sample_ratio == 10
    assert cfg.min_neg_dist == 10


def test_config_override():
    cfg = PipelineConfig(num_templates=3, template_size=5)
    assert cfg.num_templates == 3
    assert cfg.template_size == 5
    assert cfg.scale_factor == 0.9  # unchanged


# ---------------------------------------------------------------------------
# Task 3: Annotation Parsing
# ---------------------------------------------------------------------------

def test_parse_circle(tmp_path, synthetic_circle_json):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(synthetic_circle_json))
    bubbles = parse_annotations(path)
    assert len(bubbles) == 1
    cx, cy, r = bubbles[0]
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(50.0)
    assert r == pytest.approx(5.0)


def test_parse_polygon(tmp_path, synthetic_polygon_json):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(synthetic_polygon_json))
    bubbles = parse_annotations(path)
    assert len(bubbles) == 1
    cx, cy, r = bubbles[0]
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(50.0)
    assert r == pytest.approx(10.0)


def test_parse_ignores_non_bubble(tmp_path):
    ann = {
        "version": "3.3.5", "flags": {}, "description": "",
        "imagePath": "x.png", "imageWidth": 100, "imageHeight": 100,
        "shapes": [
            {"label": "other", "shape_type": "circle",
             "points": [[50.0, 50.0], [55.0, 50.0]],
             "group_id": None, "description": "", "difficult": False,
             "score": None, "flags": {}, "attributes": {}, "kie_linking": []},
        ],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(ann))
    assert parse_annotations(path) == []


# ---------------------------------------------------------------------------
# Task 4: Image Loading
# ---------------------------------------------------------------------------

def test_load_8bit_image(tmp_path):
    arr = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(tmp_path / "img.png")
    result = load_image(tmp_path / "img.png")
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, 0] == pytest.approx(1.0)


def test_load_16bit_image(tmp_path):
    arr = np.array([[0, 32768], [65535, 16384]], dtype=np.uint16)
    Image.fromarray(arr, mode="I;16").save(tmp_path / "img.png")
    result = load_image(tmp_path / "img.png")
    assert result.dtype == np.float32
    assert result.max() == pytest.approx(1.0, abs=1e-4)
    assert result.min() == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Task 5: Dataset + Train/Val Split
# ---------------------------------------------------------------------------

def test_get_session_id_standard():
    assert get_session_id("ZeroG_FlightDay_Test_C1S0014_img006001.png") == "C1S0014"


def test_get_session_id_double_underscore():
    assert get_session_id("ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png") == "C1S0004"


def test_dataset_split(tmp_dataset):
    ds = AnnotatedDataset(tmp_dataset, val_session="C1S0001")
    assert len(ds.train_images) == 0
    assert len(ds.val_images) == 1


def test_dataset_no_split(tmp_dataset):
    ds = AnnotatedDataset(tmp_dataset)
    assert len(ds.train_images) == 1
    assert len(ds.val_images) == 0


def test_dataset_load_sample(tmp_dataset):
    ds = AnnotatedDataset(tmp_dataset)
    sample = ds.load_sample(ds.train_images[0])
    assert sample.image.dtype == np.float32
    assert len(sample.bubbles) == 1
    assert sample.bubbles[0].radius == pytest.approx(5.0)
