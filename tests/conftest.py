import json
import numpy as np
import pytest


def make_dark_circle_image(size=100, cx=50, cy=50, r=5):
    """Synthetic grayscale image: light background, dark circle."""
    img = np.full((size, size), 0.9, dtype=np.float32)
    yy, xx = np.ogrid[:size, :size]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 0.1
    return img


def make_circle_annotation(cx, cy, r, image_path="test.png", width=100, height=100):
    """LabelImg-style JSON for a single circle annotation."""
    return {
        "version": "3.3.5",
        "flags": {},
        "shapes": [
            {
                "label": "bubble",
                "shape_type": "circle",
                "points": [[cx, cy], [cx + r, cy]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "score": None,
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        ],
        "imagePath": image_path,
        "imageWidth": width,
        "imageHeight": height,
        "description": "",
    }


@pytest.fixture
def synthetic_circle_json():
    return make_circle_annotation(cx=50.0, cy=50.0, r=5.0)


@pytest.fixture
def synthetic_polygon_json():
    return {
        "version": "3.3.5",
        "flags": {},
        "shapes": [
            {
                "label": "bubble",
                "shape_type": "polygon",
                "points": [[40.0, 50.0], [60.0, 50.0], [50.0, 60.0], [50.0, 40.0]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "score": None,
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        ],
        "imagePath": "test.png",
        "imageWidth": 100,
        "imageHeight": 100,
        "description": "",
    }


@pytest.fixture
def synthetic_image():
    return make_dark_circle_image(size=100, cx=50, cy=50, r=5)


@pytest.fixture
def tmp_dataset(tmp_path):
    """Minimal dataset on disk: one 100x100 PNG + one circle annotation JSON."""
    from PIL import Image

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Save image
    img_arr = (make_dark_circle_image(100, 50, 50, 5) * 255).astype(np.uint8)
    Image.fromarray(img_arr, mode="L").save(images_dir / "ZeroG_Test_C1S0001_img001.png")

    # Save annotation
    ann = make_circle_annotation(50.0, 50.0, 5.0, "ZeroG_Test_C1S0001_img001.png")
    (labels_dir / "ZeroG_Test_C1S0001_img001.json").write_text(json.dumps(ann))

    return tmp_path
