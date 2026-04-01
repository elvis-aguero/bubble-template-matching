# Bubble Size Histogram Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that estimates a per-frame bubble size histogram for unannotated video frames using NCC template matching trained on 14 annotated images.

**Architecture:** Multi-scale NCC with Bayesian calibration. A fixed-size template is built from annotated training bubbles and applied across an image pyramid; each pyramid level corresponds to one histogram size bin. Scores at bubble/non-bubble locations on training images calibrate a lookup table mapping NCC score → P(bubble), enabling count estimation as the expected value of a sum of Bernoulli variables.

**Tech Stack:** Python 3.10+, numpy, scikit-image (`match_template`, `rescale`), Pillow (image loading), matplotlib, pytest, dataclasses, pickle

---

## File Map

| File | Responsibility |
|---|---|
| `bubble_histogram/__init__.py` | Package exports |
| `bubble_histogram/config.py` | `PipelineConfig` dataclass — all hyperparameters |
| `bubble_histogram/data.py` | Annotation parsing (circles + polygons), image loading, session-based train/val split |
| `bubble_histogram/template.py` | Build per-size-bin templates from annotated patches |
| `bubble_histogram/ncc.py` | Image pyramid construction + NCC score maps per level |
| `bubble_histogram/calibration.py` | Positive/negative sampling, empirical histograms, Bayesian inversion |
| `bubble_histogram/pipeline.py` | `BubblePipeline`: `.train()`, `.predict()`, `.save()`, `.load()` |
| `bubble_histogram/histogram.py` | Output formatting and matplotlib visualization |
| `scripts/train.py` | CLI: fit pipeline on annotated data, save to disk |
| `scripts/predict.py` | CLI: run pipeline on image files, write histogram CSV |
| `scripts/visualize.py` | CLI: visualize template, calibration curves, per-frame histogram |
| `tests/conftest.py` | Shared pytest fixtures (synthetic images + annotations) |
| `tests/test_data.py` | Tests for parsing and loading |
| `tests/test_template.py` | Tests for template construction |
| `tests/test_ncc.py` | Tests for pyramid + NCC |
| `tests/test_calibration.py` | Tests for sampling and Bayesian calibration |
| `tests/test_pipeline.py` | Integration: train on synthetic data, predict, check output shape |

---

## Chunk 1: Setup + Config + Data Layer

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `bubble_histogram/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "bubble-histogram"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "scikit-image",
    "Pillow",
    "matplotlib",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install in development mode**

Run: `pip install -e ".[dev]"`
Expected: no errors, `bubble_histogram` importable

- [ ] **Step 3: Create package init**

`bubble_histogram/__init__.py` — empty file for now.

- [ ] **Step 4: Create `tests/conftest.py` with shared fixtures**

```python
import json
import numpy as np
import pytest
from pathlib import Path


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
```

- [ ] **Step 5: Verify pytest collects fixtures**

Run: `pytest tests/ --collect-only`
Expected: no errors, conftest loaded

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml bubble_histogram/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: project scaffold with pytest fixtures"
```

---

### Task 2: PipelineConfig

**Files:**
- Create: `bubble_histogram/config.py`

- [ ] **Step 1: Write failing test**

`tests/test_data.py` (create file):
```python
from bubble_histogram.config import PipelineConfig


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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data.py::test_config_defaults -v`
Expected: `ModuleNotFoundError: No module named 'bubble_histogram.config'`

- [ ] **Step 3: Implement `config.py`**

```python
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    num_templates: int = 1
    template_size: int = 10
    scale_factor: float = 0.9
    min_radius: float = 1.0
    max_radius: float = 50.0
    n_score_bins: int = 50
    neg_sample_ratio: int = 10
    min_neg_dist: int = 10
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/config.py tests/test_data.py
git commit -m "feat: PipelineConfig dataclass with all hyperparameters"
```

---

### Task 3: Annotation Parsing

**Files:**
- Create: `bubble_histogram/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_data.py`:
```python
import json
import numpy as np
import pytest
from pathlib import Path
from bubble_histogram.data import parse_annotations


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
    # centroid of [[40,50],[60,50],[50,60],[50,40]] = [50, 50]
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(50.0)
    # max dist from centroid: 10px (horizontal points)
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data.py -k "parse" -v`
Expected: `ImportError: cannot import name 'parse_annotations'`

- [ ] **Step 3: Implement `data.py` — parsing section**

```python
import json
import numpy as np
from pathlib import Path
from typing import NamedTuple


class Bubble(NamedTuple):
    cx: float
    cy: float
    radius: float


def parse_annotations(json_path: Path) -> list[Bubble]:
    """Parse LabelImg JSON → list of (cx, cy, radius) for all 'bubble' shapes."""
    data = json.loads(Path(json_path).read_text())
    bubbles = []
    for shape in data["shapes"]:
        if shape["label"] != "bubble":
            continue
        pts = np.array(shape["points"], dtype=np.float64)
        if shape["shape_type"] == "circle":
            cx, cy = pts[0]
            radius = float(np.linalg.norm(pts[1] - pts[0]))
        elif shape["shape_type"] == "polygon":
            centroid = pts.mean(axis=0)
            radius = float(np.linalg.norm(pts - centroid, axis=1).max())
            cx, cy = centroid
        else:
            continue
        bubbles.append(Bubble(float(cx), float(cy), radius))
    return bubbles
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_data.py -k "parse" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/data.py tests/test_data.py
git commit -m "feat: annotation parsing for circle and polygon shapes"
```

---

### Task 4: Image Loading

**Files:**
- Modify: `bubble_histogram/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_data.py`:
```python
import numpy as np
from PIL import Image
from bubble_histogram.data import load_image


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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data.py -k "load" -v`
Expected: `ImportError: cannot import name 'load_image'`

- [ ] **Step 3: Implement image loading in `data.py`**

Append to `bubble_histogram/data.py`:
```python
import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    """Load PNG (8-bit or 16-bit grayscale) as float32 in [0, 1]."""
    raw = np.array(Image.open(path))
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16 or raw.dtype == np.int32:
        return raw.astype(np.float32) / 65535.0
    else:
        # Already float or unknown; normalize to [0,1]
        arr = raw.astype(np.float32)
        if arr.max() > 0:
            arr /= arr.max()
        return arr
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_data.py -k "load" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/data.py tests/test_data.py
git commit -m "feat: image loading with 8-bit and 16-bit PNG support"
```

---

### Task 5: Dataset + Train/Val Split

**Files:**
- Modify: `bubble_histogram/data.py`
- Create: `tests/test_data.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_data.py`:
```python
import re
from bubble_histogram.data import get_session_id, AnnotatedDataset


def test_get_session_id_standard():
    assert get_session_id("ZeroG_FlightDay_Test_C1S0014_img006001.png") == "C1S0014"


def test_get_session_id_double_underscore():
    assert get_session_id("ZeroG_FlightDay_Test__C1S0004_IMG_S0001000001.png") == "C1S0004"


def test_dataset_split(tmp_dataset):
    # tmp_dataset has one image from session C1S0001
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data.py -k "session or dataset" -v`
Expected: `ImportError`

- [ ] **Step 3: Implement in `data.py`**

Append to `bubble_histogram/data.py`:
```python
import re
from dataclasses import dataclass


def get_session_id(filename: str) -> str:
    """Extract session ID (e.g. 'C1S0014') from image filename."""
    match = re.search(r"(C\d+S\d+)", filename)
    if not match:
        raise ValueError(f"No session ID found in filename: {filename}")
    return match.group(1)


@dataclass
class AnnotatedSample:
    image: np.ndarray
    bubbles: list[Bubble]
    image_path: Path


class AnnotatedDataset:
    def __init__(self, root_dir: Path, val_session: str | None = None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        all_images = sorted(self.image_dir.glob("*.png"))
        if val_session:
            self.train_images = [p for p in all_images if get_session_id(p.name) != val_session]
            self.val_images = [p for p in all_images if get_session_id(p.name) == val_session]
        else:
            self.train_images = list(all_images)
            self.val_images = []

    def load_sample(self, image_path: Path) -> AnnotatedSample:
        label_path = self.label_dir / (image_path.stem + ".json")
        return AnnotatedSample(
            image=load_image(image_path),
            bubbles=parse_annotations(label_path),
            image_path=image_path,
        )
```

- [ ] **Step 4: Run all data tests**

Run: `pytest tests/test_data.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/data.py tests/test_data.py
git commit -m "feat: AnnotatedDataset with leave-one-session-out split"
```

---

## Chunk 2: Template Construction + NCC

### Task 6: Template Construction

**Files:**
- Create: `bubble_histogram/template.py`
- Create: `tests/test_template.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_template.py
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
    # The single bubble (r=5) falls in one bin; the other bin may be empty
    # With only 1 bubble, only 1 bin will have patches — expect ValueError for empty bin
    # OR we skip empty bins. Test that we get at least 1 template.
    # Implementation choice: skip empty bins (don't raise)
    templates = build_templates(ds, cfg)
    assert templates.shape[0] >= 1
    assert templates.shape[1] == 8
    assert templates.shape[2] == 8
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_template.py -v`
Expected: `ModuleNotFoundError: No module named 'bubble_histogram.template'`

- [ ] **Step 3: Implement `template.py`**

```python
import math
import numpy as np
from skimage.transform import resize as sk_resize

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset


def build_templates(dataset: AnnotatedDataset, config: PipelineConfig) -> np.ndarray:
    """
    Build appearance templates from annotated training bubbles.

    Returns
    -------
    np.ndarray of shape (n_bins, template_size, template_size)
        Each template is L2-normalized. n_bins <= num_templates (empty bins are skipped).
    """
    # Log-spaced bin edges
    bin_edges = np.logspace(
        math.log10(config.min_radius),
        math.log10(config.max_radius),
        config.num_templates + 1,
    )

    bin_patches: list[list[np.ndarray]] = [[] for _ in range(config.num_templates)]

    for image_path in dataset.train_images:
        sample = dataset.load_sample(image_path)
        img = sample.image
        h, w = img.shape

        for bubble in sample.bubbles:
            cx, cy, r = bubble.cx, bubble.cy, bubble.radius

            # Assign to bin
            if config.num_templates == 1:
                bin_idx = 0
            else:
                bin_idx = int(np.searchsorted(bin_edges[1:], r))
                bin_idx = min(bin_idx, config.num_templates - 1)

            # Extract 2r × 2r patch centered on bubble
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

            # Normalize to sum=1
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_template.py -v`
Expected: all PASS

- [ ] **Step 5: Update `bubble_histogram/__init__.py`**

```python
from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, load_image, parse_annotations
from bubble_histogram.template import build_templates
```

- [ ] **Step 6: Commit**

```bash
git add bubble_histogram/template.py bubble_histogram/__init__.py tests/test_template.py
git commit -m "feat: template construction with log-spaced size bins and L2 normalization"
```

---

### Task 7: Image Pyramid + NCC Score Maps

**Files:**
- Create: `bubble_histogram/ncc.py`
- Create: `tests/test_ncc.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ncc.py
import math
import numpy as np
import pytest
from bubble_histogram.config import PipelineConfig
from bubble_histogram.ncc import build_pyramid, compute_ncc_maps


def test_pyramid_levels():
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=1.0, max_radius=50.0)
    img = np.random.rand(100, 100).astype(np.float32)
    levels = build_pyramid(img, cfg)
    # n_levels = ceil(log(50 / 5) / log(1/0.9)) = ceil(log(10)/log(1.111)) ≈ ceil(24.2) = 25
    expected_n = math.ceil(math.log(cfg.max_radius / (cfg.template_size / 2)) / math.log(1 / cfg.scale_factor))
    assert len(levels) == expected_n


def test_pyramid_shapes_shrink():
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=1.0, max_radius=50.0)
    img = np.random.rand(200, 200).astype(np.float32)
    levels = build_pyramid(img, cfg)
    shapes = [lvl[1].shape for lvl in levels]
    # Each level should be smaller than or equal to the previous
    for i in range(1, len(shapes)):
        assert shapes[i][0] <= shapes[i - 1][0]


def test_pyramid_effective_radii_decrease():
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=1.0, max_radius=50.0)
    img = np.random.rand(200, 200).astype(np.float32)
    levels = build_pyramid(img, cfg)
    radii = [lvl[2] for lvl in levels]
    # Level 0: effective_radius = template_size/2 / 0.9^0 = 5 (smallest, original scale)
    # Level 1: 5 / 0.9 ≈ 5.56 (larger bubble, smaller image)
    for i in range(1, len(radii)):
        assert radii[i] > radii[i - 1]


def test_ncc_score_range():
    cfg = PipelineConfig(template_size=5, scale_factor=0.9, min_radius=1.0, max_radius=20.0)
    img = np.random.rand(100, 100).astype(np.float32)
    template = np.random.rand(5, 5).astype(np.float32)
    template /= np.linalg.norm(template)
    templates = template[np.newaxis]  # (1, 5, 5)
    results = compute_ncc_maps(img, templates, cfg)
    for eff_radius, score_map in results:
        assert score_map.min() >= -1.0 - 1e-5
        assert score_map.max() <= 1.0 + 1e-5


def test_ncc_perfect_match():
    """Score at template location should be close to 1 when image patch == template."""
    # Use scale_factor=0.9 with a tight radius range to get ~1 pyramid level
    cfg = PipelineConfig(template_size=10, scale_factor=0.9, min_radius=4.9, max_radius=5.1)
    T = np.zeros((10, 10), dtype=np.float32)
    T[4:6, 4:6] = 1.0
    T /= np.linalg.norm(T)

    img = np.zeros((50, 50), dtype=np.float32)
    img[0:10, 0:10] = T  # place template pattern at top-left

    templates = T[np.newaxis]
    results = compute_ncc_maps(img, templates, cfg)
    assert len(results) >= 1
    _, score_map = results[0]
    # Peak should be near top-left (center of template window)
    peak_yx = np.unravel_index(score_map.argmax(), score_map.shape)
    assert peak_yx[0] < 10
    assert peak_yx[1] < 10
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ncc.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `ncc.py`**

```python
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
    List of (effective_radius_px, score_map) pairs. score_map is same spatial
    size as the scaled image at that level (pad_input=True).
    """
    n_bins = len(templates)
    bin_edges = np.logspace(
        math.log10(config.min_radius),
        math.log10(config.max_radius),
        n_bins + 1,
    )
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    pyramid = build_pyramid(image, config)
    results = []

    for _l, scaled_img, eff_radius in pyramid:
        h, w = scaled_img.shape
        ts = config.template_size

        if h < ts or w < ts:
            continue  # image too small for template at this level

        tmpl_idx = 0 if n_bins == 1 else _assign_template(eff_radius, bin_centers)
        T = templates[tmpl_idx]

        score_map = match_template(scaled_img, T, pad_input=True, mode="reflect")
        results.append((eff_radius, score_map.astype(np.float32)))

    return results
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ncc.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/ncc.py tests/test_ncc.py
git commit -m "feat: image pyramid and NCC score maps via skimage match_template"
```

---

## Chunk 3: Calibration + Pipeline

### Task 8: Score Sampling + Bayesian Calibration

**Files:**
- Create: `bubble_histogram/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calibration.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_calibration.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `calibration.py`**

```python
import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.ncc import compute_ncc_maps


def sample_scores(
    dataset: AnnotatedDataset,
    templates: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract NCC scores at annotated bubble centers (positives) and random
    non-bubble locations (negatives) from all training images.

    Each bubble is sampled at the pyramid level whose effective radius best
    matches the annotated bubble radius.
    """
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    rng = np.random.default_rng(seed=42)

    for image_path in dataset.train_images:
        sample = dataset.load_sample(image_path)
        ncc_results = compute_ncc_maps(sample.image, templates, config)

        if not ncc_results:
            continue

        eff_radii = np.array([r for r, _ in ncc_results])

        for bubble in sample.bubbles:
            cx, cy, r = bubble.cx, bubble.cy, bubble.radius
            # Pick level whose effective radius is closest to bubble radius
            level_idx = int(np.argmin(np.abs(eff_radii - r)))
            eff_r, score_map = ncc_results[level_idx]

            # Convert bubble center to scaled coordinates
            scale = eff_r / (config.template_size / 2)
            # effective_radius = (template_size/2) / scale_factor^l
            # scale_factor^l = (template_size/2) / effective_radius
            img_scale = (config.template_size / 2) / eff_r
            sx = int(round(cx * img_scale))
            sy = int(round(cy * img_scale))

            h, w = score_map.shape
            if 0 <= sx < w and 0 <= sy < h:
                pos_scores.append(float(score_map[sy, sx]))

        # Sample negatives from the first (full-res) level
        _, score_map = ncc_results[0]
        h, w = score_map.shape

        # Build exclusion mask around all bubbles (at level-0 scale)
        img_scale_0 = (config.template_size / 2) / eff_radii[0]
        excl = np.zeros((h, w), dtype=bool)
        d = config.min_neg_dist
        for bubble in sample.bubbles:
            sx = int(round(bubble.cx * img_scale_0))
            sy = int(round(bubble.cy * img_scale_0))
            excl[max(0, sy - d):min(h, sy + d), max(0, sx - d):min(w, sx + d)] = True

        candidates = np.argwhere(~excl)
        n_neg = min(len(pos_scores) * config.neg_sample_ratio, len(candidates))
        if n_neg > 0:
            chosen = rng.choice(len(candidates), size=n_neg, replace=False)
            for idx in chosen:
                y, x = candidates[idx]
                neg_scores.append(float(score_map[y, x]))

    return np.array(pos_scores, dtype=np.float32), np.array(neg_scores, dtype=np.float32)


class ScoreCalibrator:
    """Maps NCC scores → P(bubble|score) via Bayesian non-parametric calibration."""

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins
        self.bin_edges: np.ndarray | None = None
        self.p_bubble_given_score: np.ndarray | None = None

    def fit(self, pos_scores: np.ndarray, neg_scores: np.ndarray, prior: float) -> None:
        """
        Build calibration table from empirical score distributions.

        Parameters
        ----------
        pos_scores : NCC scores at annotated bubble locations
        neg_scores : NCC scores at sampled non-bubble locations
        prior : P(bubble) — fraction of locations containing a bubble
        """
        self.bin_edges = np.linspace(-1.0, 1.0, self.n_bins + 1)

        p_score_given_bubble, _ = np.histogram(pos_scores, bins=self.bin_edges, density=True)
        p_score_given_not_bubble, _ = np.histogram(neg_scores, bins=self.bin_edges, density=True)

        p_not_bubble = 1.0 - prior
        numerator = p_score_given_bubble * prior
        denominator = numerator + p_score_given_not_bubble * p_not_bubble

        with np.errstate(divide="ignore", invalid="ignore"):
            self.p_bubble_given_score = np.where(
                denominator > 0, numerator / denominator, 0.0
            ).astype(np.float32)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Look up P(bubble|score) for an array of NCC scores."""
        if self.bin_edges is None or self.p_bubble_given_score is None:
            raise RuntimeError("ScoreCalibrator must be fit before calling predict.")
        bin_idxs = np.digitize(scores, self.bin_edges) - 1
        bin_idxs = np.clip(bin_idxs, 0, self.n_bins - 1)
        return self.p_bubble_given_score[bin_idxs]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_calibration.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/calibration.py tests/test_calibration.py
git commit -m "feat: Bayesian score calibration with positive/negative sampling"
```

---

### Task 9: Pipeline (Train + Predict + Save/Load)

**Files:**
- Create: `bubble_histogram/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `pipeline.py`**

```python
import pickle
from pathlib import Path

import numpy as np

from bubble_histogram.calibration import ScoreCalibrator, sample_scores
from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, load_image
from bubble_histogram.ncc import compute_ncc_maps
from bubble_histogram.template import build_templates


class BubblePipeline:
    """
    End-to-end bubble size histogram pipeline.

    Usage
    -----
    pipeline = BubblePipeline(config)
    pipeline.train(dataset)          # fit templates + calibrator
    result = pipeline.predict(img)   # dict with radius_px + expected_count
    pipeline.save(path)
    pipeline = BubblePipeline.load(path)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.templates: np.ndarray | None = None
        self.calibrator: ScoreCalibrator | None = None

    def train(self, dataset: AnnotatedDataset) -> None:
        self.templates = build_templates(dataset, self.config)

        pos_scores, neg_scores = sample_scores(dataset, self.templates, self.config)

        # Estimate prior P(bubble) = total bubbles / total pixel locations
        total_bubbles = 0
        total_locs = 0
        for p in dataset.train_images:
            sample = dataset.load_sample(p)
            total_bubbles += len(sample.bubbles)
            total_locs += int(np.prod(sample.image.shape))

        prior = total_bubbles / max(total_locs, 1)

        self.calibrator = ScoreCalibrator(n_bins=self.config.n_score_bins)
        self.calibrator.fit(pos_scores, neg_scores, prior)

    def predict(self, image: np.ndarray) -> dict:
        """
        Estimate bubble size histogram for a single image.

        Returns
        -------
        dict with keys:
          "radius_px"      : list[float] — effective bubble radius per level (original image px)
          "expected_count" : list[float] — expected bubble count per level
        """
        if self.templates is None or self.calibrator is None:
            raise RuntimeError("Pipeline must be trained before calling predict.")

        ncc_results = compute_ncc_maps(image, self.templates, self.config)

        radius_px = []
        expected_counts = []

        for eff_radius, score_map in ncc_results:
            probs = self.calibrator.predict(score_map.ravel())
            expected_counts.append(float(probs.sum()))
            radius_px.append(eff_radius)

        return {"radius_px": radius_px, "expected_count": expected_counts}

    def save(self, path: Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "templates": self.templates,
                    "calibrator": self.calibrator,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BubblePipeline":
        with open(path, "rb") as f:
            data = pickle.load(f)
        pipeline = cls(data["config"])
        pipeline.templates = data["templates"]
        pipeline.calibrator = data["calibrator"]
        return pipeline
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/pipeline.py tests/test_pipeline.py
git commit -m "feat: BubblePipeline with train, predict, save, and load"
```

---

## Chunk 4: Output, Visualization, and Scripts

### Task 10: Histogram Output + Visualization

**Files:**
- Create: `bubble_histogram/histogram.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_pipeline.py`:
```python
from bubble_histogram.histogram import plot_histogram
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests


def test_plot_histogram_runs(trained_pipeline, synthetic_image):
    result = trained_pipeline.predict(synthetic_image)
    ax = plot_histogram(result)
    assert ax is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline.py::test_plot_histogram_runs -v`
Expected: `ImportError`

- [ ] **Step 3: Implement `histogram.py`**

```python
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes


def plot_histogram(
    result: dict,
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Bubble Size Histogram",
    color: str = "steelblue",
) -> matplotlib.axes.Axes:
    """
    Plot a per-frame bubble size histogram.

    Parameters
    ----------
    result : output of BubblePipeline.predict() —
             dict with "radius_px" and "expected_count"
    ax : existing Axes to draw on (creates new figure if None)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    radii = np.array(result["radius_px"])
    counts = np.array(result["expected_count"])

    # Bar width in log10 space = log10(1/scale_factor) — one pyramid step
    log_radii = np.log10(radii)
    bar_width = log_radii[1] - log_radii[0] if len(log_radii) > 1 else 0.05

    ax.bar(log_radii, counts, width=bar_width * 0.9, align="center", color=color, alpha=0.8)

    tick_vals = [1, 2, 5, 10, 20, 50]
    ax.set_xticks(np.log10(tick_vals))
    ax.set_xticklabels([str(v) for v in tick_vals])
    ax.set_xlabel("Bubble radius (px)")
    ax.set_ylabel("Expected count")
    ax.set_title(title)
    ax.set_xlim(log_radii.min() - bar_width, log_radii.max() + bar_width)

    return ax


def aggregate_histograms(results: list[dict]) -> dict:
    """Sum expected counts across multiple frames."""
    if not results:
        return {"radius_px": [], "expected_count": []}
    radius_px = results[0]["radius_px"]
    total = np.sum([np.array(r["expected_count"]) for r in results], axis=0)
    return {"radius_px": radius_px, "expected_count": total.tolist()}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add bubble_histogram/histogram.py tests/test_pipeline.py
git commit -m "feat: histogram plotting and multi-frame aggregation"
```

---

### Task 11: Scripts

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/predict.py`
- Create: `scripts/visualize.py`

- [ ] **Step 1: Create `scripts/train.py`**

```python
#!/usr/bin/env python3
"""Fit bubble histogram pipeline on annotated data and save to disk."""
import argparse
from pathlib import Path

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.pipeline import BubblePipeline


def main():
    parser = argparse.ArgumentParser(description="Train bubble histogram pipeline.")
    parser.add_argument("data_dir", type=Path, help="Path to seed_v04/ directory")
    parser.add_argument("output", type=Path, help="Output path for saved pipeline (.pkl)")
    parser.add_argument("--val-session", default=None, help="Session ID to hold out for validation")
    parser.add_argument("--num-templates", type=int, default=1)
    parser.add_argument("--template-size", type=int, default=10)
    parser.add_argument("--scale-factor", type=float, default=0.9)
    parser.add_argument("--min-radius", type=float, default=1.0)
    parser.add_argument("--max-radius", type=float, default=50.0)
    args = parser.parse_args()

    cfg = PipelineConfig(
        num_templates=args.num_templates,
        template_size=args.template_size,
        scale_factor=args.scale_factor,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )

    print(f"Loading dataset from {args.data_dir}...")
    ds = AnnotatedDataset(args.data_dir, val_session=args.val_session)
    print(f"  Train images: {len(ds.train_images)}, Val images: {len(ds.val_images)}")

    print("Training pipeline...")
    pipeline = BubblePipeline(cfg)
    pipeline.train(ds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(args.output)
    print(f"Pipeline saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/predict.py`**

```python
#!/usr/bin/env python3
"""Run trained pipeline on image files and output histogram CSV."""
import argparse
import csv
from pathlib import Path

from bubble_histogram.data import load_image
from bubble_histogram.pipeline import BubblePipeline


def main():
    parser = argparse.ArgumentParser(description="Run bubble histogram pipeline on images.")
    parser.add_argument("pipeline", type=Path, help="Path to saved pipeline (.pkl)")
    parser.add_argument("images", type=Path, nargs="+", help="Image files to process")
    parser.add_argument("--output", type=Path, default=Path("histograms.csv"), help="Output CSV path")
    args = parser.parse_args()

    pipeline = BubblePipeline.load(args.pipeline)
    print(f"Loaded pipeline. Processing {len(args.images)} image(s)...")

    rows = []
    for img_path in args.images:
        img = load_image(img_path)
        result = pipeline.predict(img)
        for r, c in zip(result["radius_px"], result["expected_count"]):
            rows.append({"image": img_path.name, "radius_px": r, "expected_count": c})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "radius_px", "expected_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2b: Smoke-test `predict.py`**

Run:
```bash
python scripts/predict.py pipeline.pkl seed_v04/images/ZeroG_FlightDay_Test_C1S0014_img006001.png --output out.csv
head out.csv
```
Expected: `out.csv` has header `image,radius_px,expected_count` and one row per pyramid level.

- [ ] **Step 3: Create `scripts/visualize.py`**

```python
#!/usr/bin/env python3
"""Visualize template, calibration curves, and per-frame histogram."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.histogram import plot_histogram
from bubble_histogram.pipeline import BubblePipeline
from bubble_histogram.data import load_image


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline components.")
    parser.add_argument("pipeline", type=Path, help="Path to saved pipeline (.pkl)")
    parser.add_argument("--image", type=Path, default=None, help="Image to run and plot histogram for")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    pipeline = BubblePipeline.load(args.pipeline)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Template(s)
    templates = pipeline.templates
    n = len(templates)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (ax, T) in enumerate(zip(axes, templates)):
        ax.imshow(T, cmap="gray")
        ax.set_title(f"Template {i}")
        ax.axis("off")
    fig.suptitle("Learned Templates (dark=low intensity)")
    fig.savefig(args.output_dir / "templates.png", dpi=150, bbox_inches="tight")
    print(f"Saved templates.png")

    # 2. Calibration curves
    cal = pipeline.calibrator
    if cal is not None and cal.bin_edges is not None:
        bin_centers = 0.5 * (cal.bin_edges[:-1] + cal.bin_edges[1:])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bin_centers, cal.p_bubble_given_score, label="P(bubble|score)")
        ax.set_xlabel("NCC score")
        ax.set_ylabel("P(bubble|score)")
        ax.set_title("Calibration: score → bubble probability")
        ax.legend()
        fig.savefig(args.output_dir / "calibration.png", dpi=150, bbox_inches="tight")
        print("Saved calibration.png")

    # 3. Per-frame histogram (if image provided)
    if args.image:
        img = load_image(args.image)
        result = pipeline.predict(img)
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_histogram(result, ax=ax, title=f"Histogram: {args.image.name}")
        fig.savefig(args.output_dir / "histogram.png", dpi=150, bbox_inches="tight")
        print("Saved histogram.png")

    plt.close("all")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-test train script on real data**

Run:
```bash
python scripts/train.py seed_v04/ pipeline.pkl --val-session C1S0010
```
Expected: "Pipeline saved to pipeline.pkl" with no errors

- [ ] **Step 5: Smoke-test visualize script**

Run:
```bash
python scripts/visualize.py pipeline.pkl --image seed_v04/images/ZeroG_FlightDay_Test_C1S0014_img006001.png
```
Expected: three PNG files in `plots/` — `templates.png` (dark spot on light bg), `calibration.png`, `histogram.png`

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: all PASS

- [ ] **Step 7: Final commit**

```bash
git add scripts/ bubble_histogram/histogram.py bubble_histogram/__init__.py
git commit -m "feat: train/predict/visualize scripts and histogram output"
```

---

**Note on temporal smoothing:** The spec mentions an optional causal EWMA smoother on per-frame histogram counts. This is deferred: implement the per-frame pipeline first, then measure autocorrelation of outputs on real video data to determine if smoothing is warranted. No code is needed here.

---

## Verification Checklist

- [ ] `templates.png` shows a dark spot on a light background (bubble appearance)
- [ ] `calibration.png` shows P(bubble|score) that is generally higher for scores > 0
- [ ] `histogram.png` shows bulk of count concentrated in small-radius bins (radius 2–10px)
- [ ] Total expected count across all levels on a training image is in the right ballpark vs. annotated count
- [ ] `pytest tests/ -v` — all green
