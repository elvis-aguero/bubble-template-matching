import json
import re
import numpy as np
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass

__all__ = ["Bubble", "parse_annotations", "load_image", "get_session_id", "AnnotatedSample", "AnnotatedDataset"]

from PIL import Image


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


def load_image(path: Path) -> np.ndarray:
    """Load PNG (8-bit or 16-bit grayscale) as float32 in [0, 1]."""
    raw = np.array(Image.open(path))
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16 or raw.dtype == np.int32:
        return raw.astype(np.float32) / 65535.0
    else:
        raise ValueError(f"Unsupported image dtype {raw.dtype}. Expected uint8 or uint16.")


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
    """Dataset with leave-one-session-out train/val split support."""

    def __init__(self, root_dir: Path, val_session: str | None = None,
                 template_frac: float | None = None,
                 calibration_frac: float | None = None,
                 seed: int = 42):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        all_images = sorted(self.image_dir.glob("*.png"))

        if val_session is not None:
            # LOSO mode — existing behaviour unchanged
            train = [p for p in all_images if get_session_id(p.name) != val_session]
            val   = [p for p in all_images if get_session_id(p.name) == val_session]
            self.train_images       = train
            self.val_images         = val
            self.template_images    = train
            self.calibration_images = train
            self.test_images        = val
            self.split_info = {
                "mode": "loso",
                "val_session": val_session,
                "template":    [p.name for p in train],
                "calibration": [p.name for p in train],
                "test":        [p.name for p in val],
            }

        elif template_frac is not None and calibration_frac is not None:
            # Image-level three-way split
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(all_images))
            n = len(all_images)
            n_test = max(1, round(n * (1.0 - template_frac - calibration_frac)))
            n_tmpl = round(n * template_frac)

            test_idx = sorted(idx[:n_test].tolist())
            tmpl_idx = sorted(idx[n_test:n_test + n_tmpl].tolist())
            cal_idx  = sorted(idx[n_test + n_tmpl:].tolist())

            self.test_images        = [all_images[i] for i in test_idx]
            self.template_images    = [all_images[i] for i in tmpl_idx]
            self.calibration_images = [all_images[i] for i in cal_idx]
            self.train_images       = self.template_images + self.calibration_images
            self.val_images         = self.test_images
            self.split_info = {
                "mode":             "image_level",
                "seed":             seed,
                "template_frac":    template_frac,
                "calibration_frac": calibration_frac,
                "template":         [p.name for p in self.template_images],
                "calibration":      [p.name for p in self.calibration_images],
                "test":             [p.name for p in self.test_images],
            }

        else:
            # No split — backward compat
            self.train_images       = list(all_images)
            self.val_images         = []
            self.template_images    = list(all_images)
            self.calibration_images = list(all_images)
            self.test_images        = []
            self.split_info         = {"mode": "none"}

    def load_sample(self, image_path: Path) -> AnnotatedSample:
        label_path = self.label_dir / (image_path.stem + ".json")
        return AnnotatedSample(
            image=load_image(image_path),
            bubbles=parse_annotations(label_path),
            image_path=image_path,
        )
