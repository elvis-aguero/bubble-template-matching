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
