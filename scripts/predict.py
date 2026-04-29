#!/usr/bin/env python3
"""
Run a trained bubble histogram pipeline on one or more images and write results to CSV.

USAGE
-----
  python scripts/predict.py <pipeline.pkl> <image1> [image2 ...] [--output out.csv]

OUTPUT
------
  CSV with one row per (image × size bin):

    image,radius_px,expected_count
    frame_001.png,3.2,12.4
    frame_001.png,3.6,9.1
    ...

  radius_px     — effective bubble radius for that histogram bin, in original image pixels
  expected_count — expected number of bubbles of that size (sum of P(bubble) over local maxima)

QUICK START
-----------
  # Single image
  python scripts/predict.py output/pipeline.pkl seed_v04/images/some_frame.png

  # Batch (glob)
  python scripts/predict.py output/pipeline.pkl seed_v04/images/*.png --output results.csv

  # Load results in Python
  import pandas as pd
  df = pd.read_csv("results.csv")
  total_per_frame = df.groupby("image")["expected_count"].sum()
"""
import argparse
import csv
from pathlib import Path

from bubble_histogram.data import load_image
from bubble_histogram.pipeline import BubblePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run trained bubble pipeline on images and write histogram CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pipeline", type=Path, help="Path to saved pipeline (.pkl)")
    parser.add_argument("images", type=Path, nargs="+", help="Image files to process (PNG/TIFF)")
    parser.add_argument("--output", type=Path, default=Path("histograms.csv"),
                        help="Output CSV path (default: histograms.csv)")
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
