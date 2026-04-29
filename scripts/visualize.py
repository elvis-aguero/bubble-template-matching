#!/usr/bin/env python3
"""
Visualize a trained bubble histogram pipeline: template, calibration curve, and size histogram.

USAGE
-----
  python scripts/visualize.py <pipeline.pkl> [--image <frame.png>] [--output-dir <dir>]

OUTPUT (saved to --output-dir, default: plots/)
------
  templates.png    — the averaged bubble appearance template(s); should show a dark disc
                     surrounded by a brighter background ring. A flat grey blob means
                     the averaging cancelled out (too much size variation or bad training data).
  calibration.png  — P(bubble|NCC score) vs score; should rise steeply for scores > 0.3
                     and be near zero for negative scores.
  histogram.png    — predicted bubble size histogram for --image (only if --image is given)

QUICK START
-----------
  # Inspect a trained pipeline (no image needed for template + calibration plots)
  python scripts/visualize.py output/pipeline.pkl

  # Also plot the size histogram for a specific frame
  python scripts/visualize.py output/pipeline.pkl --image seed_v04/images/some_frame.png

  # Save to a custom directory
  python scripts/visualize.py output/pipeline.pkl --image frame.png --output-dir my_plots/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bubble_histogram.histogram import plot_histogram
from bubble_histogram.pipeline import BubblePipeline
from bubble_histogram.data import load_image


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pipeline template, calibration curve, and size histogram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pipeline", type=Path, help="Path to saved pipeline (.pkl)")
    parser.add_argument("--image", type=Path, default=None,
                        help="Image to predict on and plot the size histogram for")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                        help="Directory to write PNGs into (default: plots/)")
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
