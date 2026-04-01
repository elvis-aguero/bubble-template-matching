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
