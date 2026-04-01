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
