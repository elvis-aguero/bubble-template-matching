#!/usr/bin/env python3
"""Fit bubble histogram pipeline on annotated data and save to disk."""
import argparse
from pathlib import Path

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, load_image
from bubble_histogram.pipeline import BubblePipeline


def main():
    parser = argparse.ArgumentParser(description="Train bubble histogram pipeline.")
    parser.add_argument("data_dir", type=Path, help="Path to seed_v04/ directory")
    parser.add_argument("output", type=Path, help="Output path for saved pipeline (.pkl)")
    parser.add_argument("--val-session", default=None,
                        help="LOSO: session ID to hold out for validation (disables image-level split)")
    parser.add_argument("--template-frac", type=float, default=0.30,
                        help="Fraction of images used for template construction (default 0.30)")
    parser.add_argument("--calibration-frac", type=float, default=0.65,
                        help="Fraction of images used for calibration (default 0.65)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Random seed for image-level split")
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
    if args.val_session:
        ds = AnnotatedDataset(args.data_dir, val_session=args.val_session)
        print(f"  LOSO mode — train: {len(ds.train_images)}, val: {len(ds.val_images)}")
    else:
        ds = AnnotatedDataset(
            args.data_dir,
            template_frac=args.template_frac,
            calibration_frac=args.calibration_frac,
            seed=args.split_seed,
        )
        print(f"  Image-level split — template: {len(ds.template_images)}, "
              f"calibration: {len(ds.calibration_images)}, "
              f"test: {len(ds.test_images)}")

    print("Training pipeline...")
    pipeline = BubblePipeline(cfg)
    pipeline.train(ds)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save NCC maps for up to 2 calibration images
    ncc_imgs, ncc_names = [], []
    for p in ds.calibration_images[:2]:
        ncc_imgs.append(load_image(p))
        ncc_names.append(p.stem[:40])  # truncate long stems

    pipeline.save(args.output,
                  ncc_images=ncc_imgs if ncc_imgs else None,
                  ncc_names=ncc_names if ncc_names else None)
    print(f"Pipeline saved to {args.output}")
    print(f"Artifacts written alongside: {args.output.parent}/")


if __name__ == "__main__":
    main()
