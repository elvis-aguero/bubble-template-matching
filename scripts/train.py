#!/usr/bin/env python3
"""Fit bubble histogram pipeline on annotated data and save to disk."""
import argparse
from pathlib import Path

import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset, load_image
from bubble_histogram.histogram import plot_histogram
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

    # Save NCC map + size histogram (with empirical overlay) for each test image
    test_paths = getattr(ds, "test_images", [])
    if test_paths:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for p in test_paths:
            sample = ds.load_sample(p)
            result = pipeline.predict(sample.image)

            # NCC score map — prefixed with TEST_ so it's clearly distinct
            ncc_out = args.output.with_name(
                f"{args.output.stem}_ncc_TEST_{p.stem[:40]}.png"
            )
            pipeline.save_ncc_png(ncc_out, sample.image)
            print(f"Test NCC map saved to {ncc_out}")

            # Bin annotated radii into the same pyramid-level bins
            radii = np.array(result["radius_px"])
            log_r = np.log(radii)
            half_step = (log_r[1] - log_r[0]) / 2 if len(log_r) > 1 else 0.1
            edges = np.exp(np.concatenate([
                [log_r[0] - half_step],
                (log_r[:-1] + log_r[1:]) / 2,
                [log_r[-1] + half_step],
            ]))
            ann_radii = np.array([b.radius for b in sample.bubbles])
            empirical_counts, _ = np.histogram(ann_radii, bins=edges)

            fig, ax = plt.subplots(figsize=(8, 4))
            plot_histogram(result, ax=ax,
                           title=f"Bubble size histogram — {p.stem}",
                           empirical_counts=empirical_counts)
            hist_out = args.output.with_name(
                f"{args.output.stem}_size_hist_{p.stem[:40]}.png"
            )
            fig.tight_layout()
            fig.savefig(hist_out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Size histogram saved to {hist_out}")


if __name__ == "__main__":
    main()
