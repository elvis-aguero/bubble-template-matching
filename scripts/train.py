#!/usr/bin/env python3
"""
Train the bubble size histogram pipeline on annotated images and save all artifacts.

USAGE
-----
  python scripts/train.py <data_dir> <output.pkl> [options]

WHAT IT DOES
------------
  1. Loads annotated images from <data_dir>/images/ and <data_dir>/labels/.
  2. Splits them into three non-overlapping sets:
       template    (default 30%) — bubble patches averaged into the NCC template
       calibration (default 65%) — NCC scores used to fit P(bubble|score)
       test        (default  5%) — held-out images never seen during training
  3. Trains a BubblePipeline: builds templates, fits the Bayesian calibrator,
     estimates the bubble prior from calibration images.
  4. Saves <output.pkl> plus diagnostic artifacts in the same directory:
       *_templates.png          — the learned template(s); should show a dark disc
       *_score_histograms.png   — P(score|bubble) vs P(score|background); should be separated
       *_split.json             — exact image-to-split assignment for reproducibility
       *_ncc_TEST_<name>.png    — NCC score map for the test image at the most active scale
       *_top_matches_TEST_<name>.png  — top-100 NCC peaks after 3D NMS, drawn as bounding boxes
       *_size_hist_<name>.png   — predicted size histogram vs annotated ground truth

QUICK START
-----------
  # Basic run (random 30/65/5 split, seed=42)
  python scripts/train.py seed_v04/ output/pipeline.pkl

  # Reproducible split with a specific seed
  python scripts/train.py seed_v04/ output/pipeline.pkl --split-seed 7

  # Leave-one-session-out (for cross-validation; available sessions: C1S0004 C1S0010 C1S0014 C1S0019 C1S0024)
  python scripts/train.py seed_v04/ output/pipeline_no_C1S0010.pkl --val-session C1S0010

  # Load the saved pipeline later
  from bubble_histogram.pipeline import BubblePipeline
  pipeline = BubblePipeline.load("output/pipeline.pkl")
  result = pipeline.predict(image)  # {"radius_px": [...], "expected_count": [...]}
"""
import argparse
from pathlib import Path

import numpy as np

from bubble_histogram.config import PipelineConfig
from bubble_histogram.data import AnnotatedDataset
from bubble_histogram.histogram import plot_histogram
from bubble_histogram.pipeline import BubblePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train the bubble histogram pipeline and save artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_dir", type=Path,
                        help="Root of annotated dataset (must contain images/ and labels/ subdirs)")
    parser.add_argument("output", type=Path,
                        help="Where to save the trained pipeline (.pkl). Artifacts are written alongside it.")

    split = parser.add_argument_group("data split")
    split.add_argument("--val-session", default=None,
                       help="Leave-one-session-out mode: hold out this session ID (e.g. C1S0010). "
                            "Overrides the random image-level split.")
    split.add_argument("--template-frac", type=float, default=0.30,
                       help="Fraction of images used for template construction (default 0.30)")
    split.add_argument("--calibration-frac", type=float, default=0.65,
                       help="Fraction of images used for Bayesian calibration (default 0.65). "
                            "Remaining images become the test set.")
    split.add_argument("--split-seed", type=int, default=42,
                       help="Random seed for the image-level split (default 42)")

    _d = PipelineConfig()  # pull defaults from one place
    tmpl = parser.add_argument_group("template / pyramid")
    tmpl.add_argument("--num-templates", type=int, default=_d.num_templates,
                      help=f"Number of appearance templates (size bins). Default {_d.num_templates} pools all bubbles.")
    tmpl.add_argument("--template-size", type=int, default=_d.template_size,
                      help=f"Template side length in pixels (default {_d.template_size}). "
                           "Each bubble patch is resized to this before averaging.")
    tmpl.add_argument("--scale-factor", type=float, default=_d.scale_factor,
                      help=f"Pyramid downscale factor per level (default {_d.scale_factor}). "
                           "Smaller = finer size resolution but more levels.")
    tmpl.add_argument("--min-radius", type=float, default=_d.min_radius,
                      help=f"Smallest bubble radius to detect in original image pixels (default {_d.min_radius})")
    tmpl.add_argument("--max-radius", type=float, default=_d.max_radius,
                      help=f"Largest bubble radius to detect in original image pixels (default {_d.max_radius})")
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
    pipeline.save(args.output)
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

            # Top-100 3D-NMS matches with bounding boxes
            matches_out = args.output.with_name(
                f"{args.output.stem}_top_matches_TEST_{p.stem[:40]}.png"
            )
            pipeline.save_top_matches_png(matches_out, sample.image, top_n=100)
            print(f"Top matches saved to {matches_out}")

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
