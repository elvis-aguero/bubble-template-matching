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
