from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(axis, x, y, width, height, title, text, color):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.02", fc=color, ec="#273746", lw=1.0)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height * 0.69, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
    axis.text(x + width / 2, y + height * 0.36, text, ha="center", va="center", fontsize=6.5, linespacing=1.35)


def arrow(axis, x0, y0, x1, y1, dashed=False):
    axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#34495e", linestyle="--" if dashed else "-"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.15, 2.75))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    box(axis, 0.02, 0.52, 0.18, 0.34, "Observed enterprise network", "Pre-event authentication graph\nSparse auth and DNS windows", "#edf3f7")
    box(axis, 0.27, 0.52, 0.19, 0.34, "Endpoint-choice prior", "Fit Boltzmann parameter\nEstimate Fisher curvature", "#f8f0e8")
    box(axis, 0.53, 0.52, 0.19, 0.34, "Temporal source model", "Candidate posterior\nPosterior ambiguity", "#ecf5f2")
    box(axis, 0.79, 0.52, 0.19, 0.34, "Audited controller", "Expected uncertainty reduction\nper telemetry cost", "#f8ebed")
    box(axis, 0.34, 0.08, 0.31, 0.23, "Ranked decision and audit", "Source posterior, reliability, entropy\nQuery trace and containment shortlist", "#f4f6f7")
    arrow(axis, 0.20, 0.69, 0.27, 0.69)
    arrow(axis, 0.46, 0.69, 0.53, 0.69)
    arrow(axis, 0.72, 0.69, 0.79, 0.69)
    arrow(axis, 0.885, 0.52, 0.64, 0.31)
    arrow(axis, 0.55, 0.52, 0.55, 0.31)
    arrow(axis, 0.36, 0.52, 0.48, 0.31, dashed=True)
    axis.text(0.365, 0.42, "bounded structural residual", fontsize=6.2, color="#6c3d44", rotation=35)
    axis.text(0.75, 0.40, "host and modality query", fontsize=6.2, color="#34495e", rotation=-34)
    axis.text(0.02, 0.93, "Curvature-Audited Active Telemetry", fontsize=11, fontweight="bold", color="#1f2d3d")
    axis.text(0.02, 0.89, "Structural evidence guides acquisition only when the endpoint-choice prior is locally identifiable.", fontsize=7, color="#566573")
    fig.tight_layout(pad=0.2)
    fig.savefig(args.output, dpi=350, bbox_inches="tight")


if __name__ == "__main__":
    main()
