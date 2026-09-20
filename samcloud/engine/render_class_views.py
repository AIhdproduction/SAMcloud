"""Render lightweight top and side previews of a classified binary PLY."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


CLASS_NAMES = [
    "Unknown/noise",
    "Ground",
    "Road",
    "Sidewalk",
    "Curb",
    "Low vegetation",
    "Tree",
    "Building",
    "Vehicle",
    "Construction machinery",
    "Utility pole/light",
    "Traffic sign",
    "Fence/barrier",
    "Power line/cable",
    "Scaffolding",
    "Container/material",
    "Water",
]

CLASS_COLORS = [
    "#8c8c8c", "#a6611a", "#303030", "#f4a261", "#ffd166", "#76c893",
    "#238b45", "#d73027", "#4575b4", "#6a3d9a", "#00a6d6", "#ff1493",
    "#8b4513", "#7b2cbf", "#bdb2ff", "#f77f00", "#00b4d8",
]


def read_header(path):
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Invalid PLY header")
            decoded = line.decode("ascii").strip()
            if decoded.startswith("element vertex "):
                vertex_count = int(decoded.split()[-1])
            if decoded == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY has no vertex count")
        return handle.tell(), vertex_count


def plot_view(points, class_ids, x_axis, y_axis, title, output_path):
    fig, ax = plt.subplots(figsize=(15, 9), dpi=180)
    for class_id in range(len(CLASS_NAMES)):
        selected = class_ids == class_id
        if not np.any(selected):
            continue
        ax.scatter(
            points[x_axis][selected],
            points[y_axis][selected],
            s=0.35 if class_id else 0.15,
            c=CLASS_COLORS[class_id],
            alpha=0.65 if class_id else 0.18,
            linewidths=0,
            rasterized=True,
        )

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel(f"{x_axis.upper()} coordinate (m)")
    ax.set_ylabel(f"{y_axis.upper()} coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22, linewidth=0.5)

    present = np.unique(class_ids)
    legend = [
        Line2D([0], [0], marker="o", color="w", label=CLASS_NAMES[class_id],
               markerfacecolor=CLASS_COLORS[class_id], markersize=7)
        for class_id in present
    ]
    ax.legend(handles=legend, title="Class", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-points", type=int, default=500_000)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    offset, count = read_header(input_path)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("scalar_class", "u1"),
    ])
    cloud = np.memmap(input_path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    step = max(1, int(np.ceil(count / args.max_points)))
    sampled = cloud[::step]
    class_ids = sampled["scalar_class"]

    plot_view(sampled, class_ids, "y", "x", "Classified point cloud: top view", outdir / "classes_top.png")
    plot_view(sampled, class_ids, "y", "z", "Classified point cloud: side view", outdir / "classes_side.png")
    print(f"Rendered {len(sampled):,} sampled points from {count:,} total points")


if __name__ == "__main__":
    main()
