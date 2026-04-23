# pyright: reportMissingImports=false
"""Visualize predicted point clouds from predict.py as 3D scatter plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffshape.samples_to_sdt import radius_to_xyz


def plot_case(
    xyz_k: np.ndarray,
    title: str,
    out_path: Path,
    point_size: float,
    elev: float,
    azim: float,
) -> None:
    k = xyz_k.shape[0]
    cols = min(k, 4)
    rows = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for j in range(k):
        ax = fig.add_subplot(rows, cols, j + 1, projection="3d")
        p = xyz_k[j]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=point_size, c=p[:, 2], cmap="viridis")
        ax.set_title(f"sample {j}")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mean(
    xyz_mean: np.ndarray,
    title: str,
    out_path: Path,
    point_size: float,
    elev: float,
    azim: float,
) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz_mean[:, 0],
        xyz_mean[:, 1],
        xyz_mean[:, 2],
        s=point_size,
        c=xyz_mean[:, 2],
        cmap="viridis",
    )
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    xyz_k: np.ndarray,
    title: str,
    out_path: Path,
    point_size: float,
    elev: float,
    azim: float,
) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    for j, p in enumerate(xyz_k):
        ax.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            s=point_size,
            alpha=0.35,
            color=cmap(j % 10),
            label=f"s{j}",
        )
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize predicted point clouds")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--cases",
        type=int,
        nargs="*",
        default=None,
        help="Case indices within samples array (default: all)",
    )
    p.add_argument("--mode", choices=["mean", "grid", "overlay", "all"], default="mean")
    p.add_argument("--point-size", type=float, default=1.0)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=45.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    samples = np.load(args.samples)
    n_cases, k, _ = samples.shape
    n_patch = int(meta["n_patch"])
    min_r, max_r = float(meta["normal_min_r"]), float(meta["normal_max_r"])
    center = np.asarray(meta["fixed_center"], dtype=np.float32)
    cases = list(range(n_cases)) if args.cases is None else args.cases
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i in cases:
        r = samples[i] * (max_r - min_r) + min_r
        xyz_k = np.stack([radius_to_xyz(r[j], n_patch) + center for j in range(k)])
        src_idx = int(meta["indices"][i])
        title = f"case {i} (src={src_idx}) K={k}"
        if args.mode in ("mean", "all"):
            xyz_mean = radius_to_xyz(r.mean(0), n_patch) + center
            plot_mean(
                xyz_mean,
                title + " mean",
                args.output_dir / f"{i:04d}_mean.png",
                args.point_size,
                args.elev,
                args.azim,
            )
        if args.mode in ("grid", "all"):
            plot_case(
                xyz_k,
                title,
                args.output_dir / f"{i:04d}_grid.png",
                args.point_size,
                args.elev,
                args.azim,
            )
        if args.mode in ("overlay", "all"):
            plot_overlay(
                xyz_k,
                title,
                args.output_dir / f"{i:04d}_overlay.png",
                args.point_size,
                args.elev,
                args.azim,
            )
    print(f"Wrote {len(cases)} visualizations → {args.output_dir}")


if __name__ == "__main__":
    main()
