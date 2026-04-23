# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm


def radius_to_xyz(r_samples: np.ndarray, n_patch: int) -> np.ndarray:
    phi = np.linspace(0, np.pi, n_patch + 1)
    theta = np.linspace(-np.pi, np.pi, n_patch + 1)
    phi_c = 0.5 * (phi[:-1] + phi[1:])
    theta_c = 0.5 * (theta[:-1] + theta[1:])
    tg, pg = np.meshgrid(theta_c, phi_c)
    tf, pf = tg.ravel(), pg.ravel()
    r = r_samples
    x = r * np.sin(pf) * np.cos(tf)
    y = r * np.sin(pf) * np.sin(tf)
    z = r * np.cos(pf)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def signed_distance_volume(
    points: np.ndarray, center: np.ndarray, grid: tuple[int, int, int]
) -> np.ndarray:
    zz, yy, xx = np.indices(grid, dtype=np.float32)
    grid_pts = np.stack((zz, yy, xx), -1).reshape(-1, 3)
    tree = cKDTree(points)
    dist, idx = tree.query(grid_pts)
    g_r = np.linalg.norm(grid_pts - center, axis=1)
    s_r = np.linalg.norm(points[idx] - center, axis=1)
    sign = np.where(g_r < s_r, -1.0, 1.0)
    return (dist * sign).reshape(grid).astype(np.float32)


def sdt_stats(
    point_clouds: np.ndarray,
    center: np.ndarray,
    grid: tuple[int, int, int],
    smooth: float,
) -> tuple[np.ndarray, np.ndarray]:
    sdts = np.stack(
        [
            signed_distance_volume(
                pc.astype(np.float32), center.astype(np.float32), grid
            )
            for pc in point_clouds
        ]
    )
    mu, var = sdts.mean(0), sdts.var(0)
    if smooth > 0:
        mu = gaussian_filter(mu, sigma=smooth)
        var = gaussian_filter(var, sigma=smooth)
    return mu.astype(np.float32), var.astype(np.float32)


def variance_to_confidence(
    var: np.ndarray, blur: float, tau: float | None
) -> np.ndarray:
    if blur > 0:
        var = gaussian_filter(var, sigma=blur)
    if tau is None:
        nz = var[var > 0]
        tau = float(np.median(nz)) if nz.size else 1.0
    tau = max(tau, 1e-6)
    return np.exp(-var / tau).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert diffusion samples to per-case (mu_sdt, w_conf) priors"
    )
    p.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="samples_<dataset>_K<k>.npy from predict.py",
    )
    p.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="meta_<dataset>_K<k>.json from predict.py",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--grid-size", type=int, default=192)
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--var-blur", type=float, default=2.0)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--format", choices=["h5", "nii", "both"], default="h5")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    samples = np.load(args.samples)
    n_cases, k, n_pc = samples.shape
    n_patch = int(meta["n_patch"])
    min_r, max_r = float(meta["normal_min_r"]), float(meta["normal_max_r"])
    center = np.asarray(meta["fixed_center"], dtype=np.float32)
    grid = (args.grid_size, args.grid_size, args.grid_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_cases), desc="sdt"):
        r = samples[i] * (max_r - min_r) + min_r
        xyz_samples = np.stack(
            [radius_to_xyz(r[j], n_patch) + center for j in range(k)]
        )
        mu, var = sdt_stats(xyz_samples, center, grid, args.smooth_sigma)
        w = variance_to_confidence(var, args.var_blur, args.tau)
        if args.format in ("h5", "both"):
            with h5py.File(args.output_dir / f"{i:04d}.h5", "w") as h:
                h.create_dataset("mu_sdt", data=mu, compression="gzip")
                h.create_dataset("w_conf", data=w, compression="gzip")
                h.attrs["source_index"] = int(meta["indices"][i])
                h.attrs["image_path"] = str(meta["image_paths"][i])
        if args.format in ("nii", "both"):
            affine = np.eye(4, dtype=np.float32)
            nib.save(
                nib.Nifti1Image(mu, affine),
                str(args.output_dir / f"{i:04d}_mu_sdt.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(w, affine),
                str(args.output_dir / f"{i:04d}_w_conf.nii.gz"),
            )
    print(f"Wrote {n_cases} priors → {args.output_dir}")


if __name__ == "__main__":
    main()
