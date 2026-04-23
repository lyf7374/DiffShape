from __future__ import annotations

from pathlib import Path

import mcubes
import numpy as np
import trimesh

from diffshape.preprocess import process_scan

FIXED_CENTER = np.array([96.0, 96.0, 96.0], dtype=np.float64)


def sample_point_cloud_by_angular_grid(
    vertices: np.ndarray,
    center: np.ndarray,
    n_regions_phi: int,
    n_regions_theta: int,
    cartesian: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Angular-bin sampling: assign vertices to (phi, theta) bins, pick max-radius per bin,
    interpolate empty bins from neighbors. Returns (points, empty_count, iterations)."""
    translated = np.asarray(vertices, dtype=np.float64) - np.asarray(
        center, dtype=np.float64
    )
    radii = np.linalg.norm(translated, axis=1)
    safe_radii = np.where(radii == 0, 1e-12, radii)
    theta = np.arctan2(translated[:, 1], translated[:, 0])
    phi = np.arccos(np.clip(translated[:, 2] / safe_radii, -1.0, 1.0))

    phi_bins = np.linspace(0, np.pi, n_regions_phi + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_regions_theta + 1)
    phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing="xy")

    phi_idx = np.clip(
        np.searchsorted(phi_bins, phi, side="right") - 1, 0, n_regions_phi - 1
    )
    theta_idx = np.clip(
        np.searchsorted(theta_bins, theta, side="right") - 1, 0, n_regions_theta - 1
    )
    flat_idx = phi_idx * n_regions_theta + theta_idx

    order = np.lexsort((-radii, flat_idx))
    flat_sorted = flat_idx[order]
    unique_bins, first_positions = np.unique(flat_sorted, return_index=True)
    chosen_vertex_idx = order[first_positions]

    n_bins = n_regions_phi * n_regions_theta
    selected_vertex_idx = np.full(n_bins, -1, dtype=np.int64)
    selected_vertex_idx[unique_bins] = chosen_vertex_idx
    radius_grid = np.full((n_regions_phi, n_regions_theta), np.nan, dtype=np.float64)
    radius_grid.flat[unique_bins] = radii[chosen_vertex_idx]

    def shift_grid(grid: np.ndarray, phi_shift: int, theta_shift: int) -> np.ndarray:
        shifted = np.full_like(grid, np.nan, dtype=np.float64)
        if phi_shift >= 0:
            src_phi = slice(0, grid.shape[0] - phi_shift)
            dst_phi = slice(phi_shift, grid.shape[0])
        else:
            src_phi = slice(-phi_shift, grid.shape[0])
            dst_phi = slice(0, grid.shape[0] + phi_shift)
        shifted[dst_phi, :] = grid[src_phi, :]
        if theta_shift != 0:
            shifted = np.roll(shifted, shift=theta_shift, axis=1)
        return shifted

    interpolated_grid = radius_grid.copy()
    empty_count = int(np.isnan(interpolated_grid).sum())
    iterations = 0

    if np.all(np.isnan(interpolated_grid)):
        raise RuntimeError("No mesh vertices assigned to any angular bin.")

    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    while np.isnan(interpolated_grid).any():
        nan_mask = np.isnan(interpolated_grid)
        neighbor_sum = np.zeros_like(interpolated_grid)
        neighbor_count = np.zeros_like(interpolated_grid, dtype=np.int64)
        for ps, ts in neighbor_offsets:
            shifted = shift_grid(interpolated_grid, ps, ts)
            valid = ~np.isnan(shifted)
            neighbor_sum += np.where(valid, shifted, 0.0)
            neighbor_count += valid.astype(np.int64)
        fillable = nan_mask & (neighbor_count > 0)
        if not np.any(fillable):
            raise RuntimeError(
                "Interpolation stalled: no empty bins have valid neighbors."
            )
        interpolated_grid[fillable] = neighbor_sum[fillable] / neighbor_count[fillable]
        iterations += 1

    selected_points = np.empty((n_bins, 3), dtype=np.float64)
    occupied_mask = selected_vertex_idx >= 0
    if cartesian:
        selected_points[occupied_mask] = np.asarray(vertices, dtype=np.float64)[
            selected_vertex_idx[occupied_mask]
        ]
        empty_flat = np.flatnonzero(~occupied_mask)
        if empty_flat.size:
            phi_empty = phi_grid.ravel()[empty_flat]
            theta_empty = theta_grid.ravel()[empty_flat]
            radius_empty = interpolated_grid.ravel()[empty_flat]
            x = radius_empty * np.sin(phi_empty) * np.cos(theta_empty)
            y = radius_empty * np.sin(phi_empty) * np.sin(theta_empty)
            z = radius_empty * np.cos(phi_empty)
            selected_points[empty_flat] = np.column_stack((x, y, z)) + np.asarray(
                center, dtype=np.float64
            )
    else:
        selected_points[:, 0] = interpolated_grid.ravel()
        selected_points[:, 1] = theta_grid.ravel()
        selected_points[:, 2] = phi_grid.ravel()

    return selected_points, empty_count, iterations


def extract_gi_single(
    mask_path: str | Path,
    crop_center: np.ndarray,
    crop_shape: tuple[int, int, int],
    n_patch: int,
    mesh_output_dir: Path | None = None,
    case_id: str = "",
) -> np.ndarray:
    """Extract GI (Geometric Index) point cloud from a single brain mask.
    Returns (n_patch*n_patch, 3) cartesian points."""
    mask_vol = process_scan(
        str(mask_path),
        mask=True,
        output_shape=crop_shape,
        center=crop_center,
    )
    vertices, triangles = mcubes.marching_cubes(mask_vol, 0)

    if mesh_output_dir is not None:
        mesh_output_dir = Path(mesh_output_dir)
        mesh_output_dir.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        mesh.export(mesh_output_dir / f"{case_id}.obj")

    points, _, _ = sample_point_cloud_by_angular_grid(
        vertices, FIXED_CENTER, n_patch, n_patch, cartesian=True
    )
    return points


def extract_gi_batch(
    mask_paths: list[str | Path],
    crop_centers: np.ndarray,
    crop_shape: tuple[int, int, int],
    n_patch: int,
    mesh_output_dir: Path | None = None,
    case_ids: list[str] | None = None,
) -> np.ndarray:
    """Extract GI for multiple cases. Returns (N, n_patch*n_patch, 3)."""
    n_cases = len(mask_paths)
    if case_ids is None:
        case_ids = [f"case_{i:04d}" for i in range(n_cases)]

    all_points = []
    for i in range(n_cases):
        points = extract_gi_single(
            mask_paths[i],
            crop_centers[i],
            crop_shape,
            n_patch,
            mesh_output_dir,
            case_ids[i],
        )
        all_points.append(points)
    return np.array(all_points)
