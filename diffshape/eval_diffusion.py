# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.preprocess import process_scan
from diffshape.inference import (
    EvalDataset,
    ddim_sample_k,
    load_checkpoint,
    load_stats,
)
from diffshape.train_diffusion import (
    convert2GI_fast,
    ensure_square_point_count,
    load_processed_dataset,
    normalize_radius,
    resolve_splits,
)
from models.SDmodels import ConditionalDiffusionModel_DiT_v2, DiffusionSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diffusion checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--k-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_diffusion"))
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.002)
    parser.add_argument("--n-pc", type=int, default=4096)
    parser.add_argument("--active-folds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.0,
        help="Centered-cube mask ratio for robustness test (0.0-1.0)",
    )
    return parser.parse_args()


def apply_center_mask(images: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if mask_ratio <= 0.0:
        return images
    masked = images.clone()
    D = masked.shape[-3]
    mask_size = int((mask_ratio ** (1.0 / 3.0)) * D)
    start = (D - mask_size) // 2
    end = start + mask_size
    masked[..., start:end, start:end, start:end] = 0
    return masked


def get_eval_indices(
    dataset_name: str, data: dict, active_folds: list[int] | None
) -> np.ndarray:
    n_cases = len(data["gi"])
    test_indices = resolve_splits(dataset_name, data["splits"], active_folds).get(
        "test", np.array([], dtype=np.int64)
    )
    if test_indices.size == 0:
        return np.arange(n_cases, dtype=np.int64)
    return test_indices


def prepare_dataset(
    data: dict, indices: np.ndarray, n_patch: int, min_r: float, max_r: float
) -> EvalDataset:
    gi_centered = data["gi"] - np.asarray(data["fixed_center"], dtype=np.float32)
    gi_4d = convert2GI_fast(gi_centered, n_patch)
    gi_4d[:, :, :3] += np.asarray(data["fixed_center"], dtype=np.float32)
    gi_4d = normalize_radius(gi_4d, min_r, max_r)
    return EvalDataset(
        data["image_paths"][indices], gi_4d[indices], data["centers"][indices]
    )


def radius_to_cartesian(
    r_norm: torch.Tensor,
    fixed_center: torch.Tensor,
    min_r: float,
    max_r: float,
    n_patch: int,
) -> torch.Tensor:
    r = r_norm * (max_r - min_r) + min_r
    r = r.view(-1, n_patch, n_patch)
    phi_edges = torch.linspace(0, torch.pi, n_patch + 1, device=r.device, dtype=r.dtype)
    theta_edges = torch.linspace(
        -torch.pi, torch.pi, n_patch + 1, device=r.device, dtype=r.dtype
    )
    phi = (phi_edges[:-1] + phi_edges[1:]) * 0.5
    theta = (theta_edges[:-1] + theta_edges[1:]) * 0.5
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="xy")
    sin_phi = torch.sin(phi_grid).unsqueeze(0)
    x = r * sin_phi * torch.cos(theta_grid).unsqueeze(0)
    y = r * sin_phi * torch.sin(theta_grid).unsqueeze(0)
    z = r * torch.cos(phi_grid).unsqueeze(0)
    xyz = torch.stack([x, y, z], dim=-1).view(r_norm.size(0), -1, 3)
    return xyz + fixed_center.view(1, 1, 3)


def chamfer_per_case(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred, gt, p=2)
    return dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)


def evaluate_dataset(
    dataset_name: str,
    loader: DataLoader,
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    fixed_center: np.ndarray,
    min_r: float,
    max_r: float,
    n_patch: int,
    k_samples: int,
    ddim_steps: int,
    output_dir: Path,
    mask_ratio: float = 0.0,
) -> dict:
    model.eval()
    k_values = [k for k in (1, 4, 8, 16) if k <= k_samples] or [k_samples]
    fixed_center_t = torch.as_tensor(
        fixed_center, device=next(model.parameters()).device, dtype=torch.float32
    )
    all_samples, chamfer_rows = [], {k: [] for k in k_values}
    rmse_sum = {k: 0.0 for k in k_values}
    n_total = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            images = batch["image"].to(fixed_center_t.device)
            images = apply_center_mask(images, mask_ratio)
            gt_radius = batch["radius"].to(fixed_center_t.device)
            gt_xyz = batch["gi_xyz"].to(fixed_center_t.device)
            samples = ddim_sample_k(
                model, schedule, images, gt_radius.size(1), k_samples, ddim_steps
            )
            all_samples.append(samples.permute(1, 0, 2).cpu().numpy())
            for k in k_values:
                pred_radius = samples[:k].mean(dim=0)
                rmse_sum[k] += torch.sum((pred_radius - gt_radius) ** 2).item()
                pred_xyz = radius_to_cartesian(
                    pred_radius, fixed_center_t, min_r, max_r, n_patch
                )
                chamfer_rows[k].append(chamfer_per_case(pred_xyz, gt_xyz).cpu())
            n_total += gt_radius.numel()
    all_samples_arr = np.concatenate(all_samples, axis=0)
    chamfer_arr = np.stack(
        [torch.cat(chamfer_rows[k]).numpy() for k in k_values], axis=0
    )
    suffix = f"_m{int(round(mask_ratio * 100)):02d}" if mask_ratio > 0 else ""
    np.save(output_dir / f"all_samples_{dataset_name}{suffix}.npy", all_samples_arr)
    np.save(output_dir / f"chamfer_per_sample_{dataset_name}{suffix}.npy", chamfer_arr)
    metrics = {
        str(k): {
            "rmse_x100": float(np.sqrt(rmse_sum[k] / n_total) * 100.0),
            "chamfer_mean": float(chamfer_arr[i].mean()),
            "chamfer_std": float(chamfer_arr[i].std()),
        }
        for i, k in enumerate(k_values)
    }
    print(
        f"{dataset_name:<8} | "
        + " | ".join(
            [
                f"K={k}: RMSE {metrics[str(k)]['rmse_x100']:.3f}, CD {metrics[str(k)]['chamfer_mean']:.5f}"
                for k in k_values
            ]
        )
    )
    return {
        "num_cases": int(all_samples_arr.shape[0]),
        "k_values": k_values,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_patch = ensure_square_point_count(args.n_pc)
    stats = load_stats(args.checkpoint)
    normal_min_r = cast(float, stats["normal_min_r"])
    normal_max_r = cast(float, stats["normal_max_r"])
    fixed_center = cast(np.ndarray, stats["fixed_center"])
    model = ConditionalDiffusionModel_DiT_v2(radius_dim=args.n_pc).to(device)
    load_checkpoint(model, args.checkpoint, device)
    schedule = DiffusionSchedule(
        args.timesteps, args.beta_start, args.beta_end, device=device
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "stats_path": str(stats["stats_path"]),
        "processed_dir": str(args.processed_dir),
        "ddim_steps": args.ddim_steps,
        "k_samples": args.k_samples,
        "timesteps": args.timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "n_pc": args.n_pc,
        "active_folds": args.active_folds,
        "mask_ratio": args.mask_ratio,
        "datasets": {},
    }

    print("Dataset  | Metrics")
    print("-" * 100)
    for name in args.datasets:
        data = load_processed_dataset(args.processed_dir, name, args.n_pc)
        indices = get_eval_indices(name, data, args.active_folds)
        dataset = prepare_dataset(data, indices, n_patch, normal_min_r, normal_max_r)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        summary["datasets"][name] = evaluate_dataset(
            name,
            loader,
            model,
            schedule,
            fixed_center,
            normal_min_r,
            normal_max_r,
            n_patch,
            args.k_samples,
            args.ddim_steps,
            output_dir,
            mask_ratio=args.mask_ratio,
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
