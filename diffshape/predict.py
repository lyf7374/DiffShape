# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

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
)
from diffshape.models import ConditionalDiffusionModel_DiT_v2, DiffusionSchedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run diffusion prediction on a processed dataset"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name matching processed_dataset/<name>/",
    )
    p.add_argument("--processed-dir", type=Path, default=Path("processed_dataset"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k-samples", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Specific case indices (default: all)",
    )
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.002)
    p.add_argument("--n-pc", type=int, default=4096)
    return p.parse_args()


def select_indices(n: int, indices: list[int] | None) -> np.ndarray:
    if indices is None:
        return np.arange(n, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError(f"Indices out of range [0, {n})")
    return idx


def main() -> None:
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_patch = ensure_square_point_count(args.n_pc)

    stats = load_stats(args.checkpoint)
    min_r = cast(float, stats["normal_min_r"])
    max_r = cast(float, stats["normal_max_r"])
    fixed_center = cast(np.ndarray, stats["fixed_center"])

    model = ConditionalDiffusionModel_DiT_v2(radius_dim=args.n_pc).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    schedule = DiffusionSchedule(
        args.timesteps, args.beta_start, args.beta_end, device=device
    )

    data = load_processed_dataset(args.processed_dir, args.dataset, args.n_pc)
    indices = select_indices(len(data["gi"]), args.indices)
    if indices.size == 0:
        raise ValueError("No cases selected")

    gi_centered = data["gi"] - np.asarray(fixed_center, dtype=np.float32)
    gi_4d = convert2GI_fast(gi_centered, n_patch)
    gi_4d[:, :, :3] += np.asarray(fixed_center, dtype=np.float32)
    gi_4d = normalize_radius(gi_4d, min_r, max_r)
    dataset = EvalDataset(
        data["image_paths"][indices], gi_4d[indices], data["centers"][indices]
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_samples: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            samples = ddim_sample_k(
                model, schedule, images, args.n_pc, args.k_samples, args.ddim_steps
            )
            all_samples.append(samples.permute(1, 0, 2).cpu().numpy())
    samples_arr = np.concatenate(all_samples, axis=0)

    np.save(
        args.output_dir / f"samples_{args.dataset}_K{args.k_samples}.npy", samples_arr
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "k_samples": args.k_samples,
        "ddim_steps": args.ddim_steps,
        "n_pc": args.n_pc,
        "n_patch": n_patch,
        "indices": indices.tolist(),
        "normal_min_r": min_r,
        "normal_max_r": max_r,
        "fixed_center": fixed_center.tolist(),
        "image_paths": [str(p) for p in data["image_paths"][indices]],
    }
    with open(
        args.output_dir / f"meta_{args.dataset}_K{args.k_samples}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, indent=2)
    print(f"Saved samples: shape={samples_arr.shape} → {args.output_dir}")


if __name__ == "__main__":
    main()
