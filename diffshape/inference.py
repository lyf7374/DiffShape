# pyright: reportMissingImports=false
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from diffshape.preprocess import process_scan
from diffshape.models import DiffusionSchedule


class EvalDataset(Dataset):
    def __init__(self, image_paths: np.ndarray, gi: np.ndarray, centers: np.ndarray):
        self.image_paths = image_paths
        self.gi = gi
        self.centers = centers

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = process_scan(
            str(self.image_paths[idx]),
            norm_method="zs",
            output_shape=(192, 192, 192),
            center=self.centers[idx],
        ).astype(np.float32)
        return {
            "image": torch.from_numpy(image[None, ...]),
            "radius": torch.from_numpy(self.gi[idx, :, 3].astype(np.float32)),
            "gi_xyz": torch.from_numpy(self.gi[idx, :, :3].astype(np.float32)),
        }


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and state and next(iter(state)).startswith("module."):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)


def load_stats(checkpoint_path: Path) -> dict[str, object]:
    stats_path = checkpoint_path.with_suffix("")
    stats_path = stats_path.parent / f"{stats_path.name}_stats.npz"
    stats = np.load(stats_path)
    return {
        "stats_path": stats_path,
        "normal_min_r": float(stats["normal_min_r"]),
        "normal_max_r": float(stats["normal_max_r"]),
        "r_mean": stats["r_mean"],
        "r_std": stats["r_std"],
        "fixed_center": stats["fixed_center"].astype(np.float32),
    }


def ddim_sample_k(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    n_pc: int,
    k: int,
    ddim_steps: int,
) -> torch.Tensor:
    steps = np.linspace(schedule.timesteps - 1, 0, ddim_steps, dtype=int)
    batch = images.size(0)
    images_k = images.repeat(k, 1, 1, 1, 1)
    x_t = torch.cat(
        [torch.randn(batch, n_pc, device=images.device) for _ in range(k)], dim=0
    )
    for t_cur, t_next in zip(steps[:-1], steps[1:]):
        t = torch.full((batch * k,), int(t_cur), device=images.device, dtype=torch.long)
        eps = model(x_t, images_k, t)
        a_t = schedule.alpha_bar[int(t_cur)]
        a_s = schedule.alpha_bar[int(t_next)]
        term1 = (x_t - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
        x_t = torch.sqrt(a_s) * term1 + torch.sqrt(1.0 - a_s) * eps
    return x_t.view(k, batch, n_pc)
