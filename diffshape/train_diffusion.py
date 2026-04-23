# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

from models.SDmodels import ConditionalDiffusionModel_DiT_v2, DiffusionSchedule
from datasets.preprocess import process_scan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion training")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--variant", type=str, default="diff", choices=["diff", "diff_r", "diff_r_rm"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-pc", type=int, default=4096)
    parser.add_argument("--beta-start", type=float, default=0.0001)
    parser.add_argument("--beta-end", type=float, default=0.002)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--embedding-reg", action="store_true")
    parser.add_argument("--reg-coeff", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--model-save-path", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--active-folds", nargs="*", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def convert2GI_fast(GI: np.ndarray, n_patch: int) -> np.ndarray:
    phi_bins = np.linspace(0, np.pi, n_patch + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_patch + 1)
    phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers)
    angles = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

    r = np.sqrt((GI ** 2).sum(axis=-1))
    theta_a, phi_a = angles[:, 0], angles[:, 1]
    x = r * np.sin(phi_a) * np.cos(theta_a)
    y = r * np.sin(phi_a) * np.sin(theta_a)
    z = r * np.cos(phi_a)
    return np.stack([x, y, z, r], axis=-1)


def add_noise_batch(
    schedule: DiffusionSchedule,
    radius_vectors: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_ab = torch.sqrt(schedule.alpha_bar[t]).to(radius_vectors.device)
    sqrt_1m_ab = torch.sqrt(1 - schedule.alpha_bar[t]).to(radius_vectors.device)
    noise = torch.randn_like(radius_vectors)
    noisy = sqrt_ab[:, None] * radius_vectors + sqrt_1m_ab[:, None] * noise
    return noisy, noise


def random_mask(
    imgs: torch.Tensor,
    p_cut: float = 0.5,
    cut_range: tuple = (8, 32),
    margin: int = 16,
    noise_std: float = 0.05,
) -> torch.Tensor:
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)
        squeeze_back = True
    elif imgs.dim() == 5:
        squeeze_back = False
    else:
        raise ValueError("Expect (C,D,H,W) or (B,C,D,H,W)")

    imgs = imgs.clone()
    B, C, D, H, W = imgs.shape
    device = imgs.device

    for b in range(B):
        cut = torch.rand(1, device=device) < p_cut
        sz = torch.randint(cut_range[0], cut_range[1], (1,), device=device).item()
        if min(D, H, W) - 2 * margin - sz <= 0:
            continue

        def rand_coord(max_len):
            return torch.randint(margin, max_len - margin - sz, (1,), device=device).item()

        if cut:
            z, y, x = map(rand_coord, (D, H, W))
            imgs[b, :, z:z+sz, y:y+sz, x:x+sz] = torch.randn_like(imgs[b, :, z:z+sz, y:y+sz, x:x+sz]) * noise_std
        else:
            z1, y1, x1 = map(rand_coord, (D, H, W))
            patch = imgs[b, :, z1:z1+sz, y1:y1+sz, x1:x1+sz].clone()
            scale = torch.empty(1, device=device).uniform_(0.7, 1.3)
            patch = patch * scale
            z2, y2, x2 = map(rand_coord, (D, H, W))
            imgs[b, :, z2:z2+sz, y2:y2+sz, x2:x2+sz] = patch

    return imgs.squeeze(0) if squeeze_back else imgs


class ProcessedDataset(Dataset):
    skull_paths: list[str | Path]
    gi_all: np.ndarray
    centers_pre: np.ndarray
    img_shape: tuple[int, int, int] = (192, 192, 192)
    norm_method: str = "zs"

    def __init__(
        self,
        skull_paths: Sequence[str | Path],
        gi_all: np.ndarray,
        centers_pre: np.ndarray,
        img_shape: tuple[int, int, int] = (192, 192, 192),
        norm_method: str = "zs",
    ):
        self.skull_paths = list(skull_paths)
        self.gi_all = gi_all
        self.centers_pre = centers_pre
        self.img_shape = img_shape
        self.norm_method = norm_method

    def __len__(self) -> int:
        return len(self.skull_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = process_scan(
            self.skull_paths[idx],
            norm_method=self.norm_method,
            output_shape=self.img_shape,
            center=self.centers_pre[idx],
        )
        x = x.reshape((1, *self.img_shape)).astype(np.float32)
        radius_vector = self.gi_all[idx][:, 3].astype(np.float32)
        gi_xyz = self.gi_all[idx][:, :3].astype(np.float32)
        return {
            "image": torch.from_numpy(x),
            "radius_vector": torch.from_numpy(radius_vector),
            "GI_true": torch.from_numpy(gi_xyz),
        }


def _resolve_image_paths(raw_paths: np.ndarray, processed_dir: Path) -> np.ndarray:
    resolved = []
    processed_dir = processed_dir.resolve()
    marker = "processed_dataset" + os.sep
    for p in raw_paths:
        p = str(p)
        if os.path.isabs(p):
            idx = p.find(marker)
            if idx >= 0:
                p = str(processed_dir / p[idx + len("processed_dataset") + 1:])
        else:
            p = str(processed_dir / p)
        resolved.append(p)
    return np.array(resolved, dtype=str)


def load_processed_dataset(
    processed_dir: Path,
    dataset_name: str,
    n_pc: int,
) -> dict:
    ds_dir = processed_dir / dataset_name
    with open(ds_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    splits = {}
    for split_path in ds_dir.glob("split_*_indices.npy"):
        split_name = split_path.stem.replace("split_", "").replace("_indices", "")
        splits[split_name] = np.load(split_path)

    raw_paths = np.load(ds_dir / "image_paths.npy", allow_pickle=True)
    resolved_paths = _resolve_image_paths(raw_paths, processed_dir)

    return {
        "summary": summary,
        "case_ids": np.load(ds_dir / "case_ids.npy", allow_pickle=True),
        "centers": np.load(ds_dir / "centers.npy"),
        "gi": np.load(ds_dir / f"cGI_{dataset_name}_{n_pc}rpt_preC.npy"),
        "fixed_center": np.load(ds_dir / "fixed_sampling_center.npy"),
        "image_paths": resolved_paths,
        "splits": splits,
    }


def ensure_square_point_count(n_pc: int) -> int:
    n_patch = int(n_pc ** 0.5)
    if n_patch * n_patch != n_pc:
        raise ValueError(f"n_pc must be a perfect square, got {n_pc}")
    return n_patch


def format_float_tag(value: float) -> str:
    return np.format_float_positional(value, trim="-")


def build_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.model_save_path:
        model_path = Path(args.model_save_path)
    else:
        dataset_tag = "-".join(args.datasets)
        filename = f"{args.variant}_{dataset_tag}_T{args.timesteps}_b{format_float_tag(args.beta_end)}.pth"
        model_path = Path("checkpoints") / filename
    stats_path = model_path.with_suffix("")
    stats_path = stats_path.parent / f"{stats_path.name}_stats.npz"
    return model_path, stats_path


def build_gbm125_split_indices(active_folds: list[int]) -> dict[str, np.ndarray]:
    n_folds = 5
    fold_size = 25
    train_per_fold = 17
    test_per_fold = 8

    if any(fold < 0 or fold >= n_folds for fold in active_folds):
        raise ValueError(f"gbm125 active_folds must be within [0, {n_folds - 1}]")

    train_indices = []
    test_indices = []
    unseen_indices = []
    active_fold_set = set(active_folds)

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        fold_indices = list(range(start, start + fold_size))
        if fold_idx in active_fold_set:
            train_indices.extend(fold_indices[:train_per_fold])
            test_indices.extend(fold_indices[train_per_fold:train_per_fold + test_per_fold])
        else:
            unseen_indices.extend(fold_indices)

    return {
        "train": np.array(train_indices, dtype=np.int64),
        "test": np.array(test_indices, dtype=np.int64),
        "unseen": np.array(unseen_indices, dtype=np.int64),
    }


def resolve_splits(dataset_name: str, splits: dict[str, np.ndarray], active_folds: list[int] | None) -> dict[str, np.ndarray]:
    if dataset_name == "gbm125" and active_folds is not None:
        return build_gbm125_split_indices(active_folds)
    return splits


def normalize_radius(data: np.ndarray, normal_min_r: float, normal_max_r: float) -> np.ndarray:
    normalized = data.copy()
    normalized[:, :, 3] = (normalized[:, :, 3] - normal_min_r) / (normal_max_r - normal_min_r)
    return normalized


def build_loss_fn(variant: str, r_mean: np.ndarray, r_std: np.ndarray):
    if variant == "diff":
        return nn.MSELoss()

    def custom_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        weights = torch.as_tensor(r_mean * (1 + r_std) + eps, dtype=pred.dtype, device=pred.device)
        return torch.sqrt(torch.mean((pred - target) ** 2 / weights))

    return custom_loss


def save_checkpoint(
    model: nn.Module,
    model_path: Path,
    stats_path: Path,
    normal_min_r: float,
    normal_max_r: float,
    r_mean: np.ndarray,
    r_std: np.ndarray,
    fixed_center: np.ndarray,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, model_path)
    np.savez(
        stats_path,
        normal_min_r=np.asarray(normal_min_r, dtype=np.float32),
        normal_max_r=np.asarray(normal_max_r, dtype=np.float32),
        r_mean=np.asarray(r_mean, dtype=np.float32),
        r_std=np.asarray(r_std, dtype=np.float32),
        fixed_center=np.asarray(fixed_center, dtype=np.float32),
    )


def print_config(args: argparse.Namespace, model_path: Path, stats_path: Path) -> None:
    use_random_mask = args.variant == "diff_r_rm"
    print("Training configuration")
    print(f"  variant: {args.variant}")
    print(f"  datasets: {', '.join(args.datasets)}")
    print(f"  active_folds: {args.active_folds if args.active_folds is not None else 'saved splits'}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}")
    print(f"  seed: {args.seed}")
    print(f"  n_pc: {args.n_pc}")
    print(f"  timesteps: {args.timesteps}")
    print(f"  beta_start: {args.beta_start}")
    print(f"  beta_end: {args.beta_end}")
    print(f"  embedding_reg: {args.embedding_reg}")
    print(f"  reg_coeff: {args.reg_coeff}")
    print(f"  random_mask: {use_random_mask}")
    print("  optimizer: AdamW")
    print("  scheduler: CosineAnnealingLR")
    print("  eval_every: 10")
    print(f"  early_stop_patience: {args.early_stop_patience}")
    print(f"  model_save_path: {model_path}")
    print(f"  stats_save_path: {stats_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_patch = ensure_square_point_count(args.n_pc)
    model_path, stats_path = build_checkpoint_paths(args)
    print_config(args, model_path, stats_path)

    all_skull_paths_train: list[str] = []
    all_skull_paths_test: list[str] = []
    all_gi_train: list[np.ndarray] = []
    all_gi_test: list[np.ndarray] = []
    all_centers_train: list[np.ndarray] = []
    all_centers_test: list[np.ndarray] = []
    fixed_center: np.ndarray | None = None

    for dataset_name in args.datasets:
        data = load_processed_dataset(args.processed_dir, dataset_name, args.n_pc)
        dataset_fixed_center = np.asarray(data["fixed_center"], dtype=np.float32)
        if fixed_center is None:
            fixed_center = dataset_fixed_center
        elif not np.allclose(fixed_center, dataset_fixed_center):
            raise ValueError(f"Inconsistent fixed_center between datasets: {args.datasets}")

        gi_centered = data["gi"] - dataset_fixed_center
        gi_4d = convert2GI_fast(gi_centered, n_patch)
        gi_4d[:, :, :3] += dataset_fixed_center

        dataset_splits = resolve_splits(dataset_name, data["splits"], args.active_folds)
        for split_name, indices in dataset_splits.items():
            if split_name == "train":
                all_gi_train.extend(gi_4d[indices])
                all_centers_train.extend(data["centers"][indices])
                all_skull_paths_train.extend(data["image_paths"][indices])
            elif split_name == "test":
                all_gi_test.extend(gi_4d[indices])
                all_centers_test.extend(data["centers"][indices])
                all_skull_paths_test.extend(data["image_paths"][indices])

        train_count = len(dataset_splits.get("train", []))
        test_count = len(dataset_splits.get("test", []))
        unseen_count = len(dataset_splits.get("unseen", []))
        print(f"Loaded {dataset_name}: train={train_count}, test={test_count}, unseen={unseen_count}")

    if not all_gi_train:
        raise RuntimeError("No training data found. Check datasets and active_folds.")

    if fixed_center is None:
        raise RuntimeError("No dataset metadata loaded.")

    gi_train_arr = np.asarray(all_gi_train)
    gi_test_arr = np.asarray(all_gi_test) if all_gi_test else np.empty((0, args.n_pc, 4), dtype=gi_train_arr.dtype)
    centers_train_arr = np.asarray(all_centers_train)
    centers_test_arr = np.asarray(all_centers_test) if all_centers_test else np.empty((0, 3), dtype=centers_train_arr.dtype)

    normal_min_r = float(np.min(gi_train_arr[:, :, 3]))
    normal_max_r = float(np.max(gi_train_arr[:, :, 3]))
    gi_train_arr = normalize_radius(gi_train_arr, normal_min_r, normal_max_r)
    if gi_test_arr.shape[0] > 0:
        gi_test_arr = normalize_radius(gi_test_arr, normal_min_r, normal_max_r)

    r_mean = gi_train_arr[:, :, 3].mean(axis=0)
    r_std = gi_train_arr[:, :, 3].std(axis=0)
    train_criterion = build_loss_fn(args.variant, r_mean, r_std)

    print(f"Training cases: {len(all_skull_paths_train)}")
    print(f"Test cases: {len(all_skull_paths_test)}")
    print(f"Radius range: [{normal_min_r:.6f}, {normal_max_r:.6f}]")

    train_dataset = ProcessedDataset(all_skull_paths_train, gi_train_arr, centers_train_arr, norm_method="zs")
    test_dataset = ProcessedDataset(all_skull_paths_test, gi_test_arr, centers_test_arr, norm_method="zs") if all_skull_paths_test else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset else None

    diffusion_schedule = DiffusionSchedule(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    model = ConditionalDiffusionModel_DiT_v2(radius_dim=args.n_pc).to(device)

    if torch.cuda.device_count() > 1:
        device_ids = list(range(len(args.gpu.split(","))))
        print(f"Using {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_random_mask = args.variant == "diff_r_rm"
    eval_every = 10
    best_eval_loss = float("inf")
    patience_counter = 0
    saved_any_checkpoint = False
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            images = batch["image"].to(device)
            radius_vectors = batch["radius_vector"].to(device)
            images_in = random_mask(images) if use_random_mask else images

            t = torch.randint(0, diffusion_schedule.timesteps, (images.size(0),), device=device).long()
            noisy_radius, noise = add_noise_batch(diffusion_schedule, radius_vectors, t)
            predicted_noise = model(noisy_radius, images_in, t)
            loss = train_criterion(predicted_noise, noise)

            if args.embedding_reg:
                img_emb = base_model.img_encoder(images)
                img_emb_aug = base_model.img_encoder(images_in)
                loss = loss + args.reg_coeff * F.mse_loss(img_emb.detach(), img_emb_aug)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] train_loss={avg_train_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")

        should_eval = test_loader is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == args.epochs)
        if not should_eval:
            continue

        assert test_loader is not None
        model.eval()
        eval_loss_sum = 0.0
        eval_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                radius_vectors = batch["radius_vector"].to(device)
                t = torch.randint(0, diffusion_schedule.timesteps, (images.size(0),), device=device).long()
                noisy_radius, noise = add_noise_batch(diffusion_schedule, radius_vectors, t)
                predicted_noise = model(noisy_radius, images, t)
                batch_loss = train_criterion(predicted_noise, noise)
                eval_loss_sum += batch_loss.item() * images.size(0)
                eval_samples += images.size(0)

        eval_loss = eval_loss_sum / eval_samples
        print(f"  eval_loss={eval_loss:.6f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            patience_counter = 0
            save_checkpoint(
                model,
                model_path,
                stats_path,
                normal_min_r,
                normal_max_r,
                r_mean,
                r_std,
                fixed_center,
            )
            saved_any_checkpoint = True
            print(f"  saved model to {model_path}")
            print(f"  saved stats to {stats_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    if not saved_any_checkpoint:
        save_checkpoint(
            model,
            model_path,
            stats_path,
            normal_min_r,
            normal_max_r,
            r_mean,
            r_std,
            fixed_center,
        )
        print(f"Saved final model to {model_path}")
        print(f"Saved final stats to {stats_path}")

    print(f"Training complete. Model path: {model_path}")


if __name__ == "__main__":
    main()
