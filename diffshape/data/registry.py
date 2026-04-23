from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    description: str
    layout: str
    data_root: Path
    crop_shape: tuple[int, int, int]
    crop_function: str
    norm_method: str
    n_patch: int
    split: dict

    image_dir: Optional[str] = None
    mask_dir: Optional[str] = None
    image_pattern: Optional[str] = None
    mask_pattern: Optional[str] = None
    case_id_regex: Optional[str] = None
    case_id_min_length: int = 1

    image_filename: Optional[str] = None
    mask_filename: Optional[list[str] | str] = None

    ordering_file: Optional[str] = None


@dataclass(frozen=True)
class CaseRecord:
    dataset: str
    case_id: str
    image_path: Path
    mask_path: Path


def load_config(config_path: str | Path) -> DatasetConfig:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    crop = raw.get("crop_shape", [192, 192, 192])
    mask_fn = raw.get("mask_filename")
    if isinstance(mask_fn, str):
        mask_fn = [mask_fn]

    return DatasetConfig(
        dataset_name=raw["dataset_name"],
        description=raw.get("description", ""),
        layout=raw["layout"],
        data_root=Path(raw["data_root"]),
        crop_shape=(crop[0], crop[1], crop[2]),
        crop_function=raw.get("crop_function", "legacy"),
        norm_method=raw.get("norm_method", "zs"),
        n_patch=raw.get("n_patch", 128),
        split=raw.get("split", {"method": "all_test"}),
        image_dir=raw.get("image_dir"),
        mask_dir=raw.get("mask_dir"),
        image_pattern=raw.get("image_pattern"),
        mask_pattern=raw.get("mask_pattern"),
        case_id_regex=raw.get("case_id_regex"),
        case_id_min_length=raw.get("case_id_min_length", 1),
        image_filename=raw.get("image_filename"),
        mask_filename=mask_fn,
        ordering_file=raw.get("ordering_file"),
    )


def _discover_flat_directory(cfg: DatasetConfig, project_root: Path) -> list[CaseRecord]:
    if not cfg.image_dir or not cfg.mask_dir or not cfg.case_id_regex or not cfg.mask_pattern:
        raise ValueError("flat_directory layout requires image_dir, mask_dir, case_id_regex, mask_pattern")

    image_dir = project_root / cfg.data_root / cfg.image_dir
    mask_dir = project_root / cfg.data_root / cfg.mask_dir
    pattern = re.compile(cfg.case_id_regex)

    id_to_image: dict[str, Path] = {}
    for f in sorted(image_dir.iterdir()):
        if f.name.startswith("."):
            continue
        m = pattern.match(f.name)
        if m and len(m.group("id")) >= cfg.case_id_min_length:
            id_to_image[m.group("id")] = f

    cases: list[CaseRecord] = []
    for case_id, img_path in id_to_image.items():
        mask_name = cfg.mask_pattern.format(case_id=case_id)
        mask_path = mask_dir / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for case {case_id}: {mask_path}")
        cases.append(CaseRecord(
            dataset=cfg.dataset_name,
            case_id=case_id,
            image_path=img_path,
            mask_path=mask_path,
        ))
    return cases


def _discover_folder_per_case(cfg: DatasetConfig, project_root: Path) -> list[CaseRecord]:
    if not cfg.image_filename or not cfg.mask_filename:
        raise ValueError("folder_per_case layout requires image_filename and mask_filename")

    data_dir = project_root / cfg.data_root
    cases: list[CaseRecord] = []

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        case_id = folder.name
        image_path = folder / cfg.image_filename
        if not image_path.exists():
            continue

        mask_path: Optional[Path] = None
        for mask_candidate in cfg.mask_filename:
            candidate = folder / mask_candidate
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for case {case_id} in {folder}. "
                f"Tried: {cfg.mask_filename}"
            )
        cases.append(CaseRecord(
            dataset=cfg.dataset_name,
            case_id=case_id,
            image_path=image_path,
            mask_path=mask_path,
        ))
    return cases


def _apply_ordering(
    cases: list[CaseRecord],
    ordering_file: Path,
) -> list[CaseRecord]:
    ordered_ids = np.load(ordering_file, allow_pickle=True)
    ordered_ids = [str(x) for x in ordered_ids]
    id_to_case = {c.case_id: c for c in cases}

    ordered: list[CaseRecord] = []
    for pid in ordered_ids:
        matches = [cid for cid in id_to_case if pid in cid or cid in pid]
        if len(matches) == 1:
            ordered.append(id_to_case[matches[0]])
        elif pid in id_to_case:
            ordered.append(id_to_case[pid])
        else:
            import warnings
            warnings.warn(f"Ordering ID '{pid}' not found on disk, skipping")
    return ordered


def discover_cases(
    cfg: DatasetConfig,
    project_root: str | Path,
) -> list[CaseRecord]:
    project_root = Path(project_root)

    if cfg.layout == "flat_directory":
        cases = _discover_flat_directory(cfg, project_root)
    elif cfg.layout == "folder_per_case":
        cases = _discover_folder_per_case(cfg, project_root)
    else:
        raise ValueError(f"Unknown layout: {cfg.layout}")

    if cfg.ordering_file:
        ordering_path = project_root / cfg.ordering_file
        if ordering_path.exists():
            cases = _apply_ordering(cases, ordering_path)

    return cases
