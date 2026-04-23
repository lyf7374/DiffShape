"""
Pipeline entry point: config → rigid register to MNI152 → center → GI → processed_dataset

Usage:
    python -m diffshape.prepare_data --configs diffshape/configs/dataset1.yaml diffshape/configs/dataset2.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from diffshape.data.registry import (
    load_config,
    discover_cases,
    CaseRecord,
    DatasetConfig,
)
from diffshape.data.registration import register_case
from diffshape.data.gi_extractor import extract_gi_single, FIXED_CENTER
from diffshape.data.splits import apply_split

import nibabel as nib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare processed dataset from config"
    )
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--output-dir", type=Path, default=Path("processed_dataset"))
    parser.add_argument(
        "--mni-template",
        type=Path,
        default=Path("data_utils/mni_icbm152_t1_tal_nlin_sym_09a.nii"),
    )
    parser.add_argument("--skip-gi", action="store_true")
    parser.add_argument("--active-folds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max cases per dataset (for smoke testing)",
    )
    return parser.parse_args()


def process_dataset(
    cfg: DatasetConfig,
    cases: list[CaseRecord],
    project_root: Path,
    output_dir: Path,
    mni_template: Path,
    skip_gi: bool,
    active_folds: list[int] | None,
) -> dict:
    dataset_dir = output_dir / cfg.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": cfg.dataset_name,
        "n_cases": len(cases),
        "n_patch": cfg.n_patch,
        "n_points": cfg.n_patch * cfg.n_patch,
        "crop_shape": list(cfg.crop_shape),
        "registration": "rigid_mni152",
    }

    case_ids = [c.case_id for c in cases]

    reg_dir = dataset_dir / "registered"
    reg_dir.mkdir(parents=True, exist_ok=True)
    registered_images = []
    registered_masks = []
    for case in tqdm(cases, desc=f"{cfg.dataset_name}: rigid registration to MNI152"):
        reg_img, reg_mask = register_case(
            case.image_path,
            case.mask_path,
            project_root / mni_template,
            reg_dir,
            case.case_id,
        )
        registered_images.append(reg_img)
        registered_masks.append(reg_mask)
    image_paths = registered_images
    mask_paths = registered_masks

    centers = []
    for mask_p in tqdm(
        mask_paths,
        desc=f"{cfg.dataset_name}: computing centers from registered masks",
    ):
        mdata = nib.load(str(mask_p)).get_fdata()
        centers.append(np.median(np.argwhere(mdata > 0.5), axis=0).astype(np.int64))
    centers_arr = np.array(centers, dtype=np.int64)

    np.save(dataset_dir / "centers.npy", centers_arr)
    np.save(dataset_dir / "case_ids.npy", np.array(case_ids, dtype=str))

    if not skip_gi:
        mesh_dir = dataset_dir / "meshes"
        all_points = []
        for i, case in enumerate(
            tqdm(cases, desc=f"{cfg.dataset_name}: GI extraction")
        ):
            points = extract_gi_single(
                str(mask_paths[i]),
                centers_arr[i],
                cfg.crop_shape,
                cfg.crop_function,
                cfg.n_patch,
                mesh_dir,
                case.case_id,
            )
            all_points.append(points)
        gi_array = np.array(all_points)
        np.save(
            dataset_dir
            / f"cGI_{cfg.dataset_name}_{cfg.n_patch * cfg.n_patch}rpt_preC.npy",
            gi_array,
        )
        summary["gi_shape"] = list(gi_array.shape)

    splits = apply_split(cases, cfg.split, active_folds=active_folds)
    for split_name, split_cases in splits.items():
        indices = [case_ids.index(c.case_id) for c in split_cases]
        np.save(
            dataset_dir / f"split_{split_name}_indices.npy",
            np.array(indices, dtype=np.int64),
        )
    summary["splits"] = {k: len(v) for k, v in splits.items()}

    np.save(dataset_dir / "fixed_sampling_center.npy", FIXED_CENTER)

    def _to_relative(paths, base):
        base = Path(base).resolve()
        return np.array(
            [str(Path(p).resolve().relative_to(base)) for p in paths], dtype=str
        )

    np.save(dataset_dir / "image_paths.npy", _to_relative(image_paths, output_dir))
    np.save(dataset_dir / "mask_paths.npy", _to_relative(mask_paths, output_dir))

    with open(dataset_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for config_path in args.configs:
        cfg = load_config(config_path)
        cases = discover_cases(cfg, project_root)
        if args.limit is not None:
            cases = cases[: args.limit]
        print(f"[{cfg.dataset_name}] Discovered {len(cases)} cases")

        summary = process_dataset(
            cfg,
            cases,
            project_root,
            output_dir,
            args.mni_template,
            args.skip_gi,
            args.active_folds,
        )
        all_summaries[cfg.dataset_name] = summary

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nProcessed dataset saved to: {output_dir}")
    for name, s in all_summaries.items():
        splits_str = ", ".join(f"{k}={v}" for k, v in s.get("splits", {}).items())
        print(f"  {name}: {s['n_cases']} cases, {splits_str}")


if __name__ == "__main__":
    main()
