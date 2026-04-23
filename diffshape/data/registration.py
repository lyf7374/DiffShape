from __future__ import annotations

from pathlib import Path

import numpy as np


def rigid_register_to_mni152(
    image_path: str | Path,
    mni_template_path: str | Path,
    output_dir: str | Path,
    case_id: str,
) -> tuple[Path, Path]:
    """Rigid registration to MNI152. Returns (registered_image_path, transform_path)."""
    import ants

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(str(mni_template_path), reorient="LPI")
    moving = ants.image_read(str(image_path), reorient="LPI")

    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        verbose=False,
    )

    registered_path = output_dir / f"{case_id}_mni152.nii.gz"
    ants.image_write(result["warpedmovout"], str(registered_path))

    transform_path = Path(result["fwdtransforms"][0])
    return registered_path, transform_path


def apply_transform_to_mask(
    mask_path: str | Path,
    reference_path: str | Path,
    transform_path: str | Path,
    output_dir: str | Path,
    case_id: str,
) -> Path:
    """Apply rigid transform to a binary mask using nearest-neighbor interpolation."""
    import ants

    output_dir = Path(output_dir)
    reference = ants.image_read(str(reference_path), reorient="LPI")
    mask = ants.image_read(str(mask_path), reorient="LPI")

    warped_mask = ants.apply_transforms(
        fixed=reference,
        moving=mask,
        transformlist=[str(transform_path)],
        interpolator="nearestNeighbor",
        verbose=False,
    )

    mask_out_path = output_dir / f"{case_id}_mask_mni152.nii.gz"
    ants.image_write(warped_mask, str(mask_out_path))
    return mask_out_path


def register_case(
    image_path: str | Path,
    mask_path: str | Path,
    mni_template_path: str | Path,
    output_dir: str | Path,
    case_id: str,
) -> tuple[Path, Path]:
    """Full rigid registration: image + mask → MNI152 space.
    Returns (registered_image_path, registered_mask_path)."""
    registered_image, transform = rigid_register_to_mni152(
        image_path, mni_template_path, output_dir, case_id
    )
    registered_mask = apply_transform_to_mask(
        mask_path, registered_image, transform, output_dir, case_id
    )
    return registered_image, registered_mask
