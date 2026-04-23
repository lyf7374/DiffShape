from __future__ import annotations

from pathlib import Path

import numpy as np


def find_center(
    image_path: str | Path,
    mni_template_path: str | Path,
    mni_mask_path: str | Path,
) -> np.ndarray:
    """ANTs SyN registration → warp MNI mask → median of foreground voxels = brain center."""
    import ants

    template = ants.image_read(str(mni_template_path), reorient="LPI")
    template_mask = ants.image_read(str(mni_mask_path), reorient="LPI")
    subject = ants.image_read(str(image_path), reorient="LPI")

    registration = ants.registration(
        fixed=subject,
        moving=template,
        type_of_transform="SyN",
        verbose=False,
    )
    warped_mask = ants.apply_transforms(
        fixed=registration["warpedmovout"],
        moving=template_mask,
        transformlist=registration["fwdtransforms"],
        interpolator="nearestNeighbor",
        verbose=False,
    )
    foreground = np.argwhere(warped_mask.numpy() > 0.5)
    return np.median(foreground, axis=0).astype(int)


def find_centers_batch(
    image_paths: list[str | Path],
    mni_template_path: str | Path,
    mni_mask_path: str | Path,
) -> np.ndarray:
    """Run find_center on multiple images. Returns (N, 3) array of centers."""
    centers = []
    for path in image_paths:
        centers.append(find_center(path, mni_template_path, mni_mask_path))
    return np.array(centers, dtype=np.int64)
