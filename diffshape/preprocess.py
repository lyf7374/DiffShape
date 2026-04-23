import nibabel as nib
import numpy as np


def read_nifti_file(filepath, re_ori=True):
    scan = nib.load(filepath)
    if re_ori:
        scan = nib.as_closest_canonical(scan)
    return scan.get_fdata()


def normalize(volume, method="mm"):
    if method == "zs":
        mean = np.mean(volume)
        std = np.std(volume)
        volume = (volume - mean) / std
    elif method == "mm":
        vmin = np.min(volume)
        vmax = np.max(volume)
        volume = (volume - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unsupported norm method: {method}")
    return volume.astype("float32")


def resize_volume(path, center=None, output_shape=(192, 192, 192), re_ori=True):
    img = read_nifti_file(path, re_ori=re_ori)
    img_shape = img.shape
    output_shape = list(output_shape)

    if center is None or (np.array(center) == None).any():  # noqa: E711
        center = [s // 2 for s in img_shape]
    else:
        center = [int(c) for c in center]

    half = [s // 2 for s in output_shape]
    start = [center[i] - half[i] for i in range(3)]
    end = [start[i] + output_shape[i] for i in range(3)]

    pad_before = [max(0, -start[i]) for i in range(3)]
    pad_after = [max(0, end[i] - img_shape[i]) for i in range(3)]
    cs = [max(0, start[i]) for i in range(3)]
    ce = [min(img_shape[i], end[i]) for i in range(3)]

    cropped = img[cs[0] : ce[0], cs[1] : ce[1], cs[2] : ce[2]]
    padding = tuple((pad_before[i], pad_after[i]) for i in range(3))
    return np.pad(cropped, padding, mode="constant", constant_values=0)


def process_scan(
    path,
    mask=False,
    resize=True,
    norm_method="zs",
    output_shape=(192, 192, 192),
    center=None,
    re_ori=True,
):
    volume = (
        resize_volume(path, output_shape=output_shape, center=center, re_ori=re_ori)
        if resize
        else read_nifti_file(path)
    )
    if mask:
        volume[volume > 0.5] = 1
        volume[volume <= 0.5] = 0
        return volume
    return normalize(volume, method=norm_method)
