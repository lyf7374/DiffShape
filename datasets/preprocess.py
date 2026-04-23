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
    elif method == "special":
        nz = volume[volume != 0]
        vmin, vmax = np.min(nz), np.max(nz)
        volume[volume != 0] = (volume[volume != 0] - vmin) / (vmax - vmin)
    return volume.astype("float32")


def _crop_pad(img, center, output_shape):
    img_shape = img.shape
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


def resize_volume(path, center=None, output_shape=(192, 192, 192), re_ori=True):
    img = read_nifti_file(path, re_ori=re_ori)
    if center is None or (np.array(center) == None).any():  # noqa: E711
        center = [s // 2 for s in output_shape]
    else:
        center = [max(0, min(img.shape[i], int(center[i]))) for i in range(3)]

    half = [s // 2 for s in output_shape]
    start = [max(0, center[i] - half[i]) for i in range(3)]
    end = [min(img.shape[i], center[i] + half[i]) for i in range(3)]
    for i in range(3):
        if end[i] - start[i] < output_shape[i]:
            if start[i] == 0:
                end[i] = min(img.shape[i], start[i] + output_shape[i])
            elif end[i] == img.shape[i]:
                start[i] = max(0, end[i] - output_shape[i])

    img = img[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
    diff = [output_shape[i] - (end[i] - start[i]) for i in range(3)]
    padding = [(d // 2, d - d // 2) for d in diff]
    return np.pad(img, padding)


def resize_volume_new(path, center=None, output_shape=(192, 192, 192), re_ori=True):
    img = read_nifti_file(path, re_ori=re_ori)
    if center is None or (np.array(center) == None).any():  # noqa: E711
        center = [s // 2 for s in img.shape]
    return _crop_pad(img, [int(c) for c in center], list(output_shape))


def process_scan(
    path,
    mask=False,
    resize=True,
    norm_method="mm",
    output_shape=(192, 192, 192),
    center=None,
    re_ori=True,
):
    if resize:
        volume = resize_volume(
            path, output_shape=output_shape, center=center, re_ori=re_ori
        )
    else:
        volume = read_nifti_file(path)

    if mask:
        volume[volume > 0.5] = 1
        volume[volume <= 0.5] = 0
        return volume
    return normalize(volume, method=norm_method)


def process_scan_new(
    path,
    mask=False,
    resize=True,
    norm_method="mm",
    output_shape=(192, 192, 192),
    center=None,
    re_ori=True,
):
    if resize:
        volume = resize_volume_new(
            path, output_shape=output_shape, center=center, re_ori=re_ori
        )
    else:
        volume = read_nifti_file(path)

    if mask:
        volume[volume > 0.5] = 1
        volume[volume <= 0.5] = 0
        return volume
    return normalize(volume, method=norm_method)
