import geopandas as gpd
import numpy as np
import pickle
import datetime
import torch
import argparse
from pathlib import Path
from numpy.lib.stride_tricks import as_strided


def get_doy(date: str):
    date = str(date)
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = f"{Y}.{m}.{d}"
    dt = datetime.datetime.strptime(date, "%Y.%m.%d")

    return dt.timetuple().tm_yday


def extract_windows(data, window_height, window_width):
    """
    Extracts fixed-size windows from a 3D or 4D numpy array.

    Note: You could also use a function like as_strided from numpy.lib.stride_tricks instead
    of this function, but this function is easier to understand and I would want to write some
    pretty serious unit tests before using that function for this...

    Parameters:
        - data: A numpy array of shape (c, h, w) or (t, c, h, w).
        - window_height: The height of the window.
        - window_width: The width of the window.

    Returns:
    - A list of numpy arrays, each of shape (t, c, window_height, window_width) for 4D input,
      or (1, c, window_height, window_width) for 3D input.
    """
    # Normalize 3D input to 4D
    if data.ndim == 3:
        data = data[np.newaxis, :]

    _, _, h, w = data.shape
    windows = []

    # Iterate over the spatial dimensions to extract windows
    for i in range(0, h - window_height + 1, window_height):
        for j in range(0, w - window_width + 1, window_width):
            window = data[:, :, i : i + window_height, j : j + window_width]
            windows.append(window)

    return windows


def unfold_reshape(image: np.ndarray, window_dim: int):
    """
    Convert the function to use numpy operations for unfolding and reshaping,
    correctly handling dimensions.
    image: Input image array, can be 3D (C, H, W) or 4D (T, C, H, W)
    window_dim: The height and width of the unfolding window and step size
    """
    if image.ndim == 4:  # If the input image is 4D
        T, C, H, W = image.shape
        num_windows = (H // window_dim) * (W // window_dim)
        # Calculate new shape after unfolding
        new_shape = (num_windows, T, C, window_dim, window_dim)
        img_reshaped = np.reshape(image, new_shape)

    elif image.ndim == 3:  # If the input image is 3D
        C, H, W = image.shape
        num_windows = (H // window_dim) * (W // window_dim)
        new_shape = (num_windows, C, window_dim, window_dim)
        img_reshaped = np.reshape(image, new_shape)

    elif image.ndim == 2:  # If the input image is 2D
        H, W = image.shape
        num_windows = (H // window_dim) * (W // window_dim)
        new_shape = (num_windows, window_dim, window_dim)
        img_reshaped = np.reshape(image, new_shape)

    return img_reshaped


def generate_samples(root_dir: Path, output_dir: Path, output_dim: int):
    """
    Generate samples of a given dimension from the data for Sentinel-1 and Sentinel-2 tiles
    and then save them as pickle files.
    """
    meta_patch = gpd.read_file(root_dir / "metadata.geojson")
    if not (output_dir / "samples").exists():
        (output_dir / "samples").mkdir()
    labels = []
    for i in range(meta_patch.shape[0]):
        print(f"Processing file {i} of {meta_patch.shape[0]}")
        # Load data files
        s2 = np.load(root_dir / "DATA_S2" / f"S2_{meta_patch['ID_PATCH'].iloc[i]}.npy")
        s1a = np.load(
            root_dir / "DATA_S1A" / f"S1A_{meta_patch['ID_PATCH'].iloc[i]}.npy"
        )
        s1d = np.load(
            root_dir / "DATA_S1D" / f"S1D_{meta_patch['ID_PATCH'].iloc[i]}.npy"
        )
        labels = np.load(
            root_dir / "ANNOTATIONS" / f"TARGET_{meta_patch['ID_PATCH'].iloc[i]}.npy"
        )
        if labels.dtype != np.uint8:
            labels = labels.astype(np.uint8)

        # Sort Sentinel-2 tiles by date
        s2_dates = meta_patch["dates-S2"].iloc[i]
        s2_doy = np.array([get_doy(date) for date in s2_dates.values()])
        index = np.argsort(s2_doy)
        s2 = s2[index]
        s2_doy = s2_doy[index]

        # Sort Sentinel-1A tiles by date
        s1a_dates = meta_patch["dates-S1A"].iloc[i]
        s1a_doy = np.array([get_doy(date) for date in s1a_dates.values()])
        index = np.argsort(s1a_doy)
        s1a = s1a[index]
        s1a_doy = s1a_doy[index]

        # Sort Sentinel-1D tiles by date
        s1d_dates = meta_patch["dates-S1D"].iloc[i]
        s1d_doy = np.array([get_doy(date) for date in s1d_dates.values()])
        index = np.argsort(s1d_doy)
        s1d = s1d[index]
        s1d_doy = s1d_doy[index]

        unfolded_s2 = extract_windows(s2, output_dim, output_dim)
        unfolded_s1a = extract_windows(s1a, output_dim, output_dim)
        unfolded_s1d = extract_windows(s1d, output_dim, output_dim)
        unfolded_labels = extract_windows(labels, output_dim, output_dim)

        for j, (s2_patch, s1a_patch, s1d_patch, label_patch) in enumerate(
            zip(unfolded_s2, unfolded_s1a, unfolded_s1d, unfolded_labels)
        ):
            label_patch = label_patch.squeeze(0)
            assert label_patch.shape == (3, output_dim, output_dim)
            if label_patch[0, output_dim // 2, output_dim // 2] > 20:
                print(
                    f"Label Patch value: {label_patch[output_dim // 2, output_dim // 2]}"
                )
                print(
                    f"Label value counts: {np.unique(label_patch, return_counts=True)}"
                )
                raise ValueError(
                    "Label patch value is greater than 20, something in the data processing is wrong."
                )

            out_file = (
                output_dir / "samples" / f"{meta_patch['ID_PATCH'].iloc[i]}_{j}.pickle"
            )
            with open(out_file, "wb") as f:
                pickle.dump(
                    {
                        "s2": s2_patch,
                        "s1a": s1a_patch,
                        "s1d": s1d_patch,
                        "labels": label_patch,
                        # See line here for why this class label mapping is used for the class label:
                        # https://github.com/VSainteuf/pastis-benchmark/blob/ac1830ecfd4c2922b50f8555f142392c31913c75/code/dataloader.py#L201
                        "class_label": np.uint8(
                            label_patch[0, output_dim // 2, output_dim // 2]
                        ),  # save center pixel label as class label
                        "s2_doy": s2_doy,
                        "s1a_doy": s1a_doy,
                        "s1d_doy": s1d_doy,
                    },
                    f,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate input samples for PASTIS dataset"
    )
    parser.add_argument(
        "--rootdir", type=Path, help="Root directory containing PASTIS-R dataset"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output root directory for processed data. Data will be saved in the samples directory of the rootdir by default.",
    )
    parser.add_argument(
        "--output-dim", type=int, default=24, help="Size of extracted samples"
    )
    args = parser.parse_args()

    assert args.rootdir.exists(), "Root directory does not exist"
    if args.output_dim > 128 or args.output_dim < 8:
        raise ValueError("Output dimension must be between 8 and 128")

    if args.outdir is None:
        args.outdir = args.rootdir
        (args.outdir / "samples").mkdir(exist_ok=True)

    generate_samples(args.rootdir, args.outdir, args.output_dim)
