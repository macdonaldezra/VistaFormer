import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MTLCC_FORMAT = {
    "x10/data": tf.io.FixedLenFeature([], tf.string),
    "x10/shape": tf.io.FixedLenFeature([4], tf.int64),
    "x20/data": tf.io.FixedLenFeature([], tf.string),
    "x20/shape": tf.io.FixedLenFeature([4], tf.int64),
    "x60/data": tf.io.FixedLenFeature([], tf.string),
    "x60/shape": tf.io.FixedLenFeature([4], tf.int64),
    "dates/doy": tf.io.FixedLenFeature([], tf.string),
    "dates/year": tf.io.FixedLenFeature([], tf.string),
    "dates/shape": tf.io.FixedLenFeature([1], tf.int64),
    "labels/data": tf.io.FixedLenFeature([], tf.string),
    "labels/shape": tf.io.FixedLenFeature([3], tf.int64),
}
REMAP_LABELS = {
    0: 17,  # Map unknown class to 17
    1: 0,
    2: 1,
    3: 2,
    5: 3,
    8: 4,
    9: 5,
    12: 6,
    13: 7,
    15: 8,
    16: 9,
    17: 10,
    19: 11,
    22: 12,
    23: 13,
    24: 14,
    25: 15,
    26: 16,
}


def parse_tfrecord(serial_data):
    features = tf.io.parse_single_example(serial_data, MTLCC_FORMAT)

    x10 = tf.reshape(
        tf.io.decode_raw(features["x10/data"], tf.int64),
        tf.cast(features["x10/shape"], tf.int32),
    )
    x20 = tf.reshape(
        tf.io.decode_raw(features["x20/data"], tf.int64),
        tf.cast(features["x20/shape"], tf.int32),
    )
    x60 = tf.reshape(
        tf.io.decode_raw(features["x60/data"], tf.int64),
        tf.cast(features["x60/shape"], tf.int32),
    )
    doy = tf.reshape(
        tf.io.decode_raw(features["dates/doy"], tf.int64),
        tf.cast(features["dates/shape"], tf.int32),
    )
    year = tf.reshape(
        tf.io.decode_raw(features["dates/year"], tf.int64),
        tf.cast(features["dates/shape"], tf.int32),
    )

    labels = tf.reshape(
        tf.io.decode_raw(features["labels/data"], tf.int64),
        tf.cast(features["labels/shape"], tf.int32),
    )

    return x10, x20, x60, doy, year, labels


def remove_padded_instances(max_res, mid_res, min_res, day, year, labels):
    """
    Remove instances that are padded with -1 in the day field. Padding will then need to be done
    at the batch level.
    """
    mask = day > 0
    max_res = max_res[mask]
    mid_res = mid_res[mask]
    min_res = min_res[mask]
    day = day[mask]
    year = year[mask]
    labels = labels[mask]

    return max_res, mid_res, min_res, day, year, labels


def data_to_array(max_res, mid_res, min_res, day, year, labels):
    """
    Convert the TensorFlow tensors to numpy arrays
    """
    max_res = np.array(max_res, dtype=np.float32)
    mid_res = np.array(mid_res, dtype=np.float32)
    min_res = np.array(min_res, dtype=np.float32)
    day = np.array(day, dtype=np.int16)
    year = np.array(year, dtype=np.int16)
    labels = np.array(labels, dtype=np.int16)

    return max_res, mid_res, min_res, day, year, labels


def resize_images(images, new_height, new_width):
    """
    Resize images stored in a (T, C, H, W) format using linear interpolation.

    Parameters:
    images (np.array): Input array of images with shape (T, C, H, W).
    new_height (int): The desired height of the images.
    new_width (int): The desired width of the images.

    Returns:
    np.array: Resized images of shape (T, C, new_height, new_width).
    """
    T, C, H, W = images.shape
    resized_images = np.empty((T, C, new_height, new_width), dtype=np.int32)

    for i in range(T):
        for c in range(C):
            # OpenCV expects the image in (H, W, C) format, so we need to transpose the axes.
            frame = images[i, c, :, :]
            # Resize the frame
            resized_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )

            # Store it back in the correct place
            resized_images[i, c, :, :] = resized_frame

    return resized_images


def remap_labels(labels: np.ndarray, label_dict: dict[int, int] = REMAP_LABELS):
    """
    Function to remap labels in a numpy array using a dictionary.
    """
    not_remapped = np.ones(labels.shape, dtype=np.bool_)
    for key, value in label_dict.items():
        label_idx = labels == key
        remap_idx = label_idx & not_remapped
        labels[remap_idx] = value
        not_remapped[remap_idx] = False

    return labels


def parse_mtlcc_tfrecords(data_path: Path, out_path: Path):
    """
    Function performs the following pre-processing steps on TFRecord MTLCC dataset:
    1) Load TFRecords to Numpy Array
    2)
    """
    files = list(data_path.glob("*.tfrecord.gz"))
    assert len(files) > 0, f"No tfrecord files found in {data_path}"
    print(f"Found {len(files)} tfrecord files in {data_path}")

    data = tf.data.TFRecordDataset(files, compression_type="GZIP")
    dataset = data.map(parse_tfrecord, num_parallel_calls=1)

    iterator = iter(dataset)
    for filename, data in zip(files, iterator):
        name = filename.name.split(".")[0]
        max_res, mid_res, min_res, day, year, labels = data

        max_res, mid_res, min_res, day, year, labels = data_to_array(
            max_res, mid_res, min_res, day, year, labels
        )
        max_res, mid_res, min_res, day, year, labels = remove_padded_instances(
            max_res, mid_res, min_res, day, year, labels
        )
        mid_res = np.transpose(mid_res, (0, 3, 1, 2))
        min_res = np.transpose(min_res, (0, 3, 1, 2))
        max_res = np.transpose(max_res, (0, 3, 1, 2))
        mid_res = resize_images(mid_res, max_res.shape[2], max_res.shape[3])
        min_res = resize_images(min_res, max_res.shape[2], max_res.shape[3])
        inputs = np.concatenate((max_res, mid_res, min_res), axis=1)

        # See note here regarding labels:
        # https://github.com/MarcCoru/MTLCC/blob/f6e5aed5c59af084a91428fdd285a17fcf6344f4/Dataset.py#L124
        labels = labels[0]  # Only take the first label since they're all the same...
        labels = remap_labels(
            labels
        )  # Remap the labels to have values indexed from 0-17

        out_data = {"inputs": inputs, "day": day, "year": year, "labels": labels}
        save_path = out_path / f"{name}.pkl"
        if not save_path.exists():
            with open(save_path, "wb") as f:
                pickle.dump(out_data, f)

            print(f"Processed {filename.name} and saved as {name}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate input samples for MTLCC dataset"
    )
    parser.add_argument(
        "--rootdir", type=Path, help="Root directory containing MTLCC dataset"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output root directory for processed data. Data will be saved in the samples directory of the rootdir by default.",
    )
    args = parser.parse_args()
    assert args.rootdir.exists(), "Root directory does not exist"

    if args.outdir is None:
        args.outdir = args.rootdir
        (args.outdir / "samples").mkdir(exist_ok=True)

    parse_mtlcc_tfrecords(args.rootdir, args.outdir)
