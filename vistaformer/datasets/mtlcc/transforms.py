import random
from typing import Optional
import typing as T

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

LABELS = {
    0: "sugar_beet",
    1: "summer_oat",
    2: "meadow",
    3: "rape",
    4: "hop",
    5: "winter_spelt",
    6: "winter_triticale",
    7: "beans",
    8: "peas",
    9: "potatoes",
    10: "soybeans",
    11: "asparagus",
    12: "winter_wheat",
    13: "winter_barley",
    14: "winter_rye",
    15: "summer_barley",
    16: "maize",
    17: "unknown",
}

ORIGINAL_LABELS = {
    0: "unknown",
    1: "sugar_beet",
    2: "summer_oat",
    3: "meadow",
    5: "rape",
    8: "hop",
    9: "winter_spelt",
    12: "winter_triticale",
    13: "beans",
    15: "peas",
    16: "potatoes",
    17: "soybeans",
    19: "asparagus",
    22: "winter_wheat",
    23: "winter_barley",
    24: "winter_rye",
    25: "summer_barley",
    26: "maize",
}

REMAP_LABELS = {
    0: 17,
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


def get_label_names():
    names = {}
    for label in ORIGINAL_LABELS:
        names[REMAP_LABELS[label]] = ORIGINAL_LABELS[label]

    return names


def get_train_transforms(
    seq_len: int, crop_size: Optional[int] = None
) -> T.List[T.Callable]:
    """
    Returns a list of transforms to be applied to the dataset
    """
    return transforms.Compose(
        [
            SampleSequence(seq_len=seq_len, random_sample=False, use_dates=False),
            ToTensor(),
            Normalize(),
            Rotate(),
            Flip(),
        ]
    )


def get_val_transforms(seq_len: int, crop_size: Optional[int] = None):
    """
    Val/Test transforms
    """
    return transforms.Compose(
        [
            SampleSequence(seq_len=seq_len, random_sample=False, use_dates=False),
            ToTensor(),
            Normalize(),
        ]
    )


class RandomCrop(object):
    """
    Randomly crops the input arrays if the height and width are larger than the crop size.
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Assumes that the input arrays are in the shape (seq_len, bands, height, width)
        and that the label arrays are in the shape (channels, height, width)
        """
        _, _, h, w = data["inputs"].shape
        if h > self.crop_size and w > self.crop_size:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            data["inputs"] = data["inputs"][
                :, :, top : top + self.crop_size, left : left + self.crop_size
            ]
            data["labels"] = data["labels"][
                :, top : top + self.crop_size, left : left + self.crop_size
            ]
        return data


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors and rescales them
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels

    Code Repository Source: https://github.com/michaeltrs/DeepSatModels
    Arxiv Paper Link: https://arxiv.org/abs/2301.04944
    Original License: Apache 2.0
    """

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        if "B01" in data.keys():
            x10 = torch.stack(
                [
                    torch.tensor(data[key].astype(np.float32), dtype=torch.float32)
                    for key in ["B04", "B03", "B02", "B08"]
                ]
            )
            x20 = torch.stack(
                [
                    torch.tensor(data[key].astype(np.float32), dtype=torch.float32)
                    for key in ["B05", "B06", "B07", "B8A", "B11", "B12"]
                ]
            )
            x60 = torch.stack(
                [
                    torch.tensor(data[key].astype(np.float32), dtype=torch.float32)
                    for key in ["B01", "B09", "B10"]
                ]
            )
            doy = torch.tensor(np.array(data["doy"]).astype(np.float32))
            year = torch.tensor(0.0).repeat(len(data["doy"])) + 2016
            labels = torch.tensor(data["labels"].astype(np.float32))
            data = {
                "x10": x10,
                "x20": x20,
                "x60": x60,
                "day": doy,
                "year": year,
                "labels": labels,
            }

            return data

        data["inputs"] = torch.tensor(data["inputs"]).type(torch.float32)
        data["day"] = torch.tensor(data["day"]).type(torch.float32)
        data["year"] = torch.tensor(data["year"]).type(torch.float32)
        data["labels"] = torch.from_numpy(data["labels"].astype(np.int64))

        return data


class ParseLabels(object):
    """
    Parse the labels from the input arrays.
    """

    def __init__(self, label_dict: dict[int, str] = REMAP_LABELS):
        self.label_dict = label_dict

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """
        Remaps the labels to a new label set that numbers from 0 to n_classes - 1.
        """
        labels = data["labels"]
        not_remapped = np.ones(labels.shape, dtype=np.bool_)
        for key, value in self.label_dict.items():
            label_idx = labels == key
            remap_idx = label_idx & not_remapped
            labels[remap_idx] = value
            not_remapped[remap_idx] = False

        data["labels"] = labels

        return data


class Normalize(object):
    """
    Normalize inputs as in https://arxiv.org/pdf/1802.02080.pdf
    """

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data["inputs"] = data["inputs"] * 1e-4
        data["day"] = (
            data["day"] / 365.0001
        )  # 365 + h, h = 0.0001 to avoid placing day 365 in out of bounds bin
        data["year"] = data["year"] - 2016

        return data


class Rescale(object):
    """
    Rescale the images in a sample to the dimension of the largest image.
    """

    def __init__(self, concat: bool = True):
        self.concat = concat

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Expects data to be in the shape (seq_len, bands, height, width)
        """
        h, w = data["x10"].shape[2], data["x10"].shape[3]
        data["x10"] = self.rescale(data["x10"], h, w)
        data["x20"] = self.rescale(data["x20"], h, w)
        data["x60"] = self.rescale(data["x60"], h, w)

        if self.concat:
            data["inputs"] = torch.cat(
                dim=1, tensors=[data["x10"], data["x20"], data["x60"]]
            )

        return data

    def rescale(self, image: torch.Tensor, h: int, w: int):
        img = F.interpolate(image, size=(h, w), mode="bilinear")

        return img


class SampleSequence(object):
    """
    Sample a sequence of length seq_len from the input arrays.
    """

    def __init__(
        self, seq_len: int, random_sample: bool = False, use_dates: bool = False
    ):
        self.seq_len = seq_len
        self.random_sample = random_sample
        self.use_dates = use_dates

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        data: dict[str, np.ndarray]: A dictionary containing the input and label arrays.

        Note: The input arrays are expected to be in the shape (seq_len, bands, height, width)
        """
        sampled_indices = None
        if self.random_sample:
            sampled_indices = np.random.choice(
                data["inputs"].shape[0], size=self.seq_len, replace=False
            )

        data["inputs"] = self.select_indices(data["inputs"], sampled_indices)

        return data

    def select_indices(
        self, in_array: np.ndarray, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Selects the indices from the input array. If the input array is shorter than the sequence length,
        it pads the array with zeros.

        in_array: np.ndarray (T, C, H, W)
        """
        if indices is not None:
            return in_array[indices]
        t, c, h, w = in_array.shape
        if self.seq_len <= t:
            return in_array[0 : self.seq_len]

        pad_arr = np.zeros((self.seq_len - t, c, h, w), dtype=in_array.dtype)

        return np.concatenate((in_array, pad_arr), axis=0)


class Flip(object):
    """
    Flip horizontally and/or vertically with a given probability.
    """

    def __init__(self, vflip_p: float = 0.5, hflip_p: float = 0.5):
        self.vflip_p = vflip_p
        self.hflip_p = hflip_p

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Assumes that the input arrays are in the shape (seq_len, bands, height, width)
        and that the label arrays are in the shape (channels, height, width)
        """
        if random.random() < self.vflip_p:
            # Flip vertically along the height dimension (axis=2)
            data["inputs"] = torch.flip(data["inputs"], dims=[2])
            data["labels"] = torch.flip(data["labels"], dims=[0])

        if random.random() < self.hflip_p:
            # Flip horizontally along the width dimension (axis=3)
            data["inputs"] = torch.flip(data["inputs"], dims=[3])
            data["labels"] = torch.flip(data["labels"], dims=[1])

        return data


class Rotate(object):
    """
    Rotate the input arrays with a given probability.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if random.random() < self.p:
            k = random.randint(1, 3)
            data["inputs"] = torch.rot90(data["inputs"], k=k, dims=(2, 3))
            data["labels"] = torch.rot90(data["labels"], k=k, dims=(0, 1))

        return data
