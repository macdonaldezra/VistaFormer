from typing import Optional
import random

import torch
import numpy as np
from torchvision import transforms


def get_val_transforms(
    seq_len: int, use_dates: bool = False, dtype: torch.dtype = torch.float32
):
    """
    Val/Test transforms
    """
    return transforms.Compose(
        [
            Normalize(),
            SampleSequence(seq_len=seq_len, random_sample=False, use_dates=use_dates),
            ToTensor(dtype=dtype, use_dates=use_dates),
        ]
    )


def get_train_transforms(
    seq_len: int,
    sample_seq: bool = False,
    use_dates: bool = False,
    dtype: torch.dtype = torch.float32,
):
    return transforms.Compose(
        [
            Normalize(),
            SampleSequence(
                seq_len=seq_len, random_sample=sample_seq, use_dates=use_dates
            ),
            Flip(),
            # GaussianNoise(),
            Rotate(),
            ToTensor(dtype=dtype, use_dates=use_dates),
        ]
    )


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
        Note that this transform expects the sequence length to be the first dimension of the input arrays.
        """
        sampled_indices = None
        if self.random_sample:  # Randomly sample a sequence of length seq_len
            sampled_indices = np.random.choice(
                self.seq_len, size=self.seq_len, replace=False
            )

        if "s2" in data:
            data["s2"] = self.select_indices(data["s2"], sampled_indices)
        if "s1d" in data:
            data["s1d"] = self.select_indices(data["s1d"], sampled_indices)
        if "s1a" in data:
            data["s1a"] = self.select_indices(data["s1a"], sampled_indices)
        if "s2_doy" in data and self.use_dates:
            data["s2_doy"] = self.select_indices(data["s2_doy"], sampled_indices)
        if "s1d_doy" in data and self.use_dates:
            data["s1d_doy"] = self.select_indices(data["s1d_doy"], sampled_indices)
        if "s1a_doy" in data and self.use_dates:
            data["s1a_doy"] = self.select_indices(data["s1a_doy"], sampled_indices)

        return data

    def select_indices(
        self, in_array: np.ndarray, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        t = in_array.shape[0]

        if self.seq_len <= t:
            return in_array[0 : self.seq_len]

        if in_array.ndim == 4:
            t, c, h, w = in_array.shape
            pad_arr = np.zeros((self.seq_len - t, c, h, w), dtype=in_array.dtype)
        else:
            t, c = in_array.shape
            pad_arr = np.zeros((self.seq_len - t, c), dtype=in_array.dtype)

        out_arr = np.concatenate((in_array, pad_arr), axis=0)
        if indices is not None:
            return out_arr[indices]

        return out_arr


class Normalize(object):
    """
    Normalize based on fold 1 mean and std for S2, S1A, and S1D for the PASTIS dataset.
    """

    def __init__(self):
        self.s2_mean = np.array(  # Fold 1 mean
            [
                [
                    [[1165.9398193359375]],
                    [[1375.6534423828125]],
                    [[1429.2191162109375]],
                    [[1764.798828125]],
                    [[2719.273193359375]],
                    [[3063.61181640625]],
                    [[3205.90185546875]],
                    [[3319.109619140625]],
                    [[2422.904296875]],
                    [[1639.370361328125]],
                ]
            ]
        ).astype(np.float32)
        self.s2_std = np.array(  # Fold 1 std
            [
                [
                    [[1942.6156005859375]],
                    [[1881.9234619140625]],
                    [[1959.3798828125]],
                    [[1867.2239990234375]],
                    [[1754.5850830078125]],
                    [[1769.4046630859375]],
                    [[1784.860595703125]],
                    [[1767.7100830078125]],
                    [[1458.963623046875]],
                    [[1299.2833251953125]],
                ]
            ]
        ).astype(np.float32)

        self.s1a_mean = np.array(  # Fold 1 mean
            [[[[-10.930951118469238]], [[-17.348514556884766]], [[6.417511940002441]]]]
        )
        self.s1a_std = np.array(  # Fold 1 std
            [[[[3.285966396331787]], [[3.2129523754119873]], [[3.3421084880828857]]]]
        )
        self.s1d_mean = np.array(  # Fold 1 mean
            [[[[-11.105852127075195]], [[-17.502077102661133]], [[6.407216548919678]]]]
        )
        self.s1d_std = np.array(  # Fold 1 std
            [[[[3.376193046569824]], [[3.1705307960510254]], [[3.34938383102417]]]]
        )

    def __call__(self, data: dict[str, np.ndarray]):
        if "s2" in data:
            data["s2"] = (data["s2"] - self.s2_mean) / self.s2_std
        if "s1d" in data:
            data["s1d"] = (data["s1d"] - self.s1d_mean) / self.s1d_std
        if "s1a" in data:
            data["s1a"] = (data["s1a"] - self.s1a_mean) / self.s1a_std
        if "s2_doy" in data:
            data["s2_doy"] = data["s2_doy"] / 365.0001
        if "s1a_doy" in data:
            data["s1a_doy"] = data["s1a_doy"] / 365.0001
        if "s1d_doy" in data:
            data["s1d_doy"] = data["s1d_doy"] / 365.0001

        return data


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
        items in  : x10, x20, x60, day, year, labels
        items out : x10, x20, x60, day, year, labels
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        use_dates: bool = False,
    ):
        self.dtype = dtype
        self.use_dates = use_dates

    def __call__(self, data: dict[str, np.ndarray]):
        if "class_label" in data:
            data["class_label"] = torch.tensor(data["class_label"]).to(torch.long)
        if "labels" in data:
            data["labels"] = torch.tensor(data["labels"]).to(torch.long)

        if "s2" in data:
            data["s2"] = torch.tensor(data["s2"]).to(self.dtype)
            if self.use_dates:
                height, width = data["s2"].shape[2], data["s2"].shape[3]
                date = torch.tensor(data["s2_doy"], dtype=self.dtype)
                data["s2"] = torch.cat(
                    (data["s2"], self.add_dates(date, height, width)),
                    dim=1,
                ).to(self.dtype)
        if "s1d" in data:
            data["s1d"] = torch.tensor(data["s1d"]).to(self.dtype)
            if self.use_dates:
                height, width = data["s1d"].shape[2], data["s1d"].shape[3]
                date = torch.tensor(data["s1d_doy"], dtype=self.dtype)
                data["s1d"] = torch.cat(
                    (data["s1d"], self.add_dates(date, height, width)),
                    dim=1,
                ).to(self.dtype)
        if "s1a" in data:
            data["s1a"] = torch.tensor(data["s1a"]).to(self.dtype)
            if self.use_dates:
                height, width = data["s1a"].shape[2], data["s1a"].shape[3]
                date = torch.tensor(data["s1a_doy"], dtype=self.dtype)
                data["s1a"] = torch.cat(
                    (data["s1a"], self.add_dates(date, height, width)),
                    dim=1,
                ).to(self.dtype)

        return data

    def add_dates(self, data: torch.Tensor, height: int, width: int):
        return data.repeat(1, height, width, 1).permute(3, 0, 1, 2)


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
            if "s2" in data:
                data["s2"] = np.flip(data["s2"], axis=2).copy()
            if "s1d" in data:
                data["s1d"] = np.flip(data["s1d"], axis=2).copy()
            if "s1a" in data:
                data["s1a"] = np.flip(data["s1a"], axis=2).copy()
            data["labels"] = np.flip(data["labels"], axis=1).copy()

        if random.random() < self.hflip_p:
            if "s2" in data:
                data["s2"] = np.flip(data["s2"], axis=3).copy()
            if "s1d" in data:
                data["s1d"] = np.flip(data["s1d"], axis=3).copy()
            if "s1a" in data:
                data["s1a"] = np.flip(data["s1a"], axis=3).copy()
            data["labels"] = np.flip(data["labels"], axis=2).copy()

        return data


class Rotate(object):
    """
    Rotate the input arrays with a given probability.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if random.random() < self.p:
            k = random.randint(1, 3)
            if "s2" in data:
                data["s2"] = np.ascontiguousarray(
                    np.rot90(data["s2"], k=k, axes=(2, 3))
                )
            if "s1d" in data:
                data["s1d"] = np.ascontiguousarray(
                    np.rot90(data["s1d"], k=k, axes=(2, 3))
                )
            if "s1a" in data:
                data["s1a"] = np.ascontiguousarray(
                    np.rot90(data["s1a"], k=k, axes=(2, 3))
                )
            data["labels"] = np.ascontiguousarray(
                np.rot90(data["labels"], k=k, axes=(1, 2))
            )

        return data


class GaussianNoise(object):
    """
    Add Gaussian noise to the input arrays.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.02, p: float = 0.25):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if random.random() < self.p:
            if "s2" in data:
                data["s2"] = data["s2"] + np.random.normal(
                    self.mean, self.std, data["s2"].shape
                )
            if "s1d" in data:
                data["s1d"] = data["s1d"] + np.random.normal(
                    self.mean, self.std, data["s1d"].shape
                )
            if "s1a" in data:
                data["s1a"] = data["s1a"] + np.random.normal(
                    self.mean, self.std, data["s1a"].shape
                )

        return data
