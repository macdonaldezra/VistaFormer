import os
import pickle
import typing as T
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data

from remote_cattn.config import TrainingConfig
from remote_cattn.datasets.mtlcc.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def get_dist_dataloader(
    rank: int,
    world_size: int,
    config: TrainingConfig,
) -> T.Dict[str, DataLoader]:
    """
    return a distributed dataloader
    """
    train_transform = get_train_transforms(
        seq_len=config.max_seq_len,
        crop_size=config.dataset.kwargs.get(
            "crop_size", None
        ),  # This should probably be the image size... will fix later to add comparisons and such
    )
    val_transform = get_val_transforms(
        seq_len=config.max_seq_len,
        crop_size=config.dataset.kwargs.get("crop_size", None),
    )

    train_dataset = MTLCCDataset(
        root_dir=config.dataset.path,
        split="train",
        transform=train_transform,
        return_paths=False,
        year=config.dataset.kwargs.get("year", None),
        fold=config.dataset.kwargs.get("fold", 0),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        sampler=train_sampler,
    )

    val_dataset = MTLCCDataset(
        root_dir=config.dataset.path,
        split="val",
        transform=val_transform,
        return_paths=False,
        year=config.dataset.kwargs.get("year", None),
        fold=config.dataset.kwargs.get("fold", 0),
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        sampler=val_sampler,
    )

    test_dataset = MTLCCDataset(
        root_dir=config.dataset.path,
        split="test",
        transform=val_transform,
        return_paths=False,
        year=config.dataset.kwargs.get("year", None),
        fold=config.dataset.kwargs.get("fold", 0),
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        sampler=test_sampler,
    )

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def get_dataloader(
    config: TrainingConfig,
    split: str,
    transform: T.Optional[T.Callable] = None,
    batch_size: int = 16,
    num_workers: int = 8,
    shuffle: bool = True,
    return_paths: bool = False,
):
    if transform is None:
        transform = get_val_transforms(
            seq_len=config.max_seq_len,
            crop_size=config.dataset.kwargs.get("crop_size", None),
        )

    dataset = MTLCCDataset(
        root_dir=config.dataset.path,
        split=split,
        transform=transform,
        return_paths=return_paths,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


class MTLCCDataset(Dataset):
    """
    Satellite Images dataset.
    """

    classes = [
        "sugar-beet",
        "summer-oat",
        "meadow",
        "rape",
        "hop",
        "winter-spelt",
        "winter-triticale",
        "beans",
        "peas",
        "potatoes",
        "soybeans",
        "asparagus",
        "winter-wheat",
        "winter-barley",
        "winter-rye",
        "summer-barley",
        "maize",
        "unknown",
    ]

    def __init__(
        self,
        root_dir: Path,
        split: str,
        year: T.Optional[int] = None,
        fold: int = 0,
        transform: T.Optional[T.Callable] = None,
        return_paths: bool = False,
        concat: bool = True,
    ):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be one of ['train', 'val', 'test']"
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        assert root_dir.exists(), f"Root directory {root_dir} does not exist"

        if split in ["train", "val"]:
            if year is None:
                df1 = pd.read_csv(
                    root_dir / f"2016_{split}_fold{fold}.tileids", header=None
                )[0].apply(lambda x: f"2016/{x}")
                df2 = pd.read_csv(
                    root_dir / f"2017_{split}_fold{fold}.tileids", header=None
                )[0].apply(lambda x: f"2017/{x}")
                self.data_paths = pd.concat([df1, df2], ignore_index=True)
            else:
                self.data_paths = pd.read_csv(
                    root_dir / f"{year}_{split}_fold{fold}.tileids", header=None
                )[0]
        else:
            if year is None:
                df1 = pd.read_csv(root_dir / f"2016_{split}.tileids", header=None)[
                    0
                ].apply(lambda x: f"2016/{x}")
                df2 = pd.read_csv(root_dir / f"2017_{split}.tileids", header=None)[
                    0
                ].apply(lambda x: f"2017/{x}")
                self.data_paths = pd.concat([df1, df2], ignore_index=True)
            else:
                self.data_paths = pd.read_csv(
                    root_dir / f"{year}_{split}.tileids", header=None
                )[0]

        if year is None:
            print(
                f"Using data from 2016 and 2017 for {split} split - using {len(self.data_paths)} samples"
            )
            print(f"Data paths: {self.data_paths.head()}")

        if year is None:
            self.root_dir = root_dir
        else:
            self.root_dir = root_dir / f"{year}"

        self.index_to_label = {i: label for i, label in enumerate(self.classes)}
        self.transform = transform
        self.return_paths = return_paths
        self.concat = concat
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        img_name = os.path.join(self.root_dir, f"{self.data_paths.iloc[index]}.pkl")

        with open(img_name, "rb") as handle:
            data = pickle.load(handle, encoding="latin1")

        if self.transform is not None:
            data = self.transform(data)
        else:
            return self.read(index, abs=False)

        if self.concat:
            return data["inputs"], data["labels"]

        if self.return_paths:
            return data, img_name

        return data

    def read(self, index: int, abs: bool = False):
        """
        read single dataset sample corresponding to index (index number)
        without applying any data transformations.
        """
        if type(index) == int:
            img_name = os.path.join(self.root_dir, self.data_paths.iloc[index, 0])
        if type(index) == str:
            if abs:
                img_name = index
            else:
                img_name = os.path.join(self.root_dir, index)

        with open(img_name, "rb") as handle:
            sample = pickle.load(handle, encoding="latin1")

        return sample
