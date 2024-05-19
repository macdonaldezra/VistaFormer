import pickle

from pathlib import Path
from typing import Optional, Callable

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.utils.data
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset

from vistaformer.config import TrainingConfig
from vistaformer.datasets.pastis.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def get_dist_dataloader(
    rank: int,
    world_size: int,
    config: TrainingConfig,
    transform: Optional[Callable] = None,
) -> dict[str, torch.utils.data.DataLoader]:
    """
    return a distributed data loader with the train, test, and validation datasets
    """
    if transform is None:
        train_transform = get_train_transforms(
            seq_len=config.max_seq_len,
            use_dates=config.dataset.kwargs.get("use_dates", False),
            sample_seq=config.dataset.kwargs.get("sample_seq", False),
        )
        val_transform = get_val_transforms(
            seq_len=config.max_seq_len,
            use_dates=config.dataset.kwargs.get("use_dates", False),
        )
    else:
        train_transform = transform
        val_transform = transform

    train_folds = config.dataset.kwargs.get("train_folds", [1, 2, 3])
    train_dataset = get_pastis_dataset(config, train_folds, train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        # collate_fn=filter_labels_collate_fn,
    )

    val_folds = config.dataset.kwargs.get("val_folds", [4])
    val_dataset = get_pastis_dataset(config, val_folds, val_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        # collate_fn=filter_labels_collate_fn,
    )

    test_folds = config.dataset.kwargs.get("test_folds", [5])
    test_dataset = get_pastis_dataset(config, test_folds, val_transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        sampler=test_sampler,
        # collate_fn=filter_labels_collate_fn,
    )

    return {"train": train_dataloader, "test": test_dataloader, "val": val_dataloader}


def get_dataloader(
    config: TrainingConfig,
    fold: int,
    transform: Optional[Callable] = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    if transform is None:
        # Transforms for inference
        transform = get_val_transforms(
            seq_len=config.max_seq_len,
            use_dates=config.dataset.kwargs.get("use_dates", False),
        )
    dataset = PASTISDataset(
        root_dir=config.dataset.path,
        folds=fold,
        transform=transform,
        use_dates=config.dataset.kwargs.get("use_dates", False),
        concat_data=config.dataset.kwargs.get("concat_data", False),
        target=config.task,
        ignore_label=config.dataset.kwargs.get("ignore_labels", 0),
        sats=config.dataset.kwargs.get("sats", ["S2", "S1A", "S1D"]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=shuffle,
        num_workers=config.dataset.num_workers,
        # collate_fn=filter_labels_collate_fn,
    )

    return dataloader


class PASTISDataset(Dataset):
    """
    PASTIS dataset.

    The `PASTIS <https://github.com/VSainteuf/pastis-benchmark>`__
    dataset is a dataset for time-series panoptic segmentation of agricultural parcels.

    Dataset features:

    * support for the original PASTIS and PASTIS-R versions of the dataset
    * 2,433 time-series with 10 m per pixel resolution (128x128 px)
    * 18 crop categories, 1 background category, 1 void category
    * semantic and instance annotations
    * 3 Sentinel-1 Ascending bands
    * 3 Sentinel-1 Descending bands
    * 10 Sentinel-2 L2A multispectral bands

    Dataset format:

    * time-series and annotations are in numpy format (.npy)

    Dataset classes:

    0. Background
    1. Meadow
    2. Soft Winter Wheat
    3. Corn
    4. Winter Barley
    5. Winter Rapeseed
    6. Spring Barley
    7. Sunflower
    8. Grapevine
    9. Beet
    10. Winter Triticale
    11. Winter Durum Wheat
    12. Fruits Vegetables Flowers
    13. Potatoes
    14. Leguminous Fodder
    15. Soybeans
    16. Orchard
    17. Mixed Cereal
    18. Sorghum
    19. Void Label

    If you use this dataset in your research, please cite the following papers:

    * https://doi.org/10.1109/ICCV48922.2021.00483
    * https://doi.org/10.1016/j.isprsjprs.2022.03.012
    """

    classes = [
        "background",  # all non-agricultural land
        "meadow",
        "soft-winter-wheat",
        "corn",
        "winter-barley",
        "winter-rapeseed",
        "spring-barley",
        "sunflower",
        "grapevine",
        "beet",
        "winter-triticale",
        "winter-durum-wheat",
        "fruits-vegetables-flowers",
        "potatoes",
        "leguminous-fodder",
        "soybeans",
        "orchard",
        "mixed-cereal",
        "sorghum",
        "void-label",  # for parcels mostly outside their patch
    ]
    cmap = {
        0: (0, 0, 0, 255),
        1: (174, 199, 232, 255),
        2: (255, 127, 14, 255),
        3: (255, 187, 120, 255),
        4: (44, 160, 44, 255),
        5: (152, 223, 138, 255),
        6: (214, 39, 40, 255),
        7: (255, 152, 150, 255),
        8: (148, 103, 189, 255),
        9: (197, 176, 213, 255),
        10: (140, 86, 75, 255),
        11: (196, 156, 148, 255),
        12: (227, 119, 194, 255),
        13: (247, 182, 210, 255),
        14: (127, 127, 127, 255),
        15: (199, 199, 199, 255),
        16: (188, 189, 34, 255),
        17: (219, 219, 141, 255),
        18: (23, 190, 207, 255),
        19: (255, 255, 255, 255),
    }

    def __init__(
        self,
        root_dir: Path,
        folds: Optional[list[int]] = None,
        transform: Callable = None,
        sats: list[str] = ["S2", "S1A", "S1D"],
        target: str = "classification",  # classification or semantic segmentation
        use_dates: bool = False,
        ignore_label: Optional[int] = None,
        concat_data: bool = False,
        remap_void_label: bool = False,  # Do not set this parameter to true unless this is for panoptic segmentation...
        dtype: torch.dtype = torch.float32,
    ):
        super(PASTISDataset, self).__init__()
        assert sum([s in {"S1A", "S1D", "S2"} for s in sats]) == len(
            sats
        ), "Unknown satellite name (available: S2/S1A/S1D)"
        if folds is not None:
            assert sum([s in {1, 2, 3, 4, 5} for s in folds]) == len(
                folds
            ), "Unknown fold (available: 1/2/3/4/5)"
        assert target in {
            "classification",
            "semantic",
        }, "Unknown target (available: classification/semantic segmentation)"

        self.root_dir = root_dir
        self.transform = transform
        self.sats = [s.casefold() for s in sats]
        self.folds = folds
        self.use_dates = use_dates
        self.ignore_label = ignore_label
        self.target = target
        self.concat_data = concat_data
        self.dtype = dtype
        self.remap_void_label = remap_void_label
        if self.remap_void_label:
            self.num_classes = len(self.classes) - 1
            self.index_to_label = {
                i: label for i, label in enumerate(self.classes) if i != 19
            }
        else:
            self.index_to_label = {i: label for i, label in enumerate(self.classes)}

        self.meta_df = gpd.read_file(self.root_dir / "metadata.geojson")
        self.file_dir = (
            self.root_dir / "samples"
        )  # Assumes pre-processing has occurred and outputted data has been saved to samples dir

        self.file_paths = []
        if self.folds is not None:
            self.meta_df = self.meta_df[self.meta_df.Fold.isin(self.folds)]
            for file in self.file_dir.glob("*.pickle"):
                lookup_path = int(file.name.split("_")[0])
                found = self.meta_df.loc[
                    self.meta_df.ID_PATCH.astype(int) == lookup_path
                ]
                if len(found) > 0 and int(found.Fold.values[0]) in self.folds:
                    self.file_paths.append(file.as_posix())
        else:
            self.file_paths = list(self.file_dir.glob("*.pickle"))

        self.set_colormap()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        with open(self.file_paths[index], "rb") as f:
            data = pickle.load(f)

        if self.transform is not None:
            data = self.transform(data)

        if self.target == "classification":
            if isinstance(data, torch.Tensor):
                label = data["class_label"].clone()
            else:
                label = data["class_label"]
        elif self.target == "semantic":
            if isinstance(data, torch.Tensor):
                label = data["labels"][0].clone()
            else:
                label = data["labels"][0]

        del data["class_label"]
        del data["labels"]
        # If date data is used it will be concatenated with other tensors so we remove it here...
        del data["s2_doy"]
        del data["s1d_doy"]
        del data["s1a_doy"]

        out_data = None
        if self.concat_data:
            out_data = torch.Tensor()
            if "s1d" in self.sats:
                out_data = torch.cat((out_data, data["s1d"]), dim=1)
            if "s1a" in self.sats:
                out_data = torch.cat((out_data, data["s1a"]), dim=1)
            if (
                "s2" in self.sats
            ):  # Order of operations here matters... I modified the TSViT model to expect dates on the last dimension...
                out_data = torch.cat((out_data, data["s2"]), dim=1)
        else:
            out_data = {}
            if "s2" in self.sats:
                out_data["s2"] = data["s2"]
            if "s1d" in self.sats:
                out_data["s1d"] = data["s1d"]
            if "s1a" in self.sats:
                out_data["s1a"] = data["s1a"]

        if self.remap_void_label:
            label[label == 19] = 0

        return out_data, label

    @staticmethod
    def display_images(image_dict):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.tight_layout()

        for i, (title, image) in enumerate(image_dict.items()):
            row = i // 3
            col = i % 3
            axes[row, col].imshow(image)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

        plt.show()

    def get_image_and_mask(self, index: int, time_index: int = 6):
        """
        Given an image index, return the image and mask while applying
        """
        with open(self.file_paths[index], "rb") as f:
            data = pickle.load(f)

        if self.transform is not None:
            data = self.transform(data)

        label = data["labels"][0]
        print(f"Shape of s2 data: {data['s2'].shape}")
        image = data["s2"][time_index, [2, 1, 0]].cpu().numpy()
        print(f"Shape of image data: {image.shape}")

        maxs = image.max(axis=(1, 2))
        mins = image.min(axis=(1, 2))
        image = (image - mins[:, None, None]) / (maxs - mins)[:, None, None]
        image = image.swapaxes(0, 2).swapaxes(0, 1)
        image = np.clip(image, a_max=1, a_min=0)

        return image, label

    def set_colormap(self):
        cm = matplotlib.colormaps.get_cmap("tab20")
        def_colors = cm.colors
        cus_colors = ["k"] + [def_colors[i] for i in range(1, 20)] + ["w"]
        cmap = ListedColormap(colors=cus_colors, name="agri", N=21)

        self.colormap = cmap

    def display_model_output(self, index: int, predicted_mask: np.ndarray = None):
        """
        Given an image index, display the image, mask, and predicted mask.
        """
        image, mask = self.get_image_and_mask(index)

        # Create a color image using RGB bands
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))

        # Display the image
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # Display the mask
        axes[1].imshow(mask, cmap=self.colormap, vmin=0, vmax=20)
        axes[1].set_title("Mask")
        axes[1].axis("off")

        # Display the predicted mask
        axes[2].imshow(predicted_mask, cmap=self.colormap, vmin=0, vmax=20)
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        fig.tight_layout()
        plt.show()


def get_pastis_dataset(
    config: TrainingConfig, folds: list[int], transforms: Callable
) -> PASTISDataset:
    """
    Get the PASTIS dataset.
    """
    return PASTISDataset(
        root_dir=config.dataset.path,
        folds=folds,
        transform=transforms,
        use_dates=config.dataset.kwargs.get("use_dates", False),
        concat_data=config.dataset.kwargs.get("concat_data", False),
        target=config.task,
        ignore_label=config.dataset.kwargs.get("ignore_labels", 0),
        sats=config.dataset.kwargs.get("sats", ["S2", "S1A", "S1D"]),
    )


def my_collate(batch):
    """Filter out sample where mask is zero everywhere"""
    idx = [b["unk_masks"].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
