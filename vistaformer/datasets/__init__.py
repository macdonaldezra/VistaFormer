from vistaformer.config import TrainingConfig
from vistaformer.datasets.pastis.dataloader import (
    get_dist_dataloader as pastis_dist_dataloader,
)
from vistaformer.datasets.mtlcc.dataloader import (
    get_dist_dataloader as mtlcc_dist_dataloader,
)
from typing import Optional
from torch.utils.data import DataLoader


def get_dist_dataloaders(
    rank: int,
    world_size: int,
    config: TrainingConfig,
) -> Optional[DataLoader]:
    if config.dataset.name == "pastis":
        return pastis_dist_dataloader(
            rank=rank,
            world_size=world_size,
            config=config,
        )
    elif config.dataset.name == "mtlcc":
        return mtlcc_dist_dataloader(
            rank=rank,
            world_size=world_size,
            config=config,
        )
    else:
        raise ValueError(f"Dataset {config.dataset.name} not found.")
