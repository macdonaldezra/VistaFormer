import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from remote_cattn.models import get_model
from .utils import get_output_path, cleanup, setup
from remote_cattn.config import TrainingConfig, get_model_config, config_to_yaml
from remote_cattn.datasets import get_dist_dataloaders
from remote_cattn.loss import get_loss
from remote_cattn.train_and_evaluate.train_segmentation import train_segmentation
from remote_cattn.train_and_evaluate.train_classification import train_classifier
from remote_cattn.train_and_evaluate.eval_utils import generate_test_metrics
from remote_cattn.train_and_evaluate.utils import load_ddp_model

num_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained_weights(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    """
    Load pre-trained weights from a checkpoint file.
    """
    print(f"Loading pretrained weights from {path.as_posix()}")
    checkpoint = torch.load(path.as_posix())
    current_model_dict = model.state_dict()
    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(
            current_model_dict.keys(), checkpoint["model_state_dict"].values()
        )
    }
    model.load_state_dict(new_state_dict, strict=False)

    return model


def main(rank: int, world_size: int, config: TrainingConfig, config_path: Path) -> None:
    """
    Function to configure and tear down DDP training according to configurations
    provided in the config file.
    """
    setup(rank, world_size)
    dist.barrier()  # Try and ensure that both systems use the same minute for the output path...
    out_path = get_output_path(
        config.output_path,
        config.model_name,
        config.dataset.name,
        config.image_size,
        config.epochs,
        config.learning_rate,
    )
    if rank == 0:
        out_log_path = out_path / "logs"
        out_log_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(out_log_path)
    else:
        writer = None

    device = torch.device("cuda", rank)
    dataloaders = get_dist_dataloaders(rank=rank, world_size=world_size, config=config)
    model = get_model(config)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss = get_loss(config, device=device)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=config.learning_rate, **config.optimizer_kwargs
    )

    if config.model_weights is not None:
        ddp_model = load_pretrained_weights(ddp_model, config.model_weights)

    if config.lr_scheduler == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(dataloaders["train"]),
            pct_start=0.1,
            div_factor=25,
            cycle_momentum=True,
            final_div_factor=25,
            **config.lr_scheduler_kwargs,
        )
    else:
        print(f"Using no learning rate scheduler...")
        lr_scheduler = None

    print(
        f"Running DDP process {rank} of {world_size} for {config.epochs} epochs, "
        + f" Batch Size: {config.dataset.batch_size}, Image size: {config.image_size}, "
        + f"Model Name: {config.model_name}, Dataset: {config.dataset.name}, Learning Rate: {config.learning_rate}\n"
        + f"Number of trainable parameters: {num_params(ddp_model)}\n"
    )

    if rank == 0:
        config_to_yaml(config, out_path / "config.yaml")
    if config.task == "classification":
        ddp_model = train_classifier(
            model=ddp_model,
            criterion=loss,
            optimizer=optimizer,
            dataloader=dataloaders["train"],
            device=device,
            epochs=config.epochs,
            output_path=out_path,
            rank=rank,
            lr_scheduler=lr_scheduler,
            multi_input=config.is_multi_input_model,
            logger=writer,
        )
    elif config.task == "semantic":
        ddp_model = train_segmentation(
            model=ddp_model,
            criterion=loss,
            optimizer=optimizer,
            dataloaders=dataloaders,
            device=device,
            epochs=config.epochs,
            output_path=out_path,
            rank=rank,
            config=config,
            lr_scheduler=lr_scheduler,
            multi_input=config.is_multi_input_model,
            logger=writer,
            ignore_index=config.ignore_index,
        )
        # Functionality only implemented for semantic segmentation :-(
        dist.barrier()  # Ensure all model saving events have finished before evaluating

        if rank == 0:
            print(
                f"Loading best model that has been saved so far for evaluation on the test dataset..."
            )
        model_path = out_path / "best_model.pth"
        ddp_model = load_ddp_model(ddp_model, model_path)

        generate_test_metrics(
            model=ddp_model,
            dataloader=dataloaders["test"],
            num_classes=config.num_classes,
            device=device,
            output_path=out_path,
            multi_input=config.is_multi_input_model,
            ignore_index=config.ignore_index,
            rank=rank,
        )
    else:
        raise ValueError(f"Task {config.task} not supported...")

    torch.cuda.empty_cache()
    cleanup()
    if rank == 0:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the configuration file",
    )

    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Requires at least 1 GPU to run, but got {n_gpus}"
    world_size = n_gpus

    config = get_model_config(args.config)
    if config.use_wandb:
        import wandb

        wandb.setup()

    mp.spawn(main, args=(world_size, config, args.config), nprocs=n_gpus, join=True)
