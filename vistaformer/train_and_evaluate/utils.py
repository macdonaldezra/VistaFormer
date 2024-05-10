import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours}h:{minutes}m:{seconds:.2f}s"


def get_current_date():
    return datetime.now().strftime("%y_%b_%d/%H_%M_%S")


def add_module_prefix_if_missing(state_dict):
    """Prepends 'module.' to keys in the state_dict if it's not already there."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("module."):
            new_key = f"module.{k}"
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def load_ddp_model(model: nn.Module, path: Path) -> nn.Module:
    """
    Load a model from the given path and wrap it in Distributed Data Parallel (DDP).
    """
    device = torch.device("cuda", torch.distributed.get_rank())
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {path.as_posix()}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = add_module_prefix_if_missing(checkpoint["model_state_dict"])
    # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    #     checkpoint["model_state_dict"], "module."
    # )
    model.load_state_dict(state_dict)

    return model


def save_ddp_model(model, optimizer, criterion, output_path):
    """
    Save the model module, optimizer and loss function iff the
    current process is the master process.
    """
    if torch.distributed.get_rank() == 0:
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            output_path,
        )


def get_output_path(
    output_path: Path,
    model_name: str,
    dataset: str,
    image_size: int,
    epochs: int,
    learning_rate: float,
):
    """
    Generate an output path with the following structure:

    output_path / model_name / dataset / image_size / epochs / learning_rate / <timestamp_of_experiment_start>
    """
    out_path = (
        output_path
        / model_name
        / dataset
        / f"image_size_{image_size}"
        / f"epochs_{epochs}"
        / f"lr_{learning_rate}"
        / get_current_date()
    )
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    return out_path


# Pretty print the FLOPs
def pretty_flops(num_flops: int):
    """
    Pretty print the number of FLOPs.
    """
    units = [("GFLOPs", 1e9), ("MFLOPs", 1e6), ("KFLOPs", 1e3), ("FLOPs", 1)]
    for unit_name, unit_value in units:
        if num_flops >= unit_value:
            return f"{num_flops / unit_value:.2f} {unit_name}"

    return "0 FLOPs"
