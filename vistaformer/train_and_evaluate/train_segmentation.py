import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import time

from torch.utils.tensorboard import SummaryWriter

from vistaformer.config import TrainingConfig

from .utils import format_time, save_ddp_model
from torchmetrics import Accuracy, MeanMetric, F1Score, JaccardIndex


def train_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    device: str,
    multi_input: bool = False,
):
    """
    Perform a single training step for the given model, criterion, optimizer, data, and target.

    Assumes the model has been set to training mode and that the target has been moved to the device.
    """
    optimizer.zero_grad()
    if multi_input:
        s2, s1a = data["s2"].to(device), data["s1a"].to(device)
        outputs = model(s2, s1a)
    else:
        data = data.to(device)
        outputs = model(data)

    loss = criterion(outputs, target)

    loss.backward()
    optimizer.step()

    return outputs, loss


def validate_step(
    model: nn.Module,
    criterion: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    device: str,
    multi_input: bool = False,
):
    """
    Perform a single validation step for the given model, criterion, data, and target.

    Assumes the model has been set to evaluation mode and that the target has been moved to the device.
    """
    with torch.no_grad():
        if multi_input:
            s2, s1a = data["s2"].to(device), data["s1a"].to(device)
            outputs = model(s2, s1a)
        else:
            data = data.to(device)
            outputs = model(data)

    loss = criterion(outputs, target)

    return outputs, loss


def train_segmentation(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: dict[str, DataLoader],
    device: str,
    epochs: int,
    output_path: str,
    rank: int,
    config: TrainingConfig,
    multi_input: bool = False,
    lr_scheduler: Optional[nn.Module] = None,
    logger: Optional[SummaryWriter] = None,
    ignore_index: Optional[int] = None,
):
    """
    Train a segmentation model using the given criterion and optimizer.

    Note: This function uses Distributed Data Parallel (DDP) to train the model, expects the models outputs
    to be logits, and computes the step for the learning rate scheduler after each batch.

    Args:
        model: The model to train.
        criterion: The criterion to use for the loss.
        optimizer: The optimizer to use for the training.
        dataloaders: A dictionary containing the training and validation dataloaders.
        device: The device to use for training.
        epochs: The number of epochs to train the model.
        output_path: The path to save the model and logs.
        rank: The rank of the current process.
        world_size: The total number of processes.
    """
    if (
        config.use_wandb
    ):  # I really don't want to have to build another Slurm container mkay
        import wandb

        wandb.init(project="remote-cattn", name=config.comment, group="DDP")
        wandb.config.update(config.dict())
        wandb.watch(model, log="all")

    iou = JaccardIndex(  # mIoU score
        average="macro",
        task="multiclass",
        num_classes=config.num_classes,
        validate_args=False,
        ignore_index=ignore_index,
    ).to(device)
    f1 = F1Score(
        average="macro",
        task="multiclass",
        num_classes=config.num_classes,
        validate_args=False,
        ignore_index=ignore_index,
    ).to(device)
    accuracy = Accuracy(  # overall accuracy
        average="micro",
        task="multiclass",
        num_classes=config.num_classes,
        validate_args=False,
        ignore_index=ignore_index,
    ).to(device)
    avg_loss = MeanMetric().to(device)

    model.train()
    best_iou = 0.0

    for epoch in range(epochs):
        dataloaders["train"].sampler.set_epoch(epoch)
        dataloaders["val"].sampler.set_epoch(epoch)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            start_time = time.time()
            for data, target in dataloaders[phase]:
                target = target.to(device)
                if phase == "train":
                    output, loss = train_step(
                        model, criterion, optimizer, data, target, device, multi_input
                    )
                else:
                    output, loss = validate_step(
                        model, criterion, data, target, device, multi_input
                    )

                output = output.argmax(dim=1)  # do not put before loss calculation!
                accuracy(output, target), avg_loss(loss), iou(output, target), f1(
                    output, target
                )

                if lr_scheduler is not None and phase == "train":
                    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
                    lr_scheduler.step()

            epoch_duration = time.time() - start_time
            acc, avg_l, iou_score, f1_score = (
                accuracy.compute(),
                avg_loss.compute(),
                iou.compute(),
                f1.compute(),
            )
            if rank == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} {phase.capitalize()} - {format_time(epoch_duration)} - "
                    f"{phase.capitalize()} Loss: {avg_l:.4f} - {phase.capitalize()} Accuracy: {acc*100:.4f}% - "
                    f"{phase.capitalize()} IoU: {iou_score*100:.4f}% - {phase.capitalize()} F1: {f1_score*100:.4f}%"
                )
                if logger is not None:
                    if config.use_wandb:
                        wandb.log(
                            {
                                f"{phase.capitalize()} Loss": avg_l,
                                f"{phase.capitalize()} Accuracy": acc,
                                f"{phase.capitalize()} IoU": iou_score,
                                f"{phase.capitalize()} F1": f1_score,
                            }
                        )
                    logger.add_scalar(f"{phase}/Loss", avg_l, epoch)
                    logger.add_scalar(f"{phase}/Accuracy", acc, epoch)
                    logger.add_scalar(f"{phase}/IoU", iou_score, epoch)
                    logger.add_scalar(f"{phase}/F1", f1_score, epoch)

            if iou_score > best_iou and phase == "val":
                best_iou = iou_score
                if (
                    rank == 0 and best_iou > 0.35
                ):  # save the model only if the micro-IoU is greater than 0.35...
                    print(
                        f"Saving best model in terms of mIoU score on validation dataset after {epoch} epochs..."
                    )
                    save_ddp_model(
                        model, optimizer, criterion, output_path / "best_model.pth"
                    )

            # reset all metrics
            accuracy.reset(), avg_loss.reset(), iou.reset(), f1.reset()

    if rank == 0 and best_iou > 0.35:
        print(f"Saving final model after {epoch} epochs...")
        save_ddp_model(model, optimizer, criterion, output_path / "final_model.pth")

    return model
