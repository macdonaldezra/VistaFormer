import typing as T
from pathlib import Path

import pandas as pd
import torch
from torchmetrics import (
    Accuracy,
    ClasswiseWrapper,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)


def get_classwise_metrics(
    num_classes: int,
    device: str,
    labels: T.List[str] = None,
    ignore_index: T.Optional[int] = None,
) -> MetricCollection:
    """
    Get metrics for model.
    """
    if num_classes < 2:
        raise ValueError("Number of classes must be greater than 1.")

    return MetricCollection(
        {
            "F1": ClasswiseWrapper(
                F1Score(
                    task="multiclass",
                    average=None,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                labels,
            ).to(device),
            "Precision": ClasswiseWrapper(
                Precision(
                    task="multiclass",
                    average=None,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                labels,
            ).to(device),
            "Recall": ClasswiseWrapper(
                Recall(
                    task="multiclass",
                    average=None,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                labels,
            ).to(device),
            "IoU": ClasswiseWrapper(
                JaccardIndex(
                    task="multiclass",
                    average=None,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                labels,
                prefix="iou_",
            ).to(device),
            "Accuracy": ClasswiseWrapper(
                Accuracy(
                    task="multiclass",
                    average=None,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                labels,
            ).to(device),
        }
    )


def get_overall_metrics(
    num_classes: int, device: str, ignore_index: T.Optional[int] = None
) -> MetricCollection:
    return MetricCollection(
        {
            # Macro Multi-Class Segmentation metrics
            "macro-F1": F1Score(
                task="multiclass",
                average="macro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "macro-Precision": Precision(
                task="multiclass",
                average="macro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "macro-Recall": Recall(
                task="multiclass",
                average="macro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "macro-IoU": JaccardIndex(
                task="multiclass",
                average="macro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "macro-Accuracy": Accuracy(
                task="multiclass",
                average="macro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "micro-F1": F1Score(
                task="multiclass",
                average="micro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "micro-Precision": Precision(
                task="multiclass",
                average="micro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "micro-Recall": Recall(
                task="multiclass",
                average="micro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "micro-IoU": JaccardIndex(
                task="multiclass",
                average="micro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
            "micro-Accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=num_classes,
                validate_args=False,
                ignore_index=ignore_index,
            ).to(device),
        }
    )


def get_metric_df(metrics: T.Dict[str, T.Dict[str, float]]) -> pd.DataFrame:
    """
    Create a DataFrame from all the class-wise metrics included in the metrics dict.
    """
    columns = ["F1", "Accuracy", "Precision", "Recall", "IoU"]
    indices = [
        name.split("_")[1]
        for name in metrics.keys()
        if columns[0].casefold() in name.casefold()
    ]

    metric_df = pd.DataFrame(columns=columns, index=indices)
    for metric, value in metrics.items():
        for column in columns:
            if column.casefold() in metric.casefold():
                metric_df.loc[metric.split("_")[1], column] = value.cpu().item()

    return metric_df


def generate_test_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    output_path: Path,
    device: str,
    multi_input: bool = False,
    ignore_index: T.Optional[int] = None,
    rank: T.Optional[int] = None,
) -> T.Dict[str, T.Dict[str, float]]:
    """
    Generate test metrics.
    """
    if rank == 0 or rank is None:
        print(f"Generating test metrics for {num_classes} classes...")

    class_metrics = get_classwise_metrics(
        num_classes=num_classes,
        device=device,
        labels=list(dataloader.dataset.index_to_label.values()),
        ignore_index=ignore_index,
    )
    overall_metrics = get_overall_metrics(num_classes, device, ignore_index)

    model = model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            target = target.to(device)
            if multi_input:
                s2, s1a = data["s2"].to(device), data["s1a"].to(device)
                output = model(s2, s1a)
                del data
            else:
                data = data.to(device)
                output = model(data)

            output = torch.argmax(output, dim=1)
            class_metrics.update(output, target)
            overall_metrics.update(output, target)

    out_class_metrics = class_metrics.compute()
    out_overall_metrics = overall_metrics.compute()
    class_metrics.reset()
    overall_metrics.reset()

    if rank == 0 or rank is None:
        metric_df = get_metric_df(out_class_metrics)
        print(
            f"Number of rows that have accuracy greater than 0: {len(metric_df.loc[metric_df.Accuracy > 0])}"
        )
        print(f"Saving metrics to {output_path / 'test_metrics.csv'}")
        metric_df.to_csv(output_path / "test_metrics.csv")
        print(f"Head of metrics:\n{metric_df.loc[metric_df.Accuracy > 0].head(30)}\n\n")
        print(f"Overall Metrics for test set:")
        for metric, value in out_overall_metrics.items():
            print(f"{metric}: {value.cpu().item():.4f}")
