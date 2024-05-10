import json
import os
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field
from ruamel.yaml import YAML
from ruamel.yaml.composer import ComposerError
from ruamel.yaml.parser import ParserError
from ruamel.yaml.scanner import ScannerError

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"


class DatasetConfig(BaseModel):
    path: Path = Field(Path, description="Path to the dataset")
    name: str = Field(..., description="Name of the dataset")
    batch_size: int = Field(
        4, gt=0, description="Batch size for training and validation"
    )
    num_workers: int = Field(
        4, ge=0, description="Number of subprocesses to use for data loading"
    )
    kwargs: dict = Field(
        {}, description="Additional keyword arguments for the DataLoader"
    )


class TrainingConfig(BaseModel):
    image_size: int = Field(..., gt=0, description="Size of the input image")
    task: str = Field(..., description="Task to perform")
    max_seq_len: int = Field(..., gt=0, description="Maximum sequence length")
    num_classes: int = Field(..., gt=0, description="Number of classes in the dataset")
    learning_rate: float = Field(
        ..., gt=0, description="Learning rate for the optimizer"
    )
    in_channels: int = Field(..., gt=0, description="Number of input channels")
    epochs: int = Field(..., gt=0, description="Number of epochs to train the model")
    ignore_index: Optional[int] = Field(
        None, description="Index to ignore in loss and metrics"
    )
    dataset: DatasetConfig = Field(..., description="Dataset-specific configurations")

    # LR Scheduler configuration
    lr_scheduler: str = Field("onecycle", description="Learning rate scheduler to use")
    lr_scheduler_kwargs: dict = Field(
        {}, description="Additional keyword arguments for the learning rate scheduler"
    )

    # Model configuration
    model_name: str = Field("", description="Name of the model to use")
    model_weights: Optional[Path] = Field(
        None, description="Path to the model weights to load"
    )
    model_kwargs: dict = Field(
        {}, description="Additional keyword arguments for the model"
    )
    is_multi_input_model: bool = Field(
        False, description="Whether model requires more than one input"
    )

    # Loss function configuration
    loss_fn: str = Field("cross_entropy", description="Loss function to use")
    loss_fn_kwargs: dict = Field(
        {}, description="Additional keyword arguments for the loss function"
    )

    # Optimizer configuration
    optimizer: str = Field("adam", description="Optimizer to use")
    optimizer_kwargs: dict = Field(
        {}, description="Additional keyword arguments for the optimizer"
    )

    # Additional experiment configurations
    output_path: Path = Field(..., description="Path to save the model and logs")
    comment: str = Field("", description="Additional comment for the experiment")
    use_wandb: bool = Field(
        False, description="Whether to use Weights & Biases for logging"
    )


def parse_yaml(data: Union[IO, bytes]) -> Union[None, Dict[str, Any]]:
    """
    Parse bytes or input data that ideally contains valid yaml.
    """
    try:
        yaml = YAML(typ="safe")
        return yaml.load(data)
    except (ScannerError, ParserError) as err:
        print(f"Error while trying to parse YAML:\n {err}")
        return None
    except ComposerError as err:
        print(f"Provided more than one YAML document:\n {err}")
        return None


def read_yaml(filepath: Path) -> Union[None, Dict[str, Any]]:
    """
    Read in a YAML file and return file contents in a dict.
    """
    try:
        fptr = open(filepath, "r")
        data = parse_yaml(fptr)
    except FileNotFoundError as err:
        print(f"File {err.filename} not found.")
        return None
    except IOError as err:
        print(f"Unable to parse contents of {err.filename}.")
        return None

    return data


def config_to_yaml(config: TrainingConfig, filepath: Path) -> None:
    """
    Output a config object to a yaml file.

    Args:
        filepath (Path): Path to the desired output file.
    """
    print(f"Outputting config object to {filepath.as_posix()}")
    filepath.parent.mkdir(exist_ok=True, parents=True)
    config_dict = json.loads(config.json())
    try:
        file = open(filepath, "w")
        yaml.dump(config_dict, file)
        file.close()
    except Exception as err:
        print(f"Unable to open {filepath} for writing.\n\n{err}")


def get_model_config(config_path: Optional[Path] = None) -> TrainingConfig:
    """
    Try and parse a YAML file and return the YAML file parsed as a Training Config object.
    """
    if config_path is None:
        if "MODEL_CONFIG" not in os.environ:
            print(
                "Variable MODEL_CONFIG not found. Falling back to default config file from project root."
            )
        config_path = Path(
            os.getenv("MODEL_CONFIG", PROJECT_ROOT / "model_default.yaml")
        )

    try:
        config_data = read_yaml(config_path)
    except OSError:
        print("Unable to parse config from provided filepath.")
        raise ValueError("Unable to load model settings.")

    if not config_data:
        print(
            "Returned config is empty. Please check the format of your config file and try again."
        )

    config = TrainingConfig.parse_obj(config_data)

    return config
