from vistaformer.config import TrainingConfig
from vistaformer.models.vistaformer import VistaFormer


def get_model(
    config: TrainingConfig,
):
    if config.model_name == "vistaformer":
        return VistaFormer(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            input_dim=config.image_size,
            **config.model_kwargs,
        )
    else:
        raise ValueError(f"Model {config.model_name} not supported")
