from vistaformer.config import TrainingConfig
from vistaformer.models.vistaformer import VistaFormer
from vistaformer.models.vistaformer_multi import VistaFormerMulti
from vistaformer.models.vistaformer_neighbourhood import VistaFormerNeighbourhood


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
    elif config.model_name == "vistaformer_multi":
        return VistaFormerMulti(
            first_in_channels=10,  # Channels have been fixed for working with the PASTIS dataset
            second_in_channels=6,
            num_classes=config.num_classes,
            input_dim=config.image_size,
            ignore_index=config.ignore_index,
            **config.model_kwargs,
        )
    elif config.model_name == "vistaformer_neighbourhood":
        return VistaFormerNeighbourhood(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            input_dim=config.image_size,
            **config.model_kwargs,
        )
    else:
        raise ValueError(f"Model {config.model_name} not supported")
