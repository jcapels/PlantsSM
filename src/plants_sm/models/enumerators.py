from enum import Enum


class ModelFilesConstants(Enum):
    """Enumeration of model types."""

    PYTORCH_MODEL_WEIGHTS = 'pytorch_model_weights.pt'
    PYTORCH_MODEL_PKL = 'pytorch_model.pkl'

    TENSORFLOW_MODEL = 'tensorflow_model.h5'

    MODEL_PARAMETERS_PKL = 'model_parameters.pkl'
