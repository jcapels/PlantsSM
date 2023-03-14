import os
from enum import Enum
from typing import Type

from plants_sm.models.constants import FileConstants
from plants_sm.models.model import Model
from plants_sm.models.pytorch_model import PyTorchModel
from plants_sm.models.tensorflow_model import TensorflowModel


class ModelFilesClasses(Enum):
    """Enumeration of model types."""

    PYTORCH_MODEL_WEIGHTS = PyTorchModel
    PYTORCH_MODEL_PKL = PyTorchModel

    TENSORFLOW_MODEL = TensorflowModel

    MODEL_PARAMETERS_PKL = None


class ModelFileEnumeratorsUtils:
    """Class with utilities for model file enumerators."""

    @staticmethod
    def get_class_to_load_from_directory(path: str) -> Type[Model]:
        """
        Get the class to load from the directory.

        Parameters
        ----------
        path: str
            path to the directory

        Returns
        -------
        class: Type[Model]
            class to load
        """

        files = os.listdir(path)
        for file in files:
            for enumerator, model_file in FileConstants.__members__.items():
                if file == model_file.value:
                    if ModelFilesClasses[enumerator].value:
                        return ModelFilesClasses[enumerator].value


