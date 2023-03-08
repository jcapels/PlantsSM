# Supported problem types
from enum import Enum

BINARY = 'binary'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'
SOFTCLASS = 'softclass'
QUANTILE = 'quantile'

PROBLEM_TYPES = [BINARY, MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE]


# files constants

class FileConstants(Enum):

    PYTORCH_MODEL_WEIGHTS = 'pytorch_model_weights.pt'
    PYTORCH_MODEL_PKL = 'pytorch_model.pkl'

    TENSORFLOW_MODEL = 'tensorflow_model.h5'

    MODEL_PARAMETERS_PKL = 'model_parameters.pkl'

    MODEL_CLASSES_FILES_PKL = [PYTORCH_MODEL_PKL, TENSORFLOW_MODEL]