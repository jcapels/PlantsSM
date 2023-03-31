import os
import warnings
from typing import Iterable

import numpy as np
from tensorflow import Tensor

from plants_sm.data_structures.dataset import Dataset
from plants_sm.io.pickle import write_pickle, is_pickable
from plants_sm.models.constants import REGRESSION, BINARY, FileConstants


def _convert_proba_to_unified_form(problem_type, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Ensures that y_pred_proba is in a consistent form across all models. For binary classification,
    converts y_pred_proba to a 1 dimensional array of prediction probabilities of the positive class. For
    multiclass and softclass classification, keeps y_pred_proba as a 2 dimensional array of prediction
    probabilities for each class. For regression, converts y_pred_proba to a 1 dimensional array of predictions.

    Parameters
    ----------
    y_pred_proba: np.ndarray
        Array of prediction probabilities

    Returns
    -------
    np.ndarray
        Array of prediction probabilities in a consistent form across all models
    """
    if problem_type == REGRESSION:
        if len(y_pred_proba.shape) == 1:
            return y_pred_proba
        else:
            return y_pred_proba[:, 1]
    elif problem_type == BINARY:
        if len(y_pred_proba.shape) == 1:
            return y_pred_proba
        elif y_pred_proba.shape[1] > 1:
            return y_pred_proba[:, 1]
        else:
            return y_pred_proba
    elif y_pred_proba.shape[1] > 2:  # Multiclass, Softclass
        return y_pred_proba
    else:  # Unknown problem type
        raise AssertionError(f'Unknown y_pred_proba format for `problem_type="{problem_type}"`.')


def write_model_parameters_to_pickle(model_parameters: dict, path: str) -> None:
    """
    Write the model parameters to a pickle file.

    Parameters
    ----------
    model_parameters: dict
        Dictionary of model parameters
    path: str
        Path to save the model
    """
    parameters = {}
    for key, value in model_parameters.items():
        if is_pickable(value):
            parameters[key] = value
        else:
            warning_str = f'Could not save {key} to save file. Skipping attribute {key}.'
            warnings.warn(warning_str)
    write_pickle(os.path.join(path, FileConstants.MODEL_PARAMETERS_PKL.value), parameters)


def array_from_tensor(tensor: Tensor) -> np.ndarray:
    """
    Converts a tensor to a numpy array.

    Parameters
    ----------
    tensor: Tensor
        The tensor to convert.

    Returns
    -------
    np.ndarray
        The converted array.
    """
    tensor = tensor.cpu().detach().numpy()
    return tensor


def array_reshape(array: np.ndarray) -> np.ndarray:
    """
    Reshapes a numpy array.

    Parameters
    ----------
    array: np.ndarray
        The array to reshape.

    Returns
    -------
    np.ndarray
        The reshaped array.
    """
    # Reshape array if necessary if it is a 1D array or if it is a 2D array with only 1 column
    if len(array.shape) == 1 or (len(array.shape) == 2 and array.shape[1] == 1):
        array = array.reshape(-1, 1)
    return array


def _batch_generator(dataset: Dataset) -> Iterable:
    """
    Generate batches of data.

    Parameters
    ----------
    dataset: Dataset
        Dataset to be used for training.

    Returns
    -------
    Union[list, dict]
        Batches of data.
    """
    while dataset.next_batch():
        if dataset.y is None:
            yield dataset.X
        else:
            yield dataset.X, dataset.y
