import os
import warnings

import numpy as np
import torch
from torch import nn

from plants_sm.io.pickle import read_pickle, write_pickle, is_pickable
from plants_sm.models.constants import REGRESSION, BINARY


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


def read_pytorch_model(path: str) -> nn.Module:
    """
    Read the model from the specified path.

    Parameters
    ----------
    path: str
        Path to read the model from

    Returns
    -------
    torch.nn.Module
    """
    weights_path = os.path.join(path, 'model.pt')
    model = read_pickle(os.path.join(path, 'model.pkl'))
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def save_pytorch_model(model: nn.Module, path: str) -> None:
    """
    Save the model to the specified path.

    Parameters
    ----------
    model: torch.nn.Module
        Model to be saved
    path: str
        Path to save the model

    Returns
    -------

    """
    weights_path = os.path.join(path, 'model.pt')
    torch.save(model.state_dict(), weights_path)
    write_pickle(model, os.path.join(path, 'model.pkl'))


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
    write_pickle(parameters, os.path.join(path, 'model_parameters.pkl'))
