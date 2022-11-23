import numpy as np

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