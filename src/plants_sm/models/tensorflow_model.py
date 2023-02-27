import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.models._utils import _convert_proba_to_unified_form
from plants_sm.models.constants import REGRESSION, QUANTILE
from plants_sm.models.model import Model

import tensorflow as tf


class TensorflowModel(Model):

    def __init__(self, model: tf.keras.Model, epochs: int, batch_size: int, problem_type: str,
                 callbacks: list = None):
        """
        Constructor for TensorflowModel class.

        Parameters
        ----------
        model: tf.keras.Model
            Tensorflow model to be used for training and prediction.
        epochs: int
            Number of epochs to train the model.
        batch_size: int
            Batch size to be used for training.
        problem_type: str
            Type of problem to be solved. Can be 'binary', 'multiclass','regression', 'softclass', 'quantile'.
        callbacks: list
            List of callbacks to be used during training.
        """
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.problem_type = problem_type

    def _preprocess_data(self, dataset: Dataset, **kwargs):
        pass

    def _fit_data(self, dataset: Dataset, validation_dataset: Dataset) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for training.
        validation_dataset: Dataset
            Dataset to be used for validation.
        """
        self._history = self.model.fit(dataset.X.values(), dataset.y,
                                       epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       callbacks=self.callbacks,
                                       validation_data=(validation_dataset.X.values(), validation_dataset.y))

    def _predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Predict probabilities for each class.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for prediction.

        Returns
        -------
        np.ndarray
            Predicted probabilities for each class.
        """
        return self.model.predict_proba(dataset.X.values())

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class for each sample.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for prediction.

        Returns
        -------
        np.ndarray
            Predicted class for each sample.
        """
        if self.problem_type in [REGRESSION, QUANTILE]:
            y_pred = self.model.predict(dataset.X.values())
            return y_pred
        else:
            predictions = self.model.predict(dataset.X.values())
            return _convert_proba_to_unified_form(self.problem_type, np.array(predictions))

    def _save(self, path: str) -> None:
        """
        Save the model to the specified path.

        Parameters
        ----------
        path: str
            Path to save the model.
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Load the model from the specified path.

        Parameters
        ----------
        path: str
            Path to load the model from.
        """
        self.model.load(path)

    @property
    def history(self) -> dict:
        """
        Get the history of the model.

        Returns
        -------
        dict
            History of the model.
        """
        return self._history
