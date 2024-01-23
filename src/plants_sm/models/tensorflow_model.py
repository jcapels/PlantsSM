import os
from typing import Union

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.io.pickle import read_pickle
from plants_sm.models._utils import _convert_proba_to_unified_form, write_model_parameters_to_pickle, _batch_generator
from plants_sm.models.constants import REGRESSION, QUANTILE, FileConstants
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
        self._history = None

    def _preprocess_data(self, dataset: Dataset, **kwargs):
        """
        Preprocess the data.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for training.
        kwargs: dict
            Keyword arguments to be used for preprocessing.

        """
        pass

    def _fit_data(self, train_dataset: Dataset, validation_dataset: Dataset) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        train_dataset: Dataset
            Dataset to be used for training.
        validation_dataset: Dataset
            Dataset to be used for validation.
        """
        self._preprocess_data(train_dataset)
        self._preprocess_data(validation_dataset)

        if train_dataset.batch_size is None:
            if isinstance(train_dataset.X, dict):
                train_dataset_x = train_dataset.X.values()
            else:
                train_dataset_x = train_dataset.X
            train_dataset_y = train_dataset.y
            train_dataset_tuple = (train_dataset_x, train_dataset_y)
        else:
            train_dataset_tuple = _batch_generator(train_dataset)

        if validation_dataset.batch_size is None:
            if isinstance(validation_dataset.X, dict):
                validation_dataset_x = validation_dataset.X.values()
            else:
                validation_dataset_x = validation_dataset.X
            validation_dataset_y = validation_dataset.y
            validation_dataset_tuple = (validation_dataset_x, validation_dataset_y)
        else:
            validation_dataset_tuple = _batch_generator(validation_dataset)

        if train_dataset.batch_size is None:
            self._history = self.model.fit(train_dataset_tuple[0], train_dataset_tuple[1],
                                           epochs=self.epochs,
                                           batch_size=self.batch_size,
                                           callbacks=self.callbacks,
                                           validation_data=validation_dataset_tuple)

        else:
            self._history = self.model.fit(train_dataset_tuple,
                                           epochs=self.epochs,
                                           batch_size=self.batch_size,
                                           callbacks=self.callbacks,
                                           validation_data=validation_dataset_tuple)

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
        if dataset.batch_size is None:
            if isinstance(dataset.X, dict):
                y_pred = self.model.predict(dataset.X.values())
            else:
                y_pred = self.model.predict(dataset.X)
        else:
            y_pred = self.model.predict(_batch_generator(dataset))

        return y_pred

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

        if dataset.batch_size is None:
            if isinstance(dataset.X, dict):
                y_pred = self.model.predict(dataset.X.values())
            else:
                y_pred = self.model.predict(dataset.X)

        else:
            y_pred = self.model.predict(_batch_generator(dataset))

        if self.problem_type in [REGRESSION, QUANTILE]:
            return y_pred
        else:
            return _convert_proba_to_unified_form(self.problem_type, np.array(y_pred))

    def _save(self, path: str) -> None:
        """
        Save the model to the specified path.

        Parameters
        ----------
        path: str
            Path to save the model.
        """
        self.model.save(os.path.join(path, FileConstants.TENSORFLOW_MODEL.value))
        # get the class attributes and save them
        parameters = {'epochs': self.epochs,
                      'batch_size': self.batch_size,
                      'callbacks': self.callbacks,
                      'problem_type': self.problem_type}
        write_model_parameters_to_pickle(parameters, path)

    @classmethod
    def _load(cls, path: str) -> 'TensorflowModel':
        """
        Load the model from the specified path.

        Parameters
        ----------
        path: str
            Path to load the model from.
        """
        model = tf.keras.models.load_model(os.path.join(path, FileConstants.TENSORFLOW_MODEL.value))
        parameters = read_pickle(os.path.join(path, FileConstants.MODEL_PARAMETERS_PKL.value))
        return cls(model, **parameters)

    @property
    def history(self) -> Union[dict, any, None]:
        """
        Get the history of the model.

        Returns
        -------
        dict
            History of the model.
        """
        if self._history is None:
            return None
        else:
            return self._history
