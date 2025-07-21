import os
import uuid
from abc import ABCMeta, abstractmethod

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.io.json import write_json


class Model(metaclass=ABCMeta):

    _name: str = None

    @property
    def name(self) -> str:
        if self._name is None:
            value = str(uuid.uuid4().hex)
            self._name = value
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the name of the model

        Parameters
        ----------
        value: str
            Name of the model
        """
        if value is None:
            value = str(uuid.uuid4().hex)
        self._name = value

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Preprocesses the data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to preprocess.
        kwargs:
            Additional keyword arguments.

        Returns
        -------
        Dataset
            The preprocessed dataset.
        """

    @abstractmethod
    def _fit_data(self, train_dataset: Dataset, validation_dataset: Dataset, **kwargs):
        """
        Fits the model to the data.

        Parameters
        ----------
        train_dataset: Dataset
            The dataset to fit the model to.
        validation_dataset: Dataset
            The dataset to validate the model on.
        """

    @abstractmethod
    def _predict_proba(self, dataset: Dataset, **kwargs) -> np.ndarray:
        """
        Predicts the probabilities of the classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the probabilities on.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """

    @abstractmethod
    def _predict(self, dataset: Dataset, **kwargs) -> np.ndarray:
        """
        Predicts the classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes on.

        Returns
        -------
        np.ndarray
            The predicted classes.
        """

    @abstractmethod
    def _save(self, path: str):
        """
        Saves the model to a file.

        Parameters
        ----------
        path: str
            The path to save the model to.
        """
    @classmethod
    def load(cls, path: str):
        """
        Loads the model from a file.

        Parameters
        ----------
        path: str
            The path to load the model from.
        """
        return cls._load(path)

    @classmethod
    @abstractmethod
    def _load(cls, path: str):
        """
        Loads the model from a file.

        Parameters
        ----------
        path: str
            The path to load the model from.
        """

    @property
    @abstractmethod
    def history(self):
        """
        Returns the underlying model.
        """

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset = None, **kwargs) -> 'Model':
        """
        Fits the model to the data.

        Parameters
        ----------
        train_dataset: Dataset
            The dataset to fit the model to.
        validation_dataset: Dataset
            The dataset to validate the model on.

        Returns
        -------
        self
        """
        return self._fit_data(train_dataset, validation_dataset, **kwargs)

    def predict_proba(self, dataset: Dataset, **kwargs) -> np.ndarray:
        """
        Predicts the probabilities of the classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the probabilities on.
        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """

        return self._predict_proba(dataset, **kwargs)

    def predict(self, dataset: Dataset, **kwargs) -> np.ndarray:
        """
        Predicts the classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes on.

        Returns
        -------
        np.ndarray
            The predicted classes.
        """
        return self._predict(dataset, **kwargs)

    def preprocess(self, dataset: Dataset, **kwargs):
        """
        Preprocesses the data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to preprocess.
        kwargs:
            Additional keyword arguments.

        Returns
        -------
        Dataset
            The preprocessed dataset.
        """
        return self._preprocess_data(dataset, **kwargs)

    def save(self, path: str):
        """
        Saves the model to a file.

        Parameters
        ----------
        path: str
            The path to save the model to.
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                self._save(path)
            else:
                raise ValueError(f'Path {path} is not a directory.')

        else:
            os.makedirs(path)
            self._save(path)

        write_json(os.path.join(path, 'model_type.json'), {"type": self.__class__.__name__})

