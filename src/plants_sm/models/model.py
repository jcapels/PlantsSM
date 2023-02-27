from abc import ABCMeta, abstractmethod

import numpy as np

from plants_sm.data_structures.dataset import Dataset


class Model(metaclass=ABCMeta):

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
    def _fit_data(self, dataset: Dataset, validation_dataset: Dataset):
        """
        Fits the model to the data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.
        validation_dataset: Dataset
            The dataset to validate the model on.
        """

    @abstractmethod
    def _predict_proba(self, dataset: Dataset) -> np.ndarray:
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
    def _predict(self, dataset: Dataset) -> np.ndarray:
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

    @abstractmethod
    def load(self, path: str):
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

    def fit(self, dataset: Dataset, validation_dataset: Dataset = None) -> 'Model':
        """
        Fits the model to the data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.
        validation_dataset: Dataset
            The dataset to validate the model on.

        Returns
        -------
        self
        """
        return self._fit_data(dataset, validation_dataset)

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
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

        return self._predict_proba(dataset)

    def predict(self, dataset: Dataset) -> np.ndarray:
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
        return self._predict(dataset)

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
        self._save(path)
