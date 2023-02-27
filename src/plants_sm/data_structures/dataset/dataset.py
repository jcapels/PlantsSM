from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from plants_sm.mixins.mixins import PickleMixin


class Dataset(PickleMixin):
    representation_fields: Dict[str, Any]

    def __init__(self):
        pass

    @property
    @abstractmethod
    def dataframe(self):
        """
        Returns the dataset as a pandas dataframe.
        """

    @property
    @abstractmethod
    def instances(self):
        """
        Returns the instances of the dataset.
        """

    @property
    @abstractmethod
    def identifiers(self):
        """
        Returns the identifiers of the instances of the dataset.
        """

    @property
    @abstractmethod
    def features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the features of the dataset.
        """

    @features.setter
    @abstractmethod
    def features(self, value: Dict[str, Dict[str, np.ndarray]]):
        """
        Sets the features of the dataset.

        Parameters
        ----------
        value: Dict[str, Dict[str, np.ndarray]]
            The features of the dataset. With the instance type as key and a dictionary of features as value.
            The keys of the features dictionary are the instance identifiers and the values are the features.
            For single-instance datasets, the instance type is a placeholder.
        """

    @property
    @abstractmethod
    def labels(self):
        """
        Returns the labels of the dataset.
        """

    @property
    @abstractmethod
    def X(self) -> Dict[str, np.ndarray]:
        """
        Returns the features of the dataset.

        Returns
        -------
        Dict[str, np.ndarray]
            The features of the dataset. With the instance identifiers as keys and the features as value.
        """

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        """
        Returns the labels of the dataset.

        Returns
        -------
        np.ndarray
            The labels of the dataset.
        """

    @property
    @abstractmethod
    def features_fields(self):
        """
        Returns the features fields of the dataset.
        """

    @features_fields.setter
    @abstractmethod
    def features_fields(self, value: Any):
        """
        Sets the features fields of the dataset.

        Parameters
        ----------
        value: Any
            The features fields of the dataset.
        """

    @abstractmethod
    def get_instances(self, instance_type: str = None):
        """
        Returns the instances of the dataset.
        """
