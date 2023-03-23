import tempfile
from abc import abstractmethod
from typing import Any, Dict, Union, Iterable

import numpy as np
from cached_property import cached_property

from plants_sm.data_structures.dataset.batch_manager.batch_manager import BatchManager
from plants_sm.design_patterns.observer import ConcreteSubject
from plants_sm.mixins.mixins import PickleMixin


class Dataset(ConcreteSubject, PickleMixin):
    representation_fields: Dict[str, Any]
    batch_size: Union[int, None] = None
    _dataframe_generator: Iterable = None
    variables_to_save = [
        ("dataframe", "csv"),
        ("instances", "json"),
        ("identifiers", "json"),
        ("features", "json"),
        ("labels", "csv")
    ]

    def __init__(self, batch_size: Union[int, None] = None):
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        else:
            self.batch_size = batch_size
            manager = BatchManager(batch_size=batch_size).register_class(self, self.variables_to_save)
            self.attach(manager)

    def _clear_cached_properties(self):
        """
        Clears the cached properties of the class.
        """
        for name in dir(type(self)):
            if isinstance(getattr(type(self), name), cached_property):
                vars(self).pop(name, None)

    @abstractmethod
    def __next__(self):
        """
        Returns the next batch of the dataset.
        """

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

    @abstractmethod
    def add_features(self, instance_type: str, features: Dict[str, np.ndarray]):
        """
        Adds features to the dataset.

        Parameters
        ----------
        instance_type: str
            The instance type.
        features: Dict[str, np.ndarray]
            The features to add. With the instance identifiers as keys and the features as value.
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
