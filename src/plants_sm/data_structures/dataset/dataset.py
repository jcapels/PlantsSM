import os
import shutil
from abc import abstractmethod
from typing import Any, Dict, Union, Iterable, List

import numpy as np
import pandas as pd
from cached_property import cached_property

from plants_sm.data_structures.dataset.batch_manager.batch_manager import BatchManager
from plants_sm.design_patterns.observer import ConcreteSubject
from plants_sm.io.pickle import write_pickle, read_pickle
from plants_sm.mixins.mixins import PickleMixin


class Dataset(ConcreteSubject, PickleMixin):
    representation_fields: Dict[str, Any]
    batch_size: Union[int, None] = None
    _dataframe_generator: Iterable = None
    _batch_state: bool = True
    variables_to_save = [
        ("_dataframe", "pkl"),
        ("_instances", "pkl"),
        ("_identifiers", "pkl"),
        ("features", "pkl"),
    ]

    def __init__(self, batch_size: Union[int, None] = None):
        """
        Class that represents a dataset.

        Parameters
        ----------
        batch_size: int
            the size of the batches
        """
        self.batch_size = batch_size
        if self.batch_size is not None:
            self._folder_to_load_features = None
            if self.batch_size <= 0:
                raise ValueError("Batch size must be a positive integer.")
            else:
                manager = BatchManager(batch_size=self.batch_size)
                manager.register_class(self, self.variables_to_save)
                self.attach(manager)

    def _clear_cached_properties(self):
        """
        Clears the cached properties of the class.
        """
        for name in dir(type(self)):
            if isinstance(getattr(type(self), name), cached_property):
                vars(self).pop(name, None)

    def reset_batch(self):
        """
        Method to reset the batch.
        """
        self._batch_state = False
        self._observer.end()

    def save_features(self, folder_path: str):
        """
        Method to save the features of the dataset.

        Parameters
        ----------
        folder_path: str
            The path to the folder where the features will be saved.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.batch_size is not None:
            i = 0
            while self.next_batch():
                write_pickle(os.path.join(folder_path, f"features_{i}.pkl"), self.features)
                i += 1
        else:
            write_pickle(os.path.join(folder_path, f"features.pkl"), self.features)

    def load_features(self, folder_path: str):
        """
        Method to load the features of the dataset.

        Parameters
        ----------
        folder_path: str
            The path to the file where the features are saved.
        """
        if self.batch_size is not None:
            self._folder_to_load_features = folder_path
            i = 0
            self.features = read_pickle(os.path.join(folder_path, f"features_{i}.pkl"))
            i += 1
            while self.next_batch():
                self.features = read_pickle(os.path.join(folder_path, f"features_{i}.pkl"))
                i += 1
        else:
            self.features = read_pickle(os.path.join(folder_path, f"features.pkl"))

    def save(self, folder_path: str):
        """
        Method to save the dataset.

        Parameters
        ----------
        folder_path: str
            The path to the folder where the dataset will be saved.
        """
        if self.batch_size is not None:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            temporary_folder = self._observer.temporary_folder
            # copy the content of the temporary folder to the folder where the dataset will be saved
            batches_directory = os.path.join(folder_path, "batches")
            if not os.path.exists(batches_directory):
                os.makedirs(os.path.join(batches_directory))

            for file in os.listdir(temporary_folder.name):
                if os.path.exists(batches_directory):
                    shutil.rmtree(batches_directory)
                shutil.copytree(os.path.join(temporary_folder.name, file), os.path.join(batches_directory, file))

            self.to_pickle(os.path.join(folder_path, f"{folder_path}.pkl"))

        else:
            self.to_pickle(os.path.join(f"{folder_path}.pkl"))

    @classmethod
    def load(cls, path: str):
        """
        Method to load the dataset.

        Parameters
        ----------
        path: str
            The path to the folder where the dataset is saved.
        """
        if os.path.isdir(path):
            dirs = os.listdir(path)

            if len(dirs) != 0:
                # create the dataset
                dataset = cls.from_pickle(os.path.join(path, f"{path}.pkl"))
                batches_directory = os.path.join(path, "batches")
                for folder in os.listdir(batches_directory):
                    if os.path.exists(os.path.join(dataset._observer.temporary_folder.name, folder)):
                        shutil.rmtree(os.path.join(dataset._observer.temporary_folder.name, folder))
                    shutil.copytree(os.path.join(batches_directory, folder),
                                    os.path.join(dataset._observer.temporary_folder.name, folder))
            else:
                raise ValueError("The directory is empty.")

        else:
            folder_path = path.replace(".pkl", "")
            dataset = cls.from_pickle(os.path.join(f"{folder_path}.pkl"))

        return dataset

    def __next__(self):
        """
        Method to iterate over the dataset.
        """
        if self.batch_size is not None and self._dataframe_generator is not None:
            self.notify(function="__next__")

            try:
                df = next(self._dataframe_generator)
            except StopIteration:
                self.end()
                self._dataframe_generator = None
                return False

            self.dataframe = df

            return True

        raise ValueError("The dataset is not iterable.")

    def next_batch(self):
        """
        Method to iterate over the dataset.
        """
        if self.batch_size is not None:
            self.notify(function="next_batch")
            if self._batch_state:
                return self
            else:
                self.end()
                return None
        else:
            raise ValueError("The dataset is not iterable.")

    @abstractmethod
    def select(self, ids: Union[List[str], List[int]], instance_type: str):
        """
        Method to select a subset of the dataset.

        Parameters
        ----------
        ids: List[str]
            The identifiers of the instances to select.
        instance_type: str
            The type of the instances to select.
        """

    @property
    @abstractmethod
    def dataframe(self):
        """
        Returns the dataset as a pandas dataframe.
        """

    @dataframe.setter
    @abstractmethod
    def dataframe(self, value):
        """
        Sets the dataset as a pandas dataframe.

        Parameters
        ----------
        value: pd.DataFrame
            The dataset as a pandas dataframe.
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

    def _set_dataframe(self, value: Any):
        """
        Private method to set the dataframe.
        Parameters
        ----------
        value: Any
            dataframe to be set, it can be in pd.DataFrame format, but can also be a List or Dictionary
            (it can be specific for each data type)
        """
        if isinstance(value, pd.DataFrame) or value is None:
            self._dataframe = value
        else:
            raise TypeError("It seems that the type of your input is not a pandas DataFrame."
                            "The type of the dataframe should be a pandas DataFrame")
