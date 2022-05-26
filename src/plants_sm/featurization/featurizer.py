from abc import ABCMeta, abstractmethod
from typing import Any, Union, List

import pandas as pd
from joblib import Parallel, delayed

from plants_sm.data_structures.dataset import Dataset


class FeaturesGenerator(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        """
        Abstract class for feature extraction

        Parameters
        ----------
        kwargs
        """
        self.kwargs = kwargs
        self._features_names = None
        if "n_jobs" not in kwargs:
            self.n_jobs = 1
        else:
            self.n_jobs = kwargs["n_jobs"]
            del self.kwargs["n_jobs"]

    @property
    def features_names(self) -> List[str]:
        """
        Abstract method and property that returns the names of the features.

        Returns
        -------
        features_names : List[str]
            the names of the features
        """
        return self._features_names

    @features_names.setter
    def features_names(self, value: List[str]):
        """
        Setter for features names.

        Parameters
        ----------
        value: List[str]
            the names of the features
        """
        self._features_names = value

    def featurize(self, dataset: Dataset) -> Dataset:
        """
        General method that calls _featurize that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to be transformed where instances are the representation or object to be processed.

        Returns
        -------
        dataset with features: Dataset
            dataset object with features
        """
        parallel_callback = Parallel(n_jobs=self.n_jobs)
        len_instances = len(dataset.instances)
        new_x = parallel_callback(delayed(self._featurize)(dataset.instances[i], dataset.identifiers[i],
                                                           dataset.instances_ids_field) for i in range(len_instances))
        new_x = pd.concat(new_x, axis=0)
        dataset.dataframe = dataset.dataframe.merge(new_x, how='left', on=dataset.instances_ids_field)
        dataset.features_names = self.features_names
        return dataset

    @abstractmethod
    def _featurize(self, instance: Any, identifier: str, identifier_field_name: str) -> pd.DataFrame:
        """
        Method to be implemented by all feature generators to generate features for one instance at a time

        Parameters
        ----------
        instance: Any
            representation or object to be processed by the feature generator

        Returns
        -------
        dataframe with features: pd.Dataframe
        """
        pass
