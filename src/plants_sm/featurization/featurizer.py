from abc import ABCMeta, abstractmethod
from typing import Any

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
        self._features_names = None
        if "n_jobs" not in kwargs:
            self.n_jobs = 1
        else:
            self.n_jobs = kwargs["n_jobs"]
            
    @property
    @abstractmethod
    def features_names(self):
        return self._features_names

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
        new_x = parallel_callback(delayed(self._featurize)(instance) for instance in dataset.instances)
        new_x = pd.concat(new_x, axis=0)
        dataset.dataframe = pd.concat((dataset.dataframe, new_x), axis=1)
        return dataset

    @abstractmethod
    def _featurize(self, instance: Any) -> pd.DataFrame:
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
