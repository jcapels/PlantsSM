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
        self._features_fields = None
        if "n_jobs" not in kwargs:
            self.n_jobs = 1
        else:
            self.n_jobs = kwargs["n_jobs"]
            del self.kwargs["n_jobs"]

    @property
    def features_fields(self) -> List[str]:
        """
        Abstract method and property that returns the names of the features.

        Returns
        -------
        features_names : List[str]
            the names of the features
        """
        return self._features_fields

    @features_fields.setter
    def features_fields(self, value: List[str]):
        """
        Setter for features names.

        Parameters
        ----------
        value: List[str]
            the names of the features
        """
        self._features_fields = value

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
        new_x = parallel_callback(
            delayed(self._featurize_and_add_identifier)(dataset.instances[i], dataset.identifiers[i],
                                                        dataset.instances_ids_field) for i in range(len_instances))

        features_names = list(new_x[0].columns)
        features_names.remove(dataset.instances_ids_field)

        new_x = pd.concat(new_x, axis=0)
        dataset.dataframe = dataset.dataframe.merge(new_x, how='left', on=dataset.instances_ids_field)

        self.features_fields = features_names
        if dataset.features_fields is None:
            dataset.features_fields = features_names
        else:
            dataset.features_fields.extend(features_names)
        return dataset

    def _featurize_and_add_identifier(self, instance: Any, identifier: str, identifier_field_name: str) -> pd.DataFrame:
        """
        Private method that calls the _featurize method and returns the dataframe with the features, adding the instance
        identifier to the dataframe.

        It is used to featurize a single instance.

        Parameters
        ----------
        instance: Any
            instance to be featurized.

        identifier: str
            identifier of the instance.

        identifier_field_name: str
            name of the identifier field.

        Returns
        -------
        dataframe with features: pd.DataFrame
            dataframe with features
        """
        features_df = self._featurize(instance)
        features_df[identifier_field_name] = [identifier]
        return features_df

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
