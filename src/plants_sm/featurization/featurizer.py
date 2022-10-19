from abc import abstractmethod
from collections import ChainMap
from typing import Any, List, Dict

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization._utils import call_set_features_names
from plants_sm.transformation.transformer import Transformer


class FeaturesGenerator(Transformer):
    device: str = None
    output_shape_dimension: int = 2
    features_names: List[str] = []

    @abstractmethod
    def set_features_names(self):
        """
        Abstract method that has to be implemented by all feature generators to set the features names
        """
        raise NotImplementedError

    def _fit(self, dataset: Dataset) -> 'FeaturesGenerator':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        """
        raise NotImplementedError

    @call_set_features_names
    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
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
        res = parallel_callback(
            delayed(self._featurize_and_add_identifier)(dataset.instances[i], dataset.identifiers[i])
            for i in range(len_instances))

        dataset.features = {instance_type, dict(ChainMap(*res))}

        if dataset.features_fields is None:
            dataset.features_fields = self.features_names
        else:
            dataset.features_fields.extend(self.features_names)
        return dataset

    def _featurize_and_add_identifier(self, instance: Any, identifier: str) -> Dict[str, ndarray]:
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

        Returns
        -------

        """
        try:
            features_values = self._featurize(instance)
        # TODO: catch the correct exception
        except Exception as e:
            features_values = np.zeros(len(self.features_names))
        temp_feature_dictionary = {identifier: features_values}

        return temp_feature_dictionary

    @abstractmethod
    def _featurize(self, instance: Any) -> np.ndarray:
        """
        Method to be implemented by all feature generators to generate features for one instance at a time

        Parameters
        ----------
        instance: Any
            representation or object to be processed by the feature generator

        Returns
        -------
        np.ndarray
        """
