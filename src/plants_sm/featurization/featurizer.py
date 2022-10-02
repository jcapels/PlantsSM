from abc import abstractmethod
from collections import ChainMap
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import ndarray

from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation.transformer import Transformer


class FeaturesGenerator(Transformer):
    device: str = None
    output_shape_dimension: int = 2
    feature_names: List[str] = None

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

    def _fit(self, dataset: Dataset) -> 'Estimator':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        """
        raise NotImplementedError

    def _transform(self, dataset: Dataset) -> Dataset:
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

        n_dimensional_features = dict(ChainMap(*res))
        identifiers = list(n_dimensional_features.keys())

        features_array = np.stack(list(n_dimensional_features.values()), axis=0)

        if self.output_shape_dimension <= 2:
            features = pd.DataFrame(features_array, columns=self.feature_names)
            features = pd.concat((pd.DataFrame({dataset.instances_ids_field: identifiers}), features),
                                 axis=1)

            dataset.features_dataframe = features
            dataset.features_shape = features_array.shape

        elif self.output_shape_dimension == 3:
            sequences = dataset.identifiers
            aa = np.arange(0, features_array.shape[1])
            features_names = np.array(self.feature_names)

            maj_dim = 1
            for dim in features_array.shape[:-1]:
                maj_dim = maj_dim * dim
            new_dims = (maj_dim, features_array.shape[-1])
            features = features_array.reshape(new_dims)

            midx = pd.MultiIndex.from_product([sequences, aa])

            features_3d = pd.DataFrame(data=features, index=midx, columns=features_names)

            dataset.features_dataframe = features_3d
            dataset.features_shape = features_array.shape

        if dataset.features_fields is None:
            dataset.features_fields = self.feature_names
        else:
            dataset.features_fields.extend(self.feature_names)
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
        feature_names, features_values = self._featurize(instance)
        temp_feature_dictionary = {identifier: features_values}
        if self.feature_names is None:
            self.feature_names = feature_names
        return temp_feature_dictionary

    @abstractmethod
    def _featurize(self, instance: Any) -> Tuple[List[str], np.ndarray]:
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
