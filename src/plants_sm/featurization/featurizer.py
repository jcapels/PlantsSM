from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization._utils import call_set_features_names
from plants_sm.transformation._utils import tqdm_joblib
from plants_sm.transformation.transformer import Transformer

from tqdm import tqdm


class FeaturesGenerator(Transformer):

    device: str = "cpu"
    output_shape_dimension: int = 2
    features_names: List[str] = []

    @abstractmethod
    def set_features_names(self):
        """
        Abstract method that has to be implemented by all feature generators to set the features names
        """
        raise NotImplementedError

    def _fit(self, dataset: Dataset, instance_type: str) -> 'FeaturesGenerator':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        """

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'FeaturesGenerator':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        """

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
        instances = dataset.get_instances(instance_type)
        if self.n_jobs > 1:
            parallel_callback = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="threads")
            with tqdm_joblib(tqdm(desc=self.__class__.__name__, total=len(instances.items()))):
                res = parallel_callback(
                    delayed(self._featurize_and_add_identifier)(instance_representation, instance_id)
                    for instance_id, instance_representation in instances.items())
        else:
            res = []
            pbar = tqdm(desc=self.__class__.__name__, total=len(instances.items()))
            for instance_id, instance_representation in dataset.get_instances(instance_type).items():
                res.append(self._featurize_and_add_identifier(instance_representation, instance_id))
                pbar.update(1)

        dataset.add_features(instance_type, dict(res))
        dataset.features_fields[instance_type] = self.features_names

        return dataset

    def _featurize_and_add_identifier(self, instance: Any, identifier: str) -> Tuple[str, ndarray]:
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
        except Exception:
            features_values = np.zeros(len(self.features_names))

        temp_feature_dictionary = (identifier, features_values)

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
