from typing import List

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.propythia.propythia_descriptors.presets import DESCRIPTORS_PRESETS


class PropythiaWrapper(FeaturesGenerator):

    preset: str = 'all'

    def set_features_names(self) -> List[str]:
        """
        The method features_names will return the names of the features.
        """
        self.features_names = []
        for descriptor in DESCRIPTORS_PRESETS[self.preset]:
            instantiated_descriptor = descriptor()
            self.features_names.extend(instantiated_descriptor.get_features_out())
        return self.features_names

    def _fit(self, dataset: Dataset, instance_type: str) -> 'PropythiaWrapper':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        """
        if self.preset not in DESCRIPTORS_PRESETS:
            raise ValueError(f'Preset {self.preset} is not available.')
        self.descriptors = []
        for descriptor in DESCRIPTORS_PRESETS[self.preset]:
            instantiated_descriptor = descriptor()
            self.descriptors.append(instantiated_descriptor)

        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'PropythiaWrapper':
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instances to be featurized
        """
        return self._fit(dataset, instance_type)

    def _featurize(self, protein_sequence: str) -> np.ndarray:
        """
        The method _featurize will generate the desired features for a given protein sequence

        Parameters
        ----------
        protein_sequence: str
            protein sequence string

        Returns
        -------
        features_names: List[str]
            the names of the features

        features: np.ndarray
            the features
        """
        features_list = []
        for descriptor in self.descriptors:
            features = descriptor(protein_sequence)
            features_list.extend(features)

        return np.array(features_list, dtype=np.float32)
