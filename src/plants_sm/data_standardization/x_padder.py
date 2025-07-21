from typing import Tuple

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation.transformer import Transformer


class XPadder(Transformer):
    padding_dimensions: Tuple

    @staticmethod
    def _pad_array_with_zeros(array, padding_config: Tuple) -> np.ndarray:
        padding_config_ = []
        for i in range(len(array.shape)):
            if padding_config[i] < array.shape[i]:
                padding_config_.append((0, 0))
            else:
                padding_config_.append((0, padding_config[i] - array.shape[i]))

        array = np.pad(array, tuple(padding_config_), mode='constant', constant_values=0)
        return array

    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
        padded_dict = {key: self._pad_array_with_zeros(value, self.padding_dimensions)
                       for key, value in dataset.features[instance_type].items()}

        dataset.add_features(instance_type, padded_dict)
        return dataset

    def _fit(self, dataset: Dataset, instance_type: str) -> 'XPadder':
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'XPadder':
        return self
