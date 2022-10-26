from typing import Any, List, Union, Set

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator


class OneHotEncoder(FeaturesGenerator):
    """
    Generic class to encode sequences with one-hot encoding.

    Parameters
    ----------
    alphabet: Union[Set[str], str], optional (default=None)
        The alphabet to be used in the encoding process. If None, the alphabet will be inferred from the dataset.
    output_shape_dimension: int, optional (default=3)
        The dimension of the output shape. If 3, the output shape will be (1, length of alphabet,
        length of sequence). However, the whole dataset would have 3 dimensions (number of sequences,
        length of alphabet, length of sequence)
    """
    name = "one_hot_encoder"
    alphabet: Union[Set[str], str] = []

    output_shape_dimension: int = 3

    def set_features_names(self):
        """
        Set the features names of the one-hot encoded sequence.

        Returns
        -------
        features_names: List[str]
            features names of the one-hot encoded sequence
        """
        for i, token in enumerate(self.alphabet):
            self.features_names.append(token)

    def _fit(self, dataset: Dataset, instance_type: str) -> 'OneHotEncoder':
        """
        Fit the OneHotEncoder with the alphabet of the dataset.

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.

        Returns
        -------
        self: OneHotEncoder
            fitted OneHotEncoder
        """

        if self.output_shape_dimension != 3:
            raise ValueError(f'output_shape_dimension must be 3, got {self.output_shape_dimension}')

        if not self.alphabet:
            self.alphabet = set()
            for sequence in list(dataset.get_instances(instance_type).values()):
                for char in str(sequence):
                    self.alphabet.add(char)
        else:
            if isinstance(self.alphabet, str):
                self.alphabet = set(list(self.alphabet))

        self.tokenizer = {}

        for i, token in enumerate(self.alphabet):
            one_hot = np.zeros(len(self.alphabet))
            one_hot[i] = 1
            self.tokenizer[token] = one_hot

        return self

    def _featurize(self, instance: Any) -> np.ndarray:
        """
        Encode the sequence with one-hot encoding.

        Parameters
        ----------
        instance: Any
            instance to be transformed where instances are the representation or object to be processed.

        Returns
        -------
        one_hot_encoded_sequence: np.ndarray
            one-hot encoded sequence
        """

        return np.array([self.tokenizer[i] for i in instance], dtype=np.bool)
