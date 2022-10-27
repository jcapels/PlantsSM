from copy import copy
from typing import Any, List, Union, Set

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.tokenisation.tokeniser import Tokenizer


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
    tokenizer: Tokenizer = None
    max_length: int = None

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

        self.tokens = {}
        lengths = []
        if not self.alphabet:
            self.alphabet = set()
            sequences = list(dataset.get_instances(instance_type).values())
            for sequence in sequences:
                if self.tokenizer:
                    tokenized_sequence = self.tokenizer.tokenize(sequence)
                else:
                    tokenized_sequence = copy(sequence)
                lengths.append(len(tokenized_sequence))
                for char in tokenized_sequence:
                    self.alphabet.add(char)
        else:
            if isinstance(self.alphabet, str):
                self.alphabet = set(list(self.alphabet))

        if not self.max_length:
            self.max_length = max(lengths)

        for i, token in enumerate(self.alphabet):
            if self.output_shape_dimension == 3:
                one_hot = np.zeros(len(self.alphabet))
                one_hot[i] = 1
                self.tokens[token] = one_hot
            else:
                self.tokens[token] = i + 1

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
        res = np.zeros(self.max_length, dtype=np.int32)
        if self.tokenizer:
            instance = self.tokenizer.tokenize(instance)
        for i, token in enumerate(instance[:self.max_length]):
            res[i] = self.tokens[token]
        return res
