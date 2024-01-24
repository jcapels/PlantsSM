from abc import abstractmethod
from copy import copy
from typing import Any, Union, Set, Iterable

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.tokenisation.tokeniser import Tokenizer


class Encoder(FeaturesGenerator):
    """
    Abstract method that has to be implemented by all encoders to set the tokens.

    Parameters
    ----------

    alphabet: Set[str]
        alphabet of the dataset
    tokenizer: Tokenizer
        tokenizer used to tokenize the sequences
    max_length: int
        maximum length of the sequences

    """
    alphabet: Union[Set[str], str] = set()
    tokenizer: Tokenizer = None
    max_length: int = None
    tokens: dict = {}
    _max_length_set: bool = False
    _alphabet_set: bool = False

    def __init__(self, alphabet: Union[Set[str], str] = None, tokenizer: Tokenizer = None, max_length: int = None, padding=""):
        """
        Abstract method that has to be implemented by all encoders to set the tokens.

        Parameters
        ----------

        alphabet: Set[str]
            alphabet of the dataset
        tokenizer: Tokenizer
            tokenizer used to tokenize the sequences
        max_length: int
            maximum length of the sequences
        """
        super().__init__()
        if alphabet is None:
            alphabet = set()
        else:
            self._alphabet_set = True

        self.alphabet = alphabet
        self.tokenizer = tokenizer

        if max_length is not None:
            self._max_length_set = True

        self.max_length = max_length

        self.padding = padding

    def set_features_names(self):
        """
        Set the features names of the encoded sequence.
        """
        for i, token in enumerate(set(self.alphabet)):
            self.features_names.append(token)

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'Encoder':
        """
        Fit the Encoder with the alphabet of the dataset.

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
        if len(self.alphabet) == 0:
            sequences = list(dataset.get_instances(instance_type).values())
            for sequence in sequences:
                if self.tokenizer:
                    tokenized_sequence = self.tokenizer.tokenize(sequence)
                else:
                    tokenized_sequence = copy(sequence)
                lengths.append(len(tokenized_sequence))
                for char in tokenized_sequence:
                    if char != self.padding:
                        self.alphabet.add(char)
        else:
            if not self.max_length:
                sequences = list(dataset.get_instances(instance_type).values())
                self.max_length = max([len(sequence) for sequence in sequences])
            elif not self._max_length_set:
                sequences = list(dataset.get_instances(instance_type).values())
                max_length = max([len(sequence) for sequence in sequences])
                if max_length > self.max_length:
                    self.max_length = max_length

            if self._alphabet_set:
                if isinstance(self.alphabet, str):
                    self.alphabet = set(list(self.alphabet))
            else:
                letters = list(self.alphabet)
                for letter in letters:
                    self.alphabet.add(letter)

        if not self.max_length:
            self.max_length = max(lengths)

        for i, token in enumerate(self.alphabet):
            self._set_tokens(i, token)

        return self

    def _fit(self, dataset: Dataset, instance_type: str) -> 'Encoder':
        """
        Fit the Encoder with the alphabet of the dataset.

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instances to be transformed

        Returns
        -------
        self: OneHotEncoder
            fitted OneHotEncoder
        """
        lengths = []
        if len(self.alphabet) == 0:
            sequences = list(dataset.get_instances(instance_type).values())
            for sequence in sequences:
                if self.tokenizer:
                    tokenized_sequence = self.tokenizer.tokenize(sequence)
                else:
                    tokenized_sequence = copy(sequence)
                lengths.append(len(tokenized_sequence))
                for char in tokenized_sequence:
                    if char != self.padding:
                        self.alphabet.add(char)
        else:
            if not self.max_length:
                sequences = list(dataset.get_instances(instance_type).values())
                self.max_length = max([len(sequence) for sequence in sequences])
            if isinstance(self.alphabet, str):
                self.alphabet = set(list(self.alphabet))

        if not self.max_length:
            self.max_length = max(lengths)

        alphabet = sorted(list(self.alphabet))
        for i, token in enumerate(alphabet):
            self._set_tokens(i, token)

        return self

    @abstractmethod
    def _set_tokens(self, i: int, token: str):
        """
        Set the tokens of the alphabet.

        Parameters
        ----------
        i: int
            index of the token
        token: str
            token to be set
        """
        pass

    def _featurize(self, instance: str) -> np.ndarray:
        """
        Encode the sequence with the encoding.

        Parameters
        ----------
        instance: str
            sequence to be encoded

        Returns
        -------
        encoded_sequence: np.ndarray
        """
        if self.output_shape_dimension == 3:
            res = np.zeros((self.max_length, len(self.alphabet)), dtype=np.int32)
        else:
            res = np.zeros(self.max_length, dtype=np.int32)
        if self.tokenizer:
            instance = self.tokenizer.tokenize(instance)
        for i, token in enumerate(instance[:self.max_length]):
            if token in self.tokens:
                if self.output_shape_dimension == 3:
                    res[i, :] = self.tokens[token]
                elif self.output_shape_dimension == 2:
                    res[i] = self.tokens[token]
        return res
