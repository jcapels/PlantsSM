import re
from typing import Dict, Any

import numpy as np
from gensim.models import KeyedVectors
from numpy import ndarray

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.bio_embeddings._utils import get_model_file, reduce_per_protein


class Word2Vec(FeaturesGenerator):

    name = "word2vec"
    embedding_dimension = 512
    number_of_layers = 1
    necessary_files = ["model_file"]
    _options: Dict[str, Any] = {}

    output_shape_dimension: int = 2

    def set_features_names(self):
        """
        The method features_names will return the names of the features
        """
        self.features_names = [f"{self.name}_{num}" for num in range(1, self.embedding_dimension + 1)]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'Word2Vec':
        """
        Fit the feature generator to the dataset

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.

        Returns
        -------
        self: Estimator
            the fitted FeaturesGenerator
        """

        self._model_file = self._options.get("model_file")

        if self._model_file is None:
            file_path = get_model_file(self.name, self.necessary_files[0])
            self._model = KeyedVectors.load(file_path, mmap="r")
        else:
            self._model = KeyedVectors.load(str(self._model_file), mmap="r")

        self._vector_size = 512
        self._zero_vector = np.zeros(self._vector_size, dtype=np.float32)
        self._window_size = 3

        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'Word2Vec':
        """
        Fit the feature generator to the dataset

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instances to be featurized

        Returns
        -------
        self: Estimator
            the fitted FeaturesGenerator
        """
        return self._fit(dataset, instance_type)

    def _featurize(self, sequence: str) -> np.ndarray:
        """
        The method _featurize will generate the desired features for a given protein sequence

        Parameters
        ----------
        sequence: str
            protein sequence string

        Returns
        -------
        dataframe with features: pd.DataFrame

        """

        sequence = re.sub(r"[UZOB]", "X", sequence)
        # pad sequence with special character (only 3-mers are considered)
        padded_sequence = "-" + sequence + "-"

        # container
        embedding = np.zeros((len(sequence), self._vector_size), dtype=np.float32)

        # for each aa in the sequence, retrieve k-mer
        for index in range(len(padded_sequence)):
            try:
                k_mer = "".join(padded_sequence[index: index + self._window_size])
                embedding[index, :] = self._get_kmer_representation(k_mer)
            # end of sequence reached
            except IndexError:
                if self.output_shape_dimension == 2:
                    embedding = reduce_per_protein(embedding)
                elif self.output_shape_dimension == 3:
                    continue
                else:
                    raise ValueError("Output dimension must be 2 or 3")
                return embedding

        return embedding

    def _get_kmer_representation(self, k_mer: str) -> ndarray:
        """
        Get the embedding for a given k-mer

        Parameters
        ----------
        k_mer: str
            k-mer string

        Returns
        -------
        kmer representation: ndarray
        """
        # try to retrieve embedding for k-mer
        try:
            return self._model.wv[k_mer]
        # in case of padded or out-of-vocab character
        except KeyError:
            # if single AA was not part of corpus (or no AA)
            if len(k_mer) <= 1:
                return self._zero_vector
            # handle border cases at start/end of seq
            elif "-" in k_mer:
                idx_center = int(len(k_mer) / 2)
                return self._get_kmer_representation(k_mer[idx_center])
