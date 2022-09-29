import re
from typing import Dict, Any

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from numpy import ndarray
from pandas import DataFrame

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.bio_embeddings._utils import get_model_file, reduce_per_protein


class Word2Vec(FeaturesGenerator):

    name = "word2vec"
    embedding_dimension = 512
    number_of_layers = 1
    necessary_files = ["model_file"]
    _options: Dict[str, Any] = {}

    output_dimension: int = 2

    def _fit(self, dataset: Dataset) -> 'Estimator':

        self._model_file = self._options.get("model_file")

        if self._model_file is None:
            file_path = get_model_file(self.name, self.necessary_files[0])
            self._model = KeyedVectors.load(file_path, mmap="r")
        else:
            self._model = KeyedVectors.load(str(self._model_file), mmap="r")

        self._vector_size = 512
        self._zero_vector = np.zeros(self._vector_size, dtype=np.float32)
        self._window_size = 3
        self.features_names = [f"{self.name}_{num}" for num in range(1, self.embedding_dimension + 1)]

    def _featurize(self, sequence: str) -> pd.DataFrame:

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
                if self.output_dimension == 2:
                    embedding = reduce_per_protein(embedding)

                features_df = DataFrame([embedding], index=[0], columns=self.features_names)
                return features_df

    def _get_kmer_representation(self, k_mer):
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

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding.mean(axis=0)
