import warnings
from typing import Dict, Any, Callable, Union, Tuple, List

import numpy as np
import pandas as pd
from numpy import ndarray

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from jax import vmap
from functools import partial
from jax_unirep.utils import get_embeddings

from jax_unirep.utils import load_params
from jax_unirep.layers import mLSTM
from jax_unirep.utils import validate_mLSTM_params

from plants_sm.featurization.proteins.bio_embeddings._utils import reduce_per_protein


class UniRepEmbeddings(FeaturesGenerator):
    """UniRep Embedder

    Alley, E.C., Khimulya, G., Biswas, S. et al. Unified rational protein
    engineering with sequence-based deep representation learning. Nat Methods
    16, 1315–1322 (2019). https://doi.org/10.1038/s41592-019-0598-1

    We use a reimplementation of unirep:

    Ma, Eric, and Arkadij Kummer. "Reimplementing Unirep in JAX." bioRxiv (2020).
    https://doi.org/10.1101/2020.05.11.088344
    """

    # An integer representing the size of the embedding.
    embedding_dimension = 1900
    # An integer representing the number of layers from the RAW output of the LM.
    number_of_layers = 1
    # dimension of the final embedding
    output_shape_dimension: int = 2

    _params: Dict[str, Any]
    _apply_fun: Callable

    def set_features_names(self):
        """
        The method features_names will set the names of the features
        """
        self.features_names = [f"unirep_{num}" for num in range(1, self.embedding_dimension + 1)]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'UniRepEmbeddings':
        """
        Load the parameters of the mLSTM model.
        """
        self.n_jobs = 1
        self._params = load_params()[1]
        _, self._apply_fun = mLSTM(output_dim=self.embedding_dimension)
        validate_mLSTM_params(self._params, n_outputs=self.embedding_dimension)

        if self.device:
            warnings.warn("It will run on CPU, because UniRep does not allow configuring the device")

        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'UniRepEmbeddings':
        """
        Fit the UniRep model to the dataset.

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.
        instance_type: str
            type of the instances to be featurized

        Returns
        -------
        self: Estimator
            the fitted UniRep
        """
        return self._fit(dataset, instance_type)

    def _featurize(self, sequence: str) -> ndarray:
        """
        Featurize a sequence using UniRep embedding.

        Parameters
        ----------
        sequence: str
            A string representing the sequence of a protein.

        Returns
        -------
        features_df: pd.DataFrame
            A DataFrame containing the UniRep embedding of the sequence.
        """
        # https://github.com/sacdallago/bio_embeddings/issues/117
        # Unirep only allows batching with sequences of the same length, so we don't do batching at all
        embedded_seqs = get_embeddings([sequence])
        # h and c refer to hidden and cell state
        # h contains all the hidden states, while h_final and c_final contain only the last state
        h_final, c_final, h = vmap(partial(self._apply_fun, self._params))(
            embedded_seqs
        )
        # Go from a batch of 1, which is `(1, len(sequence), 1900)`, to `len(sequence), 1900)`
        if self.output_shape_dimension <= 2:
            embedding = reduce_per_protein(h[0])
        else:
            embedding = np.asarray(h[0])
        return embedding
