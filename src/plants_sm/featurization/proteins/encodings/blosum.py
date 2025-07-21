from typing import Any

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.encodings._constants import BLOSUM_62, BLOSUM_50


class BLOSSUMEncoder(FeaturesGenerator):
    """
    It encodes protein sequences with a given BLOSUM substitution matrix.

    The number 62 refers to the percentage identity at which sequences are clustered in the analysis.
    It is possible to get 50 to get 50 % identity.
    Encoding a peptide this way means we provide the column from the blosum matrix corresponding to the amino acid
    at each position of the sequence. This produces 24 * seq_len matrix.

    The ProteinBlosum operates only over pandas DataFrame. Depending on the blosum used, it may require different
    preprocessing of protein sequences. For blosum62, protein sequences can not include the amino acid such as J, U, O.
    For blosum50, protein sequences can not include the amino acid such as U, O.

    """

    blosum: str = "blosum62"
    _blosum: dict = None
    output_shape_dimension = 3
    name = f"blosum_{blosum}"

    def set_features_names(self):
        """
        It sets the features names.
        """
        self.features_names = [f"{self.blosum}_{num}" for num in range(1, len(self._blosum["A"]) + 1)]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'BLOSSUMEncoder':
        """
        It fits the blosumEncoder.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the protein sequences to be encoded.

        Returns
        -------
        blosumEncoder

        """
        if self.blosum == "blosum62":
            self._blosum = BLOSUM_62

        elif self.blosum == "blosum50":
            self._blosum = BLOSUM_50

        else:
            ValueError(f'{self.blosum} is not a valid blosum.')

        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'BLOSSUMEncoder':
        """
        It fits the blosumEncoder.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the protein sequences to be encoded.

        Returns
        -------
        blosumEncoder

        """
        if self.blosum == "blosum62":
            self._blosum = BLOSUM_62

        elif self.blosum == "blosum50":
            self._blosum = BLOSUM_50

        else:
            ValueError(f'{self.blosum} is not a valid blosum.')

        return self

    def _featurize(self, sequence: Any) -> np.ndarray:

        return np.array([np.array(self._blosum[aa]) for aa in sequence], dtype=np.float32)
