
import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.encodings._constants import ZS


class ZScaleEncoder(FeaturesGenerator):
    """
    It encodes protein sequences using the Z scale.
    This encoding requires the preprocessing of the protein
    sequences to the 20 standard amino acids.

    This method encodes which amino acid of the sequence into a Z-scales. Each Z scale represent an amino-acid property:
        Z1: Lipophilicity
        Z2: Steric properties (Steric bulk/Polarizability)
        Z3: Electronic properties (Polarity / Charge)
        Z4 and Z5: They relate electronegativity, heat of formation, electrophilicity and hardness.
    """

    output_shape_dimension = 3
    name = "z_scale"

    def set_features_names(self):
        self.features_names = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']

    def _fit(self, dataset: Dataset, instance_type: str) -> 'ZScaleEncoder':
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'ZScaleEncoder':
        return self

    def _featurize(self, sequence: str) -> np.ndarray:
        """
        It encodes a protein sequence with the ZScale encoding.

        Parameters
        ----------
        sequence: str
            The protein sequence to be encoded.

        Returns
        -------
        encoded_sequence: np.ndarray
            The encoded sequence.
        """
        return np.array([np.array(ZS[aa]) for aa in sequence], dtype=np.float32)
