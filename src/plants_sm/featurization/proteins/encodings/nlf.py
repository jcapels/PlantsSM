import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.featurization.featurizer import FeaturesGenerator
from plants_sm.featurization.proteins.encodings._constants import NLF


class NLFEncoder(FeaturesGenerator):
    """
    It encodes protein sequences using the Fisher Transform.
    The ProteinNLF operates only over pandas DataFrame.

    Method that takes many physicochemical properties and transforms them using a Fisher Transform (similar to a PCA)
    creating a smaller set of features that can describe the amino acid just as well.
    There are 19 transformed features.
    This method of encoding is detailed by Nanni and Lumini in their paper:
    L. Nanni and A. Lumini, “A new encoding technique for peptide classification,”
    Expert Syst. Appl., vol. 38, no. 4, pp. 3185–3191, 2011
    This function just receives 20aa letters, therefore preprocessing is required.
    """
    name = 'NLFEncoder'
    encoding_dimension = 18
    output_shape_dimension = 3

    def set_features_names(self):
        """
        Set the features names.
        """
        self.features_names = [f"{self.name}_{num}" for num in range(1, self.encoding_dimension + 1)]

    def _fit(self, dataset: Dataset, instance_type: str) -> 'NLFEncoder':
        """
        Fit the transformer to the data.

        Parameters
        ----------
        dataset: Dataset
            The input data.

        Returns
        -------
        NLFEncoder: The fitted transformer.

        """
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'NLFEncoder':
        """
        Fit the transformer to the data.

        Parameters
        ----------
        dataset: Dataset
            The input data.

        Returns
        -------
        NLFEncoder: The fitted transformer.

        """
        return self

    def _featurize(self, sequence: str) -> np.ndarray:
        """
        Encode a single sequence.

        Parameters
        ----------
        sequence: str
            The input sequence.

        Returns
        -------
        np.ndarray: The encoded sequence.

        """
        return np.array([np.array(NLF[aa]) for aa in sequence], dtype=np.float32)
