from typing import Tuple

from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation._utils import transform_instances
from plants_sm.transformation.transformer import Transformer


class ProteinStandardizer(Transformer):
    """
    It transforms the protein sequences by replacing the following aminoacids
    by the closest amino acid residue available in the sequence:
        - Asparagine (B) -> Asparagine (N)
        - Glutamine(Z) -> Glutamine (Q)
        - Selenocysteine (U) -> Cysteinen(C)
        - Pyrrolysine (O) -> Lysine (K)

    It also removes the ambiguous amino acid (X) and replaces the ambiguous amino acid (J) by Isoleucine (I).

    Parameters
    ----------
    B: str, optional (default='N')
        One-letter aminoacid code to replace Asparagine (B). Default amino acid is Asparagine (N)
    Z   : str, optional (default='Q')
        One-letter aminoacid code to replace Glutamine(Z). Default amino acid is Glutamine (Q)
    U: str, optional (default='C')
        One-letter aminoacid code to replace Selenocysteine (U). Default amino acid is Cysteine (C)
    O: str, optional (default='K')
        One-letter aminoacid code to replace Pyrrolysine (O). Default amino acid is Lysine (K)
    J: str, optional (default='I')
        One-letter aminoacid code to replace ambiguous aminoacid (J). Default amino acid is Isoleucine (I)
    X: str, optional (default='')
        One-letter aminoacid code to remove ambiguous aminoacid (X). Default amino acid is removed
    """
    B: str = 'N'
    Z: str = 'Q'
    U: str = 'C'
    O: str = 'K'
    J: str = 'I'  # It can be either 'I' or 'L'
    X: str = ''

    def _fit(self, dataset: Dataset, instance_type: str) -> 'ProteinStandardizer':
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset
            Dataset to fit.

        Returns
        -------
        self : ProteinStandardizer
            Fitted transformer.
        """
        self.replace_dict = {'B': self.B, 'Z': self.Z, 'U': self.U, 'O': self.O, 'J': self.J, 'X': self.X}
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'ProteinStandardizer':
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset
            Dataset to fit.

        Returns
        -------
        self : ProteinStandardizer
            Fitted transformer.
        """
        return self._fit(dataset, instance_type)

    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
        """
        Standardize protein sequences.

        Parameters
        ----------
        dataset : Dataset
            Dataset to transform.

        Returns
        -------
        dataset : Dataset
            Transformed dataset.
        """
        return transform_instances(self.n_jobs, dataset, self._protein_preprocessing, instance_type,
                                   self.__class__.__name__)

    def _protein_preprocessing(self, sequence: str, identifier: str) -> Tuple[str, str]:
        """
        Preprocess a protein sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence to preprocess.
        identifier : str
            Protein identifier.

        Returns
        -------
        Tuple[str, str]
            Dictionary with the identifier as key and the preprocessed sequence as value.
        """
        for key, value in self.replace_dict.items():
            sequence = sequence.replace(key, value)
        return identifier, sequence


