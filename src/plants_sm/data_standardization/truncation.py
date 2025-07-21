
from typing import Tuple
from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.transformation.transformer import Transformer

from plants_sm.transformation._utils import transform_instances


class Truncator(Transformer):

    max_length: int

    def _fit(self, dataset: Dataset, instance_type: str) -> 'Truncator':
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset
            Dataset to fit.

        Returns
        -------
        self : Truncator
            Fitted transformer.
        """
        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'Truncator':
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset
            Dataset to fit.

        Returns
        -------
        self : Truncator
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
        return transform_instances(self.n_jobs, dataset, self._truncate, instance_type,
                                   self.__class__.__name__)

    def _truncate(self, sequence: str, identifier: str) -> Tuple[str, str]:
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
            Tuple with the identifier as first value and the preprocessed sequence as second.
        """
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        return identifier, sequence 



