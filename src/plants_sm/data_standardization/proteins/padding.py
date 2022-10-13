from collections import ChainMap
from typing import Dict

from joblib import Parallel, delayed

from plants_sm.data_standardization.padding_enumerators import PaddingEnumerators
from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation.transformer import Transformer


class SequencePadder(Transformer):
    """
    Padder for sequences of variable lengths.

    Attributes
    ----------
    pad_width: int
        width of the padded sequences
    padding: str
        type of padding to be applied
    """
    pad_width: int = None
    padding: str = "right"

    def _fit(self, dataset: Dataset) -> 'SequencePadder':
        """
        Method that fits the sequence padder to the dataset

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the transformer where instances are the representation or object to be processed.

        Returns
        -------
        fitted sequence padder: SequencePadder
        """
        if not self.pad_width:
            self.pad_width = dataset.dataframe.loc[:, dataset.representation_field].str.len().max()

        if self.padding not in ["right", "left", "center"]:
            raise ValueError(f"Padding type not supported: {self.padding}")

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to be transformed where instances are the representation or object to be processed.

        Returns
        -------
        dataset with features: Dataset
            dataset object with features
        """
        parallel_callback = Parallel(n_jobs=self.n_jobs)
        len_instances = len(dataset.instances)
        res = parallel_callback(
            delayed(self._pad_sequence)(dataset.instances[i], dataset.identifiers[i])
            for i in range(len_instances))

        sequences_dict = dict(ChainMap(*res))
        dataset.dataframe.loc[:, dataset.representation_field] = dataset.dataframe.index.map(
            sequences_dict)

        return dataset

    def _pad_sequence(self, instance: str, identifier: str) -> Dict[str, str]:
        """
        Pad a sequence of variable length to a fixed length

        Parameters
        ----------
        instance: str
            sequence to be padded
        identifier: str
            identifier of the sequence

        Returns
        -------
        padded sequence: Dict[str, str]
            dictionary with the padded sequence
        """

        padded = None

        try:
            assert len(instance) <= self.pad_width, f"Sequence length is greater than pad width: " \
                                                    f"{len(instance)} > {self.pad_width}"

            if self.padding == "right":
                padded = instance.rjust(self.pad_width, str(PaddingEnumerators.PROTEINS.value))

            elif self.padding == "left":
                padded = instance.ljust(self.pad_width, str(PaddingEnumerators.PROTEINS.value))

            elif self.padding == "center":
                padded = instance.center(self.pad_width, str(PaddingEnumerators.PROTEINS.value))

        except AssertionError:

            padded = str(PaddingEnumerators.PROTEINS.value) * self.pad_width

        return {identifier: padded}
