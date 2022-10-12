from abc import abstractmethod

from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation.transformer import Transformer


class SequencePadder(Transformer):
    pad_width: int = None

    def _fit(self, dataset: Dataset) -> 'SequencePadder':
        """
        Abstract method that has to be implemented by all feature generators
        """
        if not self.pad_width:
            self.max_length = dataset.dataframe.loc[:, dataset.representation_field].str.len().max()
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
        pass
