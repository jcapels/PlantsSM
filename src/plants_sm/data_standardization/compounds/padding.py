from typing import Tuple

from plants_sm.data_standardization.padding_enumerators import PaddingEnumerators
from plants_sm.data_structures.dataset import Dataset
from plants_sm.transformation._utils import transform_instances
from plants_sm.transformation.transformer import Transformer


class SMILESPadder(Transformer):
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
    padding: str = "left"
    _pad_width_set: bool = False

    def __init__(self, pad_width: int = None, padding: str = "left"):
        super().__init__()
        if pad_width is not None:
            self._pad_width_set = True
        self.pad_width = pad_width
        self.padding = padding

    def _fit(self, dataset: Dataset, instance_type: str) -> 'SMILESPadder':
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
            # get the maximum length of the sequences
            lengths = [len(instance) for instance in dataset.get_instances(instance_type).values()]
            self.pad_width = max(lengths)

        if self.padding not in ["right", "left", "center"]:
            raise ValueError(f"Padding type not supported: {self.padding}")

        return self

    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'SMILESPadder':
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
        if not self._pad_width_set:
            # get the maximum length of the sequences
            lengths = [len(instance) for instance in dataset.get_instances(instance_type).values()]
            max_length = max(lengths)
            if self.pad_width < max_length:
                self.pad_width = max_length

        if self.padding not in ["right", "left", "center"]:
            raise ValueError(f"Padding type not supported: {self.padding}")

        return self

    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
        """
        Abstract method that has to be implemented by all feature generators

        Parameters
        ----------
        dataset: Dataset
            dataset to be transformed where instances are the representation or object to be processed.
        instance_type: str
            type of instances to be transformed

        Returns
        -------
        dataset with features: Dataset
            dataset object with features
        """
        return transform_instances(self.n_jobs, dataset, self._pad_sequence, instance_type,
                                   self.__class__.__name__)

    def _pad_sequence(self, instance: str, identifier: str) -> Tuple[str, str]:
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
                padded = instance.rjust(self.pad_width, str(PaddingEnumerators.COMPOUNDS.value))

            elif self.padding == "left":
                padded = instance.ljust(self.pad_width, str(PaddingEnumerators.COMPOUNDS.value))

            elif self.padding == "center":
                padded = instance.center(self.pad_width, str(PaddingEnumerators.COMPOUNDS.value))

        except AssertionError:

            padded = instance[:self.pad_width]

        return identifier, padded
