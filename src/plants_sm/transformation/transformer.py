from abc import abstractmethod

from plants_sm.data_structures.dataset import Dataset, SingleInputDataset
from plants_sm.data_structures.dataset.single_input_dataset import FEATURES_FIELD
from plants_sm.estimation.estimator import Estimator


class Transformer(Estimator):
    n_jobs: int = 1

    @abstractmethod
    def _transform(self, dataset: Dataset, instance_type: str = FEATURES_FIELD) -> Dataset:
        """
        Abstract method that has to be implemented by all transformers
        """
        raise NotImplementedError

    def transform(self, dataset: Dataset, instance_type: str = None) -> Dataset:
        """
        Transform the dataset according to the transformer

        Parameters
        ----------
        dataset: Dataset
            dataset to transform
        instance_type: str
            type of the instances to transform. If None, it will only transform instances
            if the dataset has a single input

        Returns
        -------
        transformed_dataset: Dataset
            transformed dataset
        """
        if self.fitted:
            if instance_type is None and isinstance(dataset, SingleInputDataset):
                return self._transform(dataset, FEATURES_FIELD)
            elif isinstance(dataset, SingleInputDataset):
                return self._transform(dataset, FEATURES_FIELD)
            else:
                return self._transform(dataset, instance_type)
        else:
            # TODO : implement exception
            raise Exception("The transformer has to be fitted before transforming the dataset")

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit the transformer and transform the dataset
        """

        return self.fit(dataset).transform(dataset)
