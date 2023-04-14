from abc import abstractmethod

from plants_sm.data_structures.dataset import Dataset, SingleInputDataset
from plants_sm.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD
from plants_sm.estimation.estimator import Estimator


class Transformer(Estimator):
    n_jobs: int = 1

    @abstractmethod
    def _transform(self, dataset: Dataset, instance_type: str) -> Dataset:
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
            if dataset.batch_size is not None:
                dataset.reset_batch()
                batch = dataset.next_batch()
                while batch is not None:
                    self._transform_dataset(dataset, instance_type)
                    batch = dataset.next_batch()
            else:
                return self._transform_dataset(dataset, instance_type)

        else:
            # TODO : implement exception
            raise Exception("The transformer has to be fitted before transforming the dataset")

    def _transform_dataset(self, dataset: Dataset, instance_type: str = None) -> Dataset:
        """
        Transform the dataset according to the transformer

        Parameters
        ----------
        dataset: Dataset
            dataset to transform
        instance_type: str
            type of the instances to transform. If None, it will only transform instances

        Returns
        -------
        transformed_dataset: Dataset
            transformed dataset
        """
        if instance_type is None and isinstance(dataset, SingleInputDataset):
            return self._transform(dataset, PLACEHOLDER_FIELD)
        elif isinstance(dataset, SingleInputDataset):
            return self._transform(dataset, PLACEHOLDER_FIELD)
        else:
            return self._transform(dataset, instance_type)

    def fit_transform(self, dataset: Dataset, instance_type: str = None) -> Dataset:
        """
        Fit the transformer and transform the dataset
        """

        return self.fit(dataset, instance_type).transform(dataset, instance_type)
