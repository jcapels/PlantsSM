from abc import abstractmethod
from typing import Any

import numpy as np

from plants_sm.data_structures.dataset import Dataset
from plants_sm.estimation.estimator import Estimator


class Transformer(Estimator):
    n_jobs: int = 1
    _features_fields: list = None

    # def __init__(self, n_jobs: int = 1):
    #     """
    #     Constructor for a generic transformer
    #     """
    #
    #     if "n_jobs" not in kwargs:
    #         self.n_jobs = 1
    #     else:
    #         self.n_jobs = kwargs["n_jobs"]
    #         del kwargs["n_jobs"]
    #
    #     super().__init__(**kwargs)

    @abstractmethod
    def _transform(self, dataset: Dataset) -> np.ndarray:
        """
        Abstract method that has to be implemented by all transformers
        """
        raise NotImplementedError

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transform the dataset according to the transformer

        Parameters
        ----------
        dataset: Dataset
            dataset to transform

        Returns
        -------
        transformed_dataset: Dataset
            transformed dataset
        """
        if self.fitted:
            return self._transform(dataset)
        else:
            # TODO : implement exception
            raise Exception("The transformer has to be fitted before transforming the dataset")

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Fit the transformer and transform the dataset
        """

        return self.fit(dataset).transform(dataset)
