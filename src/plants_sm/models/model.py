from abc import ABCMeta, abstractmethod

from plants_sm.data_structures.dataset import Dataset


class Model(metaclass=ABCMeta):

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset):
        pass

    @abstractmethod
    def _fit_data(self, dataset: Dataset, validation_dataset: Dataset):
        pass

    @abstractmethod
    def _predict_proba(self, dataset: Dataset):
        pass

    def fit(self, dataset: Dataset, validation_dataset: Dataset = None):
        return self._fit_data(dataset, validation_dataset)

    def predict_proba(self, dataset: Dataset):
        return self._predict_proba(dataset)

    def preprocess(self, dataset: Dataset):
        return self._preprocess_data(dataset)
