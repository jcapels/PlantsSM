from abc import ABCMeta, abstractmethod

from plants_sm.data_structures.dataset import Dataset


class Model(metaclass=ABCMeta):

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, **kwargs):
        pass

    @abstractmethod
    def _fit_data(self, dataset: Dataset, validation_dataset: Dataset):
        pass

    @abstractmethod
    def _predict_proba(self, dataset: Dataset):
        pass

    @abstractmethod
    def _save(self, path: str):
        pass

    @abstractmethod
    def _load(self, path: str):
        pass

    def load(self, path: str):
        self._load(path)

    def fit(self, dataset: Dataset, validation_dataset: Dataset = None):
        return self._fit_data(dataset, validation_dataset)

    def predict_proba(self, dataset: Dataset):

        self.preprocess(dataset)
        return self._predict_proba(dataset)

    def preprocess(self, dataset: Dataset, **kwargs):
        return self._preprocess_data(dataset, **kwargs)

    def save(self, path: str):
        self._save(path)
