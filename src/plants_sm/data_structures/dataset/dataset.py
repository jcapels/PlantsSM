from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import numpy as np


class Dataset(metaclass=ABCMeta):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def instances(self):
        pass

    @property
    @abstractmethod
    def identifiers(self):
        pass

    @property
    @abstractmethod
    def features(self) -> Dict[str, Dict[str, np.ndarray]]:
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def features_fields(self):
        pass

    @features_fields.setter
    @abstractmethod
    def features_fields(self, value: Any):
        pass
