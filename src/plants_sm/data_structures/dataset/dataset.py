from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from plants_sm.mixins.mixins import PickleMixin


class Dataset(PickleMixin):
    representation_fields: Dict[str, Any]

    def __init__(self):
        pass

    @property
    @abstractmethod
    def dataframe(self):
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

    @features.setter
    @abstractmethod
    def features(self, value) -> Dict[str, Dict[str, np.ndarray]]:
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    @property
    @abstractmethod
    def X(self) -> Dict[str, np.ndarray]:
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

    @abstractmethod
    def get_instances(self, instance_type: str = None):
        pass
