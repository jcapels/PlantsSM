from abc import abstractmethod

from pydantic import BaseModel
from pydantic.validators import dict_validator

from plants_sm.data_structures.dataset import Dataset, SingleInputDataset
from plants_sm.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD
from plants_sm.estimation._utils import fit_status


class Estimator(BaseModel):

    _fitted: bool = False

    class Config:
        """
        Model Configuration: https://pydantic-docs.helpmanual.io/usage/model_config/
        """
        extra = 'allow'
        allow_mutation = True
        validate_assignment = True
        underscore_attrs_are_private = True

    @classmethod
    def get_validators(cls):
        # yield dict_validator
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if isinstance(value, cls):
            return value
        else:
            return cls(**dict_validator(value))

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool):
        if isinstance(value, bool):
            self._fitted = value

        else:
            raise TypeError("fitted has to be a boolean")

    @abstractmethod
    def _fit(self, dataset: Dataset, instance_type: str) -> 'Estimator':
        raise NotImplementedError

    @fit_status
    def fit(self, dataset: Dataset, instance_type: str = None) -> 'Estimator':

        if instance_type is None and isinstance(dataset, SingleInputDataset):
            self._fit(dataset, PLACEHOLDER_FIELD)
        elif isinstance(dataset, SingleInputDataset):
            self._fit(dataset, PLACEHOLDER_FIELD)
        else:
            self._fit(dataset, instance_type)

        return self
