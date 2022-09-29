from abc import abstractmethod, ABCMeta

from pydantic.generics import GenericModel
from pydantic.validators import dict_validator

from plants_sm.data_structures.dataset import Dataset
from plants_sm.estimation._utils import fit_status


class Estimator(GenericModel):

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
    def _fit(self, dataset: Dataset):
        raise NotImplementedError

    @fit_status
    def fit(self, dataset: Dataset) -> 'Estimator':

        self._fit(dataset)
        return self
