from abc import abstractmethod

from pydantic import BaseModel
from pydantic.validators import dict_validator

from plants_sm.data_structures.dataset import Dataset, SingleInputDataset
from plants_sm.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD
from plants_sm.estimation._utils import fit_status
from plants_sm.mixins.mixins import PickleMixin


class Estimator(BaseModel, PickleMixin):
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

    @abstractmethod
    def _fit_batch(self, dataset: Dataset, instance_type: str) -> 'Estimator':
        raise NotImplementedError

    @fit_status
    def fit(self, dataset: Dataset, instance_type: str = None) -> 'Estimator':

        if dataset.batch_size is not None:
            batch = dataset.next_batch()
            while batch is not None:
                self._fit_dataset(batch, instance_type, batch=True)
                batch = dataset.next_batch()
        else:
            self._fit_dataset(dataset, instance_type)

        return self

    def _fit_dataset(self, dataset: Dataset, instance_type: str = None, batch=False) -> 'Estimator':
        """
        Fit the estimator on the dataset

        Parameters
        ----------
        dataset: Dataset
            dataset to fit the estimator on
        instance_type: str
            type of the instances to fit. If None, it will only fit instances

        Returns
        -------
        fitted_estimator: Estimator
            fitted estimator
        """

        if batch:
            function = self._fit_batch
        else:
            function = self._fit

        if instance_type is None and isinstance(dataset, SingleInputDataset):
            function(dataset, PLACEHOLDER_FIELD)
        elif isinstance(dataset, SingleInputDataset):
            function(dataset, PLACEHOLDER_FIELD)
        else:
            function(dataset, instance_type)

        return self
