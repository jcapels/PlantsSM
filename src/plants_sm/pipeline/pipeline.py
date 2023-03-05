from typing import Union, List, Dict

import numpy as np

from plants_sm.data_structures.dataset import Dataset, PLACEHOLDER_FIELD
from plants_sm.models.model import Model
from plants_sm.transformation.transformer import Transformer


class Pipeline:

    def __init__(self, steps: Union[List[Transformer], Dict[str, List[Transformer]]] = None,
                 models: List[Model] = None,
                 metrics: List[callable] = None):

        if isinstance(steps, list):
            self._steps = {PLACEHOLDER_FIELD: steps}
        else:
            self._steps = steps
        self._models = models
        self._metrics = metrics
        self._models_indexes = {}
        self._best_model_name = None

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset) -> 'Pipeline':
        """
        Fit the pipeline

        Parameters
        ----------
        train_dataset: Dataset
            dataset to fit the pipeline
        validation_dataset: Dataset
            dataset to validate the pipeline

        Returns
        -------
        self: Pipeline
            fitted pipeline
        """

        for instance_type in self._steps.keys():
            for step in self._steps[instance_type]:
                step.fit_transform(train_dataset, instance_type=instance_type)
                step.transform(validation_dataset, instance_type=instance_type)

        for model in self._models:
            model.fit(train_dataset, validation_dataset)
            self._models_indexes[model.name] = model

        return self

    def _transform_dataset(self, dataset: Dataset) -> None:
        """
        Transform the dataset according to the pipeline

        Parameters
        ----------
        dataset: Dataset
            dataset to transform
        """

        for instance_type in self._steps.keys():
            for step in self._steps[instance_type]:
                step.transform(dataset, instance_type=instance_type)

    def predict(self, test_dataset: Dataset, model_name: str = None) -> np.ndarray:
        """
        Predict the dataset according to the pipeline

        Parameters
        ----------
        test_dataset: Dataset
            dataset to predict
        model_name: str
            name of the model to use

        Returns
        -------
        predictions: np.ndarray
            predictions of the dataset
        """

        self._transform_dataset(test_dataset)

        if model_name is not None:
            return self._models_indexes[model_name].predict(test_dataset)
        else:
            return self._models_indexes[self._best_model_name].predict(test_dataset)

    def predict_proba(self, test_dataset: Dataset, model_name: str = None) -> np.ndarray:
        """
        Predict the dataset according to the pipeline

        Parameters
        ----------
        test_dataset: Dataset
            dataset to predict
        model_name: str
            name of the model to use

        Returns
        -------
        predictions: np.ndarray
            predictions of the dataset
        """

        self._transform_dataset(test_dataset)

        if model_name is not None:
            return self._models_indexes[model_name].predict_proba(test_dataset)
        else:
            return self._models_indexes[self._best_model_name].predict_proba(test_dataset)
