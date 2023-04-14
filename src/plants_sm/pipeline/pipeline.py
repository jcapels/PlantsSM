import os
from typing import Union, List, Dict

import numpy as np

from plants_sm.data_structures.dataset import Dataset, PLACEHOLDER_FIELD
from plants_sm.io.json import write_json, read_json
from plants_sm.io.pickle import write_pickle, read_pickle
from plants_sm.models.enumerators import ModelFileEnumeratorsUtils
from plants_sm.models.model import Model
from plants_sm.transformation.transformer import Transformer


class Pipeline:

    steps: Dict[str, List[Transformer]]
    models: List[Model]
    metrics: List[callable]

    def __init__(self, steps: Union[List[Transformer], Dict[str, List[Transformer]]] = None,
                 models: List[Model] = None,
                 metrics: List[callable] = None):
        """
        Constructor

        Parameters
        ----------
        steps: List[Transformer] | Dict[str, List[Transformer]]
            list of steps to be executed
        models: List[Model]
            list of models to be executed
        metrics: List[callable]
            list of metrics to be executed
        """

        if isinstance(steps, list):
            self.steps = {PLACEHOLDER_FIELD: steps}
        else:
            self.steps = steps
        self.models = models
        self.metrics = metrics
        self._best_model_name = None

        self._models_indexes = {}
        for model in self.models:
            model_name = model.name
            if model_name in self._models_indexes:
                raise ValueError(f"Model with name {model_name} already exists")
            self._models_indexes[model_name] = model

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> 'Pipeline':
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

        for instance_type in self.steps.keys():
            for step in self.steps[instance_type]:
                step.fit_transform(train_dataset, instance_type=instance_type)
                if validation_dataset is not None:
                    step.transform(validation_dataset, instance_type=instance_type)

        for model in self.models:
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

        for instance_type in self.steps.keys():
            for step in self.steps[instance_type]:
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
        elif self._best_model_name is not None:
            return self._models_indexes[self._best_model_name].predict(test_dataset)
        else:
            return self._models_indexes[self.models[0].name].predict(test_dataset)

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
        elif self._best_model_name is not None:
            return self._models_indexes[self._best_model_name].predict_proba(test_dataset)
        else:
            return self._models_indexes[self.models[0].name].predict_proba(test_dataset)

    def _save(self, path: str) -> None:
        """
        Save the pipeline

        Parameters
        ----------
        path: str
            path to save the pipeline
        """

        config = {
            "steps": {},
            "models": [],
            "metrics": [],
            "_best_model_name": self._best_model_name
        }

        for instance_type in self.steps.keys():

            i = 1
            config["steps"][instance_type] = []

            for step in self.steps[instance_type]:
                if instance_type == PLACEHOLDER_FIELD:

                    step_file_path = f"step_{i}.pkl"
                    config["steps"][PLACEHOLDER_FIELD].append(step_file_path)

                    step.to_pickle(os.path.join(path, step_file_path))
                else:

                    step_file_path = f"step_{instance_type}_{i}.pkl"
                    config["steps"][instance_type].append(step_file_path)

                    step.to_pickle(os.path.join(path, f"step_{instance_type}_{i}.pkl"))
                i += 1

        for model in self.models:
            config["models"].append(model.name)
            model.save(os.path.join(path, model.name))

        if self.metrics is not None:
            for metric in self.metrics:
                config["metrics"].append(f"{metric.__name__}.pkl")
                write_pickle(metric, os.path.join(path, f"{metric.__name__}.pkl"))

        write_json(os.path.join(path, "config.json"), config)

    def save(self, path: str) -> None:
        """
        Save the pipeline

        Parameters
        ----------
        path: str
            path to save the pipeline
        """

        if os.path.exists(path):
            if os.path.isdir(path):
                self._save(path)
            else:
                raise ValueError("The path must be a directory")
        else:
            os.mkdir(path)
            self._save(path)

    @classmethod
    def _load(cls, path: str) -> 'Pipeline':
        """
        Load the pipeline

        Parameters
        ----------
        path: str
            path to load the pipeline

        Returns
        -------
        pipeline: Pipeline
            loaded pipeline
        """

        config = read_json(os.path.join(path, "config.json"))
        steps = {}
        models = []
        metrics = []

        for instance_type in config["steps"].keys():

            if instance_type == PLACEHOLDER_FIELD:
                steps[PLACEHOLDER_FIELD] = []
            else:
                steps[instance_type] = []

            for step in config["steps"][instance_type]:
                if instance_type == PLACEHOLDER_FIELD:
                    steps[PLACEHOLDER_FIELD].append(read_pickle(os.path.join(path, step)))
                else:
                    steps[instance_type].append(read_pickle(os.path.join(path, step)))

        for model in config["models"]:
            model_type = ModelFileEnumeratorsUtils.get_class_to_load_from_directory(os.path.join(path, model))
            loaded_model = model_type.load(os.path.join(path, model))
            models.append(loaded_model)

        for metric_path in config["metrics"]:
            metric = read_pickle(os.path.join(path, metric_path))
            metrics.append(metric)

        kwargs = {
            "steps": steps,
            "models": models,
            "metrics": metrics
        }

        new_pipeline = cls(**kwargs)
        new_pipeline._best_model_name = config["_best_model_name"]
        return new_pipeline

    @classmethod
    def load(cls, path: str) -> 'Pipeline':
        """
        Load the pipeline

        Parameters
        ----------
        path: str
            path to load the pipeline

        Returns
        -------
        pipeline: Pipeline
            loaded pipeline
        """

        if os.path.exists(path):
            if os.path.isdir(path):
                return cls._load(path)
            else:
                raise ValueError("The path must be a directory")
        else:
            raise ValueError("The path must exist")
