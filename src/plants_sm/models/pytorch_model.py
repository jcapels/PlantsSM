import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Callable, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
from numpy import ndarray
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from plants_sm.data_structures.dataset import Dataset
from plants_sm.io.pickle import read_pickle, write_pickle
from plants_sm.models._utils import _convert_proba_to_unified_form, \
    write_model_parameters_to_pickle, array_from_tensor, array_reshape
from plants_sm.models.constants import REGRESSION, QUANTILE, BINARY, FileConstants
from plants_sm.models.model import Model
import torch


class PyTorchModel(Model):

    def __init__(self, model: nn.Module, loss_function: _Loss, optimizer: Optimizer = None,
                 scheduler: ReduceLROnPlateau = None, epochs: int = 32, batch_size: int = 32,
                 patience: int = 4, validation_metric: Callable = None, problem_type: str = BINARY,
                 device: Union[str, torch.device] = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 trigger_times: int = 0, last_loss: int = None, progress: int = 100, logger_path: str = None):

        """
        Constructor for PyTorchModel

        Parameters
        ----------
        model: nn.Module
            PyTorch model
        loss_function: _Loss
            PyTorch loss function
        optimizer: Optimizer
            PyTorch optimizer
        scheduler: ReduceLROnPlateau
            PyTorch scheduler
        epochs: int
            Number of epochs
        batch_size: int
            Batch size
        patience: int
            Number of epochs to wait before early stopping
        validation_metric: Callable
            Sklearn metric function to use for validation
        problem_type: str
            Type of problem
        device: Union[str, torch.device]
            Device to use for training
        trigger_times: int
            Number of times the model has been triggered
        last_loss: int
            Last loss value
        progress: int
            Number of batches to wait before logging progress
        logger_path: str
            Path to save the logger
        """

        super().__init__()

        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        if logger_path:
            handler = TimedRotatingFileHandler(logger_path, when='midnight', backupCount=30)
        else:
            handler = TimedRotatingFileHandler('./pytorch_model.log', when='midnight', backupCount=20)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

        self.device = device
        self.model = model.to(self.device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.progress = progress
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_metric = validation_metric
        self.problem_type = problem_type
        self.trigger_times = trigger_times
        self.last_loss = last_loss

        loss_dataframe = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss"])
        loss_dataframe.set_index("epoch", inplace=True)

        metric_dataframe = pd.DataFrame(columns=["epoch", "train_metric_result", "valid_metric_result"])
        metric_dataframe.set_index("epoch", inplace=True)

        self._history = {'loss': loss_dataframe,
                         'metric_results': metric_dataframe}

        self.writer = SummaryWriter()

        if not self.optimizer:
            self.optimizer = Adam(self.model.parameters())
        if not self.scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

    @property
    def history(self) -> dict:
        """
        Get the history of the model

        Returns
        -------
        dict
            History of the model
        """
        return self._history

    @staticmethod
    def _read_pytorch_model(path: str) -> nn.Module:
        """
        Read the model from the specified path.

        Parameters
        ----------
        path: str
            Path to read the model from

        Returns
        -------
        torch.nn.Module
        """
        weights_path = os.path.join(path, FileConstants.PYTORCH_MODEL_WEIGHTS.value)
        model = read_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value))
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        return model

    @staticmethod
    def _save_pytorch_model(model: nn.Module, path: str) -> None:
        """
        Save the model to the specified path.

        Parameters
        ----------
        model: torch.nn.Module
            Model to be saved
        path: str
            Path to save the model

        Returns
        -------

        """
        weights_path = os.path.join(path, FileConstants.PYTORCH_MODEL_WEIGHTS.value)
        torch.save(model.state_dict(), weights_path)
        write_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value), model)

    def _save(self, path: str):
        """
        Save the model to a file
        This method is called by the save method and needs to have all the parameters one wants to save in the model.

        Parameters
        ----------
        path: str
            Path to save the model

        Returns
        -------
        """
        self._save_pytorch_model(self.model, path)

        model_parameters = {
            'loss_function': self.loss_function,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'validation_metric': self.validation_metric,
            'problem_type': self.problem_type,
            'device': self.device,
            'trigger_times': self.trigger_times,
            'last_loss': self.last_loss,
            'progress': self.progress
        }

        write_model_parameters_to_pickle(model_parameters, path)

    @classmethod
    def _load(cls, path: str) -> 'PyTorchModel':
        """
        Load the model from a file

        Parameters
        ----------
        path: str
            Path to load the model

        """
        model = cls._read_pytorch_model(path)
        model_parameters = read_pickle(os.path.join(path, FileConstants.MODEL_PARAMETERS_PKL.value))
        return cls(model, **model_parameters)

    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """
        Preprocess the data for training

        Parameters
        ----------
        dataset: Dataset
            Dataset to preprocess
        shuffle: bool
            Whether to shuffle the data

        Returns
        -------
        DataLoader
            Preprocessed data
        """

        tensors = []
        if isinstance(dataset.X, Dict):
            for instance in dataset.X.keys():
                tensor = torch.tensor(dataset.X[instance], dtype=torch.float)
                tensors.append(tensor)
        else:
            tensor = torch.tensor(dataset.X, dtype=torch.float)
            tensors.append(tensor)
        if dataset.y is not None:
            tensors.append(torch.tensor(dataset.y, dtype=torch.float))
        dataset = TensorDataset(
            *tensors
        )

        data_loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size
        )
        return data_loader

    def _validate_batches(self, validation_set: DataLoader, loss_total: float, predictions: ndarray,
                          actuals: ndarray) -> Tuple[float, ndarray, ndarray]:

        len_valid_dataset = len(validation_set)
        for i, inputs_targets in enumerate(validation_set):
            inputs, targets = inputs_targets[:-1], inputs_targets[-1]
            for j, inputs_elem in enumerate(inputs):
                inputs[j] = inputs_elem.to(self.device)

            targets = targets.to(self.device)
            output = self.model(inputs)

            y_hat = array_from_tensor(output)
            actual = array_from_tensor(targets)

            predictions = np.concatenate((predictions, y_hat))
            actuals = np.concatenate((actuals, actual))

            loss = self.loss_function(output, targets)
            loss_total += loss.item()

            if i % self.progress == 0:
                self.logger.info(f'Validation set: [{i}/{len_valid_dataset}] loss: {loss.item():.8}')

        return loss_total, predictions, actuals

    def _validate(self, validation_set: Dataset) -> Tuple[float, float]:
        """
        Validate the model

        Parameters
        ----------
        validation_set: DataLoader
            Validation set

        Returns
        -------
        Tuple[float, float]
            Validation loss and validation metric result
        """
        self.model.eval()
        loss_total = 0
        predictions, actuals = np.empty(shape=(0, 1)), np.empty(shape=(0, 1))
        with torch.no_grad():

            if validation_set.batch_size is None:
                validation_set_preprocessed = self._preprocess_data(validation_set, shuffle=False)
                len_valid_dataset = len(validation_set_preprocessed)
                loss_total, predictions, actuals = \
                    self._validate_batches(validation_set_preprocessed, loss_total, predictions, actuals)

            else:
                len_valid_dataset = 0
                while validation_set.next_batch():
                    validation_set_preprocessed = self._preprocess_data(validation_set, shuffle=False)
                    len_valid_dataset += len(validation_set_preprocessed)
                    loss_total, predictions, actuals = \
                        self._validate_batches(validation_set_preprocessed, loss_total, predictions, actuals)

        validation_metric_result = None
        if self.validation_metric:
            predictions = self.get_pred_from_proba(np.array(predictions))
            validation_metric_result = self.validation_metric(actuals, predictions)

        return loss_total / len_valid_dataset, validation_metric_result

    def _train(self, inputs_targets: Tensor) -> Tuple[np.ndarray, np.ndarray, Tensor]:
        """
        Train the model

        Parameters
        ----------
        inputs_targets: Tensor
            Inputs and targets

        Returns
        -------
        Tuple[List[float], List[float], Tensor]
            Loss, predictions, targets
        """

        inputs, targets = inputs_targets[:-1], inputs_targets[-1]

        for j, inputs_elem in enumerate(inputs):
            inputs[j] = inputs_elem.to(self.device)

        targets = targets.to(self.device)

        self.optimizer.zero_grad()

        output = self.model(inputs)
        # Zero the gradients

        # Forward and backward propagation
        loss = self.loss_function(output, targets)
        loss.backward()
        self.optimizer.step()

        actual = array_from_tensor(targets)
        actual = array_reshape(actual)

        yhat = array_from_tensor(output)
        yhat = array_reshape(yhat)

        return actual, yhat, loss

    def _register_history(self, loss: float, epoch: int, metric_result: float, train: bool = True) -> None:
        """
        Register the history of the model

        Parameters
        ----------
        loss: float
            Loss
        epoch: int
            Epoch
        metric_result: float
            Metric result
        train: bool
            Whether it is training or validation
        """
        dataset_type = "train" if train else "valid"

        self.writer.add_scalar(f"Loss/{dataset_type}", loss, epoch)
        self._history["loss"].at[epoch - 1, f"{dataset_type}_loss"] = loss
        self.writer.add_scalar(f"Metric/{dataset_type}", metric_result, epoch)
        self._history["metric_results"].at[epoch - 1, f"{dataset_type}_metric_result"] = metric_result

    def _early_stopping(self, validation_dataset: Dataset, epoch: int) -> Union[nn.Module, None]:
        """
        Early stopping

        Parameters
        ----------
        validation_dataset: DataLoader
            Validation dataset
        epoch: int
            Epoch

        Returns
        -------
        Union[nn.Module, None]
            Model or None
        """

        # Early stopping
        current_loss, validation_metric_result = self._validate(validation_dataset)
        self.logger.info(
            f'Validation loss: {current_loss:.8}; Validation metric: {validation_metric_result:.8}')
        if current_loss >= self.last_loss:
            self.trigger_times += 1

            if self.trigger_times >= self.patience:
                return self.model

        else:
            self.trigger_times = 0

        self._register_history(current_loss, epoch, validation_metric_result, train=False)

        self.last_loss = current_loss

    def _train_epoch(self, train_dataset: Dataset, epoch: int,
                     validation_dataset: Dataset = None) -> Union[nn.Module, None]:
        """
        Train the model for one epoch

        Parameters
        ----------
        train_dataset: Dataset
            Training dataset
        epoch: int
            Epoch
        validation_dataset: Dataset
            Validation dataset

        Returns
        -------
        Union[nn.Module, None]
            Model
        """
        self.model.train()
        loss_total = 0
        predictions, actuals = np.empty(shape=(0, 1)), np.empty(shape=(0, 1))
        train_dataset_preprocessed = self._preprocess_data(train_dataset, shuffle=True)

        len_train_dataset = len(train_dataset_preprocessed)

        for i, inputs_targets in enumerate(train_dataset_preprocessed):
            actual, yhat, loss = self._train(inputs_targets)

            predictions = np.concatenate((predictions, yhat))
            actuals = np.concatenate((actuals, actual))
            loss_total += loss.item()
            # Show progress
            if i % self.progress == 0 or i == len_train_dataset - 1:
                self.logger.info(f'[{epoch}/{self.epochs}, {i}/{len_train_dataset}] loss: {loss.item():.8}')

                predictions = self.get_pred_from_proba(predictions)
                validation_metric_result = self.validation_metric(actuals, predictions)

                self.logger.info(f'[{epoch}/{self.epochs}, {i}/{len_train_dataset}] '
                                 f'metric result: {validation_metric_result:.8}')

        loss = loss_total / len_train_dataset

        predictions = self.get_pred_from_proba(predictions)
        validation_metric_result = self.validation_metric(actuals, predictions)
        self.logger.info(
            f'Training loss: {loss:.8};  Metric result: {validation_metric_result:.8}')
        self._register_history(loss, epoch, validation_metric_result)

        if validation_dataset:
            assert validation_dataset != train_dataset, "Validation dataset should not be the same as training dataset"
            return self._early_stopping(validation_dataset, epoch)
        else:
            return None

    def _fit_data(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> nn.Module:
        """
        Fit the model to the data

        Parameters
        ----------
        train_dataset: Dataset
            Training dataset
        validation_dataset: Dataset
            Validation dataset

        Returns
        -------
        nn.Module
            Trained model
        """

        self.last_loss = 100
        self.trigger_times = 0

        self.logger.info("starting to fit the data...")

        if train_dataset.batch_size is None:

            for epoch in range(1, self.epochs + 1):
                self._train_epoch(train_dataset, epoch, validation_dataset)

                self._write_model_check_points(epoch)

                self.scheduler.step(self.last_loss)
        else:

            for epoch in range(1, self.epochs + 1):

                while train_dataset.next_batch():

                    self._train_epoch(train_dataset, epoch, validation_dataset)

                    self._write_model_check_points(epoch)

                    self.scheduler.step(self.last_loss)

        self.writer.flush()
        return self.model

    def _write_model_check_points(self, epoch: int) -> None:
        """
        Write the model checkpoints for tensorboard.

        Parameters
        ----------
        epoch: int
            Epoch
        """

        os.makedirs("./.model_checkpoints", exist_ok=True)
        os.makedirs(f"./.model_checkpoints/{self.model.__class__.__name__}/epoch_{epoch}", exist_ok=True)
        torch.save(self.model.state_dict(), f"./.model_checkpoints/{self.model.__class__.__name__}/epoch_{epoch}"
                                            f"/model.pt")

    def get_pred_from_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Get the prediction from the probability

        Parameters
        ----------
        y_pred_proba: list
            List of probabilities
        Returns
        -------
        np.ndarray
            Array of predictions
        """
        if self.problem_type == BINARY:
            y_pred = np.array([1 if pred >= 0.5 else 0 for pred in y_pred_proba])
        elif self.problem_type == REGRESSION:
            y_pred = y_pred_proba
        elif self.problem_type == QUANTILE:
            y_pred = y_pred_proba
        else:
            y_pred = []
            if not len(y_pred_proba) == 0:
                y_pred = np.argmax(y_pred_proba, axis=1)
                return y_pred

        y_pred = array_reshape(y_pred)
        return y_pred

    def _predict_proba_batch(self, dataset: Dataset, predictions: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each class for each sample in the dataset.

        Parameters
        ----------
        dataset: DataLoader
            Dataset to predict on
        predictions: np.ndarray
            Array of predictions

        Returns
        -------

        """
        dataset_preprocessed = self._preprocess_data(dataset, shuffle=False)
        for i, inputs_targets in enumerate(dataset_preprocessed):
            for j, inputs_elem in enumerate(inputs_targets):
                inputs_targets[j] = inputs_elem.to(self.device)

            yhat = self.model(inputs_targets)

            yhat = array_from_tensor(yhat)
            yhat = array_reshape(yhat)

            predictions = np.concatenate((predictions, yhat))

        return predictions

    def _predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the probability of each class for each sample in the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to predict on

        Returns
        -------
        np.ndarray
            Array of prediction probabilities
        """

        if self.problem_type in [REGRESSION, QUANTILE]:
            y_pred = self.model.predict(dataset)
            return y_pred

        self.model.eval()
        predictions = np.empty(shape=(0, 1))

        # the "shuffle" argument always has to be False in predicting probabilities in an evaluation context

        with torch.no_grad():

            if dataset.batch_size is None:
                predictions = self._predict_proba_batch(dataset, predictions)
            else:
                while dataset.next_batch():
                    predictions = self._predict_proba_batch(dataset, predictions)

        return _convert_proba_to_unified_form(self.problem_type, np.array(predictions))

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class for each sample in the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to predict on

        Returns
        -------
        np.ndarray
            Array of predictions
        """
        y_pred_proba = self._predict_proba(dataset)
        y_pred = self.get_pred_from_proba(y_pred_proba)
        return y_pred
