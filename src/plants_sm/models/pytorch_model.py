import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Callable, Union

import numpy as np
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from plants_sm.data_structures.dataset import Dataset
from plants_sm.models.constants import REGRESSION, QUANTILE, BINARY
from plants_sm.models.model import Model
import torch


class PyTorchModel(Model):

    def __init__(self, model: nn.Module, loss_function: _Loss, optimizer: Optimizer = None,
                 scheduler: ReduceLROnPlateau = None, epochs: int = 32, batch_size: int = 32,
                 patience: int = 4, validation_metric: Callable = None, problem_type: str = BINARY,
                 device: Union[str, torch.device] = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 trigger_times: int = 0, last_loss: int = None, progress: int = 100, logger_path: str = None):

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

        self.writer = SummaryWriter()

        if not self.optimizer:
            self.optimizer = Adam(self.model.parameters())
        if not self.scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

    def _save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True):

        tensors = []
        for instance in dataset.X.keys():
            tensor = torch.tensor(dataset.X[instance], dtype=torch.float)
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

    def _test(self, test_set):
        self.model.eval()
        predictions, actuals = list(), list()
        with torch.no_grad():
            for i, inputs_targets in enumerate(test_set):
                inputs, targets = inputs_targets[:-1], inputs_targets[-1]
                for j, inputs_elem in enumerate(inputs):
                    inputs[j] = inputs_elem.to(self.device)

                targets = targets.to(self.device)
                yhat = self.model(inputs)

                yhat = yhat.cpu().detach().numpy()
                actual = targets.cpu().numpy()
                actual = actual.reshape((len(actual),)).tolist()
                yhat = yhat.reshape((len(yhat),)).tolist()
                predictions.extend(yhat)
                actuals.extend(actual)

        predictions = self.get_pred_from_proba(predictions)
        return self.validation_metric(actuals, predictions)

    def _validate(self, validation_set: DataLoader):
        self.model.eval()
        loss_total = 0
        predictions, actuals = list(), list()
        len_valid_dataset = len(validation_set)
        with torch.no_grad():
            for i, inputs_targets in enumerate(validation_set):
                inputs, targets = inputs_targets[:-1], inputs_targets[-1]
                for j, inputs_elem in enumerate(inputs):
                    inputs[j] = inputs_elem.to(self.device)

                targets = targets.to(self.device)
                output = self.model(inputs)

                yhat = output.cpu().detach().numpy()
                actual = targets.cpu().numpy()
                actual = actual.reshape((len(actual),)).tolist()
                yhat = yhat.reshape((len(yhat),)).tolist()
                predictions.extend(yhat)
                actuals.extend(actual)

                loss = self.loss_function(output, targets)
                loss_total += loss.item()

                if i % self.progress == 0:
                    self.logger.info(f'Validation set: [{i}/{len_valid_dataset}] loss: {loss.item():.8}')

        validation_metric_result = None
        if self.validation_metric:
            predictions = self.get_pred_from_proba(predictions)
            validation_metric_result = self.validation_metric(actuals, predictions)

        return loss_total / len_valid_dataset, validation_metric_result

    def _fit_data(self, train_dataset: Dataset, validation_dataset: Dataset = None):

        last_loss = 100
        trigger_times = 0

        self.logger.info("starting to fit the data...")

        train_dataset = self._preprocess_data(train_dataset)
        if validation_dataset:
            validation_dataset = self._preprocess_data(validation_dataset)

        len_train_dataset = len(train_dataset)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loss_total = 0
            predictions, actuals = list(), list()
            for i, inputs_targets in enumerate(train_dataset):
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

                actual = targets.cpu().numpy()
                actual = actual.reshape((len(actual),)).tolist()
                yhat = output.reshape((len(output),)).tolist()
                predictions.extend(yhat)
                actuals.extend(actual)
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
            self.writer.add_scalar("Loss/train", loss, epoch)
            self.writer.add_scalar("Metric/train", validation_metric_result, epoch)

            # Early stopping
            if validation_dataset:
                current_loss, validation_metric_result = self._validate(validation_dataset)
                self.logger.info(
                    f'Validation loss: {current_loss:.8}; Validation metric: {validation_metric_result:.8}')
                if current_loss >= last_loss:
                    trigger_times += 1

                    if trigger_times >= self.patience:
                        return self.model

                else:
                    print('trigger times: 0')
                    trigger_times = 0

                self.writer.add_scalar("Loss/validation", current_loss, epoch)
                self.writer.add_scalar("Metric/validation", validation_metric_result, epoch)

                last_loss = current_loss

            os.makedirs("./.model_checkpoints", exist_ok=True)
            os.makedirs(f"./.model_checkpoints/{self.model.__class__.__name__}/epoch_{epoch}", exist_ok=True)
            torch.save(self.model.state_dict(), f"./.model_checkpoints/{self.model.__class__.__name__}/epoch_{epoch}"
                                                f"/model.pt")

            self.scheduler.step(last_loss)

        self.writer.flush()

    def get_pred_from_proba(self, y_pred_proba):
        if self.problem_type == BINARY:
            y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]
        elif self.problem_type == REGRESSION:
            y_pred = y_pred_proba
        elif self.problem_type == QUANTILE:
            y_pred = y_pred_proba
        else:
            y_pred = []
            if not len(y_pred_proba) == 0:
                y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def _convert_proba_to_unified_form(self, y_pred_proba):
        """
        Ensures that y_pred_proba is in a consistent form across all models.
        For binary classification, converts y_pred_proba to a 1 dimensional array of prediction probabilities of the positive class.
        For multiclass and softclass classification, keeps y_pred_proba as a 2 dimensional array of prediction probabilities for each class.
        For regression, converts y_pred_proba to a 1 dimensional array of predictions.
        """
        if self.problem_type == REGRESSION:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            else:
                return y_pred_proba[:, 1]
        elif self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif y_pred_proba.shape[1] > 2:  # Multiclass, Softclass
            return y_pred_proba
        else:  # Unknown problem type
            raise AssertionError(f'Unknown y_pred_proba format for `problem_type="{self.problem_type}"`.')

    def _predict_proba(self, dataset: Dataset):

        dataset = self._preprocess_data(dataset, shuffle=False)

        if self.problem_type in [REGRESSION, QUANTILE]:
            y_pred = self.model.predict(dataset)
            return y_pred

        self.model.eval()
        predictions, actuals = list(), list()

        # the "shuffle" argument always has to be False in predicting probabilities in an evaluation context

        with torch.no_grad():
            for i, inputs_targets in enumerate(dataset):
                inputs, targets = inputs_targets[:-1], inputs_targets[-1]
                for j, inputs_elem in enumerate(inputs):
                    inputs[j] = inputs_elem.to(self.device)

                targets.to(self.device)
                yhat = self.model(inputs)

                yhat = yhat.cpu().detach().numpy()
                predictions.extend(yhat)

                actuals.extend(targets.cpu().numpy())

        return self._convert_proba_to_unified_form(np.array(predictions))

    def _predict(self, dataset: Dataset):
        y_pred_proba = self._predict_proba(dataset)
        y_pred = self.get_pred_from_proba(y_pred_proba)
        return y_pred
