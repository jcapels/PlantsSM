from abc import abstractmethod
import copy
import os
from typing import Dict
from plants_sm.models.model import Model
from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.parallelisation.torch_spawner import TorchSpawner
from plants_sm.models.constants import FileConstants
from plants_sm.io.pickle import read_pickle, write_pickle
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import torch
from plants_sm.models._utils import _get_pred_from_proba, _convert_proba_to_unified_form, write_model_parameters_to_pickle
from plants_sm.models.constants import BINARY, FileConstants

from plants_sm.models._utils import _convert_proba_to_unified_form, \
    write_model_parameters_to_pickle, array_from_tensor, array_reshape, multi_label_binarize

import numpy as np

class LightningModelModule(L.LightningModule, Model):

    def __init__(self, metric: callable = None, problem_type = BINARY, 
                 batch_size: int = 32, devices="cpu", **trainer_kwargs):
        """
        Initializes the model.

        Parameters
        ----------
        batch_size: int
            The batch size to use.
        """
        super().__init__()
        self.metric = metric
        self.batch_size = batch_size
        self.problem_type = problem_type
        self._contructor_parameters = {"problem_type": problem_type, 
                                       "batch_size": batch_size}

        if isinstance(devices, list):
            self.ddp = True
            self.trainer = L.Trainer(devices=devices, **trainer_kwargs)
        else:
            self.ddp = False
            self.trainer = L.Trainer(**trainer_kwargs)

        self.devices = devices

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def compute_loss(self, batch, stage=None):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.metric is not None:
            predictions = _convert_proba_to_unified_form(self.problem_type, logits.detach().cpu().numpy())
            predictions = _get_pred_from_proba(self.problem_type, predictions)
            self.log("train_metric", self.metric(predictions, y.detach().cpu().numpy()), 
                     on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.compute_loss(output, target)
        self.log("val_loss", loss)
        if self.metric is not None:
            predictions = _convert_proba_to_unified_form(self.problem_type, output.detach().cpu().numpy())
            predictions = _get_pred_from_proba(self.problem_type, predictions)
            self.log("val_metric", self.metric(predictions, target.detach().cpu().numpy()),
                     on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch):
        inputs, target = batch
        return self(inputs)
    
    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True) -> Dataset:
        """
        Preprocesses the data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to preprocess.
        shuffle: bool
            Whether to shuffle the data

        Returns
        -------
        Dataset
            The preprocessed dataset.
        """
        tensors = []
        if isinstance(dataset.X, Dict):
            for instance in dataset.X.keys():
                tensor = torch.tensor(dataset.X[instance], dtype=torch.float)
                tensors.append(tensor)
        else:
            tensor = torch.tensor(dataset.X, dtype=torch.float)
            tensors.append(tensor)
        
        try:
            if dataset.y is not None:
                tensors.append(torch.tensor(dataset.y, dtype=torch.float))
        except ValueError:
            pass
        
        dataset = TensorDataset(
            *tensors
        )

        data_loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size
        )
        return data_loader


    def _fit_data(self, train_dataset: Dataset, validation_dataset: Dataset):
        """
        Fits the model to the data.

        Parameters
        ----------
        train_dataset: Dataset
            The dataset to fit the model to.
        validation_dataset: Dataset
            The dataset to validate the model on.
        """
        train_dataset_loader = self._preprocess_data(train_dataset)
        if validation_dataset is not None:
            validation_dataloader = self._preprocess_data(validation_dataset, shuffle=False)
        if validation_dataset:
            self.trainer.fit(model=self, train_dataloaders=train_dataset_loader, val_dataloaders=validation_dataloader)
        else:
            self.trainer.fit(model=self, train_dataloaders=train_dataset_loader)

    def _predict_proba(self, dataset: Dataset, trainer = L.Trainer(accelerator="cpu")) -> np.ndarray:
        """
        Predicts the probabilities of the classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the probabilities on.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """
        predict_dataloader = self._preprocess_data(dataset, shuffle=False)

        predictions = trainer.predict(self, predict_dataloader)
        predictions = torch.cat(predictions)
        return predictions


    def _predict(self, dataset: Dataset, trainer = L.Trainer(accelerator="cpu")) -> np.ndarray:
        """
        Predicts the classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes on.

        Returns
        -------
        np.ndarray
            The predicted classes.
        """
        predictions = self._predict_proba(dataset, trainer)
        predictions = _convert_proba_to_unified_form(self.problem_type, np.array(predictions))
        predictions = _get_pred_from_proba(self.problem_type, predictions)
        return predictions
    
    @classmethod
    def _load(cls, path: str):
        """
        Loads the model from a file.

        Parameters
        ----------
        path: str
            The path to load the model from.
        """
        weights_path = os.path.join(path, "pytorch_model_weights.ckpt")
        model = read_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value))
        model_parameters = read_pickle(os.path.join(path, FileConstants.MODEL_PARAMETERS_PKL.value))
        model = model.load_from_checkpoint(weights_path,**model_parameters)
        model = cls(**model_parameters)
        return model

    def _save(self, path: str):
        """
        Saves the model to a file.

        Parameters
        ----------
        path: str
            The path to save the model to.
        """
        weights_path = os.path.join(path, "pytorch_model_weights.ckpt")
        self.trainer.save_checkpoint(weights_path)
        write_pickle(os.path.join(path, FileConstants.PYTORCH_MODEL_PKL.value), self.__class__)
        write_model_parameters_to_pickle(self._contructor_parameters, path)

    @property
    def history(self):
        """
        Returns the underlying model.
        """