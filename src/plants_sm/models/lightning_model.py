from abc import abstractmethod
import os
from typing import Dict, Union
from plants_sm.models.model import Model
from plants_sm.data_structures.dataset.dataset import Dataset
from plants_sm.models.constants import FileConstants
from plants_sm.io.pickle import read_pickle, write_pickle
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import torch
from plants_sm.models._utils import _get_pred_from_proba, _convert_proba_to_unified_form, write_model_parameters_to_pickle
from plants_sm.models.constants import BINARY, FileConstants

from plants_sm.models._utils import _convert_proba_to_unified_form, \
    write_model_parameters_to_pickle

import numpy as np

class InternalLightningModel(Model):

    def __init__(self, module, batch_size: int = 32, devices="cpu", model_name = "lightning_model", **trainer_kwargs) -> None:
        super().__init__()

        self.name = model_name
        self.module = module
        self.batch_size = batch_size
        self.trainer_kwargs = trainer_kwargs
        if isinstance(devices, list):
            self.ddp = True
            self.trainer = L.Trainer(devices=devices, **trainer_kwargs)
        else:
            self.ddp = False
            self.trainer = L.Trainer(**trainer_kwargs)

        self.devices = devices

    def get_embedding(self, dataset):
        if isinstance(dataset, Dataset):
            predict_dataloader = self._preprocess_data(dataset, shuffle=False)
        else:
            predict_dataloader = dataset

        # the module has to have the return_embedding attribute
        self.module.return_embedding = True 
        embeddings = self.trainer.predict(self.module, predict_dataloader)
        # get a list of embeddings from the list of tuples
        embeddings = [embedding[1] for embedding in embeddings]
        embeddings = torch.cat(embeddings)
        return np.array(embeddings)
    
    def reset_weights(self):
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.module.apply(fn=weight_reset)


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
    

    def _fit_data(self, train_dataset: Union[Dataset, TensorDataset], 
                  validation_dataset: Union[Dataset, TensorDataset]):
        """
        Fits the model to the data.

        Parameters
        ----------
        train_dataset: Dataset
            The dataset to fit the model to.
        validation_dataset: Dataset
            The dataset to validate the model on.
        """

        self.trainer = L.Trainer(devices=self.devices, **self.trainer_kwargs)
        if isinstance(train_dataset, Dataset):
            train_dataset_loader = self._preprocess_data(train_dataset)
        else:
            train_dataset_loader = train_dataset

        if validation_dataset is not None:
            if isinstance(train_dataset, Dataset):
                validation_dataloader = self._preprocess_data(validation_dataset, shuffle=False)
            else:
                validation_dataloader = validation_dataset
        if validation_dataset:
            self.trainer.fit(model=self.module, train_dataloaders=train_dataset_loader, val_dataloaders=validation_dataloader)
        else:
            self.trainer.fit(model=self.module, train_dataloaders=train_dataset_loader)

    def _predict_proba(self, dataset: Union[Dataset, TensorDataset], trainer = L.Trainer(accelerator="cpu")) -> Union[Union[np.ndarray, list], list]:
        """
        Predicts the probabilities of the classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the probabilities on.

        Returns
        -------
        Union[np.ndarray, list]
            np.ndarray if the network only has one output, list of np.ndarray if the network has multiple outputs.
        Union[np.ndarray, list]
            np.ndarray if the network only has one output, list of np.ndarray if the network has multiple outputs.
        """
        if isinstance(dataset, Dataset):
            predict_dataloader = self._preprocess_data(dataset, shuffle=False)
        else:
            predict_dataloader = dataset

        predictions = trainer.predict(self.module, predict_dataloader)
        if type(predictions[0]) == tuple:
            len_tuple = len(predictions[0])
            new_predictions = [None] * len_tuple
            for i in range(len_tuple):
                new_predictions[i] = [prediction[i] for prediction in predictions]
                new_predictions[i] = torch.cat(new_predictions[i]).detach().cpu().numpy()
            predictions = new_predictions
        else:
            predictions = torch.cat(predictions)
            # convert to numpy array
            predictions = predictions.detach().cpu().numpy()
        return predictions


    def _predict(self, dataset: Union[Dataset, TensorDataset], trainer = L.Trainer(accelerator="cpu")) -> np.ndarray:
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

        if type(predictions) == list:
            for i in range(len(predictions)):
                predictions[i] = _convert_proba_to_unified_form(self.module.problem_type, np.array(predictions[i]))
                predictions[i] = _get_pred_from_proba(self.module.problem_type, predictions[i])
        else:
            predictions = _convert_proba_to_unified_form(self.module.problem_type, np.array(predictions))
            predictions = _get_pred_from_proba(self.module.problem_type, predictions)


        if type(predictions) == list:
            for i in range(len(predictions)):
                predictions[i] = _convert_proba_to_unified_form(self.module.problem_type, np.array(predictions[i]))
                predictions[i] = _get_pred_from_proba(self.module.problem_type, predictions[i])
        else:
            predictions = _convert_proba_to_unified_form(self.module.problem_type, np.array(predictions))
            predictions = _get_pred_from_proba(self.module.problem_type, predictions)

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
        weights_path = os.path.join(path, FileConstants.LIGHTNING_WEIGHTS.value)
        model = read_pickle(os.path.join(path, FileConstants.LIGHTNING_MODEL_PKL.value))
        weights_path = os.path.join(path, FileConstants.LIGHTNING_WEIGHTS.value)
        model = read_pickle(os.path.join(path, FileConstants.LIGHTNING_MODEL_PKL.value))
        model_parameters = read_pickle(os.path.join(path, FileConstants.MODEL_PARAMETERS_PKL.value))
        model = model.load_from_checkpoint(weights_path,**model_parameters)
        model = cls(module=model)
        model.trainer = read_pickle(os.path.join(path, "trainer.pk"))
        model.trainer = read_pickle(os.path.join(path, "trainer.pk"))
        return model

    def _save(self, path: str):
        """
        Saves the model to a file.

        Parameters
        ----------
        path: str
            The path to save the model to.
        """
        weights_path = os.path.join(path, FileConstants.LIGHTNING_WEIGHTS.value)
        weights_path = os.path.join(path, FileConstants.LIGHTNING_WEIGHTS.value)
        self.trainer.save_checkpoint(weights_path)
        write_pickle(os.path.join(path, "trainer.pk"), self.trainer)
        write_pickle(os.path.join(path, FileConstants.LIGHTNING_MODEL_PKL.value), self.module.__class__)
        write_model_parameters_to_pickle(self.module._contructor_parameters, path)

    @property
    def history(self):
        """
        Returns the underlying model.
        """


class InternalLightningModule(L.LightningModule):

    def __init__(self, problem_type: str = BINARY, metric = None):
        """
        Initializes the model.

        Parameters
        ----------
        batch_size: int
            The batch size to use.
        """
        super().__init__()
        self.problem_type = problem_type
        self.metric = metric

        self._constructor_parameters = { "problem_type": problem_type }

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.training_step_y_true = []
        self.validation_step_y_true = []
        self.epoch_losses = []
        self._update_constructor_parameters()

    @abstractmethod
    def _update_constructor_parameters(self):
        pass

    @abstractmethod
    def compute_loss(self, logits, y):
        pass
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        if not isinstance(x, list):
            x = [x]
        if not isinstance(x, list):
            x = [x]
        logits = self(x)
        loss = self.compute_loss(logits, y)
        logits = logits.detach().cpu()
        y = y.detach().cpu()
        logits = logits.detach().cpu()
        y = y.detach().cpu()
        
        self.training_step_outputs.append(logits)
        self.training_step_y_true.append(y)
        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(inputs, list):
            inputs = [inputs]
        output = self(inputs)

        output = output.detach().cpu()
        target = target.detach().cpu()
        output = output.detach().cpu()
        target = target.detach().cpu()

        self.validation_step_outputs.append(output)
        self.validation_step_y_true.append(target)

    def on_train_epoch_end(self) -> None:

        if self.metric is not None:

            predictions = _convert_proba_to_unified_form(self.problem_type, torch.cat(self.training_step_outputs).detach().cpu().numpy())
            predictions = _get_pred_from_proba(self.problem_type, predictions)
            self.log("train_metric", self.metric(torch.cat(self.training_step_y_true).detach().cpu().numpy(), predictions), 
                     on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.training_step_outputs = []
        self.training_step_y_true = []
    
    def on_validation_epoch_end(self) -> None:

        loss = self.compute_loss(torch.cat(self.validation_step_outputs), 
                                 torch.cat(self.validation_step_y_true))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if self.metric is not None:
            predictions = _convert_proba_to_unified_form(self.problem_type, torch.cat(self.validation_step_outputs).detach().cpu().numpy())
            predictions = _get_pred_from_proba(self.problem_type, predictions)
            self.log("val_metric", self.metric(torch.cat(self.validation_step_y_true).detach().cpu().numpy(), predictions),
                    on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.validation_step_outputs = []
        self.validation_step_y_true = []

    
    def predict_step(self, batch):
        if len(batch) == 2:
            inputs, target = batch
        else:
            inputs = batch
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(batch) == 2:
            inputs, target = batch
        else:
            inputs = batch
        if not isinstance(inputs, list):
            inputs = [inputs]
        return self(inputs)
    