import os
from typing import Union

from joblib import Parallel, delayed
from tqdm import tqdm
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS, ESM_LAYERS
from plants_sm.models.fc.fc import DNN
import lightning as L
from plants_sm.models.lightning_model import InternalLightningModule

from plants_sm.models.ec_number_prediction._esm_model_architectures import FineTuneESM2

from torch.utils.data import DataLoader, TensorDataset

from plants_sm.data_structures.dataset import Dataset, PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS

import torch

from torch import nn


from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import torch

from plants_sm.transformation._utils import tqdm_joblib 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR

class EC_ESM_Lightning(InternalLightningModule):

    def __init__(self, model_name, hidden_layers, num_classes, freeze_layers=True, use_embedding = False, **kwargs):
        super(EC_ESM_Lightning, self).__init__(**kwargs)
        self.model_name = model_name
        
        self._contructor_parameters.update({"model_name": model_name, 
                                      "hidden_layers": hidden_layers, 
                                      "num_classes": num_classes})
        esm_callable = ESM_FUNCTIONS[model_name]
        self.layers = ESM_LAYERS[model_name]

        self.use_embedding = use_embedding
        self.freeze_layers = freeze_layers

        model, self.alphabet = esm_callable()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model = FineTuneESM2(alphabet=self.alphabet, num_layers=model.num_layers, embed_dim=model.embed_dim,
                                      attention_heads=model.attention_heads, token_dropout=model.token_dropout)
        
        self.esm_model.load_state_dict(model.state_dict())

        self.dnn = DNN(model.embed_dim, hidden_layers, num_classes, batch_norm=True, last_sigmoid=True)


    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.freeze_layers:
            if self.trainer.current_epoch < 14:
                output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True, layers_to_freeze=self.layers)
            elif self.trainer.current_epoch >= 14 and self.trainer.current_epoch < 29:
                output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True, layers_to_freeze=self.layers*3//4)
            else:
                output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True, layers_to_freeze=self.layers*2//3)
        else:
            output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True)

        output = output["representations"][self.layers]
        if not self.use_embedding:
            output = output[:, 0, :]

        else:
            batch_lens = (x != self.alphabet.padding_idx).sum(1)
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(output[i, 1 : tokens_len - 1].mean(0))

            output = torch.stack(sequence_representations)
        logits = self.dnn([output])

        loss = self.compute_loss(logits, y)
        
        self.training_step_outputs.append(logits)
        self.training_step_y_true.append(y)
        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch

        output = self.esm_model(inputs, repr_layers=[self.layers], need_head_weights=True)
        output = output["representations"][self.layers]

        if not self.use_embedding:
            output = output[:, 0, :]

        else:
            batch_lens = (inputs != self.alphabet.padding_idx).sum(1)
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(output[i, 1 : tokens_len - 1].mean(0))

            output = torch.stack(sequence_representations)

        output = self.dnn([output])
        self.validation_step_outputs.append(output)
        self.validation_step_y_true.append(target)

    # def forward(self, data):

    #     output = self.esm_model(data, repr_layers=[self.layers], need_head_weights=True)
    #     output = output["representations"][self.layers]
    #     output = output[:, 0, :]
    #     x = self.dnn([output])

    #     return x


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.esm_model.parameters(), 'lr': 1e-7},
                                      {'params': self.dnn.parameters()}], lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        # Define a custom learning rate scheduler using LambdaLR
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}

        return [optimizer], [scheduler]

    
    def compute_loss(self, logits, targets):
        return nn.BCELoss()(logits, targets)
    
    def contructor_parameters(self):
        return self._contructor_parameters
    
class EC_ESM1b_Lightning(InternalLightningModule):
    def __init__(self, hidden_layers, num_classes, freeze_layers,use_embedding, **kwargs):
        super(EC_ESM1b_Lightning, self).__init__(**kwargs)
        esm_callable = ESM_FUNCTIONS["esm1b_t33_650M_UR50S"]
        self.layers = ESM_LAYERS["esm1b_t33_650M_UR50S"]

        self.model_name = "esm1b_t33_650M_UR50S"

        self.use_embedding = use_embedding
        self.freeze_layers = freeze_layers

        model, self.alphabet = esm_callable()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model = model

        self.dnn = DNN(1280, hidden_layers, num_classes, batch_norm=False, last_sigmoid=True)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.freeze_layers:
            if self.trainer.current_epoch < 14:
                output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True, layers_to_freeze=self.layers)
            elif self.trainer.current_epoch >= 14 and self.trainer.current_epoch < 29:
                output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True, layers_to_freeze=self.layers*3//4)
            else:
                output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True, layers_to_freeze=self.layers*2//3)
        else:
            output = self.esm_model(x, repr_layers=[self.layers], need_head_weights=True)

        output = output["representations"][self.layers]
        if not self.use_embedding:
            output = output[:, 0, :]

        else:
            batch_lens = (x != self.alphabet.padding_idx).sum(1)
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(output[i, 1 : tokens_len - 1].mean(0))

            output = torch.stack(sequence_representations)
        logits = self.dnn([output])

        loss = self.compute_loss(logits, y)
        
        self.training_step_outputs.append(logits)
        self.training_step_y_true.append(y)
        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch

        output = self.esm_model(inputs, repr_layers=[self.layers], need_head_weights=True)
        output = output["representations"][self.layers]

        if not self.use_embedding:
            output = output[:, 0, :]

        else:
            batch_lens = (inputs != self.alphabet.padding_idx).sum(1)
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(output[i, 1 : tokens_len - 1].mean(0))

            output = torch.stack(sequence_representations)

        output = self.dnn([output])
        self.validation_step_outputs.append(output)
        self.validation_step_y_true.append(target)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.esm_model.parameters(), 'lr': 1e-5},
                                      {'params': self.dnn.parameters()}], lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        # Define a custom learning rate scheduler using LambdaLR
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}

        return [optimizer], [scheduler]
    
    def compute_loss(self, logits, targets):
        return nn.BCELoss()(logits, targets)

    def _preprocess_data(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        tensors = []
        sequences = [(sequence_id, dataset.instances[PLACEHOLDER_FIELD][sequence_id]) for sequence_id in
                     dataset.dataframe[dataset.instances_ids_field]]
        
        esm_callable = ESM_FUNCTIONS[self.model_name]

        _, alphabet = esm_callable()
        batch_converter = alphabet.get_batch_converter()

        # _, _, tokens = batch_converter(sequences)

        batch_size = 10000  # You can adjust this based on your preferences

        # Initialize the progress bar
        progress_bar = tqdm(total=len(sequences), desc="Processing sequences", position=0, leave=True)

        # Define the function to be parallelized
        def process_batch(batch):
            _, _, tokens = batch_converter(batch)
            return tokens

        # Process sequences in parallel with a progress bar in batches
        result_tokens = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_results = Parallel(n_jobs=-1)(delayed(process_batch)(batch) for batch in [batch])
            result_tokens.extend(batch_results)
            progress_bar.update(len(batch))

        # Close the progress bar
        progress_bar.close()

        # Use joblib to parallelize the function across sequences
        result_tokens = torch.cat(result_tokens, dim=0)
        tensors.append(result_tokens)

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
    