import os
from typing import Union

from joblib import Parallel, delayed
from tqdm import tqdm
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS, ESM_LAYERS
from plants_sm.models.fc.fc import DNN
import lightning as L
from plants_sm.models.lightning_model import InternalLightningModule

from torch.utils.data import DataLoader, TensorDataset

from plants_sm.data_structures.dataset import Dataset, PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.bio_embeddings.constants import ESM_FUNCTIONS

import torch
import torch.nn.functional as F
from esm.model.esm1 import ProteinBertModel

from torch import nn
from esm.model.esm2 import ESM2

import esm

from plants_sm.io.pickle import read_pickle, write_pickle
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import torch
from plants_sm.models._utils import _get_pred_from_proba, _convert_proba_to_unified_form, write_model_parameters_to_pickle
from plants_sm.models.constants import BINARY, FileConstants

from plants_sm.transformation._utils import tqdm_joblib 

class ESM2Model(ESM2):

    def __init__(self, num_layers: int = 33,
                 embed_dim: int = 1280,
                 attention_heads: int = 20,
                 alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
                 token_dropout: bool = True) -> None:
        super().__init__(num_layers, embed_dim, attention_heads, alphabet, token_dropout)

    def forward(self, tokens, repr_layers=None, need_head_weights=True, return_contacts=False):

        if repr_layers is None:
            repr_layers = []
        if return_contacts:
            need_head_weights = True

        # assert tokens.ndim == 2
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

class ESM1Model(ProteinBertModel):

    def __init__(self, args, alphabet):
        super().__init__(args, alphabet)
        for param in self.parameters():
            # Check if parameter dtype is  Half (float16)
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

    def forward(self, tokens, repr_layers=[], need_head_weights=True, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        x = x + self.embed_positions(tokens)

        if self.model_version == "ESM-1b":
            if self.emb_layer_norm_before:
                x = self.emb_layer_norm_before(x)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        if self.model_version == "ESM-1b":
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            x = self.lm_head(x)
        else:
            x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

class EC_ESM_Lightning(InternalLightningModule):

    def __init__(self, model_name, hidden_layers, num_classes, **kwargs):
        super(EC_ESM_Lightning, self).__init__(**kwargs)
        self.model_name = model_name
        
        # self._contructor_parameters.update({"model_name": model_name, 
        #                               "hidden_layers": hidden_layers, 
        #                               "num_classes": num_classes})
        esm_callable = ESM_FUNCTIONS[model_name]
        self.layers = ESM_LAYERS[model_name]

        model, self.alphabet = esm_callable()

        self.batch_converter = self.alphabet.get_batch_converter()
        # self.esm_model = ESM2Model(alphabet=self.alphabet, num_layers=model.num_layers, embed_dim=model.embed_dim,
        #                       attention_heads=model.attention_heads, token_dropout=model.token_dropout)
        
        self.esm_model = model

        # self.esm_model.load_state_dict(model.state_dict())

        self.dnn = DNN(model.embed_dim, hidden_layers, num_classes, batch_norm=True, last_sigmoid=True)

    def forward(self, data):

        output = self.esm_model(data, repr_layers=[self.layers], need_head_weights=True)
        output = output["representations"][self.layers]
        output = output[:, 0, :]
        x = self.dnn([output])

        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
    
    def compute_loss(self, logits, targets):
        targets = targets.unsqueeze(0)
        return nn.BCELoss()(logits, targets)
    
    def contructor_parameters(self):
        return self._contructor_parameters
    
class EC_ESM1b_Lightning(InternalLightningModule):
    def __init__(self, hidden_layers, num_classes, **kwargs):
        super(EC_ESM1b_Lightning, self).__init__(**kwargs)
        esm_callable = ESM_FUNCTIONS["esm1b_t33_650M_UR50S"]
        self.layers = ESM_LAYERS["esm1b_t33_650M_UR50S"]

        model, self.alphabet = esm_callable()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model = ESM1Model(model.args, alphabet=self.alphabet)

        self.esm_model.load_state_dict(model.state_dict())

        self.dnn = DNN(1280, hidden_layers, num_classes, batch_norm=False, last_sigmoid=True)

    def forward(self, data):
        output = self.esm_model(data, repr_layers=[self.layers])
        output = output["representations"][self.layers]
        output = output[:, 0, :]
        x = self.dnn([output])
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.trainer.model.parameters(), lr=1e-3)
    
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
    