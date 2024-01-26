import esm

from typing import Union
import torch
from esm.model.esm2 import ESM2
from esm.model.esm1 import ProteinBertModel


class ESM2Model(ESM2):

    def __init__(self, num_layers: int = 33,
                 embed_dim: int = 1280,
                 attention_heads: int = 20,
                 alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
                 token_dropout: bool = True, is_ddf=False,
                 num_gpus=1, 
                 devices=None) -> None:
        self.is_ddf = is_ddf
        self.num_gpus = num_gpus
        if devices is None and num_gpus is not None:
            self.devices = [f"cuda:{i}" for i in range(num_gpus)]
        else:
            self.devices = devices
        super().__init__(num_layers, embed_dim, attention_heads, alphabet, token_dropout)

    def forward(self, tokens, repr_layers=None, need_head_weights=False, return_contacts=False):

        if repr_layers is None:
            repr_layers = []
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        self.embed_tokens = self.embed_tokens.to(tokens.device)
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

        if self.is_ddf:
            gpus = self.devices
        i = 0
        for layer_idx, layer in enumerate(self.layers):
            if self.is_ddf:
                gpu = gpus[i % len(gpus)]
                x.to(gpu, dtype=torch.float32)
                layer.to(gpu)
            elif self.num_gpus == 0:
                x = x.to("cpu")
                layer = layer.to("cpu")
            else:
                x = x.to(self.devices[0])
                layer = layer.to(self.devices[0])

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
            i += 1

        if self.is_ddf:
            gpu = gpus[i % len(gpus)]
            x.to(gpu)
            self.emb_layer_norm_after.to(gpu)
        elif self.num_gpus == 0:
            x = x.to("cpu")
            self.emb_layer_norm_after = self.emb_layer_norm_after.to("cpu")
        else:
            x = x.to(self.devices[0])
            self.emb_layer_norm_after = self.emb_layer_norm_after.to(self.devices[0])

        i += 1
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        if self.is_ddf:
            gpu = gpus[i % len(gpus)]
            x.to(gpu)
            self.lm_head.to(gpu)
        elif self.num_gpus == 0:
            x = x.to("cpu")
            self.lm_head = self.lm_head.to("cpu")
        else:
            x = x.to(self.devices[0])
            self.lm_head = self.lm_head.to(self.devices[0])
        i += 1
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


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


class ESM1Model(ProteinBertModel):

    def __init__(self, args, alphabet, device="cpu"):
        super().__init__(args, alphabet)
        self.device = device
        for param in self.parameters():
            # Check if parameter dtype is  Half (float16)
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        self.embed_tokens = self.embed_tokens.to(tokens.device)

        x = self.embed_scale * self.embed_tokens(tokens)
        x = x.float()

        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        self.embed_positions = self.embed_positions.to(tokens.device)
        x = x + self.embed_positions(tokens)


        if self.model_version == "ESM-1b":
            if self.emb_layer_norm_before:
                x = x.to(self.device)
                self.emb_layer_norm_before = self.emb_layer_norm_before.to(self.device)
                self.emb_layer_norm_before.weight = self.emb_layer_norm_before.weight.float()
                self.emb_layer_norm_before.bias = self.emb_layer_norm_before.bias.float()
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

            x = x.to(self.device)
            layer.to(self.device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)

            layer.buffer_dtype = torch.float32
            layer.compute_dtype = torch.float32
            x, attn = layer(
                x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        if self.model_version == "ESM-1b":
            x = x.to(self.device)
            self.emb_layer_norm_after = self.emb_layer_norm_after.to(self.device)

            self.emb_layer_norm_after.weight = self.emb_layer_norm_after.weight.float()
            self.emb_layer_norm_after.bias = self.emb_layer_norm_after.bias.float()
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x

            x = x.to(self.device)
            self.lm_head = self.lm_head.to(self.device)

            self.lm_head.weight = self.lm_head.weight.float()
            self.lm_head.bias = self.lm_head.bias.float()
            self.lm_head.dense.weight = self.lm_head.dense.weight.float()
            self.lm_head.dense.bias = self.lm_head.dense.bias.float()
            self.lm_head.layer_norm.weight = self.lm_head.layer_norm.weight.float()
            self.lm_head.layer_norm.bias = self.lm_head.layer_norm.bias.float()
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
