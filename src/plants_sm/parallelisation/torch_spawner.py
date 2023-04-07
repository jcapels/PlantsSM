import esm

from typing import Union
import torch
from esm.model.esm2 import ESM2


class ESMModel(ESM2):

    def __init__(self, num_layers: int = 33,
                 embed_dim: int = 1280,
                 attention_heads: int = 20,
                 alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
                 token_dropout: bool = True, is_ddp=False,
                 num_gpus=1) -> None:
        self.is_ddp = is_ddp
        self.num_gpus = num_gpus
        super().__init__(num_layers, embed_dim, attention_heads, alphabet, token_dropout)

    def forward(self, tokens, repr_layers=None, need_head_weights=False, return_contacts=False):

        if repr_layers is None:
            repr_layers = []
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
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

        if self.is_ddp:
            gpus = list(range(torch.cuda.device_count()))
            if len(gpus) >= self.num_gpus:
                gpus = gpus[:self.num_gpus]
            gpus = [f"cuda:{i}" for i in range(len(gpus))]

        i = 0
        for layer_idx, layer in enumerate(self.layers):
            if self.is_ddp:
                gpu = gpus[i % len(gpus)]
                x.to(gpu)
                layer.to(gpu)
            else:
                x = x.to("cpu")
                layer = layer.to("cpu")

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

        if self.is_ddp:
            gpu = gpus[i % len(gpus)]
            x.to(gpu)
            self.emb_layer_norm_after.to(gpu)
        else:
            x = x.to("cpu")
            self.emb_layer_norm_after = self.emb_layer_norm_after.to("cpu")

        i += 1
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        if self.is_ddp:
            gpu = gpus[i % len(gpus)]
            x.to(gpu)
            self.lm_head.to(gpu)
        else:
            x = x.to("cpu")
            self.lm_head = self.lm_head.to("cpu")
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
