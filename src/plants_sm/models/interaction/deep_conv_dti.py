from copy import copy

import torch
from torch import nn
from torch.nn.functional import elu, sigmoid

import torch.nn.functional as F


class DeepConvDTI(nn.Module):

    def __init__(self, protein_strides, proteins_layers, compounds_layers, interaction_layers, filters):
        super().__init__()
        self.proteins_embedding = nn.Embedding(num_embeddings=21, embedding_dim=20)
        self.protein_strides = protein_strides
        self.filters = filters
        self.proteins_layers = proteins_layers
        self.compounds_layers = compounds_layers
        self.interaction_layers = interaction_layers

    class PLayer(nn.Module):

        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, stride, stride=stride)
            self.bn = nn.BatchNorm1d(out_channels)
            self.elu = nn.ELU()

        def forward(self, x):
            x = self.conv.cuda()(x)
            x = F.normalize(x, p=2, dim=1)
            x = self.bn.cuda()(x)
            x = self.elu(x)
            x = nn.AdaptiveMaxPool1d(2)(x)
            return x

    def forward(self, x):
        proteins = x[0]
        compounds = x[1]

        proteins = proteins.to(torch.int64)
        proteins = self.proteins_embedding(proteins)

        proteins = F.normalize(proteins, p=2, dim=1)

        proteins = nn.BatchNorm1d(proteins.shape[1]).to("cuda")(proteins)
        proteins = nn.Dropout1d(0.2)(proteins)
        conv_results = []
        conv_proteins = copy(proteins)
        for i, stride in enumerate(self.protein_strides):
            conv_results.append(self.PLayer(proteins.shape[1], self.filters, stride)(conv_proteins))

        proteins = torch.cat(conv_results, dim=1)
        proteins = proteins.view(proteins.shape[0], -1)
        for i, layer in enumerate(self.proteins_layers):
            proteins = nn.Linear(proteins.shape[1], layer).cuda()(proteins)
            proteins = F.normalize(proteins, p=2, dim=1)
            proteins = nn.BatchNorm1d(proteins.shape[1]).cuda()(proteins)
            proteins = elu(proteins).cuda()

        for i, layer in enumerate(self.compounds_layers):
            compounds = nn.Linear(compounds.shape[1], layer).cuda()(compounds)
            compounds = F.normalize(compounds, p=2, dim=1)
            compounds = nn.BatchNorm1d(compounds.shape[1]).cuda()(compounds)
            compounds = elu(compounds).cuda()

        interaction = torch.cat([proteins, compounds], dim=1)
        for i, layer in enumerate(self.interaction_layers):
            interaction = nn.Linear(interaction.shape[1], layer).cuda()(interaction)
            interaction = F.normalize(interaction, p=2, dim=1)
            interaction = nn.BatchNorm1d(interaction.shape[1]).cuda()(interaction)
            interaction = elu(interaction).cuda()

        yhat = sigmoid(nn.Linear(interaction.shape[1], 1).cuda()(interaction))

        return yhat