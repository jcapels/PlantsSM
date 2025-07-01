from copy import copy

import torch
from torch import nn, relu


class BaselineModel(nn.Module):
    def __init__(self, input_size_proteins, input_size_compounds, hidden_layers_proteins,
                 hidden_layers_compounds, hidden_layers_interaction):
        super().__init__()
        self.hidden_layers_proteins = hidden_layers_proteins
        self.hidden_layers_compounds = hidden_layers_compounds
        self.hidden_layers_interaction = hidden_layers_interaction
        layers_neurons = input_size_proteins * 2
        linear_input = copy(layers_neurons)
        self.dense_proteins1 = nn.Linear(input_size_proteins, layers_neurons)
        for i, hidden_size in enumerate(hidden_layers_proteins):
            self.add_module(
                'fc_proteins{}'.format(i),
                nn.Linear(linear_input, hidden_size)
            )
            linear_input = copy(hidden_size)

        proteins_final_layer = copy(linear_input)

        layers_neurons = input_size_compounds * 2
        self.dense_compounds1 = nn.Linear(input_size_compounds, layers_neurons)
        linear_input = copy(layers_neurons)
        for i, hidden_size in enumerate(hidden_layers_compounds):
            self.add_module(
                'fc_compounds{}'.format(i),
                nn.Linear(linear_input, hidden_size)
            )
            linear_input = copy(hidden_size)
        compounds_final_layer = copy(linear_input)

        self.dense_interaction_layer1 = nn.Linear(proteins_final_layer + compounds_final_layer, layers_neurons)
        linear_input = copy(layers_neurons)
        for i, hidden_size in enumerate(hidden_layers_interaction):
            self.add_module(
                'fc_interaction{}'.format(i),
                nn.Linear(linear_input, hidden_size)
            )
            linear_input = copy(hidden_size)

        self.final_layer = nn.Linear(linear_input, 1)

    def forward(self, x):
        x_proteins = x[0]
        x_proteins = relu(self.dense_proteins1(x_proteins))
        for i, layer in enumerate(self.hidden_layers_proteins):
            x_proteins = relu(getattr(self, 'fc_proteins{}'.format(i))(x_proteins))

        x_compounds = x[1]
        x_compounds = relu(self.dense_compounds1(x_compounds))
        for i, layer in enumerate(self.hidden_layers_compounds):
            x_compounds = relu(getattr(self, 'fc_compounds{}'.format(i))(x_compounds))

        x_interaction = torch.cat([x_proteins, x_compounds], dim=1)
        x_interaction = relu(self.dense_interaction_layer1(x_interaction))
        for i, layer in enumerate(self.hidden_layers_interaction):
            x_interaction = relu(getattr(self, 'fc_interaction{}'.format(i))(x_interaction))
        y = torch.sigmoid(self.final_layer(x_interaction))
        return y