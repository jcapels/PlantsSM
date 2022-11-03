import torch
from torch import nn, relu


class DeepDTA(nn.Module):

    def __init__(self, protein_shape, compounds_shape, protein_char_set_n, compound_char_set_n, filters):
        super().__init__()
        self.proteins_embedding = nn.Embedding(num_embeddings=protein_char_set_n + 1, embedding_dim=128, padding_idx=0)
        self.compounds_embedding = nn.Embedding(num_embeddings=compound_char_set_n + 1, embedding_dim=128,
                                                padding_idx=0)
        self.conv1_proteins_1 = nn.Conv1d(protein_shape[1], filters, 4, stride=1,
                                          padding='valid')
        self.conv1_proteins_2 = nn.Conv1d(filters, filters * 2, 6, stride=1, padding='valid')
        self.conv1_proteins_3 = nn.Conv1d(filters * 2, filters * 3, 8, stride=1, padding='valid')
        self.maxpool1_proteins = nn.MaxPool1d(2)

        self.conv1_compounds_1 = nn.Conv1d(compounds_shape[1], filters, 4, stride=1, padding='valid')
        self.conv1_compounds_2 = nn.Conv1d(filters, filters * 2, 6, stride=1, padding='valid')
        self.conv1_compounds_3 = nn.Conv1d(filters * 2, filters * 3, 8,
                                           stride=1, padding='valid')
        self.maxpool1_compounds = nn.MaxPool1d(2)

        self.dense1_interaction = nn.Linear(filters * 3 + filters * 3, 1024)
        self.dense2_interaction = nn.Linear(1024, 1024)
        self.dense3_interaction = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.1)
        self.final_layer = nn.Linear(512, 1)

    def forward(self, x):
        x_proteins = x[0]
        x_proteins = x_proteins.to(torch.int32)
        x_proteins = self.proteins_embedding(x_proteins)
        y = relu(self.conv1_proteins_1(x_proteins))
        y = relu(self.conv1_proteins_2(y))
        y_proteins = relu(self.conv1_proteins_3(y))
        y_proteins = nn.MaxPool1d(y_proteins.shape[2] - 1)(y_proteins)

        x_compounds = x[1]
        x_compounds = x_compounds.to(torch.int32)
        x_compounds = self.compounds_embedding(x_compounds)
        y = relu(self.conv1_compounds_1(x_compounds))
        y = relu(self.conv1_compounds_2(y))
        y_compounds = relu(self.conv1_compounds_3(y))
        y_compounds = nn.MaxPool1d(y_compounds.shape[2] - 1)(y_compounds)

        y = torch.cat([y_proteins, y_compounds], dim=1)
        y = y.reshape(y.shape[0], y.shape[1])
        y = relu(self.dense1_interaction(y))
        y = self.dropout(y)
        y = relu(self.dense2_interaction(y))
        y = self.dropout(y)
        y = relu(self.dense3_interaction(y))
        y = self.dropout(y)
        y = torch.sigmoid(self.final_layer(y))
        return y