import torch
import torch.nn as nn


class HDMLF(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(HDMLF, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, 1280)

        # Bidirectional GRU
        self.bi_gru = nn.GRU(1280, hidden_size, bidirectional=True, batch_first=True)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=1)

        # Dense layer 1
        self.dense1 = nn.Linear(hidden_size * 2, 64)

        # Linear layer (output layer)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)

        # Bidirectional GRU
        x, _ = self.bi_gru(x)

        # Attention layer
        x, _ = self.attention(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Dense layer 1
        x = self.dense1(x)

        # Linear layer (output layer)
        x = self.linear(x)

        return x