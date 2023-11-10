import torch
import torch.nn as nn


class HDMLF(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(HDMLF, self).__init__()
        self.bidirectional_gru = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 256)
        self.dense1 = nn.Linear(256, 64)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        # Assuming x is the input tensor of shape (batch_size, sequence_length, input_size)
        # Pass through bidirectional GRU
        gru_output, _ = self.bidirectional_gru(x)

        # Apply attention
        attention_weights = torch.softmax(self.attention(gru_output), dim=1)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)

        # Pass through dense layers
        dense_output = torch.relu(self.dense1(context_vector))
        output = self.linear(dense_output)

        return output