import torch
from torch import nn


class EC_ESM1b(nn.Module, hidden_layers, num_classes):
    def __init__(self, in_dim, model):
        super(EC_ESM1b, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, data, subs, rep_layer, gpu, use_cuda):
        output = self.model(data, repr_layers=[rep_layer])
        output = output["representations"][rep_layer]
        if use_cuda:
            output = output.cuda(gpu, non_blocking=True)
        output = output[:, 0, :]
        x = torch.cat((output, subs), dim=1)
        if use_cuda:
            x = x.cuda(gpu, non_blocking=True)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x