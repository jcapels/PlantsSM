import torch
from torch import nn
import torch.nn.functional as F

from plants_sm.models.pytorch_model import PyTorchModel


class DSPACEModel(nn.Module):

    def __init__(self, num_aa, sequence_dimension, num_classes):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels=num_aa, out_channels=16, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(16)
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(24)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(48)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=7, padding=3)
        self.bn5 = nn.BatchNorm1d(96)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.fc0 = nn.Linear(96 * (sequence_dimension // 8), 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.embedding_bn = nn.BatchNorm1d(256)

        self.classification_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x[0]
        x = F.elu(self.bn0(self.conv0(x)))
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)

        x = F.elu(self.bn4(self.conv4(x)))
        x = F.elu(self.bn5(self.conv5(x)))
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.elu(self.bn6(self.fc0(x)))
        x = F.elu(self.bn7(self.fc1(x)))
        x = F.elu(self.bn8(self.fc2(x)))
        x = F.elu(self.fc3(x))
        x = self.embedding_bn(x)

        x = torch.sigmoid(self.classification_layer(x))

        return x


class DSPACE(PyTorchModel):

    def __init__(self, num_aa, num_columns, num_classes,
                 loss_function, validation_loss_function,
                 batch_size,
                 optimizer=torch.optim.Adam, learning_rate=0.009999999776482582,
                 epochs=30, device="cuda:0", patience=4, **kwargs):
        model = DSPACEModel(num_aa, num_columns, num_classes)
        self.optimizer = optimizer(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)

        super().__init__(model=model,
                         loss_function=loss_function,
                         validation_loss_function=validation_loss_function,
                         optimizer=self.optimizer,
                         device=device,
                         epochs=epochs,
                         patience=patience,
                         batch_size=batch_size,
                         **kwargs
                         )
