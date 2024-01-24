import torch

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from torch import nn


class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x[0]
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu2(out)
        out = nn.Sigmoid()(out)
        return out


class TestPytorchBaselineModel(nn.Module):

    def __init__(self, input_dim, output_dim, intermediate_dim=128):
        super(TestPytorchBaselineModel, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.final_layer = nn.Linear(output_dim, 1)

    def forward(self, x):
        x = x[0]
        x = self.fc1(x)
        x = self.fc2(x)
        y = torch.sigmoid(self.final_layer(x))
        return y


class ToyTensorflowModel:

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_dim, dtype='int32')

        # Fully connected
        FC1 = Dense(1024, activation='relu')(inputs)
        FC2 = Dense(512, activation='relu')(FC1)

        # And add a logistic regression on top
        predictions = Dense(1, activation='sigmoid')(FC2)

        model = Model(inputs=inputs, outputs=[predictions])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
