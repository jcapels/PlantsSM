import torch
import torch.nn as nn

from PlantsSM.src.plants_sm.models.pytorch_model import PyTorchModel



class DeepECCNN(nn.Module):
    def __init__(self, num_filters, num_columns, kernel_sizes, num_dense_layers, dense_layer_size, num_classes):
        super(DeepECCNN, self).__init__()

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_dense_layers = num_dense_layers

        for i in range(len(kernel_sizes)):
            setattr(self, f"conv{i}", nn.Conv2d(1, num_filters, kernel_size=(kernel_sizes[i], num_columns)))
            setattr(self, f"maxpool{i}", nn.MaxPool2d((kernel_sizes[i], 1)))
            setattr(self, f"flatten{i}", nn.Flatten())
            setattr(self, f"batchnorm{i}", nn.BatchNorm1d(num_filters))

        self.concat = nn.Linear(num_filters * len(kernel_sizes), dense_layer_size)

        self.batchnorm4 = nn.BatchNorm1d(dense_layer_size)
        self.activation1 = nn.ReLU()

        for i in range(num_dense_layers):
            setattr(self, f"dense{i}", nn.Linear(dense_layer_size, dense_layer_size))
            setattr(self, f"batchnorm{i}", nn.BatchNorm1d(dense_layer_size))
            setattr(self, f"activation{i}", nn.ReLU())

        self.dense_final = nn.Linear(dense_layer_size, num_classes)

    def forward(self, x):
        x = x[0]
        x = x.unsqueeze(1)
        for i in range(self.num_filters):
            conv = getattr(self, f"conv{i}")
            x = conv(x)
            x = getattr(self, f"maxpool{i}")(x)
            x = getattr(self, f"flatten{i}")(x)
            x = getattr(self, f"batchnorm{i}")(x)
            x = getattr(self, f"activation{i}")(x)

        x = self.concat(x)
        x = self.batchnorm4(x)
        x = self.activation1(x)

        for i in range(self.num_dense_layers):
            x = getattr(self, f"dense{i}")(x)
            x = getattr(self, f"batchnorm{i}")(x)
            x = getattr(self, f"activation{i}")(x)

        return x


class DeepECCNNOptimal(DeepECCNN):

    def __init__(self, num_columns, num_classes):
        super().__init__(128, num_columns, [4, 8, 16], 2, 512, num_classes)

class DeepEC(PyTorchModel):

    def __init__(self, num_columns, num_classes, 
                 loss_function, validation_loss_function,
                 batch_size,
                 optimizer=torch.optim.Adam, learning_rate=0.009999999776482582, 
                 epochs=30):
        
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.validation_loss_function = validation_loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        # self.device = device
        


