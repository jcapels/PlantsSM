from copy import deepcopy
import torch
import torch.nn as nn

from plants_sm.models.pytorch_model import PyTorchModel


class DeepECCNN(nn.Module):
    def __init__(self, num_filters, input_size, num_columns, kernel_sizes, num_dense_layers, dense_layer_size,
                 num_classes):
        super(DeepECCNN, self).__init__()

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_dense_layers = num_dense_layers

        for i in range(len(kernel_sizes)):
            setattr(self, f"conv{i}", nn.Conv2d(1, num_filters, kernel_size=(kernel_sizes[i], num_columns),
                                                stride=(1, 1), padding="valid"))
            setattr(self, f"maxpool{i}", nn.MaxPool2d((input_size - kernel_sizes[i], 1)))
            setattr(self, f"flatten{i}", nn.Flatten())
            setattr(self, f"batchnorm{i}", nn.BatchNorm1d(num_filters))

        self.concat = nn.Linear(num_filters * len(kernel_sizes), dense_layer_size)

        self.batchnorm4 = nn.BatchNorm1d(dense_layer_size)
        self.activation1 = nn.ReLU()

        self.last_sigmoid = False

        for i in range(num_dense_layers):
            setattr(self, f"dense{i}", nn.Linear(dense_layer_size, dense_layer_size))
            setattr(self, f"batchnorm_dense{i}", nn.BatchNorm1d(dense_layer_size))
            setattr(self, f"activation_dense{i}", nn.ReLU())

        self.dense_final = nn.Linear(dense_layer_size, num_classes)

    def forward(self, x):
        x = x[0]
        x = x.unsqueeze(1)
        to_concat = []
        for i in range(len(self.kernel_sizes)):
            x_copy = deepcopy(x)
            conv = getattr(self, f"conv{i}")
            x_copy = conv(x_copy)
            x_copy = getattr(self, f"maxpool{i}")(x_copy)
            x_copy = getattr(self, f"flatten{i}")(x_copy)
            x_copy = getattr(self, f"batchnorm{i}")(x_copy)
            to_concat.append(x_copy)

        x = torch.cat(to_concat, dim=1)
        x = self.concat(x)
        x = self.batchnorm4(x)
        x = self.activation1(x)

        for i in range(self.num_dense_layers):
            x = getattr(self, f"dense{i}")(x)
            x = getattr(self, f"batchnorm_dense{i}")(x)
            x = getattr(self, f"activation_dense{i}")(x)

        x = self.dense_final(x)
        if self.last_sigmoid:
            x = torch.sigmoid(x)

        return x


class DeepECCNNOptimal(DeepECCNN):

    def __init__(self, num_columns, input_size, num_classes):
        super().__init__(128, input_size, num_columns, [4, 8, 16], 2, 512, num_classes)


class DeepEC(PyTorchModel):

    def __init__(self, num_columns, input_size, num_classes,
                 loss_function, validation_loss_function,
                 batch_size,
                 optimizer=torch.optim.Adam, learning_rate=0.009999999776482582,
                 epochs=30, device="cuda:0", patience=4, **kwargs):
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.input_size = input_size

        model = DeepECCNN(128, input_size, num_columns, [4, 8, 16], 2, 512, num_classes)
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
