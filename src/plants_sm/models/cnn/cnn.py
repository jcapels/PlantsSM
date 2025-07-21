from torch import nn


class CNN1D(nn.Module):
    def __init__(self, fc_hidden_sizes, cnn_layers, kernel_size, num_classes, last_sigmoid=True):
        super(CNN1D, self).__init__()

        self.fc_hidden_sizes = fc_hidden_sizes
        self.cnn_layers = cnn_layers

        self.last_sigmoid = last_sigmoid

        # Convolutional layer
        for i in range(0, len(cnn_layers)):
            if i == 0:
                setattr(self, f"conv{i}",
                        nn.Conv1d(in_channels=1, out_channels=cnn_layers[i], kernel_size=kernel_size[i]))
            else:
                setattr(self, f"conv{i}", nn.Conv1d(in_channels=cnn_layers[i - 1], out_channels=cnn_layers[i],
                                                    kernel_size=kernel_size[i]))
            setattr(self, f"pool{i}", nn.MaxPool1d(kernel_size=kernel_size[i]))

            setattr(self, f"relu_conv{i}", nn.ReLU())

        self.flatten = nn.Flatten()

        # Fully-connected input layer
        self.fc_initial = nn.LazyLinear(fc_hidden_sizes[0])
        self.relu_initial = nn.ReLU()

        # Fully-connected output layer
        for i in range(1, len(fc_hidden_sizes)):
            setattr(self, f"fc{i}", nn.Linear(fc_hidden_sizes[i - 1], fc_hidden_sizes[i]))
            setattr(self, f"relu{i}", nn.ReLU())

        self.fc_final = nn.Linear(fc_hidden_sizes[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply convolutional layer
        x = x[0]
        x = x.unsqueeze(1)
        for i in range(0, len(self.cnn_layers)):
            conv = getattr(self, f"conv{i}")
            x = conv(x)
            x = getattr(self, f"relu_conv{i}")(x)
            x = getattr(self, f"pool{i}")(x)

        x = self.flatten(x)

        x = self.fc_initial(x)
        x = self.relu_initial(x)
        # Apply fully-connected layer
        for i in range(1, len(self.fc_hidden_sizes)):
            x = getattr(self, f"fc{i}")(x)
            x = getattr(self, f"relu{i}")(x)

        x = self.fc_final(x)

        if self.last_sigmoid:
            x = self.sigmoid(x)

        return x
