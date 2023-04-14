from torch import nn

class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, last_sigmoid=True):
        super(DenseNet, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.fc_initial = nn.Linear(input_size, hidden_sizes[0])
        self.relu_initial = nn.ReLU()
        self.last_sigmoid = last_sigmoid

        for i in range(1, len(hidden_sizes)):
            setattr(self, f"fc{i}", nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            setattr(self, f"relu{i}", nn.ReLU())
        self.fc_final = nn.Linear(hidden_sizes[-1], output_size)
        self.final_relu = nn.ReLU() 

    def forward(self, x):
        x = x[0]
        out = self.fc_initial(x)
        out = self.relu_initial(out)
        for i in range(1, len(self.hidden_sizes)):
            out = getattr(self, f"fc{i}")(out)
            out = getattr(self, f"relu{i}")(out)
            
        out = self.fc_final(out)
        if self.last_sigmoid:
            out = nn.Sigmoid()(out)
        return out