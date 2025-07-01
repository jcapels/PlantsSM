from torch import nn
import torch

class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, batch_norm=False, last_sigmoid=True, dropout=None, 
                 layers_to_freeze=0):
        super(DNN, self).__init__()

        self.batch_norm = batch_norm

        self.hidden_sizes = hidden_sizes
        self.layers_to_freeze = layers_to_freeze
        
        if len(hidden_sizes) != 0:
            self.fc_initial = nn.Linear(input_size, hidden_sizes[0])
            self.batch_norm_initial = nn.BatchNorm1d(hidden_sizes[0])
            self.relu_initial = nn.ReLU()
            self.last_sigmoid = last_sigmoid
            if dropout is not None:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = None
            
            for i in range(1, len(hidden_sizes)):
                setattr(self, f"fc{i}", nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                setattr(self, f"relu{i}", nn.ReLU())
                setattr(self, f"batch_norm_layer{i}", nn.BatchNorm1d(hidden_sizes[i]))
        

            self.fc_final = nn.Linear(hidden_sizes[-1], output_size)
        else:
            self.fc_initial = nn.Linear(input_size, output_size)
            self.batch_norm_initial = nn.BatchNorm1d(output_size)
            self.relu_initial = nn.ReLU()
            self.last_sigmoid = last_sigmoid
            if dropout is not None:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = None

            self.fc_final = nn.Linear(input_size, output_size)

        self.final_relu = nn.ReLU()
        self.final_batch_norm = nn.BatchNorm1d(output_size)

    def _forward_initial_layers(self, x):

        if type(x) == list:
            out = self.fc_initial(x)
        if self.batch_norm and x.shape[0] > 1:
            out = self.batch_norm_initial(out)
        out = self.relu_initial(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    
    def _forward_hidden_layer(self, out, i):
        out = getattr(self, f"fc{i}")(out)
        if self.batch_norm and out.shape[0] > 1:
            out = getattr(self, f"batch_norm_layer{i}")(out)
        out = getattr(self, f"relu{i}")(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def forward(self, x, return_embedding=False):
        x = x[0]

        if len(self.hidden_sizes) != 0:
            if self.layers_to_freeze > 0:
                with torch.no_grad():
                    out = self._forward_initial_layers(x)
            else:
                out = self._forward_initial_layers(x)

            for i in range(1, len(self.hidden_sizes)):
                if i < self.layers_to_freeze-1:
                    with torch.no_grad():
                        out = self._forward_hidden_layer(out, i)
                else:
                    out = self._forward_hidden_layer(out, i)

            if return_embedding:
                embedding = torch.clone(out)
            out = self.fc_final(out)
        else:
            if return_embedding:
                embedding = torch.clone(x)
            out = self.fc_final(x)

        if return_embedding:
            if self.last_sigmoid:
                out = nn.Sigmoid()(out)
            return out, embedding
        
        if self.last_sigmoid:
            out = nn.Sigmoid()(out)
        return out
