from torch import nn

import torch


class Conv1D(nn.Module):

    def __init__(self, input_size, size, output_size, num_layers, conv1_padding=0,
                 conv1_dilation=1, conv1_kernel_size=12, conv1_stride=1, max_pool_padding=0,
                 max_pool_dilation=1, max_pool_kernel_size=5, max_pool_stride=12):
        super(Conv1D, self).__init__()
        self.num_layers = num_layers
        original_hidden_size = size

        # ------------------- Calculation of max pool output size -------------------

        # l_out = ((sequence_length + 2 * conv1_padding - conv1_dilation * (
        #         conv1_kernel_size - 1) - 1) / conv1_stride + 1)
        # max_pool_output = int(
        #     (l_out + 2 * max_pool_padding - max_pool_dilation * (max_pool_kernel_size - 1) - 1) / max_pool_stride + 1)

        # max_pool_output *= hidden_size
        # ---------------------------------------------------------------------------

        self.conv1 = nn.Conv1d(input_size, size, kernel_size=conv1_kernel_size,
                               stride=conv1_stride, padding=conv1_padding, dilation=conv1_dilation)
        self.max_pool = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=max_pool_stride,
                                     padding=max_pool_padding, dilation=max_pool_dilation)

        self.fc_last = torch.nn.Linear(original_hidden_size * (2 ** num_layers), output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for i in range(self.num_layers):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'relu{}'.format(i))(x)
            x = getattr(self, 'dropout{}'.format(i))(x)
        x = self.fc_last(x)

        return x
