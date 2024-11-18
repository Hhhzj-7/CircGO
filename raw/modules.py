import torch
from torch import nn


class FunctionPredictor(torch.nn.Module):
    def __init__(self, input_size, output_size, drop_rate=0.1):
        super(FunctionPredictor, self).__init__()

        hidden_size = 4 * output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
