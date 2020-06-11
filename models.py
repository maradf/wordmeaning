import torch
import torch.nn as nn
from torch.autograd import Variable

class myLSTM(nn.Module):
    def __init__(self, in_features, out_features, batch_size, num_layers):
        super(myLSTM, self).__init__()
        self.lstm = nn.LSTM(in_features, out_features, num_layers, batch_first=True)
        self.batch_size = batch_size

    
    def forward(self, inputs, returnHidden=False):
        out, hidden = self.lstm(inputs.view(inputs.size(0), -1, 1))
        if returnHidden:
            return out, hidden
        else:
            return out




class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)