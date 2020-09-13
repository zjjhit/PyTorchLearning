import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), -1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class RNNBatch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_len, batch_size):
        super(RNNBatch, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_len = max_len
        self.batch_size = batch_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs, inputs_len):
        '''

        :param inputs:  B*N*D
        :param inputs_len: B*1
        :param hddien: B*1*D

        :return:
        '''

        hidden = Variable(torch.zeros(self.batch_size, 1, self.hidden_size))

        assert self.batch_size == inputs.size()[0]

        tmp_out = Variable(torch.zeros(self.batch_size, self.max_len, self.output_size))
        # tmp_hidden = Variable(torch.zeros(inputs.size()[0], self.max_len, self.hidden_size))

        for i in range(self.max_len):
            print(inputs.size(), hidden.squeeze(1).size())
            combined = torch.cat((inputs[:, i, :], hidden.squeeze(1)), -1)
            assert combined.dim() == 2, print(combined.size(), inputs.size())

            hidden = self.i2h(combined)
            output = self.i2o(combined)
            output = self.softmax(output)
            # print('Test')
            # print(output)

            tmp_out[:, i, :] = output

        return tmp_out[range(self.batch_size), inputs_len.squeeze()]
