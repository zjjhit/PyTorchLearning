# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         train_new
# Description:  
# Author:       lenovo
# Date:         2020/6/10
# -------------------------------------------------------------------------------

import sys, os

from rnn_class.data import *
from rnn_class.model import *
import random
import time
import math

n_hidden = 128
n_epochs = 10000
print_every = 5000
plot_every = 1000
learning_rate = 0.0005  # If you set this too high, it might explode. If too low, it might not learn


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingPair(category_lines, all_categories):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)  # Tensor out of Variable with .data
    category_i = top_i.item()  # 1D
    return all_categories[category_i], category_i


from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class NameDataSet(Dataset):
    def __init__(self, category_lines, all_categories):
        super(NameDataSet, self).__init__()
        self.tmp_ = []
        for k in category_lines:
            for one in category_lines[k]:
                # self.tmp_.append([one, k, Variable(lineToTensor(one)), Variable(torch.LongTensor([all_categories.index(k)]))])
                self.tmp_.append([one, k])

    def __getitem__(self, index):
        return self.tmp_[index]

    def __len__(self):
        return len(self.tmp_)


from tqdm import tqdm, trange


def train(category_tensor, line_tensor, model, opt, crite, device):
    hidden = model.initHidden().to(device)
    opt.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = crite(output, category_tensor)
    loss.backward()
    opt.step()

    return output, loss


def main():
    start = time.time()
    # Keep track of losses for plotting

    device = torch.device("cuda")

    category_lines, all_categories = getData()
    n_categories = len(all_categories)
    data_set = NameDataSet(category_lines, all_categories)

    rnn = RNN(n_letters, n_hidden, n_categories)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    rnn.to(device)

    train_dataloader = DataLoader(data_set, batch_size=32)

    rnn.train()
    for epoch in range(1, n_epochs + 1):
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            print(batch[0])
            # batch = tuple(t.to(device) for t in batch)
            # category, line, category_tensor, line_tensor = batch
            #
            # output, loss = train(category_tensor, line_tensor, rnn, optimizer, criterion, device)
            #
            # # Print epoch number, loss, name and guess
            # if epoch % print_every == 0:
            #     guess, guess_i = categoryFromOutput(output, all_categories)
            #     correct = '✓' if guess == category else '✗ (%s)' % category
            #     print('%d %d%% (%s) %.4f %s / %s %s' % (
            #         epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # torch.save(rnn, './char-rnn-classification.pt')


def copyPara(src_rnn, dst_rnn):
    from torch.nn.parameter import Parameter

    dst_rnn.i2o.weight = Parameter(src_rnn.i2o.weight.clone().detach().requires_grad_(True))
    dst_rnn.i2o.bias = Parameter(src_rnn.i2o.bias.clone().detach().requires_grad_(True))

    dst_rnn.i2h.weight = Parameter(src_rnn.i2h.weight.clone().detach().requires_grad_(True))
    dst_rnn.i2h.bias = Parameter(src_rnn.i2h.bias.clone().detach().requires_grad_(True))

    return dst_rnn


if __name__ == '__main__':
    # main()

    # tmp()

    t = RNN(3, 3, 2)  # input_size, hidden_size, output_size

    criterion = nn.NLLLoss()
    hidden = t.initHidden()

    print('Begin')
    for n, p in t.named_parameters():
        print(n, p)

    category_tensor = torch.tensor([1])

    line_tensor_nb = torch.rand(2, 5, 1, 3)
    line_tensor = line_tensor_nb[0]

    for i in range(line_tensor.size()[0]):
        output, hidden = t(line_tensor[i], hidden)
    print(output)

    loss = criterion(output, category_tensor)
    print('B')
    print(loss)
    loss.backward(retain_graph=True)

    print('Loss')
    for n, p in t.named_parameters():
        print(n, p.grad)

    #########################
    t.zero_grad()
    hidden = t.initHidden()
    line_tensor = line_tensor_nb[1]

    for i in range(line_tensor.size()[0]):
        output, hidden = t(line_tensor[i], hidden)
    print(output)

    loss = criterion(output, category_tensor)
    print('B')
    print(loss)
    loss.backward()

    print('Loss')
    for n, p in t.named_parameters():
        print(n, p.grad)

    ###########################
    t_nb = RNNBatch(3, 3, 2, 5, 2)  # input_size, hidden_size, output_size, max_len, batch_size
    t_nb = copyPara(t, t_nb)

    print('Begin')
    for n, p in t_nb.named_parameters():
        print(n, p)

    # line_tensor = line_tensor.squeeze().unsqueeze(0)
    # line_tensor = torch.repeat_interleave(line_tensor, 2, dim=0)

    line_len = torch.tensor([[4], [4]])
    # line_len = torch.tensor([[4]])
    p = t_nb(line_tensor_nb.squeeze(), line_len)
    print(p)

    category_tensor = torch.tensor([1, 1])
    loss = criterion(p, category_tensor)
    print('BN')
    print(loss)
    loss.backward()

    print('AAA')
    for n, p in t_nb.named_parameters():
        print(n, p.grad)
