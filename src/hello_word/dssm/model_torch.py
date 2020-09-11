# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         model_torch
# Description:  
# Author:       lenovo
# Date:         2020/9/8
# -------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# from transformers import BertTokenizer, BertModel, BertConfig


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class DSSMOne(nn.Module):

    def __init__(self, config, device='cpu'):
        super(DSSMOne, self).__init__()

        self.device = device
        ###此部分的信息有待处理
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.to(self.device)  ###????

        self.latent_out = config.latent_out_1
        self.hidden_size = config.hidden_size
        self.kernel_out = config.kernel_out_1
        self.kernel_size = config.kernel_size
        self.kmax = config.kmax

        # layers for query
        self.query_conv = nn.Conv1d(self.hidden_size, self.kernel_out, self.kernel_size)
        self.query_sem = nn.Linear(self.kernel_out, self.latent_out)  ## config.latent_out_1  需要输出的语义维度中间
        # layers for docs
        self.doc_conv = nn.Conv1d(self.hidden_size, self.kernel_out, self.kernel_size)
        self.doc_sem = nn.Linear(self.kernel_out, self.latent_out)
        # learning gamma
        self.learn_gamma = nn.Conv1d(self.latent_out * 2, 1, 1)

    def forward(self, data):
        # data_loader = tqdm.tqdm(enumerate(data_set),
        #                         total=len(data_set))

        # for i, data in data_loader:
        ## Batch*Len*Dim --> Batch*Dim* Len
        # data = {key: value.to(self.device).transpose(1, 2) for key, value in data.items()}
        # data = {key: value.to(self.device) for key, value in data_set.items()}
        q, d = self.embeddings(data['query_']).transpose(1, 2), self.embeddings(data['doc_']).transpose(1, 2)  ###待匹配的两个句子

        ### query
        q_c = F.tanh(self.query_conv(q))
        q_k = kmax_pooling(q_c, 2, self.kmax)
        q_k = q_k.transpose(1, 2)
        q_s = F.tanh(self.query_sem(q_k))
        # q_s = q_s.resize(self.latent_out)
        b_, k_, l_ = q_s.size()
        q_s = q_s.contiguous().view((b_, k_ * l_))

        ###doc
        d_c = F.tanh(self.doc_conv(d))
        d_k = kmax_pooling(d_c, 2, self.kmax)
        d_k = d_k.transpose(1, 2)
        d_s = F.tanh(self.doc_sem(d_k))
        # d_s = d_s.resize(self.latent_out)
        d_s = d_s.contiguous().view((b_, k_ * l_))

        ###双塔结构向量拼接
        out_ = torch.cat((q_s, d_s), 1)
        out_ = out_.unsqueeze(2)

        with_gamma = self.learn_gamma(out_)  ### --> B * 2 * 1
        with_gamma = with_gamma.contiguous().view(b_, -1)
        return with_gamma

    # def train(self, data):
    #
    #
    # def evaluate(self, data):
    #     pass


def test():
    # Build a random data set.
    import numpy as np
    from torch.autograd import Variable
    from transformers import BertConfig

    config = BertConfig.from_pretrained('./config.json')
    sample_size = 10
    l_Qs = []
    pos_l_Ds = []

    for i in range(sample_size):
        query_len = np.random.randint(1, 10)
        l_Q = np.random.rand(5, query_len, config.hidden_size)
        l_Qs.append(l_Q)

        doc_len = np.random.randint(50, 500)
        l_D = np.random.rand(5, doc_len, config.hidden_size)
        pos_l_Ds.append(l_D)

    # Till now, we have made a complete numpy dataset
    # Now let's convert the numpy variables to torch Variable

    for i in range(len(l_Qs)):
        l_Qs[i] = Variable(torch.from_numpy(l_Qs[i]).float())
        pos_l_Ds[i] = Variable(torch.from_numpy(pos_l_Ds[i]).float())

    model = DSSMOne(config)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # output variable, remember the cosine similarity with positive doc was at 0th index
    y = torch.randn(10, 5)
    y = (y > 0).int().float()

    for i in range(sample_size):
        y_pred = model({'query_': l_Qs[i], 'doc_': pos_l_Ds[i]})

        b_, _ = y_pred.shape
        print(i, y_pred.shape, y_pred.view(b_, -1).shape)
        loss = criterion(y_pred.view(b_, -1), y[i].view(b_, -1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
