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
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.word_embeddings.to(self.device)  ###????

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
            ## Batch*Len*Dim --> Batch*Dim* Len
            data = {key: value.to(self.device).transpose(1, 2) for key, value in data.items()}
            q, d = data['query_'], data['doc_']  ###待匹配的两个句子

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

            with_gamma = self.learn_gamma(out_)
            return with_gamma


# Build a random data set.
import numpy as np

sample_size = 10
l_Qs = []
pos_l_Ds = []

(query_len, doc_len) = (5, 100)

for i in range(sample_size):
    query_len = np.random.randint(1, 10)
    l_Q = np.random.rand(1, query_len, WORD_DEPTH)
    l_Qs.append(l_Q)

    doc_len = np.random.randint(50, 500)
    l_D = np.random.rand(1, doc_len, WORD_DEPTH)
    pos_l_Ds.append(l_D)

neg_l_Ds = [[] for j in range(J)]
for i in range(sample_size):
    possibilities = list(range(sample_size))
    possibilities.remove(i)
    negatives = np.random.choice(possibilities, J, replace=False)
    for j in range(J):
        negative = negatives[j]
        neg_l_Ds[j].append(pos_l_Ds[negative])

# Till now, we have made a complete numpy dataset
# Now let's convert the numpy variables to torch Variable

for i in range(len(l_Qs)):
    l_Qs[i] = Variable(torch.from_numpy(l_Qs[i]).float())
    pos_l_Ds[i] = Variable(torch.from_numpy(pos_l_Ds[i]).float())
    for j in range(J):
        neg_l_Ds[j][i] = Variable(torch.from_numpy(neg_l_Ds[j][i]).float())

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# output variable, remember the cosine similarity with positive doc was at 0th index
y = np.ndarray(1)
# CrossEntropyLoss expects only the index as a long tensor
y[0] = 0
y = Variable(torch.from_numpy(y).long())

for i in range(sample_size):
    y_pred = model(l_Qs[i], pos_l_Ds[i], [neg_l_Ds[j][i] for j in range(J)])
    loss = criterion(y_pred.resize(1, J + 1), y)
    print(i, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    pass
