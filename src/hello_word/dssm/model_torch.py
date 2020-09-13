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
        # self.embeddings.to(self.device)  ###????

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
        if config.loss == 'bce':
            self.learn_gamma = nn.Conv1d(self.latent_out * 2, 1, 1)
        else:
            self.learn_gamma = nn.Conv1d(self.latent_out * 2, 2, 1)

    def forward(self, data):
        # data_loader = tqdm.tqdm(enumerate(data_set),
        #                         total=len(data_set))

        # for i, data in data_loader:
        ## Batch*Len*Dim --> Batch*Dim* Len
        # data = {key: value.to(self.device).transpose(1, 2) for key, value in data.items()}
        # data = {key: value.to(self.device) for key, value in data_set.items()}
        # print(data['query_'].shape)
        q, d = self.embeddings(data['query_']).transpose(1, 2), self.embeddings(data['doc_']).transpose(1,
                                                                                                        2)  ###待匹配的两个句子
        # print(q.shape)
        ### query
        q_c = F.relu(self.query_conv(q))
        q_k = kmax_pooling(q_c, 2, self.kmax)
        q_k = q_k.transpose(1, 2)
        q_s = F.relu(self.query_sem(q_k))
        # q_s = q_s.resize(self.latent_out)
        b_, k_, l_ = q_s.size()
        q_s = q_s.contiguous().view((b_, k_ * l_))

        ###doc
        d_c = F.relu(self.doc_conv(d))
        d_k = kmax_pooling(d_c, 2, self.kmax)
        d_k = d_k.transpose(1, 2)
        d_s = F.relu(self.doc_sem(d_k))
        # d_s = d_s.resize(self.latent_out)
        d_s = d_s.contiguous().view((b_, k_ * l_))

        ###双塔结构向量拼接
        out_ = torch.cat((q_s, d_s), 1)
        out_ = out_.unsqueeze(2)

        with_gamma = F.tanh(self.learn_gamma(out_))  ### --> B * 2 * 1
        with_gamma = with_gamma.contiguous().view(b_, -1)
        return with_gamma


class DSSMTwo(nn.Module):

    def __init__(self, config, device='cpu'):
        super(DSSMTwo, self).__init__()

        self.device = device
        ###此部分的信息有待处理
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.embeddings.to(self.device)  ###????

        self.latent_out = config.latent_out_1
        self.hidden_size = config.hidden_size
        self.kernel_out = config.kernel_out_1
        self.kernel_size = config.kernel_size
        self.max_len = config.max_len
        self.kmax = config.kmax

        # layers for query
        self.query_conv = nn.Conv1d(self.hidden_size, self.kernel_out, self.kernel_size)
        self.query_sem = nn.Linear(self.max_len, self.latent_out)  ## config.latent_out_1  需要输出的语义维度中间
        # layers for docs
        self.doc_conv = nn.Conv1d(self.hidden_size, self.kernel_out, self.kernel_size)
        self.doc_sem = nn.Linear(self.max_len, self.latent_out)
        # learning gamma
        if config.loss == 'bce':
            self.learn_gamma = nn.Conv1d(self.latent_out * 2, 1, 1)
        else:
            self.learn_gamma = nn.Linear(self.latent_out * 2, 2)

    def forward(self, data):

        q, d = self.embeddings(data['query_']).permute(0, 2, 1), self.embeddings(data['doc_']).permute(0, 2,
                                                                                                       1)  ###待匹配的两个句子
        ### query
        q_c = F.relu(self.query_conv(q))
        q_k = kmax_pooling(q_c, 1, self.kmax)  ## B 1 L
        q_s = F.relu(self.query_sem(q_k))
        b_, k_, l_ = q_s.size()
        q_s = q_s.contiguous().view((b_, -1))

        ###doc
        d_c = F.relu(self.doc_conv(d))
        d_k = kmax_pooling(d_c, 1, self.kmax)
        d_s = F.relu(self.doc_sem(d_k))
        d_s = d_s.contiguous().view((b_, -1))

        ###双塔结构向量拼接
        out_ = torch.cat((q_s, d_s), 1)
        # out_ = out_.unsqueeze(2)

        with_gamma = self.learn_gamma(out_)  ### --> B * 2 * 1
        return with_gamma


class DSSMFour(nn.Module):
    """
     卷积层共享？
    """

    def __init__(self, config, device='cpu'):
        super(DSSMFour, self).__init__()

        self.device = device
        # 此部分的信息有待处理
        self.latent_out = config.latent_out_1
        self.hidden_size = config.hidden_size
        self.kernel_out = config.kernel_out_1
        self.kernel_size = config.kernel_size
        self.max_len = config.max_len
        self.kmax = config.kmax

        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)
        # layers for query
        self.query_conv = nn.Conv1d(self.hidden_size, self.kernel_out, self.kernel_size)
        self.query_sem = nn.Linear(self.max_len, self.latent_out)  ## config.latent_out_1  需要输出的语义维度中间
        # learning gamma
        if config.loss == 'bce':
            self.learn_gamma = nn.Conv1d(self.latent_out * 2, 1, 1)
        else:
            self.learn_gamma = nn.Linear(self.latent_out * 2, 2)

    def forward(self, data):

        q, d = self.embeddings(data['query_']).permute(0, 2, 1), self.embeddings(data['doc_']).permute(0, 2,
                                                                                                       1)  # 待匹配的两个句子
        # query
        q_c = F.relu(self.query_conv(q))
        q_k = kmax_pooling(q_c, 1, self.kmax)  ## B 1 L
        q_s = F.relu(self.query_sem(q_k))
        b_, k_, l_ = q_s.size()
        q_s = q_s.contiguous().view((b_, -1))

        ###doc
        d_c = F.relu(self.query_conv(d))
        d_k = kmax_pooling(d_c, 1, self.kmax)
        d_s = F.relu(self.query_sem(d_k))
        d_s = d_s.contiguous().view((b_, -1))

        ###双塔结构向量拼接
        out_ = torch.cat((q_s, d_s), 1)
        # out_ = out_.unsqueeze(2)

        with_gamma = self.learn_gamma(out_)  ### --> B * 2 * 1
        return with_gamma


class DSSMThree(nn.Module):

    def __init__(self, config, device='cpu'):
        super(DSSMThree, self).__init__()

        self.device = device
        ###此部分的信息有待处理

        # self.embeddings.to(self.device)  ###????

        self.latent_out = config.latent_out_1
        self.hidden_size = config.hidden_size
        self.kernel_out = config.kernel_out_1
        self.kernel_size = config.kernel_size
        self.max_len = config.max_len
        self.kmax = config.kmax
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)
        self.query_sem = nn.Linear(self.max_len * self.hidden_size,
                                   self.latent_out)  ## config.latent_out_1  需要输出的语义维度中间
        # layers for docs
        self.doc_sem = nn.Linear(self.max_len * self.hidden_size, self.latent_out)
        # learning gamma
        self.learn_gamma = nn.Linear(self.latent_out * 2, 2)

    def forward(self, data):
        q = self.embeddings(data['query_'])
        d = self.embeddings(data['doc_'])

        b_, l_, d_ = q.shape
        q = q.view(b_, -1)
        d = d.view(b_, -1)

        # print(q.shape)
        ### query
        q_s = F.relu(self.query_sem(q))

        ###doc
        d_s = F.relu(self.doc_sem(d))

        ###双塔结构向量拼接
        out_ = torch.cat((q_s, d_s), 1)

        with_gamma = self.learn_gamma(out_)  ### --> B * 2
        return with_gamma


class DSSMFive(DSSMFour):
    def __init__(self, config, device='cpu', vocab=None):
        # super(DSSMFive, self).__init__()
        DSSMFour.__init__(self, config, device)

        self.vocab = vocab
        self.hidden_size = 7
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)
        self.query_conv = nn.Conv1d(self.hidden_size, self.kernel_out, self.kernel_size)
        tmp_ = self.__toBcode__()
        self.embeddings.weight.data.copy_(tmp_)
        self.embeddings.weight.requires_grad = False

    def __toBcode__(self):
        t_ = []
        for char_ in self.vocab:
            b_ = bin(int(self.vocab[char_]))[2:]
            p_ = [int(k) for k in '0' * (7 - len(b_)) + b_]
            t_.append(p_)
        return torch.tensor(t_)


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
