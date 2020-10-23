# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         model_torch
# Description:  
# Author:       lenovo
# Date:         2020/9/8
# -------------------------------------------------------------------------------

import torch
import torch.nn as nn


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
        self.soft = nn.Softmax(dim=1)

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

        with_gamma = self.soft(self.learn_gamma(out_))  ### --> B * 2)
        return with_gamma


class DSSMFour(nn.Module):
    """
     卷积层共享？
    """

    def __init__(self, config, device='cpu'):
        super(DSSMFour, self).__init__()

        # self.device = device
        # # 此部分的信息有待处理
        # self.latent_out = config.latent_out_1
        # self.hidden_size = config.hidden_size
        # self.kernel_out = config.kernel_out_1
        # self.kernel_size = config.kernel_size
        # self.max_len = config.max_len
        # self.kmax = config.kmax
        #
        # self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)
        # # layers for query
        # self.query_conv = nn.Conv1d(self.hidden_size, self.kernel_out, self.kernel_size)  # 16* 64 * 1
        # self.pool_1 = nn.MaxPool1d(2)
        # self.query_conv_2 = nn.Conv1d(16, 32, self.kernel_size)
        # self.query_sem = nn.Linear(self.max_len, self.latent_out)  ## config.latent_out_1  需要输出的语义维度中间
        #
        # # learning gamma
        # if config.loss == 'bce':
        #     self.learn_gamma = nn.Conv1d(self.latent_out * 2, 1, 1)
        # else:
        #     self.learn_gamma = nn.Linear(2 * self.latent_out, 2)
        # # self.soft = nn.Softmax(dim=1)
        #
        # self.norm = nn.BatchNorm1d(2 * self.latent_out)

        # 此部分的信息有待处理
        self.latent_out = config.latent_out_1
        self.hidden_size = config.hidden_size
        self.kernel_out = config.kernel_out_1
        self.kernel_size = config.kernel_size
        self.max_len = config.max_len

        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)

        self.convs = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                             out_channels=self.kernel_out,
                                             kernel_size=self.kernel_size),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm1d(self.kernel_out),
                                   nn.Conv1d(in_channels=self.kernel_out,
                                             out_channels=16,
                                             kernel_size=1),
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2),
                                   nn.BatchNorm1d(16))
        #                              nn.BatchNorm1d(num_features=config.feature_size),
        # nn.ReLU(),
        # nn.MaxPool1d(kernel_size=self.max_len + 1))]))

        self.learn_gamma = nn.Conv1d(32, 16, 32)
        self.lrelu = nn.LeakyReLU()
        self.norm = nn.BatchNorm1d(16)
        self.fc = nn.Linear(in_features=16, out_features=2)

    def forward(self, data):
        q = self.embeddings(data['query_']).permute(0, 2, 1)
        d = self.embeddings(data['doc_']).permute(0, 2, 1)  # 待匹配的两个句子

        out_q = self.convs(q)
        out_d = self.convs(d)

        out_ = torch.cat((out_q, out_d), dim=1)  # B 32 32
        # print(out_.shape)
        out_ = self.learn_gamma(out_)  # B 16 1
        out_ = self.lrelu(out_)

        b_, _, _ = out_.shape

        out_ = out_.view(b_, -1)  # B 16
        out_ = self.norm(out_)
        return self.fc(out_)

        # # query
        # q_c = F.relu(self.query_conv(q))
        # q_k = kmax_pooling(q_c, 1, self.kmax)  ## B 1 L
        # q_s = F.relu(self.query_sem(q_k))
        # b_, k_, l_ = q_s.size()
        # q_s = q_s.contiguous().view((b_, -1))
        #
        # ###doc
        # d_c = F.relu(self.query_conv(d))
        # d_k = kmax_pooling(d_c, 1, self.kmax)
        # d_s = F.relu(self.query_sem(d_k))
        # d_s = d_s.contiguous().view((b_, -1))
        #
        # ###双塔结构向量拼接
        # out_ = torch.cat((q_s, d_s), 1)
        # out_ = self.norm(out_)
        #
        # with_gamma = F.relu(self.learn_gamma(out_))  ### --> B * 2 * 1
        # return with_gamma


class DSSMFive(nn.Module):
    """
    不同层 不同学习率,不做级联
    """

    def __init__(self, config):
        super(DSSMFive, self).__init__()

        # 此部分的信息有待处理
        self.hidden_size = config.hidden_size  #
        self.kernel_out = config.kernel_out_1  # 32
        self.max_len = config.max_len  # 97

        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size, padding_idx=0)  # + pad unk

        self.convs = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                             out_channels=self.kernel_out,
                                             kernel_size=3,  # 2
                                             stride=3),
                                   nn.LeakyReLU(),  # B 32 32
                                   nn.BatchNorm1d(self.kernel_out),  # B*L*D->B*D*L->B*Kout*((L-size)/stride+1)
                                   # nn.MaxPool1d(2),  # -> B*Kout* || / 2
                                   nn.Conv1d(in_channels=self.kernel_out,
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=2),
                                   nn.LeakyReLU(),  ## B 32 16
                                   nn.MaxPool1d(2),  ## B 32 8
                                   nn.BatchNorm1d(32),
                                   nn.Conv1d(in_channels=32,
                                             out_channels=8,
                                             kernel_size=2,
                                             stride=2),
                                   nn.LeakyReLU()  ## B * 8 * 4
                                   )

        self.convs_attention = nn.Sequential(nn.Conv1d(in_channels=config.max_len,
                                                       out_channels=32,
                                                       kernel_size=3,
                                                       stride=3),  # B 32 (max_len - ksize ) / stride + 1 : B 32 32
                                             nn.LeakyReLU(),
                                             nn.Conv1d(in_channels=32,
                                                       out_channels=8,
                                                       kernel_size=4,
                                                       stride=4),  # B 8 8
                                             nn.LeakyReLU(),
                                             nn.BatchNorm1d(8)
                                             )
        # (B 8 4) * 2 + (B 8 8)
        self.fc_1 = nn.Linear(in_features=128, out_features=16)
        self.relu_1 = nn.LeakyReLU()
        self.fc_2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, data):
        q = self.embeddings(data['query_'])
        d = self.embeddings(data['doc_'])
        att_ = torch.matmul(q, d.transpose(-1, -2))

        q = q.permute(0, 2, 1)
        d = d.permute(0, 2, 1)  # 待匹配的两个句子

        out_q = self.convs(q)
        out_d = self.convs(d)
        out_att = self.convs_attention(att_)
        b_, _, _ = out_q.shape
        # print(out_q.shape, out_att.shape)

        out_ = torch.cat((out_q.view(b_, -1), out_d.view(b_, -1), out_att.view(b_, -1)), dim=1)  # B 16 24
        # print(out_.shape)
        out_ = self.relu_1(self.fc_1(out_.view(b_, -1)))
        out_ = self.fc_2(out_)

        return out_


class DSSMSix(nn.Module):

    def __init__(self, config, device='cpu', vocab=None):
        # super(DSSMThree, self).__init__()
        super(DSSMSix, self).__init__()

        self.device = device
        ###此部分的信息有待处理

        # self.embeddings.to(self.device)  ###????
        self.vocab = vocab
        self.latent_out = config.latent_out_1
        self.hidden_size = config.hidden_size
        self.kernel_out = config.kernel_out_1
        self.kernel_size = config.kernel_size
        self.max_len = config.max_len
        self.kmax = config.kmax

        self.hidden_size = 7
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)

        # layers for docs
        self.doc_sem = nn.Linear(self.max_len * self.hidden_size, self.latent_out)
        # learning gamma
        self.learn_gamma = nn.Linear(self.latent_out * 2, 2)
        self.soft = nn.Softmax(dim=1)

        tmp_ = self.__toBcode__()
        self.embeddings.weight.data.copy_(tmp_)
        self.embeddings.weight.requires_grad = False

    def __toBcode__(self):
        t_ = []
        for char_ in self.vocab:
            b_ = bin(int(self.vocab[char_]))[2:]
            p_ = [int(k) for k in '0' * (7 - len(b_)) + b_]
            t_.append(p_)
        t_ = [t_[-1]] + t_[:-1]
        return torch.tensor(t_)

    def forward(self, data):
        q = self.embeddings(data['query_'])
        d = self.embeddings(data['doc_'])

        b_, l_, d_ = q.shape
        q = q.view(b_, -1)
        d = d.view(b_, -1)

        ### query
        q_s = F.relu(self.doc_sem(q))

        ###doc
        d_s = F.relu(self.doc_sem(d))

        ###双塔结构向量拼接
        out_ = torch.cat((q_s, d_s), 1)

        with_gamma = self.soft(self.learn_gamma(out_))  ### --> B * 2)
        return with_gamma


import torch.nn.functional as F


class DSSMSeven(nn.Module):
    """
    数据需要调换
    """

    def __init__(self, config):
        super(DSSMSeven, self).__init__()

        # 此部分的信息有待处理
        self.hidden_size = config.hidden_size  #
        self.kernel_out = config.kernel_out_1  # 32
        self.max_len = config.max_len  # 96

        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size, padding_idx=0)

        self.convs_ = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                              out_channels=self.kernel_out,
                                              kernel_size=3,  # 2
                                              stride=3),
                                    nn.LeakyReLU(),  # B 32 32
                                    nn.BatchNorm1d(self.kernel_out),  # B*L*D->B*D*L->B*Kout*((L-size)/stride+1)
                                    # nn.MaxPool1d(2),  # -> B*Kout* || / 2
                                    nn.Conv1d(in_channels=self.kernel_out,
                                              out_channels=32,
                                              kernel_size=2,
                                              stride=2),
                                    nn.LeakyReLU(),  ## B 32 16
                                    nn.MaxPool1d(2),  ## B 32 8
                                    nn.BatchNorm1d(32),
                                    nn.Conv1d(in_channels=32,
                                              out_channels=8,
                                              kernel_size=2,
                                              stride=2),
                                    nn.LeakyReLU()  ## B * 8 * 4
                                    )

        self.convs_attention = nn.Sequential(nn.Conv1d(in_channels=config.max_len,
                                                       out_channels=32,
                                                       kernel_size=3,
                                                       stride=3),  # B 32 (max_len - ksize ) / stride + 1 : B 32 32
                                             nn.LeakyReLU(),
                                             nn.Conv1d(in_channels=32,
                                                       out_channels=8,
                                                       kernel_size=4,
                                                       stride=4),  # B 8 8
                                             nn.LeakyReLU(),
                                             nn.BatchNorm1d(8)
                                             )

        self.att_linear = nn.Linear(config.max_len, config.max_len)  #
        # # (B 8 4) * 2 + (B 8 8)
        self.fc_1 = nn.Linear(in_features=96, out_features=8)
        self.relu_1 = nn.LeakyReLU()
        self.fc_2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, data):
        q = self.embeddings(data['query_'])
        d = self.embeddings(data['doc_'])

        att_ = self.attention_matrix(q, d)

        q = q.permute(0, 2, 1)
        d = d.permute(0, 2, 1)  # 待匹配的两个句子

        out_q = self.convs_(q)
        out_d = self.convs_(d)

        att_q = self.att_linear(att_.permute(0, 2, 1))
        att_d = self.att_linear(att_)
        out_att_q = self.convs_attention(att_q)
        out_att_d = self.convs_attention(att_d)

        b_, _, _ = out_q.shape

        out_q = torch.cat((out_q.view(b_, -1), out_att_q.view(b_, -1)), dim=1).view(b_, -1)  # B 8 12
        out_d = torch.cat((out_d.view(b_, -1), out_att_d.view(b_, -1)), dim=1).view(b_, -1)  # B 8 12

        out_q = self.relu_1(self.fc_1(out_q))
        out_d = self.relu_1(self.fc_1(out_d))
        out_ = torch.cat((out_q, out_d), dim=1)
        out_ = self.fc_2(out_)

        # dis_ = 1 / (1 + F.pairwise_distance(out_q, out_d, keepdim=True))

        return out_

    def attention_matrix(self, x_1, x_2, eps=1e-6):
        '''compute attention matrix using match score
        1 / (1 + |x · y|)
        |·| is euclidean distance
        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, sentence_length, width)
        Returns
        -------
        output : 3-D torch Tensor
            match score result of size (batch_size, sentence_length(for x2), sentence_length(for x1))
        '''
        x_1 = x_1.unsqueeze(1)  # size (batch_size, 1, sentence_length, width)
        x_2 = x_2.unsqueeze(1)
        eps = torch.tensor(eps)
        one = torch.tensor(1.)
        euclidean = (torch.pow(x_1 - x_2.permute(0, 2, 1, 3), 2).sum(dim=3) + eps).sqrt()
        return (euclidean + one).reciprocal()


class DSSMEight(DSSMSeven):
    def __init__(self, config):
        super(DSSMEight, self).__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size, padding_idx=3)

    def test(self):
        print(self.embeddings.weight[:5])


class DSSMNine(DSSMFive):
    def __init__(self, config):
        super(DSSMNine, self).__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size, padding_idx=3)

    def test(self):
        print(self.embeddings.weight[:5])


from dssm.ABCNN import Abcnn2


class DSSMAbcnn1(nn.Module):
    def __init__(self, config):
        super(DSSMAbcnn1, self).__init__()
        # emb_dim, sentence_length, filter_width, filter_channel=100, layer_size=2, match='cosine',inception = True

        self.emb = nn.Embedding(config.vocab_size, config.emb_dim, config.pad_id)
        self.abcnn2 = Abcnn2(config.emb_dim, config.sentence_length, config.filter_width, layer_size=config.layer_size)

    # def to(self, device):
    #     self.emb.to(device)
    #     self.abcnn2.to(device)

    def forward(self, data):
        q = self.emb(data['query_'])
        d = self.emb(data['doc_'])
        x1 = q.unsqueeze(1)
        x2 = d.unsqueeze(1)
        # x1 = x1.permute(0, 1, 3, 2)
        # x2 = x2.permute(0, 1, 3, 2)  # 待匹配的两个句子

        # print(x1.shape)
        return self.abcnn2(x1, x2)
