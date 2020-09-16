# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Description:  
# Author:       lenovo
# Date:         2020/9/11
# -------------------------------------------------------------------------------

import os

from torch.utils.data import DataLoader

from dssm.data_process import *
from dssm.model_torch import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
BASE_DATA_PATH = '../data/'


def test():
    dataset = pd.read_csv(BASE_DATA_PATH + '/test.csv')
    print('begin0')
    device = 'cpu'
    vacab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))

    print('begin')

    data_base = DSSMCharDataset(dataset, vacab, max_len=64)
    data = DataLoader(data_base, batch_size=100)

    # model = DSSMOne(config, device)
    model = torch.load(BASE_DATA_PATH + '/final_model_4_0_100ford.pt').to(device)

    with torch.no_grad():
        for i, data_ in enumerate(data):
            y_pred = model(data_)
            b_, _ = y_pred.shape

            org_ = data_['origin_']
            label_ = data_['label_']

            k_ = torch.max(y_pred, 1)[1]
            tmp_ = torch.cat((y_pred, k_.float().unsqueeze(1), label_.float().unsqueeze(1)), dim=1)

            print(tmp_)
            # print('\n'.join(org_))
            # print(label_)
            acc_ = sum(k_ == label_)

            # print(org_)
            print(acc_.item(), b_, float(acc_.item()) / b_)

    # print(model.embeddings.weight)


import pandas as pd

BASE_DATA_PATH = '../data/'
from transformers import BertConfig


def testLL():
    config_path = BASE_DATA_PATH + '/config.json_5'

    config = BertConfig.from_pretrained(config_path)

    dataset = pd.read_csv(BASE_DATA_PATH + '/t.csv')  # processed_train.csv

    device = 'cpu'
    vocab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))
    train = dataset.iloc[range(len(dataset))]

    train_base = DSSMCharDataset(train, vocab)
    for _ in range(3):
        train_d = DataLoader(train_base, batch_size=3, shuffle=True)
        for i, d in enumerate(train_d):
            print(d)
        print('\n\n')

    model = DSSMFive(config, device, vocab).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # for i, k in enumerate(train_d):
    #     y = model(k)
    #     c_ = criterion(y, k['label_'])
    #     print(y, c_)
    #     print(y.shape)
    #     print(model.embeddings.weight[0])
    # print('model_5')


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


if __name__ == '__main__':
    pass
    # testLL()
    test()
