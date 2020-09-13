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

# config = BertConfig.from_pretrained('./config.json')
# model = DSSMOne(config)
# stat(model, (256, 256))

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BASE_DATA_PATH = '../data/'


def test():
    dataset = pd.read_csv(BASE_DATA_PATH + '/test.csv')
    print('begin0')
    device = 'cpu'
    vacab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))

    print('begin')

    data_base = DSSMCharDataset(dataset, vacab)
    data = DataLoader(data_base, batch_size=50)

    # model = DSSMOne(config, device)
    model = torch.load(BASE_DATA_PATH + '/model/best_model_3_0_ford.pt').to(device)

    with torch.no_grad():
        for i, data_ in enumerate(data):
            y_pred = model(data_)
            b_, _ = y_pred.shape

            org_ = data_['origin_']
            label_ = data_['label_']
            # query_ = data_['query_'].unsqueeze(1)
            # doc_ = data_['doc_'].unsqueeze(1)
            print(y_pred)
            k_ = torch.max(y_pred, 1)[1]
            tmp_ = torch.stack((k_, label_), dim=0)

            # print(tmp_)
            acc_ = sum(k_ == label_)
            # print(org_)
            print(acc_.item(), b_, float(acc_.item()) / b_)


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


import pandas as pd

BASE_DATA_PATH = '../data/'


def makeTnmp():
    dataset = pd.read_csv('../data/processed_train.csv')  # processed_train.csv
    # train = random.sample(dataset, 50000)
    # test = random.sample(dataset, 100)

    df = pd.DataFrame(columns=['origin', 'label'], data=dataset.take(range(1, 50000)))
    df.to_csv(BASE_DATA_PATH + '/train.csv', index=False)
    df = pd.DataFrame(columns=['origin', 'label'], data=dataset.take(range(60000, 60100)))
    df.to_csv(BASE_DATA_PATH + '/test.csv', index=False)


def testVocab():
    dic_ = {}
    fout = open(BASE_DATA_PATH + '/' + 'char2id.vocab', 'rb')

    dic_ = pickle.load(fout)
    fout.close()

    for k in dic_:
        print(k, dic_[k])


if __name__ == '__main__':
    test()
    # testVocab()
