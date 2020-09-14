# -*- coding: utf-8 -*-#

import time

from transformers import BertConfig

# -------------------------------------------------------------------------------
# Name:         trainDssm
# Description:
# Author:       lenovo
# Date:         2020/9/11
# -------------------------------------------------------------------------------
from dssm.data_process import *
from dssm.model_torch import *


def saveModel(model, file_path):
    print('Model save {} , {}'.format(file_path, time.asctime()))
    torch.save(model, file_path)
    # self.model.to(self.device)


BASE_DATA_PATH = '../data/'
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = BASE_DATA_PATH + '/config.json_5'

    config = BertConfig.from_pretrained(config_path)

    dataset = pd.read_csv(BASE_DATA_PATH + '/train.csv')  # processed_train.csv

    vocab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))

    if '-1' not in config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        print('CUDA_VISIBLE_DEVICES\t' + config.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    if config.loss == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss().to(device)  ###需要调整 网罗结构
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    print('begin')

    # kf = KFold(n_splits=15, shuffle=True)
    # nums_ = config.nums
    # for k, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):
    #     if k > 2:
    #         break
    #
    #     train = dataset.iloc[train_index]
    #     val = dataset.iloc[val_index]
    #
    #     print('Start train {} ford {}'.format(k, len(train)))
    #
    #     train_base = DSSMCharDataset(train, vocab)
    #     val = DSSMCharDataset(val, vocab)
    #     val = DataLoader(val, batch_size=config.batch_size, num_workers=2)
    #
    #     for n_ in range(nums_):
    #
    #         train = DataLoader(train_base, batch_size=config.batch_size, shuffle=True)
    #
    #         for i, data_set in enumerate(train):
    #             data = {key: value.to(device) for key, value in data_set.items() if key != 'origin_'}
    #
    #             print(data['query_'].shape)
    #             print(data['query_'].data[0])

    model_ = torch.load(BASE_DATA_PATH + '/' + config.reload).to(device)
    print(model_)

    for name, parms in model_.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
              ' -->grad_value:', torch.mean(parms.grad))
