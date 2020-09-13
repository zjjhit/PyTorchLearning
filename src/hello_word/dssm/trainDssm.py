# -*- coding: utf-8 -*-#

import time

import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
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
        config_path = BASE_DATA_PATH + '/config.json'

    config = BertConfig.from_pretrained(BASE_DATA_PATH + '/config.json')

    dataset = pd.read_csv(BASE_DATA_PATH + '/train.csv')  # processed_train.csv

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    nums_ = config.nums  ## 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))

    if config.loss == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss().to(device)  ###需要调整 网罗结构
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    print('begin')

    kf = KFold(n_splits=15, shuffle=True)

    for k, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):
        if k == 2:
            break

        train = dataset.iloc[train_index]
        val = dataset.iloc[val_index]

        print('Start train {} ford {}'.format(k, len(train)))

        train_base = DSSMCharDataset(train, vocab)
        val = DSSMCharDataset(val, vocab)
        val = DataLoader(val, batch_size=config.batch_size, num_workers=2)

        if config.id == 1:
            model = DSSMOne(config, device).to(device)
            print('model_1')
        elif config.id == 2:
            model = DSSMTwo(config, device).to(device)
            print('model_2')
        elif config.id == 3:
            model = DSSMThree(config, device).to(device)
            print('model_3')
        elif config.id == 4:
            model = DSSMFour(config, device).to(device)
            print('model_4')
        elif config.id == 5:
            model = DSSMFive(config, device, vocab).to(device)
            print('model_4')

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        best_loss = 100000
        model.train()
        for n_ in range(nums_):
            print('k ford and nums ,{} ,{}'.format(k, n_))
            train = DataLoader(train_base, batch_size=config.batch_size, shuffle=True)

            data_loader = tqdm.tqdm(enumerate(train),
                                    total=len(train))
            for i, data_set in data_loader:
                data = {key: value.to(device) for key, value in data_set.items() if key != 'origin_'}

                y_pred = model(data)
                b_, _ = y_pred.shape

                if config.loss == 'bce':
                    loss = criterion(y_pred.view(b_, -1), data['label_'].view(b_, -1))
                else:
                    loss = criterion(y_pred.view(b_, -1), data['label_'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if n_ % 5 == 0:
                with torch.no_grad():
                    data_loader = tqdm.tqdm(enumerate(val),
                                            total=len(val))
                    loss_val = 0
                    for ii, data_set in data_loader:
                        data = {key: value.to(device) for key, value in data_set.items() if key != 'origin_'}
                        y_pred = model(data)
                        b_, _ = y_pred.shape
                        if config.loss == 'bce':
                            a = criterion(y_pred.view(b_, -1), data['label_'].view(b_, -1))
                        else:
                            a = criterion(y_pred.view(b_, -1), data['label_'])
                        loss_val += a.item()

                    print('val_loss,best_los,{},{}'.format(loss_val, best_loss))
                    if best_loss > loss_val:
                        best_loss = loss_val
                        saveModel(model, BASE_DATA_PATH + '/best_model_{}_{}_ford.pt'.format(config.id, k))
                        print('Best val loss {},{},{},{}'.format(best_loss, config.id, k, time.asctime()))
