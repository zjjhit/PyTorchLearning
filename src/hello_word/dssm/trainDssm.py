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

if __name__ == '__main__':

    dataset = pd.read_csv(BASE_DATA_PATH + '/train.csv')  # processed_train.csv
    config = BertConfig.from_pretrained(BASE_DATA_PATH + '/config.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    nums_ = config.nums  ## 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vacab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))

    if config.loss == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss().to(device)  ###需要调整 网罗结构
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    print('begin')

    kf = KFold(n_splits=15, shuffle=True)

    for k, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):
        print('Start train {} ford'.format(k))

        train = dataset.iloc[train_index]
        val = dataset.iloc[val_index]

        train_base = DSSMCharDataset(train, vacab)
        # train = DataLoader(train, batch_size=256, shuffle=True)
        val = DSSMCharDataset(val, vacab)
        val = DataLoader(val, batch_size=256, num_workers=2)

        model = DSSMOne(config, device).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        best_loss = 100000

        model.train()

        for i in range(nums_):
            print('k ford and nums ,{} ,{}'.format(k, i))
            train = DataLoader(train_base, batch_size=256, shuffle=True)

            data_loader = tqdm.tqdm(enumerate(train),
                                    total=len(train))
            for i, data_set in data_loader:
                data = {key: value.to(device) for key, value in data_set.items() if key != 'origin_'}
                # print(data['query_'])

                y_pred = model(data)
                b_, _ = y_pred.shape

                if config.loss == 'bce':
                    loss = criterion(y_pred.view(b_, -1), data['label_'].view(b_, -1))
                else:
                    loss = criterion(y_pred.view(b_, -1), data['label_'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % 5 == 0:
                with torch.no_grad():
                    # model.eval()
                    data_loader = tqdm.tqdm(enumerate(val),
                                            total=len(val))
                    loss_val = 0
                    for i, data_set in data_loader:
                        data = {key: value.to(device) for key, value in data_set.items() if key != 'origin_'}
                        y_pred = model(data)
                        b_, _ = y_pred.shape

                        if config.loss == 'bce':
                            loss = criterion(y_pred.view(b_, -1), data['label_'].view(b_, -1))
                        else:
                            loss = criterion(y_pred.view(b_, -1), data['label_'])

                        loss_val += loss.item()

                    if best_loss > loss_val:
                        best_loss = loss_val
                        saveModel(model, BASE_DATA_PATH + '/best_model_{}_ford.pt'.format(k))
                        print('Best val loss {} ,{},{}'.format(best_loss, k, time.asctime()))

                    # model.to(device)

        # trainer.load('best_model_{}ford.pt'.format(k))
        # for i in trainer.inference(val):
        #     print(i)
        #     print('\n')
