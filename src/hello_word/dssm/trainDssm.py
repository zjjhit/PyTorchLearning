# -*- coding: utf-8 -*-#

import time

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


BASE_DATA_PATH = './data/'
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        dataset = pd.read_csv(BASE_DATA_PATH + '/train.csv')  # processed_train.csv
        vocab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))
    else:
        config_path = '../data/config.json_4'
        dataset = pd.read_csv('../data/tt.csv')  # processed_train.csv
        vocab = pickle.load(open('../data/char2id.vocab', 'rb'))

    config = BertConfig.from_pretrained(config_path)

    if '-1' not in config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        print('CUDA_VISIBLE_DEVICES\t' + config.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    if config.loss == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss().to(device)  ###需要调整 网罗结构
    elif config.loss == 'cross':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.BCELoss().to(device)

    print('begin')

    kf = KFold(n_splits=15, shuffle=True)
    nums_ = config.nums
    for k, (train_index, val_index) in enumerate(kf.split(range(len(dataset)))):
        if k > 2:
            break

        train = dataset.iloc[train_index]
        val = dataset.iloc[val_index]

        print('Start train {} ford {}'.format(k, len(train)))

        train_base = DSSMCharDataset(train, vocab, max_len=config.max_len)
        val = DSSMCharDataset(val, vocab, max_len=config.max_len)
        val = DataLoader(val, batch_size=config.batch_size, num_workers=2)

        if 'pt' in config.reload:
            model = torch.load(BASE_DATA_PATH + '/' + config.reload).to(device)
        else:
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
                print('model_5')
            elif config.id == 6:
                model = DSSMSix(config, device, vocab).to(device)
                print('model_6')
            elif config.id == 7:
                model = DSSMSeven(config, device, vocab).to(device)
                print('model_')

            print('para init')
            for m in model.modules():
                if isinstance(m, (nn.Conv1d)):
                    nn.init.xavier_uniform_(m.weight)
                    print('para init', m.weight.shape, m.weight)
                    # nn.init.kaiming_normal_(m.weight, mode='fan_in')

        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        best_loss = 100000

        model.train()
        for n_ in range(nums_):

            train = DataLoader(train_base, batch_size=config.batch_size, shuffle=True)

            # data_loader = tqdm.tqdm(enumerate(train),
            #                         total=len(train))

            total_loss = 0
            for i, data_set in enumerate(train):
                data = {key: value.to(device) for key, value in data_set.items() if key != 'origin_'}

                optimizer.zero_grad()

                y_pred = model(data)
                b_, _ = y_pred.shape

                if i % 1000 == 0:
                    print(y_pred.data[0:5])

                if config.loss == 'bce':
                    loss = criterion(y_pred.view(b_, -1), data['label_'].view(b_, -1))
                elif config.loss == 'cross':
                    loss = criterion(y_pred, data['label_'])
                else:
                    tmp_ = torch.ones(b_).to(device).view(b_, -1) - data['label_'].view(b_, -1)
                    y_target = torch.cat((tmp_, data['label_'].view(b_, -1).float()), dim=1)
                    loss = criterion(y_pred, y_target)

                loss.backward()
                optimizer.step()

                # print('label,{},{}'.format(sum(data['label_']), len(data['label_'])))

                if i % 500 == 0:
                    print('k ford and nums ,{} ,{},loss is {}'.format(n_, i, loss.data.item()))
                    for name, parms in model.named_parameters():
                        print('-->name:', name, '\t\t-->grad_requirs:', parms.requires_grad)
                        if parms.requires_grad:
                            print('--weight', torch.mean(parms.data),
                                  '\t\t-->grad_value:', torch.mean(parms.grad), '\n')
                    print('\n\n')

                total_loss += loss.data.item()

            print('total_loss ford and nums ,{} ,{},loss is {}'.format(k, n_, total_loss))
            if n_ % 100 == 0 and n_ > 0:
                saveModel(model, BASE_DATA_PATH + '/final_model_{}_{}_{}ford.pt'.format(config.id, k, n_))

            # if n_ % 10 == 0:
            #     # with torch.no_grad():
            #     model.eval()
            #     data_loader = tqdm.tqdm(enumerate(val),
            #                             total=len(val))
            #     loss_val = 0
            #     for ii, data_set in data_loader:
            #         data = {key: value.to(device) for key, value in data_set.items() if key != 'origin_'}
            #         y_pred = model(data)
            #         b_, _ = y_pred.shape
            #         if config.loss == 'bce':
            #             a = criterion(y_pred.view(b_, -1), data['label_'].view(b_, -1))
            #         else:
            #             a = criterion(y_pred.view(b_, -1), data['label_'])
            #         loss_val += a.item()
            #
            #     print('val_loss,best_los,{},{}'.format(loss_val, best_loss))
            #     if best_loss > loss_val:
            #         best_loss = loss_val
            #         saveModel(model, BASE_DATA_PATH + '/best_model_{}_{}_ford.pt'.format(config.id, k))
            #         print('Best val loss {},{},{},{}'.format(best_loss, config.id, k, time.asctime()))
