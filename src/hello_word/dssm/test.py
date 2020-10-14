# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Description:  
# Author:       lenovo
# Date:         2020/9/25
# -------------------------------------------------------------------------------

import os

######################################
from torch.utils.data import DataLoader

from dssm.data_process import *
from dssm.model_torch import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DATA_PATH = '../data/'
from transformers import BertConfig


def test():
    print('begin0')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    print('begin')
    config_path = '../data/config.json_5'
    dataset = pd.read_csv('../data/test_avg.csv')  # processed_train.csv

    config = BertConfig.from_pretrained(config_path)

    vacab = pickle.load(open(BASE_DATA_PATH + '/' + config.vocab_name, 'rb'))
    data_base = DSSMCharDataset(dataset, vacab, config)  # same with the config_4
    data = DataLoader(data_base, batch_size=100)

    model = torch.load(BASE_DATA_PATH + 'model/name__best_model_char_5_9_0_6_ford.pt').to(device)

    with torch.no_grad():
        num_ = []
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

            num_.append(float(acc_.item()) / b_)
            # print(org_)
            print(acc_.item(), b_, float(acc_.item()) / b_)

            # for ii, d_ in enumerate(tmp_):
            #     if d_[2] != d_[3]:
            #         print(org_[ii], label_[ii], tmp_[ii])
            #     if tmp_[ii][0] * tmp_[ii][1] > 0:
            #         print('Test , {} {}'.format(org_[ii], tmp_[ii]))
        # print(model.embeddings.weight)

        print(num_, sum(num_) / len(num_))


from dssm.clusterProcess import model_init, sameLogic


def testSigle(s1, s2):
    # model = torch.load(BASE_DATA_PATH + 'model/loc__best_model_5_1_8_49_ford.pt').to('cpu')
    model_dict, char_vocab, word_vocab, max_len, device = model_init()
    name_sim = sameLogic(s1, s2, model_dict, char_vocab, word_vocab, max_len, device)

    # name_1, name_2 = dataPro(s1, s2, 96, vocab)
    # name_sim = isSameModel([name_1, name_2], model)

    print(name_sim)


from dssm.clusterProcess import dataPro


def testOne(s1, s2):
    device = 'cpu'
    config_path = '../data/config.json_5'

    config = BertConfig.from_pretrained(config_path)
    model = torch.load(BASE_DATA_PATH + 'model/name__best_model_char_5_0_3_27_ford.pt').to(device)
    vacob = pickle.load(open(BASE_DATA_PATH + '/' + config.vocab_name, 'rb'))
    name_1, name_2 = dataPro(s1, s2, config.max_len, vacob)

    with torch.no_grad():
        data_ = {'query_': torch.tensor(name_1).unsqueeze(0).to(device), 'doc_': torch.tensor(name_2).unsqueeze(0).to(device)}
        pred = model(data_)
        k_ = torch.max(pred, 1)[1][0]
        print(k_.data.item())

        data_ = {'query_': torch.tensor(name_2).unsqueeze(0).to(device), 'doc_': torch.tensor(name_1).unsqueeze(0).to(device)}
        pred = model(data_)
        k_ = torch.max(pred, 1)[1][0]
        print(k_.data.item())


from dssm.runData import distance_edit, distance_jacaard

if __name__ == '__main__':
    pass
    # test()

    s1 = ['CAFFE BORBONE SRL', 'NULL']
    s2 = ['CAFFE PERFETTO LTD', 'NULL']
    # testSigle(s1, s2)
    #
    testOne(s1[0], s2[0])

    print(distance_edit(s1[0], s2[0]))

    print(distance_jacaard('1 2 3', '2 3 5'))

    # word_vocab = pickle.load(open('../data/emb_word_174215.vocab', 'rb'))
    # for k in word_vocab:
    #     print(k, word_vocab[k])

    # s1, s2 = s2, s1
    # print(s1, s2)
    # testSigle(s1, s2)
