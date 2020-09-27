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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
BASE_DATA_PATH = '../data/'


def test():
    dataset = pd.read_csv(BASE_DATA_PATH + '/tt.csv')
    print('begin0')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    vacab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))

    print('begin')

    data_base = DSSMCharDataset(dataset, vacab, max_len=64)  # same with the config_4
    data = DataLoader(data_base, batch_size=100)

    model = torch.load(BASE_DATA_PATH + 'model/best_model_7_0_6_75_ford.pt').to(device)

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

            for ii, d_ in enumerate(tmp_):
                if d_[2] != d_[3]:
                    print(org_[ii], label_[ii], tmp_[ii])
                if tmp_[ii][0] * tmp_[ii][1] > 0:
                    print('Test , {} {}'.format(org_[ii], tmp_[ii]))
        # print(model.embeddings.weight)

        print(num_, sum(num_) / len(num_))


if __name__ == '__main__':
    pass
    test()
