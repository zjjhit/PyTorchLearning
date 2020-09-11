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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BASE_DATA_PATH = '../data/'


def test():
    dataset = pd.read_csv(BASE_DATA_PATH + '/test.csv')
    print('begin0')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    vacab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))

    print('begin')

    data_base = DSSMCharDataset(dataset, vacab)
    data = DataLoader(data_base, batch_size=16)

    # model = DSSMOne(config, device)
    model = torch.load(BASE_DATA_PATH + '/best_model_0_ford.pt').to(device)

    with torch.no_grad():
        # model.eval()
        # data_loader = tqdm.tqdm(enumerate(data),
        #                         total=len(data))
        for i, data_ in enumerate(data):
            # data_ = {key: value.to(device) for key, value in data_set.items()}
            y_pred = model(data_)
            b_, _ = y_pred.shape
            # print(data_['label_'].shape)

            org_ = data_['origin_']
            label_ = data_['label_'].unsqueeze(1)
            # query_ = data_['query_'].unsqueeze(1)
            # doc_ = data_['doc_'].unsqueeze(1)
            print(y_pred)
            k_ = torch.max(y_pred, 1)[1].unsqueeze(1)
            tmp_ = torch.cat((k_, label_), dim=1)

            print(tmp_)
            print(org_)


if __name__ == '__main__':
    test()
