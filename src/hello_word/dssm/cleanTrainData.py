# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         cleanTrainData
# Description:  
# Author:       lenovo
# Date:         2020/10/12
# -------------------------------------------------------------------------------

'''
第二次清洗训练数据
'''

cleanFilterList = ['LTD', 'CO.,LTD', 'LIMITED', 'LTD.', 'CO', 'CO.,LTD.', 'CO.LTD', 'INC', 'LT', 'CORP', 'COMPANY', 'GROUP', 'FACTORY',
                   'INC.', 'CORPORATION', 'CO.LTD.', 'BRANCH', 'LLC', 'CHINA', 'INTERNATIONAL', 'TRADE', 'EXPORT', 'CO.', 'CO.,',
                   'TECHNOLOGY', 'CO.,LIMITED', 'PRODUCTS', 'GMBH', 'INDUSTRIAL', 'CORP.', 'CO,LTD', 'LIMITED.', 'S.A.', 'LTD"',
                   'LTD,',
                   'LTD.,', 'EXP', 'CO..LTD', 'CO,.LTD', 'CO.,LT', 'CO.,LTD"', 'CO.LIMITED', 'LIMITE', 'CO.,L', '.,LTD', 'COLTD',
                   'LIMIT',
                   'LI', 'INDUSTRY', 'C.V.', 'EXP.CO.,LTD', 'LIMI', 'LIM', 'B.V.', 'S.L.', '.', ',LTD', 'LTD."', 'LLC.', 'CO..LTD.',
                   'LTD)', 'S.P.A.', 'LIMITED,', 'IMP', 'S.A', 'S.R.L.', 'COMP', 'IMP.&EXP.CO.,LTD', 'TEC', 'SPA', 'LTDNO',
                   'CORP.,LTD',
                   'Ltd', 'CO.LT', 'CO.LTD.,', 'CO.,LTD)', 'LTD.-', 'CO,LTD.', 'CO.,LTD,', 'CO,', 'EXP.CO.,LTD.', 'CENTRE', 'GARMENT',
                   'INDUSTR', '.LTD', 'NORTH', 'A/S', 'PROD', 'LIMITED"', 'SCIENCE', 'BUSINESS', 'TEXTILES', 'COR', 'CO.,LTD."',
                   'CORP.LTD', 'CO.L', 'EXP.CO.LTD', 'LIMTED', 'GLOBAL', 'CO.,LTD.,', 'CO.LTD,']

import pandas as pd


def cleanMethold_1(data_):
    tmp_ = []
    for k in data_.split(' '):
        if k not in cleanFilterList:
            tmp_.append(k)
    return ' '.join(tmp_)


from dssm.runData import isSameNew
import pickle


def cleanPosData(path_):
    data_ = pd.read_csv(path_)
    dic_ = {}
    for i in range(len(data_)):
        item = data_.iloc[i]

        label_ = item['label']
        if label_ == 1:
            continue

        txt_ = item['origin'].split('\001\002')
        # edt, edt_dis, tf, tf_dis, jaca, jaca_dis
        flags_ = isSameNew(cleanMethold_1(txt_[0]), cleanMethold_1(txt_[1]))
        dic_[item['origin']] = flags_
        # if flags_[1] + flags_[3] + flags_[5] < 0 and label_ == 0:
        #     if txt_[0] in txt_[1] or txt_[1] in txt_[0]:
        #         continue
        #     info_ = [str(k) for k in flags_]
        #     if flags_[1] < -10 or (flags_[-1] < -10 and flags_[1] < 10):
        #         print(data_.iloc[i]['origin'] + '\001\002' + str(data_.iloc[i]['label']) + '\t\t' + ' '.join(info_))

    with open('../train.info.pk', 'wb')  as f:
        pickle.dump(dic_, f)


def lc_(path_):
    with open(path_, 'rb') as f:
        dic_ = pickle.load(f)

    for k in dic_:
        flags_ = dic_[k]
        txt_ = k.split('\001\002')
        # edt, edt_dis, tf, tf_dis, jaca, jaca_dis
        if flags_[1] + flags_[3] + flags_[5] < 0:
            if txt_[0] in txt_[1] or txt_[1] in txt_[0]:
                continue
            info_ = [str(k) for k in flags_]
            num_ = 0
            if flags_[1] < 0:
                num_ += 1
            if flags_[3] < 0:
                num_ += 1
            if flags_[5] < 0:
                num_ += 1

            t_1 = set(cleanMethold_1(txt_[0]).split())
            t_2 = set(cleanMethold_1(txt_[1]).split())
            if t_1 - t_2 == set() or t_2 - t_1 == set():
                continue
            if num_ >= 2 and flags_[1] < 0:
                print(str(flags_[1]) + '\t\t' + k + '\001\002' + str(0) + '\t\t' + ' '.join(info_))


def cleanFirstManue(path_fake, path_manue):
    '''
    清洗第一次标注的数据
    '''

    with open(path_fake, 'r')  as f:
        dic_fake = {one.split('\t\t')[0]: 1 for one in f.readlines() if len(one.split('\t\t')) == 2}

    with open(path_manue, 'r')  as f:
        dic_man = {one.split('\t\t')[0]: 1 for one in f.readlines() if len(one.split('\t\t')) == 2}

    for k in dic_fake:
        tmp_ = k.split('\002')
        if len(tmp_) != 2:
            continue

        if k not in dic_man:
            t_1 = set(tmp_[0].split())
            t_2 = set(tmp_[1].split())

            if t_1 - t_2 == set() or t_2 - t_1 == set() or tmp_[0] in tmp_[1] or tmp_[1] in tmp_[0]:
                print('Small\t\t' + '\001\002'.join(tmp_))
                continue

            print('Diff\t\t' + '\001\002'.join(tmp_))
        else:
            print('Same\t\t' + '\001\002'.join(tmp_))


from dssm.runData import distance_edit


def makeSecondTrainData(path_train, pos_, neg_):
    """
    基于第二次的标注修整结果构建新的训练数据集,
    注意同时调整模型的卷积的步幅，以使其可以有效抽取位置特征
    :param path_train:
    :param neg_:
    :param pos_:
    :return:
    """

    data_ = pd.read_csv(path_train)
    dic_pos = {}
    dic_neg = {}

    with open(neg_, 'r')  as f:
        neg_ = {one.split('\001\002')[0]: 1 for one in f.readlines() if len(one.split('\001\002')) == 2}

    with open(pos_, 'r')  as f:
        pos_ = {one.split('\001\002')[0]: 1 for one in f.readlines() if len(one.split('\001\002')) == 2}

    for i in range(len(data_)):
        item = data_.iloc[i]

        label_ = item['label']
        text_ = item['origin']
        txt_ = text_.split('\001\002')

        if len(txt_) != 2:
            continue

        if label_ == 1:
            t_1 = set(txt_[0].split())
            t_2 = set(txt_[1].split())

            if t_1 - t_2 == set() or t_2 - t_1 == set() or txt_[0] in txt_[1] or txt_[1] in txt_[0]:
                continue
            edf_ = distance_edit(txt_[0], txt_[1])
            if edf_ < 0.05:
                print('Test\t' + item['origin'])
                continue

            if txt_[1] + '\001\002' + txt_[0] not in dic_neg:
                dic_neg[item['origin']] = 1
        else:
            if text_ not in neg_:
                dic_pos[text_] = 1

    print(len(dic_neg), len(dic_pos))

    with open('../Train/2020-10/train_10.pk', 'wb') as f:
        pickle.dump(dic_pos, f)
        pickle.dump(dic_neg, f)

    data_ = list(dic_pos.keys()) + list(dic_neg.keys())
    label_ = [0] * len(dic_pos.keys()) + [1] * len(dic_neg.keys())
    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = data_
    df['label'] = label_
    df.to_csv('../Train/2020-10/' + 'train_10.csv', index=False)


def infodData(path_):
    data_ = pd.read_csv(path_)

    num_neg = 0
    num_pos = 0
    num_n = 0
    for i in range(len(data_)):
        item = data_.iloc[i]

        label_ = item['label']
        text_ = item['origin']
        txt_ = text_.split('\001\002')

        n_ = abs(len(txt_[0]) - len(txt_[1]))
        if label_ == 0:
            num_pos += n_
        else:
            num_neg += n_
            if n_ <= 5:
                num_n += 1

    print(num_neg, num_pos, num_n)


import random


def makeThirdTrainData():
    '''
    调整 正负列的比例，同时要求负例的长度差占比与正例的要持平
    :return:
    '''
    with open('../Train/2020-10/pos.data', 'r')  as f:
        pos_ = {one.rstrip(): 1 for one in f.readlines() if len(one.split('\001\002')) == 2}

    with open('../Train/2020-10/neg.data', 'r')  as f:
        neg_ = {one.rstrip(): 1 for one in f.readlines() if len(one.split('\001\002')) == 2}

    with open('../Train/2020-10/train_10.pk', 'rb') as f:
        dic_pos = pickle.load(f)
        dic_neg = pickle.load(f)

    tmp_pos = {k: 1 for k in dic_pos if k not in pos_}
    tmp_neg_small = {}
    tmp_neg_big = {}  # len mux

    for k in dic_neg:
        if k in neg_:
            continue
        t_ = k.split('\001\002')
        if abs(len(t_[0]) - len(t_[1])) <= 19:
            tmp_neg_small[k] = 1
        else:
            tmp_neg_big[k] = 1

    print(len(tmp_pos), len(tmp_neg_small), len(tmp_neg_big))
    data_pos = list(pos_.keys()) + random.sample(list(tmp_pos.keys()), 18000)
    data_neg = list(neg_.keys()) + random.sample(list(tmp_neg_small.keys()), 20000) + random.sample(list(tmp_neg_big.keys()), 1500)

    print(len(data_pos), len(data_neg))

    num_pos = 0
    num_neg = 0
    for i in range(len(data_pos)):
        txt_ = data_pos[i].split('\001\002')

        n_ = abs(len(txt_[0]) - len(txt_[1]))
        num_pos += n_

    for i in range(len(data_neg)):
        txt_ = data_neg[i].split('\001\002')

        n_ = abs(len(txt_[0]) - len(txt_[1]))
        num_neg += n_

    print(num_pos, num_neg)

    random.shuffle(data_pos)
    random.shuffle(data_neg)

    train_pos = data_pos[1000:]
    train_neg = data_neg[1000:]
    test_pos = data_pos[:1000]
    test_neg = data_neg[:1000]

    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = train_pos + train_neg
    df['label'] = [0] * len(train_pos) + [1] * len(train_neg)
    df.to_csv('../Train/2020-10/' + 'train_avg.csv', index=False)

    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = test_pos + test_neg
    df['label'] = [0] * len(test_pos) + [1] * len(test_neg)
    df.to_csv('../Train/2020-10/' + 'test_avg.csv', index=False)


def cleanNegData(path_0, path_1):
    dic_ = {}

    with open(path_0, 'r') as f:
        for k in f:
            t = k.rstrip().split('\001\001')
            if len(t) != 2:
                continue
            if t[1] not in dic_:
                dic_[t[1]] = t[0]

    with open(path_1, 'r') as f:
        for k in f:
            t = k.rstrip().split('\001\001')
            if len(t) != 2:
                continue
            if t[1] not in dic_:
                dic_[t[1]] = t[0]

    tmp_small = {}
    tmp_big = {}
    tmp_diff_big = {}
    tmp_other = {}
    print(len(dic_))
    for k in dic_:
        w = dic_[k].split('\001')
        t_ = k.split('\001\002')
        sen_1 = t_[0].split()
        sen_2 = t_[1].split()

        flag_ = True
        for i in range(len(w)):
            if i >= len(sen_1) or i >= len(sen_2):
                flag_ = False
                break
            if w[i] != sen_1[i] or w[i] != sen_2[i]:
                flag_ = False
                break

        if flag_:
            if abs(len(sen_2) - len(sen_1)) <= 12:
                tmp_small[k] = dic_[k]

            if len(w) > 1:
                tmp_big[k] = dic_[k]
        else:
            if len(w) > 1:
                tmp_diff_big[k] = dic_[k]
            else:
                tmp_other[k] = dic_[k]

    print(len(tmp_small), len(tmp_big), len(tmp_diff_big), len(tmp_other))
    t_small = random.sample(list(tmp_small.keys()), 36200)
    t_big = list(tmp_big.keys())
    t_diff_big = random.sample(list(tmp_diff_big.keys()), 12000)
    t_other = random.sample(list(tmp_other.keys()), 30200)

    t_ = t_small + t_big + t_diff_big + t_other

    random.shuffle(t_)

    with open('../Train/2020-10/Verison2/neg.data', 'r') as f:
        manu_neg = [one.rstrip() for one in f]

    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = t_[1000:] + manu_neg
    df['label'] = [1] * (len(t_) - 1000 + len(manu_neg))
    df.to_csv('../Train/2020-10/' + 'neg_avg.csv', index=False)

    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = t_[:1000]
    df['label'] = [1] * 1000
    df.to_csv('../Train/2020-10/' + 'neg_test_avg.csv', index=False)


def cleanPosData(path_):
    data_ = pd.read_csv(path_)
    tmp_ = []
    tmp_1 = []
    tmp_2 = []
    tmp_3 = []
    tmp_4 = []

    for i in range(len(data_)):
        item = data_.iloc[i]

        label_ = item['label']
        text_ = item['origin']
        txt_ = text_.split('\001\002')

        edt_ = distance_edit(txt_[0], txt_[1])
        set_a = set(txt_[0].split())
        set_b = set(txt_[1].split())

        if edt_ > 0.2 and abs(len(txt_[0]) - len(txt_[1])) < 10 and len(set_a & set_b) <= 2:
            print(text_)
            tmp_.append(i)

        if edt_ < 0.07:
            print('Same_07\t' + text_)
            tmp_1.append(text_)
        elif edt_ < 0.12:
            print('Same_12\t' + text_)
            tmp_2.append(text_)
        elif edt_ < 0.17:
            print('Same_17\t' + text_)
            tmp_3.append(text_)
        else:
            print('Same_else\t' + text_)
            tmp_4.append(text_)

    t_ = random.sample(tmp_1, 3000) + random.sample(tmp_2, 1500) + random.sample(tmp_3, 1500) + tmp_4
    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = t_
    df['label'] = [0] * len(t_)
    df.to_csv('../Train/2020-10/' + 'pos_avg_5.csv', index=False)


if __name__ == '__main__':
    # cleanPosData('../data/train_new.csv')
    # lc_('../train.info.pk')
    pass
    # cleanFirstManue('../Train/2020-10/fake.data.pos', '../Train/2020-10/manua.data.pos')
    # neg.data  pos.data  train_new.csv
    # cProfile.run(makeSecondTrainData('../Train/2020-10/train_new.csv', '../Train/2020-10/pos.data', '../Train/2020-10/neg.data'))
    # makeSecondTrainData('../Train/2020-10/train_new.csv', '../Train/2020-10/pos.data', '../Train/2020-10/neg.data')
    # cleanSecondData('../Train/2020-10/train_10.csv')
    # makeThirdTrainData()
    # cleanNegData('../data/Data/data.train.neg0', '../data/Data/data.train.neg1')

    cleanPosData('../Train/2020-10/pos_avg.csv')
