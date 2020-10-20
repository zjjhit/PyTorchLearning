import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DATA_PATH = '../data/'


class DSSMCharDataset(Dataset):
    """
    基于字符的 数据集合
    """

    def __init__(self, dataset, vocab, config):
        self.dataset = dataset
        self.vocab = vocab

        self.max_len = config.max_len
        self.overturn = config.overturn
        self.segment_type = config.segment_type  # char  or  word

        self.model = config.model
        self.DataDict = {}
        self.__preData__()
        self.data_size = len(self.DataDict)

    def __len__(self):
        return len(self.DataDict)

    def convert_tokens_to_ids(self, query):
        ids_ = []
        if self.segment_type == 'word':
            for one in query.split():
                if one.isalpha():
                    one = one.upper()
                one = cleanWord(one)
                if one in self.vocab:
                    ids_.append(self.vocab[one])
                else:
                    ids_.append(self.vocab['<UNK>'])
        else:
            for one in query:
                if one.isalpha():
                    one = one.upper()
                if one in self.vocab:
                    ids_.append(self.vocab[one])
                else:
                    ids_.append(self.vocab['<UNK>'])
        return ids_

    def __preData__(self):
        '''
        pre cleaning data
        :return:
        '''

        for i in range(len(self.dataset)):
            item = self.dataset.iloc[i]

            text_ = item['origin'].split('\001\002')

            query_ = text_[0]
            query_ = query_[:min(len(query_), self.max_len)]
            # convert to ids
            query_ids = self.convert_tokens_to_ids(query_)
            query_ids = query_ids + [0] * (self.max_len - len(query_ids))

            doc_ = text_[1]
            doc_ = doc_[:min(len(doc_), self.max_len)]
            doc_ids = self.convert_tokens_to_ids(doc_)
            doc_ids = doc_ids + [0] * (self.max_len - len(doc_ids))

            output = {
                'origin_': item['origin'],
                'query_': torch.tensor(query_ids),
                'doc_': torch.tensor(doc_ids)
            }
            # print(type(output['doc_']))
            if self.model == 'train':
                label_ = item['label']
                output['label_'] = torch.tensor(np.long(label_))  # torch.tensor(label_)

            self.DataDict[i] = output

            if self.overturn == True:
                output_turn = {
                    'origin_': text_[1] + '\001\002' + text_[0],
                    'query_': torch.tensor(doc_ids),
                    'doc_': torch.tensor(query_ids)
                }
                if self.model == 'train':
                    label_ = item['label']
                    output_turn['label_'] = torch.tensor(np.long(label_))  # torch.tensor(label_)

                self.DataDict[i + len(self.dataset)] = output_turn

    def __getitem__(self, item):
        return self.DataDict[item]


def vocab_build(dict_path, min_count=-float("inf")):
    """
    :param dict_  字符集合与对应频率
    :param min_count: 最小词频
    :return:  word2id = {'<PAD>':0, 'word1':id_1, ……， '<UNK>':id_n}
    """
    with open(dict_path, 'rb') as f:
        dict_ = pickle.load(f)

    word2id = {}
    new_id = 1
    for char in dict_.keys():
        if dict_[char] < min_count:
            continue
        if char.isalpha():
            char = char.upper()
        word2id[char] = new_id  # word2id = {'<PAD>':0, 'word1':id_1, ......, '<UNK>':id_n}
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    print("len(word2id):", len(word2id))
    for k in word2id:
        print(k, word2id[k])
    fout = open(BASE_DATA_PATH + '/' + 'char2id.vocab', 'wb')
    pickle.dump(word2id, fout)
    fout.close()
    return word2id


cleanFilterList = set(['LTD', 'CO.,LTD', 'LIMITED', 'LTD.', 'CO', 'CO.,LTD.', 'CO.LTD', 'INC', 'LT', 'CORP', 'COMPANY', 'GROUP',
                       'FACTORY',
                       'INC.', 'CORPORATION', 'CO.LTD.', 'BRANCH', 'LLC', 'CHINA', 'INTERNATIONAL', 'TRADE', 'EXPORT', 'CO.', 'CO.,',
                       'TECHNOLOGY', 'CO.,LIMITED', 'PRODUCTS', 'GMBH', 'INDUSTRIAL', 'CORP.', 'CO,LTD', 'LIMITED.', 'S.A.', 'LTD"',
                       'LTD,',
                       'LTD.,', 'EXP', 'CO..LTD', 'CO,.LTD', 'CO.,LT', 'CO.,LTD"', 'CO.LIMITED', 'LIMITE', 'CO.,L', '.,LTD', 'COLTD',
                       'LIMIT',
                       'LI', 'INDUSTRY', 'C.V.', 'EXP.CO.,LTD', 'LIMI', 'LIM', 'B.V.', 'S.L.', '.', ',LTD', 'LTD."', 'LLC.',
                       'CO..LTD.',
                       'LTD)', 'S.P.A.', 'LIMITED,', 'IMP', 'S.A', 'S.R.L.', 'COMP', 'IMP.&EXP.CO.,LTD', 'TEC', 'SPA', 'LTDNO',
                       'CORP.,LTD',
                       'Ltd', 'CO.LT', 'CO.LTD.,', 'CO.,LTD)', 'LTD.-', 'CO,LTD.', 'CO.,LTD,', 'CO,', 'EXP.CO.,LTD.', 'CENTRE',
                       'GARMENT',
                       'INDUSTR', '.LTD', 'NORTH', 'A/S', 'PROD', 'LIMITED"', 'SCIENCE', 'BUSINESS', 'TEXTILES', 'COR', 'CO.,LTD."',
                       'CORP.LTD', 'CO.L', 'EXP.CO.LTD', 'LIMTED', 'GLOBAL', 'CO.,LTD.,', 'CO.LTD,'])


def cleanWord(word_):
    filter_char = set([',', '.', '/', '!'])
    while len(word_) > 0:
        if word_[-1] in filter_char:
            word_ = word_[:-1]
        else:
            break
    return word_


def buildWordCab(data_path, min_count=5):
    """
    清洗数据 word
    :param data_path:
    :return:
    """

    word_dict = {}
    with open(data_path) as f:
        for one in f:
            tmp_ = one.rstrip().split('\001\002')
            if len(tmp_) != 5:
                continue
            words_ = tmp_[-2].split(' ')
            for k in words_:
                k = cleanWord(k).upper()
                if k not in word_dict:
                    word_dict[k] = 1
                else:
                    word_dict[k] += 1

    fout = open(BASE_DATA_PATH + '/' + 'origin_word.vocab', 'wb')
    pickle.dump(word_dict, fout)
    fout.close()
    use_word = {}
    use_word['<PAD>'] = 0
    use_word['<UNK>'] = 1
    num_ = 2
    for k in word_dict:
        if word_dict[k] >= min_count:
            use_word[k] = num_
            num_ += 1

    fout = open(BASE_DATA_PATH + '/' + 'emb_word_{}.vocab'.format(str(num_)), 'wb')
    pickle.dump(use_word, fout)
    fout.close()
    print('emb num {}'.format(num_))


import re

filter_word = set(
    {'LTD', 'CO.,LTD', 'LIMITED', 'LTD.', 'CO', 'CO.,LTD.', 'CO.LTD', 'COMPANY', 'GROUP', 'INC.', 'CO.LTD.', 'CO.',
     'CO.,', 'CO.,',
     'LIMITED', 'LIMITED.', 'LTD,', 'LTD.,', 'LIMITE', '.,LTD', ',LTD', 'LTD.', 'LLC.', 'CO..LTD.'})


def filterData(query, doc):
    """
    依照一定规则筛选候选相似对
    :param query:
    :param doc:
    :return:
    """
    ### 0 same  1 litte diff 2 diff
    query = re.sub(r'\s+|\t', ' ', query).rstrip('"')
    doc = re.sub(r'\s+|\t', ' ', doc).rstrip('"')
    if query == doc:
        return 0

    q_set = set(query.split(' ')) - filter_word
    d_set = set(doc.split(' ')) - filter_word

    tmp_ = q_set & d_set
    # print(float(len(tmp_)), float(min(len(q_set), len(d_set))) * 0.5)
    if float(len(tmp_)) >= float(min(len(q_set), len(d_set))) * 0.5:
        return 1
    else:
        return 2


def makeTrainPosData(path):
    """
    构建训练数据集
    :param path:
    :return:
    """
    train = set()
    with open(path, 'rb') as f:
        dict_ = pickle.load(f)
        for k in dict_:
            t_ = set(dict_[k])
            if len(t_) == 1 or len(t_) > 10:
                continue

            t_ = list(t_)
            for i in range(len(t_)):
                if i + 1 == len(t_):
                    break
                for j in t_[i + 1:]:
                    f_ = filterData(t_[i], j)
                    if f_ != 1:
                        continue
                    else:
                        train.add(t_[i] + '\001\002' + j)

    print(len(train))
    fout = open(BASE_DATA_PATH + '/data.train.pos', 'w')
    # pickle.dump(train, fout)
    for k in train:
        fout.write(k + '\n')
    fout.close()


def makeTrainNegData(path_data, data_len=500000, type_=0):
    """
    构建负样本
    :param path_:
    :param data_len: 数据集大小
    :return:
    """
    data_ = {}  ## id --> sen
    num_ = 0
    with open(path_data, 'r') as f:
        for one in f:
            tmp_ = one.rstrip().split('\001\002')
            if len(tmp_) != 5 or tmp_[-2] == 'NULL' or tmp_[-1] == 'NULL':
                continue
            data_[num_] = tmp_[-2]
            num_ += 1

    neg_ = set()  ### train_data

    if type_ == 0:
        '''
        Base 随机生成数据
        '''
        data_len = min(data_len, len(data_))
        list_ = range(len(data_))
        num_ = 0
        while num_ < data_len:
            tmp_ = random.sample(list_, 3)
            if filterData(data_[tmp_[0]], data_[tmp_[1]]) == 2:
                neg_.add(data_[tmp_[0]] + '\001\002' + data_[tmp_[1]])
            elif filterData(data_[tmp_[0]], data_[tmp_[2]]) == 2:
                neg_.add(data_[tmp_[0]] + '\001\002' + data_[tmp_[2]])
            else:
                continue
            num_ += 1
    elif type_ == 1:
        '''
        基于关键词共现的比率生成负样本
        '''
        ###基于词的倒排
        word2sen = {}  ### word--> set[sen_ids]
        ###原始词表
        words_ = {}  ### word --> nums
        for k in data_:
            tmp_ = set(data_[k].split(' ')) - filter_word
            for o in tmp_:
                if o not in words_:
                    words_[o] = 1
                else:
                    words_[o] += 1
                if o not in word2sen:
                    word2sen[o] = set({k})
                else:
                    word2sen[o].add(k)

        te_num = 2000  ###共现词的词频要求
        candidate_words = {}  ### candidate_word --> nums
        for k in words_:
            if words_[k] > te_num and len(k) >= 3:
                candidate_words[k] = words_[k]

        # for k in candidate_words:
        #     print(k, candidate_words[k])

        ###############################

        ### 单个词共现
        neg_one = set({})
        num_one = int(data_len * 0.6) + 1
        i_ = 0
        while i_ < num_one:
            for word in candidate_words:
                randn_a = random.sample(word2sen[word],
                                        min(int(num_one / len(candidate_words)) + 1, len(word2sen[word])))
                randn_b = random.sample(word2sen[word],
                                        min(int(num_one / len(candidate_words)) + 1, len(word2sen[word])))
                for i in range(len(randn_a)):
                    f_ = filterData(data_[randn_a[i]], data_[randn_b[i]])
                    if f_ == 2:
                        neg_one.add(word + '\001\001' + data_[randn_a[i]] + '\001\002' + data_[randn_b[i]])

                i_ = len(neg_one)

                if i_ > num_one:
                    break

        ### 双词共现
        print('build two same word')
        neg_two = set({})
        num_two = int(data_len * 0.3) + 1
        candidate_words_list = list(candidate_words.keys())
        candidate_two = {}
        for i in range(len(candidate_words_list)):
            if i + 1 == len(candidate_words_list):
                break
            for j in candidate_words_list[i + 1:]:
                tmp_ = word2sen[candidate_words_list[i]] & word2sen[j]
                if len(tmp_) >= 10:
                    candidate_two[(candidate_words_list[i], j)] = tmp_

        i_ = 0
        while i_ < num_two:
            for t_word in candidate_two:
                randn_a = random.sample(candidate_two[t_word],
                                        min(len(candidate_two[t_word]), int(num_two / len(candidate_two)) + 1))
                randn_b = random.sample(candidate_two[t_word],
                                        min(len(candidate_two[t_word]), int(num_two / len(candidate_two)) + 1))
                for i in range(len(randn_a)):
                    f_ = filterData(data_[randn_a[i]], data_[randn_b[i]])
                    if f_ == 2:
                        neg_two.add(
                            '\001'.join(t_word) + "\001\001" + data_[randn_a[i]] + '\001\002' + data_[randn_b[i]])

                i_ = len(neg_two)

                if i_ > num_two:
                    break

        ### 三词共现
        print('build three same word ')
        neg_three = set({})
        num_three = int(data_len * 0.2) + 1
        candidate_words_list = list(candidate_words.keys())
        candidate_three = {}
        # for i in range(len(candidate_words_list)):
        #     if i + 2 == len(candidate_words_list):
        #         break
        #     for j in range(i + 1, len(candidate_words_list)):
        #         if j + 1 == len(candidate_words_list):
        #             break
        #         for h in candidate_words_list[j + 1:]:
        #             tmp_ = word2sen[candidate_words_list[i]] & word2sen[candidate_words_list[j]] & word2sen[h]
        #         if len(tmp_) >= 3:
        #             candidate_three[(candidate_words_list[i], candidate_words_list[j], h)] = tmp_

        ###采用采样的形式
        total_three_num = 0
        tmp_dict = {}
        epoch = 0
        # num_three = 2000
        while total_three_num < num_three:
            a = random.sample(candidate_words_list, len(candidate_words_list))
            b = random.sample(candidate_words_list, len(candidate_words_list))
            c = random.sample(candidate_words_list, len(candidate_words_list))

            for i in range(len(a)):
                if a[i] == b[i] or b[i] == c[i] or a[i] == c[i]:
                    continue

                t_ = [a[i], b[i], c[i]]
                t_.sort()
                t_ = '\001'.join(t_)
                if t_ not in tmp_dict:

                    tmp_dict[t_] = 1

                    tmp_ = word2sen[a[i]] & word2sen[b[i]] & word2sen[c[i]]
                    if len(tmp_) >= 2:
                        # print("Test\t" + t_ + "\t\t" + data_[list(tmp_)[0]] + '\t' + data_[list(tmp_)[1]])
                        candidate_three[t_] = tmp_
                        total_three_num += len(tmp_)

            if total_three_num > num_three:
                break

            epoch += 1
            if epoch > 100:
                print('random three word less')
                break

        print('build three same data ')
        i_ = 0
        epoch = 0
        while i_ < num_three:
            for t_word in candidate_three:
                randn_a = random.sample(candidate_three[t_word],
                                        min(len(candidate_three[t_word]), int(num_three / len(candidate_three)) + 1))
                randn_b = random.sample(candidate_three[t_word],
                                        min(len(candidate_three[t_word]), int(num_three / len(candidate_three)) + 1))
                for i in range(len(randn_a)):
                    f_ = filterData(data_[randn_a[i]], data_[randn_b[i]])
                    if f_ == 2:
                        # print(
                        #     'TestAAA\t\t' + t_word + "\t" + data_[randn_a[i]] + '\t\t' + data_[randn_b[i]] + '\t\t' + str(randn_a[i]) +
                        #     '\t' +
                        #     str(randn_b[i]))
                        neg_three.add(t_word + "\001\001" + data_[randn_a[i]] + '\001\002' + data_[randn_b[i]])

                i_ = len(neg_three)

                if i_ > num_three:
                    break
            epoch += 1
            if epoch > 15:
                print('Three same word less ')
                break
    if type_ != 0:
        neg_ = neg_one | neg_two | neg_three

    # neg_ = neg_three
    fout = open(BASE_DATA_PATH + '/data.train.neg' + str(type_), 'w')
    for k in neg_:
        fout.write(k + '\n')
    fout.close()


from sklearn.model_selection import train_test_split


def makeTrainData(pos_path, neg_path, neg_path_0=''):
    """

    :param pos_path:
    :param neg_path:
    :return:
    """
    pos_ = set({})
    with open(pos_path, 'r')  as f:
        for k in f:
            pos_.add(k.rstrip())
    neg_ = set({})
    with open(neg_path, 'r') as f:
        for k in f:
            tmp_ = k.rstrip().split('\001\001')
            neg_.add(tmp_[1])

    if neg_path_0 != '':
        neg_0 = set({})
        with open(neg_path_0, 'r') as f:
            for k in f:
                neg_0.add(k.rstrip())
        neg_ = random.sample(neg_, int(len(neg_) * 0.9)) + random.sample(neg_0, min(int(len(neg_) * 0.15), len(neg_0)))

    pos_ = list(pos_)
    neg_ = list(neg_)
    data_ = pos_ + neg_
    label_ = [0] * len(pos_) + [1] * len(neg_)
    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = data_
    df['label'] = label_
    df.to_csv(BASE_DATA_PATH + 'processed_data.csv', index=False)

    dataset = df.values
    train, test = train_test_split(dataset, test_size=0.125)
    df = pd.DataFrame(columns=['origin', 'label'], data=train)
    df.to_csv(BASE_DATA_PATH + '/processed_train.csv', index=False)
    df = pd.DataFrame(columns=['origin', 'label'], data=test)
    df.to_csv(BASE_DATA_PATH + '/processed_test.csv', index=False)


# makeTrainPosData('../data/train.data.loc')


# makeTrainNegData('../data/data.log')

# makeTrainData('../data/data.train.pos', '../data/data.train.neg1', '../data/data.train.neg0')
# vocab_build('../data/char.dict')

from dssm.runData import isSameNew
import pandas as pd


def filterSen(sen_1):
    """
    对部分句子进行过滤
    :param sen_1:
    :return:
    """
    tmp_ = [k for k in sen_1.split(' ') if k not in filter_word]
    return ' '.join(tmp_)


def checkTrainData(path_):
    """
    验证训练集质量
    :param path_:
    :return:
    """
    # df = pd.DataFrame(columns=['origin', 'label'])
    data_ = pd.read_csv(path_)

    for i in range(len(data_)):
        txt_ = data_.iloc[i]['origin'].split('\001\002')
        lable_ = data_.iloc[i]['label']
        # flag_, _ = isSame(txt_[0], txt_[1])
        # info_ = [str(k) for k in _]
        # if flag_ == 0 and lable_ == 0:
        #     continue
        # elif flag_ == 1 and lable_ == 1:
        #     continue
        # else:
        #     print(data_.iloc[i]['origin'] + '\001\002' + str(data_.iloc[i]['label']) + '\t' + str(flag_) + '\t' + ' '.join(info_))

        flags_ = isSameNew(txt_[0], txt_[1])
        if flags_[1] + flags_[3] + flags_[5] < 0 and lable_ == 0:
            if txt_[0] in txt_[1] or txt_[1] in txt_[0]:
                continue
            info_ = [str(k) for k in flags_]
            print(data_.iloc[i]['origin'] + '\001\002' + str(data_.iloc[i]['label']) + '\t\t' + ' '.join(info_))


# checkTrainData('../data/train.csv')

import random


def makeHQTrainData(path_csv, path_manua, path_fake):
    '''
    base the manua label data , make new HG  train data
    :return:
    '''

    f_m = open(path_manua, 'r')
    f_f = open(path_fake, 'r')
    dic_manua = {one.rstrip().split('\t\t')[0].split('\002')[0]: one.rstrip().split('\t\t')[0].split('\002')[1] for one in
                 f_m.readlines() if
                 len(
                     one.rstrip(

                     ).split(
                         '\t\t')) > 1}

    dic_fake = {one.rstrip().split('\002')[0]: 1 for one in f_f.readlines()}

    print('manua')

    print('panda')
    pos_ = set({})
    neg_ = set({})
    neg_manua_ = set({})
    pos_manua_ = set({})

    data_ = pd.read_csv(path_csv)
    for i in range(len(data_)):
        txt_ = data_.iloc[i]['origin']
        lable_ = data_.iloc[i]['label']
        if lable_ == 1:
            neg_.add(txt_)
        else:
            d_ = txt_.split('\001\002')
            if d_[0] in dic_manua and dic_manua[d_[0]] == d_[1]:
                pos_manua_.add(txt_)
            elif d_[0] not in dic_manua and d_[0] in dic_fake:
                neg_manua_.add(txt_)
            elif d_[0] not in dic_fake:
                pos_.add(txt_)

    print('pos {},neg {},pos_manua {},neg_manua {}'.format(len(pos_), len(neg_), len(pos_manua_), len(neg_manua_)))
    print('\n'.join(list(neg_manua_)[:10]))

    # make manua Train Data
    train_pos_ = list(pos_manua_) + random.sample(list(pos_), 45000)
    train_neg_ = list(neg_manua_) + random.sample(list(neg_), 45000)

    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = train_pos_ + train_neg_
    df['label'] = [0] * len(train_pos_) + [1] * len(train_neg_)
    df.to_csv(BASE_DATA_PATH + 'train_new.csv', index=False)

# makeHQTrainData('../data/train.csv', '../manua.data.pos', '../fake.data.pos')
# buildWordCab('../data/data.log')
