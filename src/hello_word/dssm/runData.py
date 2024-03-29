# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         runData
# Description:  
# Author:       lenovo
# Date:         2020/9/14
# -------------------------------------------------------------------------------

'''
集成相关的方法，完成初始版本结果
'''

import distance
from scipy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer


def distance_edit(s1, s2):
    l = distance.levenshtein(s1, s2)
    return l / (len(s1) + len(s2))


def distance_jacaard(s1, s2):
    """

    :param s1:
    :param s2:
    :return:
    """
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator


def distance_tf(s1, s2):
    """

    :param s1:
    :param s2:
    :return:
    """
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


# ================================================================================================
filter_word = set(
    {'LTD', 'CO.,LTD', 'LIMITED', 'LTD.', 'CO', 'CO.,LTD.', 'CO.LTD', 'COMPANY', 'GROUP', 'INC.', 'CO.LTD.', 'CO.',
     'CO.,', 'CO.,',
     'LIMITED', 'LIMITED.', 'LTD,', 'LTD.,', 'LIMITE', '.,LTD', ',LTD', 'LTD.', 'LLC.', 'CO..LTD.'})


def preprocessData(path_):
    """

    :param path_:
    :return:
    """
    id2sentence = {}  # id-> sentence,flag_
    word2sentenceid = {}  # word->sentenceid
    cluster2set = {}  ## sentenceid --> sameSet
    num_ = 0
    with open(path_, 'r') as f:
        for one in f:
            tmp_ = one.rstrip().split('\001\002')
            if len(tmp_) != 5 or tmp_[-2] == 'NULL':
                continue
            for k in tmp_[-2].split():
                if k not in filter_word:
                    if k not in word2sentenceid:
                        word2sentenceid[k] = set({num_})
                    else:
                        word2sentenceid[k].add(num_)

            id2sentence[num_] = [tmp_[-2], False]  # 暂时不使用 list
            cluster2set[num_] = False
            num_ += 1

    return id2sentence, word2sentenceid, cluster2set


def isSame(s1, s2):
    """
    判定集合， 判定两个句子是否相等
    :param s1:
    :param s2:
    :return:
    """
    '''
    dis_tf_0,0.6921163264951035,0.024525917547916054,0.15660835805931944
    dis_tf_1,0.23781942784513277,0.016627282964165754,0.12894743183495194
    
    dis_edt_0,4.064512332198533,79.8665607628803,8.936833025622311
    dis_edt_1,17.101864891869035,413.114292867084,20.325261274503497
    
    dis_edt_0,0.09731670026484407,0.012646541083219937,0.11245743766781173
    dis_edt_1,0.38612644044545114,0.008270554715730887,0.090943020153682
    
    dis_jaca_0,0.5405942759052353,0.03168846957404664,0.17801349906205652
    dis_jaca_1,0.13869462282078243,0.007344366661623194,0.08569968554996209
    '''
    threshold_ = {'tf': {0: {'mean': 0.692, 'var': 0.024, 'std_var': 0.156},
                         1: {'mean': 0.237, 'var': 0.016, 'std_var': 0.128}},
                  'edt': {0: {'mean': 0.1, 'var': 0.012, 'std_var': 0.112},
                          1: {'mean': 0.386, 'var': 0.008, 'std_var': 0.091}},
                  'jaca': {0: {'mean': 0.54, 'var': 0.031, 'std_var': 0.178},
                           1: {'mean': 0.128, 'var': 0.007, 'std_var': 0.085}}}

    edt = distance_edit(s1, s2)
    tf = distance_tf(s1, s2)
    jaca = distance_jacaard(s1, s2)

    flag_0 = [threshold_['tf'][0]['mean'] <= tf, threshold_['edt'][0]['mean'] >= edt, threshold_['jaca'][0]['mean'] <= jaca]

    if flag_0.count(True) >= 2:
        return True

    flag_3 = [threshold_['tf'][1]['mean'] > tf, threshold_['edt'][0]['mean'] < edt, threshold_['jaca'][0]['mean'] > jaca]
    if flag_3.count(True) >= 2:
        return False

    flag_1 = [threshold_['tf'][0]['mean'] - threshold_['tf'][0]['std_var'] <= tf, \
              threshold_['edt'][0]['mean'] - threshold_['edt'][0]['std_var'] >= edt, \
              threshold_['jaca'][0]['mean'] - threshold_['jaca'][0]['std_var'] <= jaca]

    if flag_1.conut(True) >= 2:
        return True


def processSet(sen_set, id2sentence):
    """
    处理set集合，获取子聚类类别
    :param sen_set:
    :return:
    """
    tmp_list = list(sen_set)
    set_list = []
    while len(tmp_list) >= 1:
        a = tmp_list[0]
        a_set = set({a})

        flag_ = []
        for k in range(1, len(tmp_list[1:])):
            if isSame(id2sentence[a][0], id2sentence[tmp_list[k]][0]):
                flag_.append(tmp_list[k])
                a_set.add(tmp_list[k])

        flag_.append(a)
        for one in flag_:
            tmp_list.remove(one)

        set_list.append(a_set)

    return set_list


def processCluster(set_list, cluster_set, info_=''):
    """
    针对word2set的计算结果，更新cluster2set
    :param set_list:
    :param cluster_set: sentenceid --> ???
    :return:
    """
    for tmp_ in set_list:
        for id_ in tmp_:
            if not cluster_set[id_]:
                cluster_set[id_] = [tmp_]
            else:
                # print(cluster_set[id_], tmp_)
                flag_ = 0
                for i in range(len(cluster_set[id_])):
                    if len(cluster_set[id_][i] & tmp_) * 2 >= min(len(tmp_), len(cluster_set[id_][i])):
                        cluster_set[id_][i] = cluster_set[id_][i] | tmp_
                        flag_ = 1
                        break
                if flag_ == 0:
                    cluster_set[id_].append(tmp_)

    print(BASE_PATH + 'cluster.{}.pk'.format(info_))
    pickleDumpFile(BASE_PATH + 'cluster.{}.pk'.format(info_), cluster_set)
    return BASE_PATH + 'cluster.{}.pk'.format(info_)


BASE_PATH = '../cluster/'
import pickle


def pickleDumpFile(pickname, *awks):
    with open(BASE_PATH + '/' + pickname, 'wb') as f:
        for k in awks:
            pickle.dump(k, f)


def runCluster(path_, info_=''):
    id2sentence, word2sentenceid, cluster2set = preprocessData(path_)
    pickleDumpFile('baseDicta.pk', id2sentence, word2sentenceid, cluster2set)

    # for word in word2sentenceid:
    #     word_ssen_et = word2sentenceid[word]
    #     set_list = processSet(word_ssen_et, id2sentence)
    #     processCluster(set_list, cluster2set)


# =============================================================================

import multiprocessing
import copy


def arr_size(arr, size_):
    s = []
    for i in range(0, int(len(arr)), size_):
        l_ = min(i + size_, len(arr))
        c = arr[i:l_]
        if c != []:
            s.append(c)
    return s


def func(word_list, word2sentenceid, id2sentence, cluster2set, info):
    for i, word in enumerate(word_list):
        word_ssen_et = word2sentenceid[word]
        if len(word_ssen_et) > 15:
            continue
        print(i, info, word)
        set_list = processSet(word_ssen_et, id2sentence)
    return processCluster(set_list, cluster2set, str(info))


# import multiprocessing.Process as Process

import random


def multiRun():
    process_num = 3
    with open(BASE_PATH + '/baseDict.pk', 'rb') as f:
        id2sentence = pickle.load(f)
        word2sentenceid = pickle.load(f)
        cluster2set = pickle.load(f)

    word_list = random.sample(list(word2sentenceid.keys()), 30000)
    wordlist_len = int(len(word_list) / process_num)

    w_list = arr_size(word_list, 20)
    print('w_list', len(w_list), len(word_list))

    pool = multiprocessing.Pool(processes=process_num + 2)
    result_ = []
    for i in range(process_num):
        msg = "run %d" % (i)
        print(msg, len(w_list[i]))
        c_cluster2set = copy.deepcopy(cluster2set)
        c_word2sentenceid = copy.deepcopy(word2sentenceid)
        c_id2sentence = copy.deepcopy(id2sentence)

        result_.append(pool.apply_async(func, args=(w_list[i], c_word2sentenceid, c_id2sentence, c_cluster2set, i,)))

    pool.close()
    pool.join()

    for k in result_:
        print(k.get())

    # ps = []
    # for i in range(process_num):
    #     msg = "run %d" % (i)
    #     print(msg, len(w_list[i]))
    #     c_cluster2set = copy.deepcopy(cluster2set)
    #     c_word2sentenceid = copy.deepcopy(word2sentenceid)
    #     c_id2sentence = copy.deepcopy(id2sentence)
    #
    #     p = Process(target=func, name="worker" + str(i), args=(w_list[i], c_word2sentenceid, c_id2sentence, c_cluster2set, i))
    #     ps.append(p)
    #
    # for i in range(process_num):
    #     ps[i].start()
    #
    # for i in range(process_num):
    #     ps[i].join()


# ===============================================================================================
import numpy as np


def getThreshold(path_):
    dis_tf = {0: [], 1: []}
    dis_edt = {0: [], 1: []}
    dis_jaca = {0: [], 1: []}

    dataset = pd.read_csv(path_)  # processed_train.csv
    # dataset = dataset.iloc[range(len(dataset))]
    for i in range(len(dataset)):
        if i % 2000 == 0:
            print(i)
        d_ = dataset.iloc[i]
        txt_ = d_['origin'].split('\001\002')

        dis_jaca[d_['label']].append(distance_jacaard(txt_[0], txt_[1]))

        dis_tf[d_['label']].append(distance_tf(txt_[0], txt_[1]))

        dis_edt[d_['label']].append(distance_edit(txt_[0], txt_[1]))

    with open(BASE_PATH + '/thredhold.pk', 'wb') as f:
        pickle.dump(dis_tf, f)
        pickle.dump(dis_edt, f)
        pickle.dump(dis_jaca, f)

    l = npInfo(dis_tf[0])
    print('dis_tf_0,{},{},{}'.format(l[0], l[1], l[2]))
    l = npInfo(dis_tf[1])
    print('dis_tf_1,{},{},{}'.format(l[0], l[1], l[2]))

    l = npInfo(dis_edt[0])
    print('dis_edt_0,{},{},{}'.format(l[0], l[1], l[2]))
    l = npInfo(dis_edt[1])
    print('dis_edt_1,{},{},{}'.format(l[0], l[1], l[2]))
    #
    l = npInfo(dis_jaca[0])
    print('dis_jaca_0,{},{},{}'.format(l[0], l[1], l[2]))
    l = npInfo(dis_jaca[1])
    print('dis_jaca_1,{},{},{}'.format(l[0], l[1], l[2]))
    #


def npInfo(a_):
    l_ = [np.mean(a_), np.var(a_), np.std(a_, ddof=1)]
    return l_


# ===============================================================================================


def test(s1, s2):
    print(distance_jacaard(s1, s2))
    print(distance_tf(s1, s2))
    print(distance_edit(s1, s2))


tmp_ = ['BIGBYTE', 'LIMITED./SHINER', 'HISAFEW', 'NO162', 'Muinjanovna"', 'APPLAINC', 'LTD701', 'C/O,', 'EHAOLOGY',
        'SHENZHENHUASHUNCHANGTOYSCO', 'LTD./ON', '1211,CO', 'LONGDISTRICT', 'INT`T', 'TRADENO6', '401,FL.4,BUILDING', 'ALLASFE',
        'HONGFUYU', 'PATHY', 'HSINCHU', '207,A', 'PROCES', 'TZ1276808', 'DEYICHENG', '1616,', 'MIRRACK', 'HONE-STRONG', 'ZHENCUI',
        'ICELL', '05/11/20', 'ZHAODA', 'ПРОИЗ-ВА', 'PROCESS-', 'Harbin', 'FREYESLEBENSTRASSE', 'TECHNOLOGY(SUZHOU)CO.LTD.',
        'SHANGSHAO', 'KUNZHUO', 'VLAARDINGEN', 'CO..LTD.JINAN', '"РЕДЕКС', 'TECHCHEM', 'ELOVTROPNIC', '200124,', 'CHATINGBEI', 'Jukun',
        'MARUKOME', 'KNUTHTRADE', 'CUMSUN']


def mergeData(words):
    """
    基于原始的word 进行 数据合并
    :param words:
    :return:
    """
    with open(BASE_PATH + '/baseDict.pk', 'rb') as f:
        id2sentence = pickle.load(f)
        word2sentenceid = pickle.load(f)
        cluster2set = pickle.load(f)

    c_0 = pickle.load(open(BASE_PATH + '/cluster.0.pk', 'rb'))
    c_1 = pickle.load(open(BASE_PATH + '/cluster.1.pk', 'rb'))
    c_2 = pickle.load(open(BASE_PATH + '/cluster.2.pk', 'rb'))

    for w in words:
        sen_ = word2sentenceid.get(w, False)
        print(w, sen_)
        for sen_id in sen_:
            t_ = c_0.get(sen_id, False)
            if t_:
                for k in t_:
                    print(k)
                print('List')


def pianMen():
    with open(BASE_PATH + '/baseDicta.pk', 'rb') as f:
        id2sentence = pickle.load(f)
        word2sentenceid = pickle.load(f)
        cluster2set = pickle.load(f)

    words_ = random.sample(word2sentenceid.keys(), 5000)
    fout = open('tmp_result', 'w')
    for word in words_:
        set_ = word2sentenceid[word]
        print(word, len(set_))
        if len(set_) >= 2 and len(set_) <= 1000:
            result_ = processSet(set_, id2sentence)
            num_ = 0
            for s_ in result_:
                print(word, s_)
                tmp_ = ''
                for id_ in s_:
                    tmp_ += id2sentence[id_][0] + '\001\002'
                fout.write(word + '\001\001' + str(num_) + '\001\001' + tmp_.rstrip('\001\002') + '\n')
    fout.close()


import pandas as pd


def writeExcel(path_):
    df = pd.DataFrame(columns=['class_name', 'text'])

    name_ = []
    text_ = []
    dic_ = {}
    with open(path_, 'r') as f:
        for one in f:
            tmp_ = one.rstrip().split('\001\001')
            if len(tmp_) != 3:
                continue
            if tmp_[0] not in dic_:
                dic_[tmp_[0]] = [tmp_[2]]
            else:
                dic_[tmp_[0]].append(tmp_[2])

    for k in dic_:
        for i, a in enumerate(dic_[k]):

            txt_ = a.split('\001\002')
            for t in txt_:
                name_.append(k + '_' + str(i))
                text_.append(t)

    print(len(name_))
    df['class_name'] = name_
    df['text'] = text_

    df.to_excel('sample.xlsx')


######################################

import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cpu'
BASE_DATA_PATH = '../data/'
vacab = pickle.load(open(BASE_DATA_PATH + '/char2id.vocab', 'rb'))
model = torch.load(BASE_DATA_PATH + '/final_model_4_0_100ford.pt').to(device)
model.eval()
max_len = 64


def convert_tokens_to_ids(query, vocab):
    ids_ = []
    for one in query:
        if one.isalpha():
            one = one.upper()
        if one in vocab:
            ids_.append(vocab[one])
        else:
            ids_.append(vocab['<UNK>'])
    return ids_


def singleOne(s1, s2):
    q = s1[:min(len(s1), max_len)]
    d = s2[:min(len(s2), max_len)]

    q = convert_tokens_to_ids(q, vacab)
    q = q + [0] * (64 - len(q))

    d = convert_tokens_to_ids(d, vacab)
    d = d + [0] * (64 - len(d))

    data_ = {'query_': torch.tensor(q).unsqueeze(0), 'doc_': torch.tensor(d).unsqueeze(0)}
    with torch.no_grad():
        y_pred = model(data_)
        # print(y_pred)
        k_ = torch.max(y_pred, 1)[1][0]
        return k_.data.item()
    return 1


def processSetTwo(sen_list):
    """
    处理set集合，获取子聚类类别
    :param sen_set:  sen_list的集合
    :return:
    """
    tmp_list = list(sen_list)
    set_list = []
    while len(tmp_list) >= 1:
        a = tmp_list[0]
        a_set = [a]

        flag_ = []
        for k in range(1, len(tmp_list[1:])):
            if singleOne(a, tmp_list[k]) == 0 and distance_edit(a, tmp_list[k]) < 0.35:
                flag_.append(tmp_list[k])
                a_set.append(tmp_list[k])

        flag_.append(a)
        for one in flag_:
            # print(one,tmp_list)
            tmp_list.remove(one)
        set_list.append(a_set)

    return set_list


def XXC(path_):
    dic_ = {}
    with open(path_, 'r') as f:
        for one in f:
            tmp_ = one.rstrip().split('\001\001')
            if len(tmp_) != 3:
                continue
            if tmp_[0] not in dic_:
                dic_[tmp_[0]] = tmp_[2].split('\001\002')
            else:
                dic_[tmp_[0]] += tmp_[2].split('\001\002')

    all_ = 0
    for k in dic_:
        print(k, len(dic_[k]))
        all_ += len(dic_[k])
    print(all_)

    name_ = []
    text_ = []
    df = pd.DataFrame(columns=['class_name', 'text'])
    for k in dic_:
        result_ = processSetTwo(dic_[k])
        for i, l_ in enumerate(result_):
            for one in l_:
                name_.append(k + '_' + str(i))
                text_.append(one)

    print(len(name_))
    df['class_name'] = name_
    df['text'] = text_

    df.to_excel('sample.xlsx')


if __name__ == '__main__':
    pass
    # multiRun()

    # runCluster('../data/data.log')
    # pianMen()
    # writeExcel('tmp_result')
    XXC('tmp_result')
    a = 'ROHM AND HAAS INTERNATIONAL TRADE SHANGHAI CO LTD'
    b = 'ROHM AND HAAS  (FOSHAN) SPECIALTY MATERIALS CO., LTD.'
    print(singleOne(a, b))
    print(isSame(a, b))
    print(distance_edit(a, b))
