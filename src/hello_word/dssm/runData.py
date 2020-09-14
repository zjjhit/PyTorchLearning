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

            id2sentence[num_] = [tmp_[-2], False]
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
    if flag_3.count(True) >= 1:
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
            if isSame(id2sentence[a], id2sentence[tmp_list[k]]):
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
    :param cluster_set:
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

    pickleDumpFile(BASE_PATH + 'cluster.{}.pk'.format(info_), cluster_set)


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

process_num = 1


def arr_size(arr, size_):
    s = []
    for i in range(0, int(len(arr)), size_):
        l_ = min(i + size_, len(arr))
        c = arr[i:l_]
        if c != []:
            s.append(c)
    return s


def func(word_list, word2sentenceid, id2sentence, cluster2set, i):
    for word in word_list:
        word_ssen_et = word2sentenceid[word]
        set_list = processSet(word_ssen_et, id2sentence)
        processCluster(set_list, cluster2set, str(i))


def multiRun():
    with open(BASE_PATH + '/baseDict.pk', 'rb') as f:
        id2sentence = pickle.load(f)
        word2sentenceid = pickle.load(f)
        cluster2set = pickle.load(f)

    word_list = list(word2sentenceid.keys())
    wordlist_len = int(len(word_list) / process_num)

    w_list = arr_size(word_list, wordlist_len)
    # print(len(word_list))

    pool = multiprocessing.Pool(processes=process_num)

    for i in range(process_num):
        msg = "run %d" % (i)
        print(msg)
        c_cluster2set = copy.deepcopy(cluster2set)
        pool.apply_async(func, (w_list[i], word2sentenceid, id2sentence, c_cluster2set, i))

    pool.close()
    pool.join()


# ===============================================================================================
import pandas as pd
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


if __name__ == '__main__':
    pass
    # s1 = 'YUEQING ZHENGLI MACHINERY CO LTD'
    # s2 = 'YUEQING XINDALI AND EXPORT CO.LTD'
    #
    # test(s1, s2)
    #
    # s1 = 'HANGZHOU SOYANG TECHNOLOGIES CO LTD'
    # s2 = 'HANGZHOU SOYANG TECH CO LTD'
    # test(s1, s2)
    #
    # k_ = {i: str(k) for i, k in enumerate([1])}
    #
    # b = processSet(set(k_.keys()), k_)
    # for k in b:
    #     print(k)

    # runCluster(BASE_PATH + '/cluster.csv')
    #
    # id2sen = pickle.load(open(BASE_PATH + '/baseDict.pk', 'rb'))
    # dic_ = pickle.load(open(BASE_PATH + '/cluster.pk', 'rb'))
    # for k in dic_:
    #     if len(dic_[k]) >= 2:
    #         tmp_ = ''
    #         for i in dic_[k]:
    #             tmp_ += '\t' + id2sen[i][0] + '\001\001' + str(i)
    #         print(tmp_)

    # getThreshold('../data/train.csv')
    # runCluster('../data/data.log')
    multiRun()
