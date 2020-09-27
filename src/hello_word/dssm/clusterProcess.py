# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         clusterProcess
# Description:  
# Author:       lenovo
# Date:         2020/9/27
# -------------------------------------------------------------------------------

'''
基于 公司名称 与  地址信息的 名称聚类 过程
基于三个数据词典：
1、id2sentence:id->sen
2、word2sentenceid:word->sen_id
3、cluster2set:sen_id -> cluster_set
'''

# ========================
filter_word = set(
    {'LTD', 'CO.,LTD', 'LIMITED', 'LTD.', 'CO', 'CO.,LTD.', 'CO.LTD', 'COMPANY', 'GROUP', 'INC.', 'CO.LTD.', 'CO.',
     'CO.,', 'CO.,',
     'LIMITED', 'LIMITED.', 'LTD,', 'LTD.,', 'LIMITE', '.,LTD', ',LTD', 'LTD.', 'LLC.', 'CO..LTD.', 'CO.,LIMITED'})


def preprocessData(path_):
    """
    构建上述三个词典数据
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

            id2sentence[num_] = tmp_[-2]  # 暂时不使用 list
            cluster2set[num_] = False
            num_ += 1

    return id2sentence, word2sentenceid, cluster2set


BASE_PATH = '../cluster/'
import pickle


def pickleDumpFile(pickname, *awks):
    with open(BASE_PATH + '/' + pickname, 'wb') as f:
        for k in awks:
            pickle.dump(k, f)


def getDictData(path_, info_=''):
    id2sentence, word2sentenceid, cluster2set = preprocessData(path_)
    pickleDumpFile('baseDictData.pk', id2sentence, word2sentenceid, cluster2set)


###########################################################################################

from dssm.runData import isSame


def processSet(sen_set, id2sentence):
    """
    处理set集合，获取子聚类类别
    :param sen_set:  sen_id_set
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
    :param set_list:  子聚类结果 列表
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


if __name__ == '__main__':
    pass
