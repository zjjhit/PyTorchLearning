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
import sys

if len(sys.argv) == 2:
    BASE_PATH = './cluster/'
    BASE_DATA_PATH = './data/'
else:
    BASE_PATH = '../cluster/'
    BASE_DATA_PATH = '../data/'

import pickle

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

            id2sentence[num_] = [tmp_[-2], tmp_[-1]]  # 暂时不使用 list
            cluster2set[num_] = False
            num_ += 1

    return id2sentence, word2sentenceid, cluster2set


def pickleDumpFile(pickname, *awks):
    with open(BASE_PATH + '/' + pickname, 'wb') as f:
        for k in awks:
            pickle.dump(k, f)


def makeDictData(path_, info_=''):
    id2sentence, word2sentenceid, cluster2set = preprocessData(path_)
    pickleDumpFile('baseDictData.pk', id2sentence, word2sentenceid, cluster2set)


import os, torch
from transformers import BertConfig


def model_init():
    model_path = BASE_DATA_PATH + '/use_model/'
    conf = BertConfig.from_pretrained(model_path + '/' + 'init.json')

    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_id
    device = 'cpu'

    char_vocab = pickle.load(open(model_path + conf.char_vocab, 'rb'))
    model_name_char_1 = torch.load(model_path + conf.model_name_char_1).to(device)
    model_name_char_1.eval()

    model_name_char_2 = torch.load(model_path + conf.model_name_char_2).to(device)
    model_name_char_2.eval()

    word_vocab = pickle.load(open(model_path + conf.word_vocab, 'rb'))
    model_name_word = torch.load(model_path + conf.model_name_word).to(device)
    model_name_word.eval()

    model_loc = torch.load(model_path + conf.model_loc).to(device)
    model_loc.eval()

    max_len = conf.max_len

    model_dict = {
        'name_char_1': model_name_char_1,
        'name_char_2': model_name_char_2,
        'name_word': model_name_word,
        'loc': model_loc
    }

    return model_dict, char_vocab, word_vocab, max_len


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


from dssm.data_process import cleanWord


def dataPro(s1, s2, max_len, vocab, flag_=False):
    '''

    :param s1:
    :param s2:
    :param max_len:
    :param vocab:
    :param flag_: segment_type  True-->word level
    :return:
    '''

    def clean_(sen_):
        tmp_ = []
        for k in sen_.split():
            if flag_:
                k = cleanWord(k)
            tmp_.append(k)

        if len(tmp_) <= 5:
            tmp_ = tmp_ * 2

        return ' '.join(tmp_)

    s1 = clean_(s1)
    s2 = clean_(s2)

    q = s1[:min(len(s1), max_len)]
    d = s2[:min(len(s2), max_len)]

    if flag_:
        q = q.split()
        d = d.split()

    q = convert_tokens_to_ids(q, vocab)
    q = q + [0] * (max_len - len(q))

    d = convert_tokens_to_ids(d, vocab)
    d = d + [0] * (max_len - len(d))

    return q, d


def isSameModel(data_, model):
    """
    基于模型处理相似度
    :param data_:
    :param model:
    :return:
    """
    data_ = {'query_': torch.tensor(data_[0]).unsqueeze(0), 'doc_': torch.tensor(data_[1]).unsqueeze(0)}
    with torch.no_grad():
        pred = model(data_)
        k_ = torch.max(pred, 1)[1][0]
        return k_.data.item()


def sameLogic(data_1, data_2, model_dict, char_vocab, word_vocab, max_len):
    """
    基于名称与地址的相似判定逻辑
    :param sim_1:
    :param sim_2:
    :return:
    """

    name_1, name_2 = dataPro(data_1[0], data_2[0], max_len, char_vocab)
    name_1_w, name_2_w = dataPro(data_1[0], data_2[0], max_len, word_vocab, True)

    name_sim_char_1 = isSameModel([name_1, name_2], model_dict['name_char_1'])  # model 有偏，后续可优化
    name_sim_word = isSameModel([name_1_w, name_2_w], model_dict['name_word'])

    if len(data_1[1]) <= 5 or len(data_2[1]) <= 5:  ### 地址为空 则略过
        loc_sim = -1
    else:
        loc_1, loc_2 = dataPro(data_1[1], data_2[1], max_len, char_vocab)
        loc_sim = isSameModel([loc_1, loc_2], model_dict['loc'])

    print('Test processSet', data_1, data_2, name_sim_char_1, name_sim_word, loc_sim)

    if name_sim_char_1 + name_sim_word == 0 and loc_sim <= 0:
        return True
    # 对此部分的处理逻辑有待评测
    elif name_sim_char_1 + name_sim_word == 0 and loc_sim == 1:
        name_sim_char_2 = isSameModel([name_1, name_2], model_dict['name_char_2'])  # model 有偏，后续可优化
        if name_sim_char_2 == 0:
            return True
        else:
            return False
    elif name_sim_char_1 + name_sim_word == 1 and loc_sim <= 0:  ### 基于地址相似判定相似
        name_sim_char_2 = isSameModel([name_2, name_1], model_dict['name_char_2'])  # model 有偏，后续可优化
        if name_sim_char_2 == 0:
            return True
        else:
            return False
    else:
        return False


def infoSense(path_):
    '''
    Tmp
    :param path_:
    :return:
    '''
    with open(path_, 'rb') as f:
        id2sentence = pickle.load(f)
        word2sentenceid = pickle.load(f)
        cluster2set = pickle.load(f)
        #
        # fout = open('./tmp.info', 'w')
        # for k in word2sentenceid:
        #     fout.write('{}-->{}\n'.format(k, len(word2sentenceid[k])))
        # fout.close()

        fout = open('./tmp_info.info', 'w')
        for k in word2sentenceid:
            if len(word2sentenceid[k]) < 500:
                fout.write('{}-->{}\n'.format(k, ' '.join([str(one) for one in list(word2sentenceid[k])])))
        fout.close()


###########################################################################################


def processSet(sen_set, id2sentence, model_dict, char_vocab, word_vocab, max_len):
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
        for k in range(1, len(tmp_list)):
            if sameLogic(id2sentence[a], id2sentence[tmp_list[k]], model_dict, char_vocab, word_vocab,
                         max_len):  # 核心  相似判定
                flag_.append(tmp_list[k])
                a_set.add(tmp_list[k])

        flag_.append(a)
        for one in flag_:
            tmp_list.remove(one)

        set_list.append(a_set)

    return set_list


def processCluster(word_list, word2sentenceid, id2sentence, cluster_set):
    """
    针对word2set的计算结果，更新cluster2set
    :param word_list:  子聚类结果 列表
    :param cluster_set: sentenceid --> ???
    :return:
    """

    model_dict, char_vocab, word_vocab, max_len = model_init()

    cluster_info_ = {}

    for word_ in word_list:  ###对于word 的顺序要有要求
        word_sen_set = word2sentenceid[word_]
        set_list = processSet(word_sen_set, id2sentence, model_dict, char_vocab, word_vocab, max_len)  # 分割成不同的集合
        for set_ in set_list:
            for id_ in set_:
                if cluster_set[id_] == False:
                    cluster_set[id_] = set_
                    cluster_info_[id_] = [word_]
                elif set_ == cluster_set[id_]:
                    continue
                else:
                    # 原有与 现有集合的关系
                    merge_ = set_ | cluster_set[id_]
                    if len(merge_) > len(cluster_set[id_]):
                        cluster_set[id_] = merge_
                        for id_m in merge_:
                            cluster_set[id_m] = merge_  # 此步骤为粗筛
                            cluster_info_[id_].append(word_)

    pickleDumpFile('cluster_set_{}.pk'.format(len(word_list)), cluster_set)

    return cluster_info_


def runPart(path_=BASE_PATH + '/baseDictData.pk', word_sen_path=BASE_PATH + '/word_sen_num'):
    with open(path_, 'rb') as f:
        id2sentence = pickle.load(f)
        word2sentenceid = pickle.load(f)
        cluster2set = pickle.load(f)

    word_dic = {}
    with open(word_sen_path, 'r') as f:
        for one in f:
            tmp_ = one.rstrip().split('\t')
            word_dic[tmp_[0]] = int(tmp_[1])

    word_list = []
    threld = [5, 100]
    for k in word_dic:
        if word_dic[k] >= threld[0] and word_dic[k] <= threld[1]:
            word_list.append(k)

    # tmp_list = random.sample(word_list, 100)
    # XXX TODO
    tmp_list = ['KARRSEN', 'Devices', 'ZHONGSAI', 'JINWEIXIN', 'XUYA', 'JIAOWAY', 'SMARTTSAI', 'SANITY', 'CHEERIO', 'House',
                'MEIBAOJIE', 'TROLAND', 'LINGGU', 'SMITT', 'EQUIPMENT&TECHNOLOGY', 'DOLLY', 'KONG,CHINA', 'GAOLU', 'FIRMSTOCK',
                'JAWNA', 'PLATIRID', 'SCHIEFFER', 'MEEZAN', 'JIEMAOIMPORTEX', 'GMBHNEULANDER', 'KAOFU', 'KHIND', 'PATTYN', 'PLASTIE',
                'ЛЕСОПРОМЫШЛЕННОСТЬ"', 'LILIANA', 'AFRIKON', 'Liqiang', 'KERSEN', 'CDN', 'ШЕНГ', 'CAFFE', 'EVERYTHING', 'SUNSEEKER',
                'SHZ', 'SEPSTAR', 'LEADERMOUNT', 'SONGDA', 'SENBA', 'ROSTE', 'FEIFEI', 'SINLY', 'CORPORATION/HUAFANG', 'BUDWEISER',
                'DUOMI', 'WELLOAD', 'HANDTOOLS&HARDWARE', 'GUZNGZHOU', '2-5F', 'BAGS&APPAREL', 'BEINGMATE', 'VIBOTEX', 'ITTA',
                'RAILING&FENCING', 'NONGYE', 'SCLENCE', 'ARLANXEO', 'KAIXIANGTONG', 'TYP', '(RELIANCE)', 'GUANGDONG.', 'HONGCAO',
                'BONATE', 'TSINKING', 'STEINEMANN', 'РЕШЕНИЯ', 'HEXIHE', 'BUSIESS', 'KATAMAN', 'ENGLISH', 'J.TOP', '"ЛЮЛИН"', 'NEWDA',
                'Center,', 'SCARLETT', 'ALMUQADIMAH', 'TISHIELD', 'BOHOLY', 'GLODEN', 'VONWELT', 'MASTER&FRANK', 'LABORATORIES(INNER',
                'DARVEEN', 'NORWICH', 'JOINTPOWER', 'AMERAS', 'KOLN', 'QSSIELECTRIC', 'GILMAN', 'ELECTRONICS(HUI', 'MEFU',
                'KANGCHUNAN', 'IND.CO.,LTD.', 'SINO-CHTC', 'SHUANGMU']

    # tmp_list = ['DUOMI']

    cluster_info_ = processCluster(tmp_list, word2sentenceid, id2sentence, cluster2set)

    infoCluster(tmp_list, cluster2set, id2sentence, cluster_info_)


def infoCluster(word_list, cluster2set, id2sentence, cluster_info_):
    '''
    打印相关聚类
    :param cluster2set:
    :param id2sentence:
    :return:
    '''
    print('\t'.join(word_list))

    info_ = set({})
    for k in cluster2set:

        if cluster2set[k] == False:
            continue

        t_set = tuple(cluster2set[k])
        if t_set in info_:
            continue
        else:
            info_.add(t_set)
            line_ = []
            h_ = str(hash(t_set))
            for id_ in cluster2set[k]:
                line_.append(h_ + '\001\002' + '\t'.join(cluster_info_[id_]) + '\001\001' + '\001\001'.join(id2sentence[id_]))
            if len(line_) < 10:
                print('\n'.join(line_))


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


#############TMP######
import random
import pandas as pd


# from dssm.runData import isSameNew


def makePosTestData(path_):
    """
    构建地点信息训练数据集，用于测试 基于名称的模型效果
    :param path_:
    :return:
    """

    dic_ = {}
    pos_ = []
    num_ = 20000
    with open(path_, 'r')  as f:
        for one in f:
            tmp_ = one.rstrip()
            if len(tmp_) < 3:
                continue
            tmp_ = tmp_.split('\001\002')
            loc_ = [k for k in tmp_[1].split('\002') if k != 'NULL' and len(k) > 5]
            if len(loc_) < 1:
                continue
            if len(loc_) > 5 and num_ > 0:
                t_ = random.sample(loc_, 2)
                flags_ = isSameNew(t_[0], t_[1])
                if flags_[1] * flags_[3] * flags_[5] > 0:
                    # if distance_jacaard(t_[0], t_[1]) > 0.5:
                    pos_.append('\001\002'.join(t_))
                    num_ -= 1
            dic_[tmp_[0]] = loc_

    neg_ = []
    key_list = dic_.keys()
    while num_ < 20000:
        tmp_ = random.sample(key_list, 2)
        if dic_[tmp_[0]][0] != dic_[tmp_[1]][0] and len(dic_[tmp_[0]][0]) > 10 and len(dic_[tmp_[1]][0]) > 10:
            neg_.append(dic_[tmp_[0]][0] + "\001\002" + dic_[tmp_[1]][0])
            num_ += 1

    df = pd.DataFrame(columns=['origin', 'label'])
    df['origin'] = pos_ + neg_
    df['label'] = [0] * len(pos_) + [1] * len(neg_)
    df.to_csv('../data/train_loc.csv', index=False)


if __name__ == '__main__':
    pass
    # makeDictData('../data/data.log')
    # infoSense(BASE_PATH + '/baseDictData.pk')
    # makePosTestData('../data/posi.data')

    # a = model_init()

    runPart()
