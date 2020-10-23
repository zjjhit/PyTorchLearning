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
from dssm.utils import SegmentWord


def model_init():
    model_path = BASE_DATA_PATH + '/use_model/'
    conf = BertConfig.from_pretrained(model_path + '/' + 'init.json')

    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('{}_{}_{}_{}'.format(conf.model_name_char_1, conf.model_name_char_2, conf.model_name_word, conf.model_loc))

    char_vocab = pickle.load(open(model_path + conf.char_vocab, 'rb'))
    model_name_char_1 = torch.load(model_path + conf.model_name_char_1).to(device)
    model_name_char_1.eval()

    model_name_char_2 = torch.load(model_path + conf.model_name_char_2).to(device)
    model_name_char_2.eval()

    # word_vocab = pickle.load(open(model_path + conf.word_vocab, 'rb'))
    model_name_word = torch.load(model_path + conf.model_name_word).to(device)
    model_name_word.eval()

    model_loc = torch.load(model_path + conf.model_loc).to(device)
    model_loc.eval()

    max_len = conf.max_len

    segment_ = SegmentWord(model_path + conf.segment_model)
    segment_loc = None
    if 'word' in conf.model_loc:
        segment_loc = SegmentWord(model_path + conf.segment_loc_model)

    model_dict = {
        'name_char_1': model_name_char_1,
        'name_char_2': model_name_char_2,
        'name_word': model_name_word,
        'loc': model_loc,
        'segment': segment_,
        'segment_loc': segment_loc,
        'pad_id': 3
    }

    return model_dict, char_vocab, max_len, device


def convert_tokens_to_ids_char(query, vocab):
    ids_ = []
    for one in query:
        if one.isalpha():
            one = one.upper()
        if one in vocab:
            ids_.append(vocab[one])
        else:
            ids_.append(vocab['<UNK>'])
    return ids_


from dssm.data_process import cleanWord, cleanFilterList


def dataProcessWord(s1, s2, max_len, max_word, pad_id, segment_):
    q_ = segment_.encodeAsIds(s1[:min(len(s1), max_len)])
    if len(q_) > max_word:
        q_ = q_[:max_word]
    else:
        q_ = q_ + [pad_id] * (max_word - len(q_))

    d_ = segment_.encodeAsIds(s2[:min(len(s2), max_len)])
    if len(d_) > max_word:
        d_ = d_[:max_word]
    else:
        d_ = d_ + [pad_id] * (max_word - len(d_))

    return q_, d_


def dataProcessChar(s1, s2, max_len, vocab, clean_flag_=False):
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
            if k in cleanFilterList:
                continue
            k = cleanWord(k)
            tmp_.append(k)

        if len(tmp_) <= 3:
            tmp_ = tmp_ * 2

        return ' '.join(tmp_)

    # if flag_:
    #     s1 = clean_(s1)
    #     s2 = clean_(s2)

    q = s1[:min(len(s1), max_len)]
    d = s2[:min(len(s2), max_len)]

    q = convert_tokens_to_ids_char(q, vocab)
    q = q + [0] * (max_len - len(q))  # 0 pad_id

    d = convert_tokens_to_ids_char(d, vocab)
    d = d + [0] * (max_len - len(d))

    return q, d


def isSameModel(data_, model, device):
    """
    基于模型处理相似度
    :param data_:
    :param model:
    :return:
    """
    data_ = {'query_': torch.tensor(data_[0]).unsqueeze(0).to(device), 'doc_': torch.tensor(data_[1]).unsqueeze(0).to(device)}
    with torch.no_grad():
        pred = model(data_)
        k_ = torch.max(pred, 1)[1][0]
        return k_.data.item()


import distance
from dssm.utils import distance_jacaard, distance_edit


def ruleEditForTrue(s1, s2):
    diff_len = abs(len(s1) - len(s2))
    if diff_len < (len(s1) + len(s2)) * 0.2:
        edt_ = distance.levenshtein(s1, s2) / (len(s1) + len(s2))
        if edt_ < 0.1:
            return True
    else:
        if len(s1) < len(s2):  # 保持s1长度优先
            s1, s2 = s2, s1

        edt_ = distance_edit(s1[:len(s2)], s2)
        if edt_ < 0.1:
            return True

        edt_ = distance_edit(s1[:-len(s2)], s2)
        if edt_ < 0.1:
            return True

    return False


def ruleEditForFalse(s1, s2):
    """
    基于规则判定名称不相等
    :param s1:
    :param s2:
    :return:
    """

    diff_len = abs(len(s1) - len(s2))
    if len(s1) == 0 or len(s2) == 0:
        return True
    edt_ = distance.levenshtein(s1, s2) / (len(s1) + len(s2))
    if diff_len < (len(s1) + len(s2)) * 0.2:
        if edt_ > 0.3:
            return True
    else:
        if len(s1) < len(s2):  # 保持s1长度优先
            s1, s2 = s2, s1
        edt_a = distance_edit(s1[:len(s2)], s2)
        if edt_a < 0.1:
            return False

        edt_a = distance_edit(s1[:-len(s2)], s2)
        if edt_a < 0.1:
            return False
        if edt_ > 0.3:
            return True
    return False


def ruleLoc(s1, s2):
    """
    :param s1:
    :param s2:
    :return:
    """

    tmp1 = ' '.join([k for k in s1.replace(',', ' ', -1).split() if len(k) != 0])
    tmp2 = ' '.join([k for k in s2.replace(',', ' ', -1).split() if len(k) != 0])

    if len(tmp1) == 0 or len(tmp2) == 0:
        return None

    jac_ = distance_jacaard(tmp1, tmp2)
    edt_ = distance_edit(s1, s2)
    if jac_ > 0.4 or edt_ < 0.1:
        return True

    if jac_ < 0.15 and edt_ > 0.3:
        return False

    return None


def sameLogic(key_word, data_1, data_2, model_dict, char_vocab, max_len, device):
    """

    :param key_word:
    :param data_1:
    :param data_2:
    :param model_dict:
    :param char_vocab:
    :param max_len:
    :param device:
    :return:
    """
    if len(data_1[1]) <= 5 and len(data_2[1]) <= 5:  ### 地址为空 则略过
        loc_sim = -2
    elif len(data_1[1]) <= 5 or len(data_2[1]) <= 5:  ### 地址为空 则略过
        loc_sim = -1
    else:
        rule_loc_ = ruleLoc(data_1[1], data_2[1])
        if rule_loc_ is not None:
            if rule_loc_:
                loc_sim = 0
            else:
                loc_sim = 1
        else:
            if model_dict['segment_loc'] is None:
                loc_1, loc_2 = dataProcessChar(data_1[1], data_2[1], max_len, char_vocab, True)  # :XXX max_len 未统一
            else:
                loc_1, loc_2 = dataProcessWord(data_1[1], data_2[1], max_len, max_len, model_dict['pad_id'],
                                               model_dict['segment_loc'])  # :XXX max_len 未统一
            loc_sim = isSameModel([loc_1, loc_2], model_dict['loc'], device)

    edt_flag = ruleEditForTrue(data_1[0], data_2[0])  # 对编辑距离明显小的 归于相似
    if edt_flag and loc_sim != 1:  # 存在风险
        # print('EDT_RULE_{},{}'.format(data_1, data_2))
        return True

    # 对编辑距离明显大的 归于不相似
    if ruleEditForFalse(data_1[0].replace(key_word, '').replace(' ', '', -1),
                        data_2[0].replace(key_word, '').replace(' ', '', -1)) and loc_sim != 0:
        # print('FASLE_RULE_{},{}'.format(data_1, data_2))
        return False

    name_1, name_2 = dataProcessChar(data_1[0], data_2[0], max_len, char_vocab, True)
    name_1_w, name_2_w = dataProcessWord(data_1[0], data_2[0], max_len, max_len, model_dict['pad_id'], model_dict['segment'])
    name_sim_char_1 = isSameModel([name_1, name_2], model_dict['name_char_1'], device)  # model 有偏，后续可优化
    name_sim_word = isSameModel([name_1_w, name_2_w], model_dict['name_word'], device)
    name_sim_char_2 = isSameModel([name_2, name_1], model_dict['name_char_2'], device)  # model 有偏，后续可优化
    # name_sim_word = 0

    # print('Test processSet,{},{}'.format(data_1, data_2), name_sim_char_1, name_sim_word, name_sim_char_2, loc_sim)
    # print(name_1_w, name_2_w)

    ###相似的逻辑处理 趋于更严格
    # Version_1
    # if name_sim_char_1 + name_sim_word == 0 and loc_sim <= 0:
    #     return True
    # # 对此部分的处理逻辑有待评测
    # elif name_sim_char_1 + name_sim_word == 0 and loc_sim == 1:
    #     if name_sim_char_2 == 0:
    #         return True
    #     else:
    #         return False
    # elif name_sim_char_1 + name_sim_word == 1 and loc_sim <= 0:  ### 基于地址相似判定相似
    #     if name_sim_char_2 == 0:
    #         return True
    #     else:
    #         return False
    # else:
    #     return False

    # Version_2
    if name_sim_char_1 + name_sim_word + name_sim_char_2 == 0 and loc_sim != 1:
        return True
    elif name_sim_char_1 + name_sim_word + name_sim_char_2 == 1 and loc_sim == 0:
        return True
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


def processSet(sen_set, id2sentence, model_dict, char_vocab, max_len, device):
    """
    处理set集合，获取子聚类类别
    :param sen_set:  sentence_ids
    :return:
    """

    tmp_list = list(sen_set)
    set_list = []
    while len(tmp_list) >= 1:
        a = tmp_list[0]
        a_set = set({a})

        flag_ = []
        for k in range(1, len(tmp_list)):
            if sameLogic(id2sentence[a], id2sentence[tmp_list[k]], model_dict, char_vocab,
                         max_len, device):  # 核心  相似判定
                flag_.append(tmp_list[k])
                a_set.add(tmp_list[k])

        flag_.append(a)
        for one in flag_:
            tmp_list.remove(one)

        print('Test Make process Set', id2sentence[a], '\001'.join([id2sentence[tk][0] for tk in a_set]))
        set_list.append(a_set)

    return set_list


import functools


def itemCmp(a1, a2):
    """
    针对 名称、地址条目的排序
    :param a1:
    :param a2:
    :return:
    """
    l_1 = len(a1[0])
    l_2 = len(a2[0])
    if l_1 != l_2:
        return l_1 - l_2
    else:
        d_1 = len(a1[1])
        d_2 = len(a2[1])
        return d_2 - d_1


def processSetSort(key_word, sen_set, id2sentence, model_dict, char_vocab, max_len, device):
    """

    :param key_word: same word
    :param sen_set: sentence_ids
    :param id2sentence: For Sim, sentence,address
    :param model_dict:  For Sim
    :param char_vocab:  For Sim
    :param word_vocab:  For Sim
    :param max_len:     For Sim
    :param device:      For Sim
    :param cluster_set: flag_use
    :return:
    """
    tmp_data = []
    for id_ in sen_set:
        tmp_data.append(id2sentence[id_] + [id_])  # 增加一个维度 id_便于后期追溯

    tmp_sort_data = sorted(tmp_data, key=functools.cmp_to_key(itemCmp), reverse=True)

    part_cluster_set = []
    while len(tmp_sort_data) >= 1:
        guard_item = tmp_sort_data[0]
        tmp_set_id = set({guard_item[2]})

        del_flag = []
        for k in range(1, len(tmp_sort_data)):
            if sameLogic(key_word, guard_item, tmp_sort_data[k], model_dict, char_vocab,
                         max_len, device):  # 核心  相似判定
                del_flag.append(tmp_sort_data[k])
                tmp_set_id.add(tmp_sort_data[k][2])

        del_flag.append(guard_item)
        for one in del_flag:
            tmp_sort_data.remove(one)

        # print('Test Make process Set', guard_item, '\001'.join([id2sentence[tk][0] for tk in tmp_set_id]))
        part_cluster_set.append(tmp_set_id)

    return part_cluster_set


def processCluster(word_list, word2sentenceid, id2sentence, cluster_set):
    """
    针对word2set的计算结果，更新cluster2set
    :param word_list:  子聚类结果 列表
    :param cluster_set: sentenceid --> ???
    :return:
    """

    model_dict, char_vocab, max_len, device = model_init()

    cluster_info_ = {}
    num_ = 1
    for word_ in word_list:  ###对于word 的顺序要有要求
        word_sen_set = word2sentenceid[word_]  # word 对应的 sentence list
        word_sen_set = list(word_sen_set)
        if len(word_sen_set) == 1 and cluster_set[word_sen_set[0]] is False:
            cluster_set[word_sen_set[0]] = set([word_sen_set[0]])
            cluster_info_[word_sen_set[0]] = [word_]
        else:
            todo_cluster = []
            for id_ in word_sen_set:
                if not cluster_set[id_]:
                    todo_cluster.append(id_)

            # 分割成不同的集合
            print('Word_{}-len_{}', word_, len(todo_cluster))
            set_list = processSetSort(word_, todo_cluster, id2sentence, model_dict, char_vocab, max_len, device)

            ###粗筛合并的必要性
            for set_ in set_list:
                for id_ in set_:
                    if cluster_set[id_] is False:
                        cluster_set[id_] = set_
                        cluster_info_[id_] = [word_]
                    elif set_ == cluster_set[id_]:
                        continue
                    else:
                        # 原有与 现有集合的关系  TODO   此处处理的有风险
                        merge_ = set_ | cluster_set[id_]
                        if len(merge_) > len(cluster_set[id_]):
                            cluster_set[id_] = merge_
                            print("merge_\t" + str(id_))
                            for id_m in merge_:
                                cluster_set[id_m] = merge_  # 此步骤为粗筛
                                cluster_info_[id_].append(word_)

        num_ += 1
        if num_ % 10000 == 0:
            pickleDumpFile('cluster_set_{}.pk'.format(num_), cluster_set)

    return cluster_info_


import random


def runPart(reload_cluster="", path_=BASE_PATH + '/baseDictData.pk', word_sen_path=BASE_PATH + '/word_sen_num'):
    """

    :param path_:
    :param word_sen_path: 倒序排列
    :return:
    """
    with open(path_, 'rb') as f:
        id2sentence = pickle.load(f)
        word2sentenceid = pickle.load(f)
        cluster2set = pickle.load(f)

    word_dic = {}
    word_list = []
    word_num = []
    with open(word_sen_path, 'r') as f:
        for one in f:
            tmp_ = one.rstrip().split('\t')
            word_dic[tmp_[0]] = int(tmp_[1])
            word_list.append(tmp_[0])
            word_num.append(tmp_[1])

    word_list = word_list[::-1]
    word_num = word_num[::-1]

    if reload_cluster != "":
        cluster2set = pickle.load(open(reload_cluster, 'rb'))
        num_ = int(reload_cluster.rstrip('.pk').split('_')[-1])
        word_list = word_list[num_:]

    word_tmp = []
    threld = [5, 100]
    for k in word_dic:
        if word_dic[k] >= threld[0] and word_dic[k] <= threld[1]:
            word_tmp.append(k)

    tmp_list = random.sample(word_tmp, 50)
    print('TMP_LIST', tmp_list)
    # XXX TODO
    '''
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
    '''

    tmp_list = ['000000005', 'SHD']

    cluster_info_ = processCluster(word_list, word2sentenceid, id2sentence, cluster2set)

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
            # if len(line_) < 10:
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


import pandas as pd


def toExcel(path_):
    a, b, c, d = [], [], [], []

    with open(path_, 'r') as f:
        for one in f:
            tmp_ = one.rstrip().replace('\001\002', '\001\001').split('\001\001')
            if len(tmp_) != 4:
                continue

            a.append(tmp_[0])
            b.append(tmp_[1])
            c.append(tmp_[2])
            d.append(tmp_[3])

    print(len(a))
    df = pd.DataFrame(columns=['cluster_name', 'key_word', 'name', 'loc'])
    df['cluster_name'] = a
    df['key_word'] = b
    df['name'] = c
    df['loc'] = d
    df.to_excel('../data_1022.xlsx')


if __name__ == '__main__':
    pass
    # makeDictData('../data/data.log')
    # infoSense(BASE_PATH + '/baseDictData.pk')
    # makePosTestData('../data/posi.data')

    # a = model_init()

    runPart(sys.argv[1])
    # print(BASE_DATA_PATH)
    # toExcel('../log')
