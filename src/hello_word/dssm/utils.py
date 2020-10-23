# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         utils
# Description:  
# Author:       lenovo
# Date:         2020/10/20
# -------------------------------------------------------------------------------

import pickle

import sentencepiece as spm

"""
 构建训练集的词表，原始char级别的词表效果一般
"""

import distance
import numpy as np
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
    try:
        vectors = cv.fit_transform(corpus).toarray()
        # print(vectors)
        # 求交集
        numerator = np.sum(np.min(vectors, axis=0))
        # 求并集
        denominator = np.sum(np.max(vectors, axis=0))
        # 计算杰卡德系数
    except ValueError as e:
        print(e)
        return 0
    return 1.0 * numerator / denominator


def buildSubWordVocab(data_path):
    spm.SentencePieceTrainer.Train('--input={}  --model_prefix=loc_32k --vocab_size=32000 --pad_id=3'.format(data_path))


class SegmentWord():
    _instance = None

    def __init__(self, seg_model_path):
        self.segment_ = spm.SentencePieceProcessor()
        self.segment_.load(seg_model_path)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SegmentWord, cls).__new__(cls)
        return cls._instance

    def encodeAsPieces(self, sentence):
        return self.segment_.EncodeAsPieces(sentence)

    def encodeAsIds(self, sentence):
        return self.segment_.EncodeAsIds(sentence)

    def test(self):
        print(self.segment_.GetPieceSize())
        for i in range(32000):
            # print(self.segment_.IdToPiece(i))
            # if ' ' in self.segment_.IdToPiece(i) or '<' in self.segment_.IdToPiece(i):
            print(self.segment_.IdToPiece(i))


BASE_DATA_PATH = '../data/'


def charVocabBuild(dict_path, min_count=-float("inf")):
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

# if __name__ == '__main__':
# buildSubWordVocab('../data/vocab/name.data')
# buildSubWordVocab('../data/vocab/loc.data')

# pass
# testModel('../data/config.json_8')

# v = pickle.load(open('../data/char2id.vocab', 'rb'))
# for k in v:
#     print(k, v[k])

# l_ = SegmentWord('./m.model')
# print(l_.encodeAsPieces('HENFEN IMP. AND EXP. CO.,LTD. BUILDING A5-B,XINMA INDUSTRIAL'))
# print(l_.encodeAsPieces('HENFEN IMP. & EXP. CO.,LTD. BUILDING A5-B XINMA INDUSTRIAL'))
