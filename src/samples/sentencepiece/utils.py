# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         utils
# Description:  
# Author:       lenovo
# Date:         2020/7/13
# -------------------------------------------------------------------------------

import sys, os

import sentencepiece as spm
import argparse

Path_ = '/home/zjj/TMP/SentencePiece'

parser = argparse.ArgumentParser(description='Character level CNN text classifier training')
# data
parser.add_argument('--input', metavar='DIR',
                    help='path to training data csv [default: data/ag_news_csv/train.csv]',
                    default=Path_ + '/info.txt')
parser.add_argument('--model_prefix', metavar='DIR',
                    help='path to validation data csv [default: data/ag_news_csv/test.csv]',
                    default=Path_ + '/m')


def train():
    # SentencePieceTrainer_Train(args)

    spm.SentencePieceTrainer.train(input=Path_ + '/tmp2', model_prefix=Path_ + '/m_tmp2', vocab_size=3000)


def test():
    sp = spm.SentencePieceProcessor(Path_ + '/m.model')
    print(sp.encode('怎么企业阅读使用不了', out_type=str))


##语气词前缀,需要注意处理“我”的重复出现问题,“一下”
MODE_PRE = ['哎你好', '诶您好', '你好', '麻烦你', '麻烦你帮我', '那个啥', '那个什么', '能不能', '那你帮我看下', '那你给我说下', '咨询一下']

TYPE_PRE1 = ['帮我查下', '帮我查询一下', '帮我查一下', '帮我开', '帮我开通', '帮我开一下', '帮我看一下', '帮我取消', '帮我取消掉', '帮我取消一下查询一下', '查一下', '给我办个', '给我查一下',
             '给我打电话',
             '给我发', '给我发了', '给我改', '给我开', '给我取消', '给我取消掉', '给我取消了', '给我说', '给我说说', '给我说一下', '跟我说说', '开通那个', '开通一个', '开通一下', '开一个',
             '你好我想问下', '怎么设置']

###可直接接产品名
TYPE_PRE2 = ['你帮我查一下', '你帮我查一下我', '你帮我看一下', '你帮我提供一下', '你给我', '你给我查一下', '你给我介绍下', '你给我介绍一下', '你跟我说下', '请帮我提供一下', '请给我', '请给我说一下',
             '请您帮我提供一下', '请您帮我提供一下', '请说一下', '请问一下', '如何办', '什么是', '使用不了', '使用异常', '是怎么办理的', '收费标准是什么', '移动公司是不是有', '在哪可以办理',
             '在哪里办']

###可直接接产品名
TYPE_PRE3 = ['我们厂需要办理', '我们厂要办', '我们厂要办理', '我们单位', '我们单位需要', '我们单位要开通', '我们单位要办', '我们单位要办个', '我们单位要了解', '我们单位要取消', '我们单位要用', '我们公司想开通',
             '我们公司需要办', '我们公司要办', '我们公司要取消', '我们公司要用', '我们学校需要', '我们学校要办', '我们学校要开通', '我们银行要办', '']

TYPE_END = ['的收费标准', '都有什么', '多少钱', '是什么', '还有多少', '还有没有', '还有什么', '那个怎么操作', '取消什么时候生效', '取消什么时候失效', '如何办理', '用不了是怎么回事',
            '有什么功能', '有问题了', '怎么不能用了', '怎么收费的', '如何使用', '是干啥的', '需要下载APP吗', '需要下载软件吗']

TYPE_SINGLE = ['你好，*不好使', '你好，你们移动公司是不是有*', '您好，你们的*不好使', '问一下，我办理了*怎么不能用了', '我的*使用不了', '我订的*老是出问题', '我刚办的*怎么用不了', '再把那个*给我介绍一下',
               '怎么*使用不了']

###搭配、不好使\使用不了\不能用了
TYPE_PP = ['是怎么回事啊', '是什么情况', '怎么回事啊']


def putMode(result_):
    '''

    :param result_:
    :return:
    '''

    return result_
    '''
    for i in range(0, len(result_)):
        tmp_ = '['
        for k in MODE_PRE:
            if '我' in k and '我' in result_[i]:
                continue
            if '一下' in k and '一下' in result_[i]:
                continue
            tmp_ += k + ' | '
        result_[i] = tmp_.rstrip(' | ') + '], ' + result_[i]

    return result_
    '''


def getPre(txt_, model=0):
    '''

    :param txt_:
    :param model: 0 --> MODE_PRE + TYPE_PER1\TYPE_PRE2\TYPR_PRE3  ,1--> TYPE_PER1\TYPE_PRE2\TYPR_PRE3
    :return:
    '''

    result_ = []

    for k in TYPE_PRE1 + TYPE_PRE2 + TYPE_PRE3:
        result_.append(k + txt_)

    if model == 0:
        return putMode(result_)

    return result_


def getEnd(txt_, model=0):
    result_ = []

    for k in TYPE_END:
        result_.append(txt_ + k)

    if model == 0:
        return putMode(result_)

    return result_


def putEnd(result_):
    '''
    补充部分尾串
    :param result_:
    :return:
    '''
    for i in range(0, len(result_)):
        tmp_ = '['
        for k in TYPE_PP:
            # 不好使\使用不了\不能用了
            if '不好使' in result_[i] or '使用不' in result_[i] or '不能用' in result_[i]:
                tmp_ += k + ' | '
        if len(tmp_) > 2:
            result_[i] += ' ,' + tmp_.rstrip(' | ') + '] '

    return result_


def getInfo(txt_, model=0):
    '''

    :param txt_:
    :return:
    '''
    result_ = []
    for one in TYPE_SINGLE:
        result_.append(one.replace('*', txt_))

    if model == 0:
        return putEnd(result_)

    return result_


def makeTxt(dic_):
    fout = open('./data.txt', 'w')
    for k in dic_:
        for name_ in dic_[k]:
            name_ = ' {' + name_ + '} '
            d1 = getPre(name_)
            d2 = getEnd(name_)
            d3 = getInfo(name_)
            fout.write(name_ + '\t' + '|'.join(MODE_PRE) + '\n')
            for one in d1 + d2 + d3:
                fout.write(one + '\n')
            fout.write('\n')

    fout.close()


from simpleName import *

if __name__ == '__main__':
    # train()
    # test()
    pass
    # print(getInfo('和视宝'))
    makeTxt(simple_dic)
