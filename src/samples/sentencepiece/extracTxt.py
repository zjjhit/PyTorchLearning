# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         extracTxt
# Description:  
# Author:       lenovo
# Date:         2020/7/13
# -------------------------------------------------------------------------------

import sys, os
import chardet

'''
text grid  内容抽取
'''

Txt_ = '/home/zjj/DATA/SentencePiece/data'


def get_encoding(file):
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']


def getFileList(dir_):
    l_ = os.listdir(dir_)
    with open(Txt_, 'w') as f:
        for one in l_:
            path_ = dir_ + '/' + one

            try:
                result_ = run(path_)
            except UnicodeError:
                print('Wrong ' + path_)
                continue

            if len(result_) > 0:
                f.write(one + '\t\t' + '\001\001'.join(result_) + '\n')


import traceback


def run(path_):
    '''

    :param path_:
    :return:
    '''
    encode_ = get_encoding(path_)
    with open(path_, 'r', encoding=encode_, errors='ignore') as f:
        lines_ = [one.strip() for one in f.readlines()]
        if len(lines_) < 5:
            return []
        try:
            # print(lines_[5])
            if 'GLOBAL' in lines_[8]:
                result_ = typeOne(lines_)
            elif 'xmin' in lines_[3]:
                result_ = typeTwo(lines_)
            elif 'exists' in lines_[5]:
                result_ = typeThree(lines_)
            else:
                result_ = []
        except:
            print("Wrong INFO \t" + path_)
            traceback.print_exc()
            return []

    return result_


def typeOne(lines):
    '''
    ooTextFile
    1 --> server
    2 --> customer
    :param path_:
    :return:
    '''

    # assert 'GLOBAL' in lines[8], 'Type Wrong '
    nums_ = int(lines[19].strip())

    start_role = 19
    start_txt = 19 + nums_ * 3 + 5

    result_ = []
    for i in range(1, nums_ + 1):

        start_role += 3
        start_txt += 3

        if '""' in lines[start_role]:
            pass
        elif '1' in lines[start_role]:
            result_.append("server\t" + lines[start_txt])
        elif '2' in lines[start_role]:
            result_.append("customer\t" + lines[start_txt])

    return result_


def typeTwo(lines):
    '''
    3:xmin
    :param lines:
    :return:
    '''

    nums_ = int(lines[13].strip().split(' ')[-1])

    start_role = 13 + nums_ * 4 + 6
    start_txt = 13

    result_ = []
    for i in range(1, nums_ + 1):

        start_role += 4
        start_txt += 4
        print(start_role, i)
        if 'NOISE' in lines[start_role]:
            pass
        elif 'A' in lines[start_role]:
            result_.append("server\t" + lines[start_txt].lstrip('text = '))
        elif 'B' in lines[start_role]:
            result_.append("customer\t" + lines[start_txt].lstrip('text = '))

    return result_


def typeThree(lines):
    nums_ = int(lines[11].strip())

    start_txt = 11

    start_role = 11 + nums_ * 3 + 5

    result_ = []
    for i in range(1, nums_ + 1):

        start_role += 3
        start_txt += 3

        if 'NOISE' in lines[start_role]:
            pass
        elif 'A' in lines[start_role]:
            result_.append("server\t" + lines[start_txt])
        elif 'B' in lines[start_role]:
            result_.append("customer\t" + lines[start_txt])

    return result_


def getCoustmer(path_):
    # result_ = []
    with open(path_, 'r') as f:
        for one in f.readlines():
            tmp = one.rstrip().split('\t\t')
            if len(tmp) != 2:
                continue
            tmp_ = tmp[1].split('\001\001')
            for k in tmp_:
                if 'customer' in k:
                    k = k.lstrip('customer    \t"').rstrip('"')
                    if len(k) > 3:
                        print(tmp[0] + '\t\t' + k)


if __name__ == '__main__':
    # a = run('./dd')
    # print(a)
    getFileList(sys.argv[1])
