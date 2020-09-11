# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         wltx_202009
# Description:  
# Author:       lenovo
# Date:         2020/9/7
# -------------------------------------------------------------------------------

'''
处理网罗天下相关的英文公司数据集
'''

import re


def preDataTwo(path):
    """
    纯文件形式解析xml
    :param path:
    :return:
    """
    num_ = 0
    dic_ = {}
    order_ = [1]
    r_ = re.compile(r'(^\d{1,7})\001\002.*')
    with open(path, 'r', encoding='utf-8') as f:
        for one in f:
            num_ += 1
            m = r_.match(one)
            if m:
                dic_[num_] = m.group(1)
                order_.append(int(m.group(1)))
                if order_[-1] - 1 != order_[-2]:
                    print('Wrong\t' + str(order_[-1]) + '\t' + str(order_[-2]))

    num_ = 0
    data_ = {}
    lable_ = num_
    with open(path, 'r', encoding='utf-8') as f:
        for one in f:
            num_ += 1
            if num_ in dic_:
                data_[num_] = one.rstrip()
                lable_ = num_
            else:
                data_[lable_] = data_[lable_] + ' ' + one.rstrip()

    for key in data_:
        one = data_[key]
        m = r_.match(one)
        id = m.group(1)
        d_ = one[len(id):].split('\001\002', 1)
        if len(d_) != 2:
            print('SomeThingWrong\t' + one)
        else:
            print('AAA\001\002' + id + '\001\002' + '\001\002'.join(d_))

    print(len(data_))


import pyodbc


def preDataThree(path):
    # 链接数据库
    conn = pyodbc.connect(u'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + path)
    # 创建游标
    cursor = conn.cursor()
    # tb_name是access数据库中的表名
    tb_name = "com"
    cursor.execute('select * from %s' % tb_name)
    # 获取数据库中表的全部数据
    field_list = []
    fout = open('./data.dbc', 'w', encoding='utf-8')
    for field in cursor.fetchall():
        # field_list.append(field[0])

        info_ = str(field[0])
        if field[1] == None:
            info_ += '\001\002' + 'NULL'
        else:
            info_ += '\001\002' + field[1].replace('')

        if field[2] == None:
            info_ += '\001\002' + 'NULL'
        else:
            info_ += '\001\002' + field[2]
        fout.write(info_ + '\n')
    # 关闭游标和链接
    cursor.close()
    fout.close()


import pickle


def makeData(path_):
    """
    data.log 经windows 本地提取后，再进行加工
    构建训练数据中正样本数据
    :return:
    """
    dic_name = {}
    dic_loc = {}
    num_ = 0
    with open(path_, 'r') as f:
        for one in f:
            num_ += 1
            if num_ % 50000 == 0:
                print(num_)
            tmp_ = one.rstrip().split('\001\002')
            # print(one.rstrip())

            if len(tmp_) != 5 or tmp_[-2] == 'NULL' or tmp_[-1] == 'NULL':
                continue
            if tmp_[-2] not in dic_name:
                dic_name[tmp_[-2]] = set({tmp_[-1]})
            elif tmp_[-1] not in dic_name[tmp_[-2]]:
                dic_name[tmp_[-2]].add(tmp_[-1])

            if tmp_[-1] not in dic_loc:
                dic_loc[tmp_[-1]] = set({tmp_[-2]})
            elif tmp_[-2] not in dic_loc[tmp_[-1]]:
                dic_loc[tmp_[-1]].add(tmp_[-2])

    tmp_ = []
    for k in dic_name:
        if len(dic_name[k]) < 2:
            tmp_.append(k)
    for k in tmp_:
        del dic_name[k]

    tmp_.clear()
    for k in dic_loc:
        if len(dic_loc[k]) < 2:
            tmp_.append(k)
    for k in tmp_:
        del dic_loc[k]

    print(len(dic_name), len(dic_loc))
    fout = open('./data/train.data.name', 'wb')
    pickle.dump(dic_name, fout)
    fout.close()

    fout = open('./data/train.data.loc', 'wb')
    pickle.dump(dic_loc, fout)
    fout.close()


def makeNegData(path_):
    """
    构建训练数据中负样本数据
    :param path_:
    :return:
    """
    pass


def tmpData(path_):
    with open(path_, 'rb') as f:
        dic_ = pickle.load(f)
        num_ = 0
        for k in dic_:
            num_ += 1
            print(k + '\t\t' + '\001'.join(dic_[k]))
            if num_ > 1000:
                break


def makeCharDict(path_):
    """
    以字符为单位，构建词典
    :param path_:
    :return:
    """
    dict_ = {}
    with open(path_) as f:
        for one in f:
            tmp_ = one.rstrip().split('\001\002')
            if len(tmp_) != 5 or tmp_[-2] == 'NULL' or tmp_[-1] == 'NULL':
                continue
            for chr_ in tmp_[-2]:
                if chr_.isalpha():
                    chr_ = chr_.lower()
                if chr_ not in dict_:
                    dict_[chr_] = 1
                else:
                    dict_[chr_] += 1

    print(len(dict_))
    tmp_ = []
    for k in dict_:
        if dict_[k] < 1000:
            tmp_.append(k)
        else:
            print(k, dict_[k])

    for k in tmp_:
        del dict_[k]

    fout = open('./data/char.dict', 'wb')
    pickle.dump(dict_, fout)
    fout.close()


if __name__ == '__main__':
    pass
    # makeData('./data/data.log')
    # preData('./data/com.xml')
    # preDataTwo('./data/data.dbc')
    # preDataThree('./data/companys.accdb')
    tmpData('./data/train.data.loc')
    # makeCharDict('./data/data.log')
