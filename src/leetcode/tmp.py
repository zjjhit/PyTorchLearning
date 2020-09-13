# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         tmp
# Description:  
# Author:       lenovo
# Date:         2020/6/16
# -------------------------------------------------------------------------------

import sys, os


def check(an, cn):
    fin = open(an, 'r')
    dic_ = {}
    for one in fin.readlines():
        tmp = one.rstrip().split('\t')
        assert len(tmp) == 2, print(one)
        dic_[tmp[1]] = tmp[0]
    fin.close()

    fin = open(cn, 'r')
    for one in fin.readlines():
        tmp = one.rstrip().split('\t')
        if tmp[1] not in dic_:
            print('Wrong\t' + one)
        else:

            print(tmp[1] + '\t' + dic_[tmp[1]] + '\t\t' + tmp[0] + '\t' + str(tmp[0] == dic_[tmp[1]]))


from torch import nn
import torch


def test():
    shared = nn.Linear(3, 3)
    net1 = nn.Sequential(nn.Linear(2, 3), nn.ReLU(),
                         shared, nn.ReLU())
    net2 = nn.Sequential(
        shared, nn.ReLU(),
        nn.Linear(3, 1))

    x = torch.rand(1, 2)

    y_1 = net1(x)

    y_2 = net2(y_1)

    loss = nn.L1Loss()
    l_ = loss(y_2, torch.tensor(10))
    print(net1[2].weight.grad)
    l_.backward()


    print(net2[0].weight.grad)
    print(net1[2].weight.grad)


if __name__ == '__main__':
    test()
