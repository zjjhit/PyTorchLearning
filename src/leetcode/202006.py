# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         202006
# Description:  
# Author:       lenovo
# Date:         2020/6/1
# -------------------------------------------------------------------------------

import sys, os


def num_1431(a_):
    a_[1] = 1
    return a_


def spiralOrder(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """

    def getRight(i_, j_, matrix, flag_):
        data_ = []
        for j in range(j_, len(matrix[i_])):
            if flag_[i_][j]:
                break
            if [i_, j] not in data_:
                data_.append([i_, j])
                flag_[i_][j] = True
        return data_

    def getLeft(i_, j_, matrix, flag_):
        data_ = []
        for j in range(j_, -1, -1):
            if flag_[i_][j]:
                break
            if [i_, j] not in data_:
                data_.append([i_, j])
                flag_[i_][j] = True
        return data_

    def getUp(i_, j_, matrix, flag_):
        data_ = []
        for i in range(i_, -1, -1):
            if flag_[i][j_]:
                break
            if [i, j_] not in data_:
                data_.append([i, j_])
                flag_[i][j_] = True
        return data_

    def getDown(i_, j_, matrix, flag_):
        data_ = []
        for i in range(i_, len(matrix)):
            if flag_[i][j_]:
                break
            if [i, j_] not in data_:
                data_.append([i, j_])
                flag_[i][j_] = True
        return data_

    data_ = []
    n = len(matrix)
    m = len(matrix[0])

    # flag_ = [[False] * m] * n
    flag_ = [[0 for i in range(m)] for i in range(n)]
    i_ = 0
    j_ = 0

    while True:
        tmp_ = getRight(i_, j_, matrix, flag_)
        if len(tmp_) == 0:
            break
        data_ += tmp_
        i_, j_ = tmp_[-1][0] + 1, tmp_[-1][1]
        # print(i_, j_)

        tmp_ = getDown(i_, j_, matrix, flag_)
        if len(tmp_) == 0:
            break
        data_ += tmp_
        i_, j_ = tmp_[-1][0], tmp_[-1][1] - 1
        # print(i_, j_)
        tmp_ = getLeft(i_, j_, matrix, flag_)
        if len(tmp_) == 0:
            break
        data_ += tmp_
        i_, j_ = tmp_[-1][0] - 1, tmp_[-1][1]
        # print(i_, j_)
        tmp_ = getUp(i_, j_, matrix, flag_)
        if len(tmp_) == 0:
            break
        data_ += tmp_
        i_, j_ = tmp_[-1][0], tmp_[-1][1] + 1
        # print(i_, j_)
    info_ = []
    for one in data_:
        info_.append(matrix[one[0]][one[1]])

    return info_


def maxArea(height):
    """
    :type height: List[int]
    :rtype: int
    """

    def min_(a, b):
        if a <= b:
            return a
        else:
            return b

    def max_(a, b):
        if a >= b:
            return a
        else:
            return b

    num_ = 0
    i = 0
    j = len(height) - 1

    while True:
        # t_ = num_
        if i + 1 <= j - 1:
            # tmp_a = min_(height[i + 1], height[j]) * (j - i - 1)
            # tmp_b = min_(height[i], height[j - 1]) * (j - i - 1)

            tmp_d = min_(height[i], height[j]) * (j - i)
            num_ = max_(tmp_d, num_)
            # num_ = max_(tmp_a, num_)
            # num_ = max_(tmp_b, num_)

            if height[i] < height[j]:
                i += 1
            else:
                j -= 1

        elif i + 1 == j:
            tmp_d = min_(height[i], height[j]) * (j - i)
            num_ = max_(tmp_d, num_)
            i += 1
        else:
            break

    return num_


def canJump_55(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    flag_ = ['a'] * len(nums)
    len_ = len(nums)
    if len_ == 1:
        return True
    if nums[0] == 0:
        return False

    flag_[0] = 'b'
    for i in range(len_):
        if flag_[i] == 'b':
            if nums[i] > 0:
                if i + nums[i] >= len_ - 1:
                    t_ = len_ - 1
                    if flag_[i] == 'b':
                        return True
                else:
                    t_ = i + nums[i]

                for j in range(i, t_ + 1):
                    flag_[j] = 'b'
            # else:
            #     return False
        # else:
        #     return False

    print(flag_)
    if 'a' in flag_:
        return False
    return True


def merge(nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    i = 0
    j = 0

    for l in range(0, m):
        while nums2[j] <= nums1[l]:
            nums1 = nums1[:l] + [nums2[j]] + nums1[l:]
            j += 1
    print(nums1)


def strWithout3a3b(A, B):
    """
    :type A: int
    :type B: int
    :rtype: str
    """
    i = A
    j = B
    info_ = ''
    while i > 0 and j > 0:
        while i > j and j >= 1:
            info_ += 'aab'
            i -= 2
            j -= 1
        while j > i and i >= 1:
            info_ += 'bba'
            j -= 2
            i -= 1
        while i == j and i > 0:
            info_ += 'ab'
            i -= 1
            j -= 1

    if i > 0:
        info_ += 'a' * i
    if j > 0:
        info_ += 'b' * j
    return info_


def findContentChildren(g, s):
    """
    :type g: List[int]
    :type s: List[int]
    :rtype: int
    """

    g_ = sorted(g)
    s_ = sorted(s)
    num_ = 0
    j = 0
    for one in s_:
        if j < len(g_) and g_[j] <= one:
            num_ += 1
            j += 1

    return num_


if __name__ == '__main__':
    g = [21, 2, 3]
    s = [1, 1]
    print(findContentChildren(g, s))
