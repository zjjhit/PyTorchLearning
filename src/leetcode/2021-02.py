# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         2021-02
# Description:  
# Author:       zjjhit
# Date:         2021/2/7
#-------------------------------------------------------------------------------

import sys, os


class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        '''
        liangbianpaixu
        '''

        if len(nums) < 2:
            return

        i = j = len(nums)
        p = 0
        num_ = 0
        while p < len(nums):
            if nums[p] == 0:
                i = p
                if i > j:
                    nums[i], nums[j] = nums[j], nums[i]

                    num_ += 1
                    p = j +1
                    j = len(nums)
                    continue
            elif j == len(nums):
                j = p
            p += 1

        j = len(nums)
        p = num_
        while p < len(nums):
            if nums[p] == 1:
                i = p
                if i > j:
                    nums[i], nums[j] = nums[j], nums[i]
                    p = j +1
                    j = len(nums)
                    continue
            elif j == len(nums) and nums[p] == 2:
                j = p
            p += 1

        return nums


if __name__ == '__main__':
    a = Solution()
    b =  [2,0]
    # print(a.sortColors(b))
    b.sort()
    print(b)