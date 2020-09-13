# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Description:  
# Author:       lenovo
# Date:         2020/5/19
# -------------------------------------------------------------------------------

import sys, os

import torchsummary
from transformerT.transformer import *

m = MultiHeadedAttention(50, 100)

torchsummary(m, (200, 300, 1000), (200, 300, 1000), (200, 300, 1000))
