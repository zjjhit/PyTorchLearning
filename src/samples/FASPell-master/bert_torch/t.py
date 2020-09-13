# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         t
# Description:  
# Author:       lenovo
# Date:         2020/8/11
# -------------------------------------------------------------------------------

import sys, os

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

model_dir = "../model/pre-trained/bert_model.ckpt"

ckpt = tf.train.get_checkpoint_state(model_dir)
# ckpt_path = ckpt.model_checkpoint_path

# reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
# param_dict = reader.get_variable_to_shape_map()
#
# for key, val in param_dict.items():
#     try:
#         print(key, val)
#     except:
#         pass

# from bert_modified import tokenization
import masked_lm

p_ = masked_lm.Processor('../model/pre-trained/vocab.txt', 128)
# input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights
for one in p_.create_single_instance('。国际电台苦名丰持人。'):
    print(one)
