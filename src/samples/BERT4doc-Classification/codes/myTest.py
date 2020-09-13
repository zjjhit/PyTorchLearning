# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         myTest
# Description:  
# Author:       lenovo
# Date:         2020/8/6
# -------------------------------------------------------------------------------

import sys, os
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import finetuning.tokenization
from finetuning.modeling import BertConfig, BertForSequenceClassification
from finetuning.optimization import BERTAdam

from finetuning.run_classifier import argsInfo

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = '../data/bert-base-cased/'


def testRun():
    bert_config = BertConfig.from_json_file(DATA_PATH + 'bert_config.json')


# if __name__ == '__main__':
testRun()
