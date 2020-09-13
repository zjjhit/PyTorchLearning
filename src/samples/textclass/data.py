# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         data
# Description:  
# Author:       lenovo
# Date:         2020/5/11
# -------------------------------------------------------------------------------

import sys, os

import logging
import torch
from torchtext.vocab import Vocab
import torchtext
from torchtext.datasets import text_classification as torch_text

NGRAMS = 2
import os

DATA_PATH = '../data/ag_news_csv/'


def prepairData(path, ngrams=NGRAMS, vocab=None):
    if not os.path.isdir(path):
        logging.error('Data path err')
        return

    train_csv_path = path + 'train.csv'
    test_csv_path = path + 'test.csv'

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = torch_text.build_vocab_from_iterator(torch_text._csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    train_data, train_labels = torch_text._create_data_from_iterator(
        vocab, torch_text._csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk=False)
    logging.info('Creating testing data')
    test_data, test_labels = torch_text._create_data_from_iterator(
        vocab, torch_text._csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk=False)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (torch_text.TextClassificationDataset(vocab, train_data, train_labels),
            torch_text.TextClassificationDataset(vocab, test_data, test_labels))


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


# train_dataset, test_dataset = torch_text.DATASETS['AG_NEWS'](
#     root='./.data', ngrams=NGRAMS, vocab=None)

# train_dataset, test_dataset = prepairData(DATA_PATH)

# torch.save(train_dataset, DATA_PATH + '/ag_news_train.pkl')
# torch.save(test_dataset, DATA_PATH + '/ag_news_test.pkl')

train_dataset = torch.load(DATA_PATH + '/ag_news_train.pkl')
test_dataset = torch.load(DATA_PATH + '/ag_news_test.pkl')

BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
