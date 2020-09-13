# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         torch_model_lm
# Description:  
# Author:       lenovo
# Date:         2020/8/11
# -------------------------------------------------------------------------------

import sys, os

# from transformers import tokenization_bert
from transformers import BertModel

import tensorflow.contrib.keras as kr
from transformers import BertConfig
import torch

import torch_tokenization

MODEL_PATH = '../model/torch_train/'


class Data(object):
    """
    Load data.

    """

    def __init__(self, data, processor):

        self.data = data
        self.pos = 0  # records the iterating progress for df
        self.processor = processor

    def next_predict_batch(self, batch_size):
        """
        Produce the next batch for predicting.

        Args
        ----------------
        batch_size: batch_size for predicting

        Returns
        ----------------
        features_padded_batch, tags_padded_batch, length_batch
        or
        None if the data is exhausted
        """
        print(f'processed {self.pos} entries...')
        if self.pos >= len(self.data):
            self.pos = 0  # get ready for the next round of prediction

            return None

        else:
            batch = self.data[self.pos: self.pos + batch_size]
            self.pos += batch_size

            input_ids_batch, \
            input_mask_batch, \
            segment_ids_batch, \
            masked_lm_positions_batch, \
            masked_lm_ids_batch, \
            masked_lm_weights_batch = self.parse(batch)

            input_ids_batch = kr.preprocessing.sequence.pad_sequences(input_ids_batch,
                                                                      self.processor.max_seq_length,
                                                                      padding='post')
            input_mask_batch = kr.preprocessing.sequence.pad_sequences(input_mask_batch,
                                                                       self.processor.max_seq_length,
                                                                       padding='post')
            segment_ids_batch = kr.preprocessing.sequence.pad_sequences(segment_ids_batch,
                                                                        self.processor.max_seq_length,
                                                                        padding='post')

            masked_lm_positions_batch = kr.preprocessing.sequence.pad_sequences(masked_lm_positions_batch,
                                                                                self.processor.max_seq_length - 2,
                                                                                padding='post')
            masked_lm_ids_batch = kr.preprocessing.sequence.pad_sequences(masked_lm_ids_batch,
                                                                          self.processor.max_seq_length - 2,
                                                                          padding='post')
            masked_lm_weights_batch = kr.preprocessing.sequence.pad_sequences(masked_lm_weights_batch,
                                                                              self.processor.max_seq_length - 2,
                                                                              padding='post')

            return input_ids_batch, input_mask_batch, segment_ids_batch, masked_lm_positions_batch, masked_lm_ids_batch, masked_lm_weights_batch

    def parse(self, batch):
        input_ids_batch, \
        input_mask_batch, \
        segment_ids_batch, \
        masked_lm_positions_batch, \
        masked_lm_ids_batch, \
        masked_lm_weights_batch = [], [], [], [], [], []
        for sentence in batch:
            input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = \
                self.processor.create_single_instance(sentence)
            input_ids_batch.append(input_ids)
            input_mask_batch.append(input_mask)
            segment_ids_batch.append(segment_ids)
            masked_lm_positions_batch.append(masked_lm_positions)
            masked_lm_ids_batch.append(masked_lm_ids)
            masked_lm_weights_batch.append(masked_lm_weights)

        return input_ids_batch, input_mask_batch, segment_ids_batch, masked_lm_positions_batch, masked_lm_ids_batch, masked_lm_weights_batch


class Processor(object):
    def __init__(self, vocab_file, max_seq_length):
        self.tokenizer = torch_tokenization.FullTokenizer(vocab_file=vocab_file)
        self.idx_to_word = self.inverse_vocab(self.tokenizer.vocab)
        self.max_seq_length = max_seq_length

    @staticmethod
    def inverse_vocab(vocab):
        idx_to_word = {}
        for word in vocab:
            idx_to_word[vocab[word]] = word
        return idx_to_word

    def create_single_instance(self, sentence):
        # tokenization
        tokens_raw = self.tokenizer.tokenize(torch_tokenization.convert_to_unicode(sentence))

        # add [CLS] and [SEP]
        assert len(sentence) <= self.max_seq_length - 2
        tokens = ["[CLS]"] + tokens_raw + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # produce pseudo ground truth, since the truth is unknown when it comes to spelling checking.
        input_tokens, masked_lm_positions, masked_lm_labels = self.create_pseudo_ground_truth(tokens)

        # convert to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(segment_ids)

        masked_lm_positions = list(masked_lm_positions)
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        # print(input_tokens)

        return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights

    @staticmethod
    def create_pseudo_ground_truth(tokens):
        input_tokens = list(tokens)
        masked_lm_positions = []
        masked_lm_labels = []

        for index, token in enumerate(tokens):

            if token == "[CLS]" or token == "[SEP]":
                continue

            masked_token = tokens[index]  # keep the original token

            input_tokens[index] = masked_token
            masked_lm_positions.append(index)
            masked_lm_labels.append(tokens[index])

        return input_tokens, masked_lm_positions, masked_lm_labels


import numpy as np


class Config(object):
    max_seq_length = 16
    vocab_file = MODEL_PATH + "vocab.txt"
    bert_config_file = MODEL_PATH + "bert_config.json"
    # init_checkpoint = MODEL_PATH+"bert_model.bin"
    bert_config = BertConfig.from_json_file(bert_config_file)
    topn = 5
    bigrams = None  # pickle.load(open('bigram_dict_simplified.sav', 'rb'))


class ModelMy(BertModel):
    def __init__(self, config):
        super(ModelMy, self).__init__()
        self.model = ModelMy(config.bert_config)
        self.processor = Processor(config.vocab_file, config.max_seq_length)

    def topn_predict(self, batch):

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = batch

        self.model()

    def find_topn_candidates(self, sentences, batch_size=1):
        """
        Args
        -----------------------------
        sentences: a list of sentences, e.g., ['the man went to the store.', 'he bought a gallon of milk.']
        batch_size: default=1

        Return
        -----------------------------
        candidates for each token in the sentences, e.g., [[[('the', 0.88), ('a', 0.65)], ...], [...]]

        """
        data = Data(sentences, self.processor)
        stream_res = []
        stream_probs = []
        lengths = []
        while True:
            batch = data.next_predict_batch(batch_size)
            if batch is not None:
                _, id_mask_batch, _, _, _, _ = batch
                topn_probs, topn_predictions = self.model.topn_predict(batch)
                lengths.extend(list(np.sum(id_mask_batch, axis=-1)))
                stream_res.extend(topn_predictions)
                stream_probs.extend(topn_probs)
            else:
                break

        res = []
        pos = 0
        length_id = 0

        while pos < len(stream_res):  # 126  写的比较费劲
            sen = []
            for i in range(self.config.max_seq_length - 2):  # 126
                if i < lengths[length_id] - 2:  # to account for [CLS] and [SEP]  实际的长度
                    token_candidates = []
                    for token_idx, prob in zip(stream_res[pos], stream_probs[pos]):
                        token_candidates.append((self.processor.idx_to_word[token_idx], prob))
                    sen.append(token_candidates)
                pos += 1
            length_id += 1
            res.append(sen)  ### 双层嵌套

        return res


def test():
    config_ = BertConfig().from_json_file(MODEL_PATH + 'bert_config.json')

    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))

    from transformers import BertTokenizer

    # tokenizer = BertTokenizer(MODEL_PATH + 'vocab.txt')
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # print(inputs)

    model_ = BertModel(config_).from_pretrained(MODEL_PATH)

    for name, param in model_.named_parameters():
        print(name)


test()

# if __name__ == '__main__':
#     pass
