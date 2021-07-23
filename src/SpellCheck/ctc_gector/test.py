# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Description:  
# Author:       lenovo
# Date:         2021/7/21
# -------------------------------------------------------------------------------

import json

import tokenization


def test(path_):
    source_ = []
    target_ = []
    with open(path_, 'r') as f:
        for one in f.readlines():
            one = one.rstrip('\n')
            tmp_ = json.loads(one)
            # print(tmp_['source'])
            source_.append(tmp_['source'])
            target_.append((tmp_['target']))

    with open(path_ + '_s', 'w') as f:
        f.write('\n'.join(source_) + '\n')

    with open(path_ + '_t', 'w') as f:
        f.write('\n'.join(target_) + '\n')


def fenci(path_):
    tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)

    with open(path_, 'r') as f, open(path_ + '.tok', 'w') as f_out:
        for line in tqdm(f):
            line = line.strip()
            items = line.split('\t')

            line = tokenization.convert_to_unicode(items[0])
            if not line:
                print()
                continue

            tokens = tokenizer.tokenize(line)
            f_out.write(' '.join(tokens) + '\n')


from utils.preprocess_data import *


def mergeData(path_s, path_t, path_o):
    convert_data_from_raw_files(path_s, path_t, path_o, 1000000, 128)


if __name__ == '__main__':
    path_ = '/home/zjj/Data/SpellCheck/Chinese/TrainData/train_small.json'
    test(path_)
    fenci(path_ + '_s')
    fenci(path_ + '_t')
    mergeData(path_ + '_s.tok', path_ + '_t.tok', path_ + '_.out')
