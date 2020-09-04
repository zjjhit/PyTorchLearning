# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Description:  
# Author:       lenovo
# Date:         2020/8/26
# -------------------------------------------------------------------------------

import numpy as np
import torch


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."

    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        # src = torch.tensor(data, requires_grad=False)
        # tgt = torch.tensor(data, requires_grad=False)
        src = data.clone().detach().requires_grad_(False)
        tgt = data.clone().detach().requires_grad_(False)
        yield Batch(src, tgt, 0)


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
        self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


if __name__ == '__main__':
    data_ = data_gen(100, 10, 5)
    for d in data_:
        print(d.trg_mask.shape)
        t = d.trg_mask
    print(t)
