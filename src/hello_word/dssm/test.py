# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Description:  
# Author:       lenovo
# Date:         2020/9/11
# -------------------------------------------------------------------------------

from torchstat import stat
from transformers import BertConfig

from dssm.model_torch import *

config = BertConfig.from_pretrained('./config.json')
model = DSSMOne(config)
stat(model, (256, 256))

if __name__ == '__main__':
    pass
