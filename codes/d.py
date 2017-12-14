import pickle

import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn


word_vec = None
with open('filename.pickle', 'rb') as handle:
    word_vec = pickle.load(handle)


nli_net = torch.load("savedir/model.pickle")


s1 = "This is the real deal"
s2 = "This is the real deal for real"

s1_batch, s1_len = get_batch(s1, word_vec)
s2_batch, s2_len = get_batch(s2, word_vec)
s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

# model forward
output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
print(output)

