import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from fastai.text import *


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTM,self).__init__()




        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.doc_LSTM = nn.LSTM(embedding_dim, hidden_size, 1, batch_first = True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, )
            nn.ReLu(),
            nn.Dropout(p=0.5),
            nn.Linear(,20)
        )
    def forward(self, input):
        pass
