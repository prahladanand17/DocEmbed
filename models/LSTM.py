import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import numpy as np
from fastai.text import *


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, word_to_idx, glove_path):
        super(LSTM,self).__init__()
        #load pretrained gloVe embeddings

        word_embeds = self.load_glove(glove_path)
        embeddings = torch.randn((vocab_size, embedding_dim))
        for word in word_embeds.keys():
            try:
                index = word_to_idx[word]
                embeddings[index] = torch.from_numpy(word_embeds[word])
            except:
                pass
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_((embeddings))
        self.word_embeddings.weight.requires_grad=False
        self.doc_LSTM = nn.LSTM(embedding_dim, hidden_size, 1, batch_first = False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(50,20)
        )
        self.initial_states = self.initialize_states()

    def forward(self, input):
        word_embeddings = self.word_embeddings(input)
        hidden_features, _ = self.doc_LSTM(word_embeddings, self.initial_states)
        doc_embeds = hidden_features.sum(0) / hidden_features.shape[0]
        output = self.classifier(doc_embeds)
        output = F.softmax(output, dim=0)
        return output

    def load_glove(self, fpath):
        words = {}
        with open(fpath, 'r') as fr:
            for line in fr:
                line = line.rstrip("\n")
                sp = line.split(" ")
                emb = [float(sp[i]) for i in range(1, len(sp))]
                assert len(emb) == 300
                words[sp[0]] = np.array(emb, dtype=np.float32)
        return words
    def initialize_states(self, bs):
        #initialize tuple of (h0,c0)
        return (torch.zeros(self.doc_LSTM.num_layers, bs, self.hidden_size),
                torch.zeros(self.doc_LSTM.num_layers, bs, self.hidden_size))
