import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from fastai.text import *


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, word_to_idx, glove_path):
        super(LSTM,self).__init__()
        #load pretrained gloVe embeddings

        word_embeds = load_glove(glove_path)
        embeddings = np.zeros((vocab_size, embedding_size))
        for word in word_embeds.keys():
            index = word_to_idx[word]
            embeddings[index] = word_embeds[word]
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size).cuda()
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))


        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.doc_LSTM = nn.LSTM(embedding_dim, hidden_size, 1, batch_first = True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 50)
            nn.ReLu(),
            nn.Dropout(p=0.5),
            nn.Linear(50,20)
        )
    def forward(self, input):
        word_embeddings = self.word_embeddings(input)
        import pdb; pdb.set_trace()

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
