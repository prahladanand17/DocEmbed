import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path
from data.build_dataset import build_dataset
from fastai.text import *
import os
from models.LSTM import LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to csv file with data')
parser.add_argument('--embedding', type=str, help='path to gloVe file with pretrained embeddings')
args = parser.parse_args()



class ModelTrainer():
    def __init__(self):
        #Build dataloaders, vocabulary, and numericalize texts
        self.databunch = TextClasDataBunch.from_csv(args.data, bs = 200)


        '''
        Build word_to_idx and idx_to_word dictionaries
        for the dataset's vocabulary
        '''

        def build_word_to_idx(idx_to_word):
            word_to_idx = {}
            for i in range(len(idx_to_word)):
                word_to_idx[idx_to_word[i]] = i
            return word_to_idx
        idx_to_word = self.databunch.vocab.itos
        word_to_idx = build_word_to_idx(idx_to_word)

        self.model = LSTM(vocab_size = len(idx_to_word), embedding_dim = 300, hidden_size = 20, word_to_idx = word_to_idx, glove_path = args.embedding)

        self.train_dataloader = self.databunch.train_dl
        self.valid_dataloader = self.databunch.valid_dl
    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            self.model.forward(Variable(data))



if __name__ == "__main__":
     LSTM = ModelTrainer()
     LSTM.train()
