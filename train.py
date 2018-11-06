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

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to csv file with data')
args = parser.parse_args()



class ModelTrainer():
    def __init__(self):

        self.batch_size = 100
        self.num_epochs = 10

        datapath = args.data
        self.databunch = TextClasDataBunch.from_csv(datapath, bs = 100)
        self.train_dataloader = self.databunch.train_dl
        import pdb; pdb.set_trace()
        self.train_dataloader.batch_size = self.batch_size
        self.train_dataloader.shuffle = True
        self.valid_dataloader = self.databunch.valid_dl



if __name__ == "__main__":
     LSTM = ModelTrainer()
