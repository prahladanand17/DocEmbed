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
        cuda_device = torch.device('cuda')

        self.batch_size = 200
        self.num_epochs = 10


        datapath = args.data
        self.databunch = TextClasDataBunch.from_csv(datapath, bs = self.batch_size)
        self.train_dataloader = self.databunch.train_dl
        self.valid_dataloader = self.databunch.valid_dl
        print torch.cuda.is_available()
    def train(self):
        for batch_idx, (data,target) in enumerate(self.train_dataloader):
            import pdb; pdb.set_trace()



if __name__ == "__main__":
     LSTM = ModelTrainer()
     LSTM.train()
