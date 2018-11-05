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

        datapath = args.data
        self.databunch = TextClasDataBunch.from_csv(datapath)
        self.train_dataloader = databunch.train_dl
        self.valid_dataloader = databunch.valid_dl



    def train():
	pass
