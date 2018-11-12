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
parser.add_argument('--model_state', type=str, help='path to director with saved model states')
args = parser.parse_args()



class ModelTrainer():
    def __init__(self):
        #Build dataloaders, vocabulary, and numericalize texts
        self.databunch = TextClasDataBunch.from_csv(args.data, bs = 10)


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

        self.model = LSTM(vocab_size = len(idx_to_word), embedding_dim = 300, hidden_size = 300, word_to_idx = word_to_idx, glove_path = args.embedding)
      #  self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0")
        self.model.to(self.device)

        self.train_dataloader = self.databunch.train_dl
        self.valid_dataloader = self.databunch.valid_dl

        self.epochs = 10
        self.learning_rate = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        for e in range(self.epochs):
            num_correct = 0
            for batch_idx, (data, target) in enumerate(self.train_dataloader):

                #Wrap inputs and targets in Variable
                data, target = Variable(data), Variable(target)


                #Zero Gradients for each batch
                self.optimizer.zero_grad()

                #Forward, backward, and step of optimizer
                result = self.model.forward(data)
                loss = self.loss_function(result, target)
                loss.backward()
                self.optimizer.step()

                #Calculate predicted output
                value, index = torch.max(result.data, 1)
                for i in range(0, 5):
                    if index[i] == target.data[i]:
                        num_correct += 1

                #Print model statistics
                print ('Epoch: ' + str(e + 1) + "\t" + "Progress: " + str(((batch_idx + 1) * 5)) + " / " +  str(len(self.train_dataloader.dataset)) + "\t" + "Loss: " + str(loss.item()))

                #Save model state for ease of access/training later
                torch.save(self.model.state_dict(), args.model_state + "/checkpoint.pth.tar")

            print ("Total Correct:" + num_correct)
            print ("Accuracy: " + num_correct + "/" + len(self.train_dataloader))




if __name__ == "__main__":
     LSTM = ModelTrainer()
     LSTM.train()
