import os
import torch
import argparse
import torch.nn as nn
from pathlib import Path
from fastai.text import *
import torch.optim as optim
from models.LSTM import LSTM
from models.GloVe_embed import Word_Vector_Model
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


parser = argparse.ArgumentParser()

parser.add_argument('--model', type='str', help='type of model we are training')
parser.add_argument('--data', type=str, help='path to csv file with data')
parser.add_argument('--embedding', type=str, help='path to gloVe file with pretrained embeddings')
parser.add_argument('--save', type=str, help='path to director with saved model states')

args = parser.parse_args()



class ModelTrainer():
    def __init__(self):
        #Build dataloaders, vocabulary, and numericalize texts
        self.databunch = TextClasDataBunch.from_csv(args.data, bs = 10, csv_name='data.csv', pad_first=True, pad_idx = 1)

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

        models = {}

        models['LSTM'] = LSTM(vocab_size = len(idx_to_word), embedding_dim = 300, hidden_size = 300, word_to_idx = word_to_idx, glove_path = args.embedding)
        models['GloVe'] = Word_Vector_Model(vocab_size = len(idx_to_word), embedding_dim = 300, word_to_idx = word_to_idx, glove_path = args.embedding)

        self.model = models[args.model]
        #self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0")
        self.model.to(self.device)

        self.train_dataloader = self.databunch.train_dl
        self.valid_dataloader = self.databunch.valid_dl

        self.epochs = 20
        self.learning_rate = 0.005
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        losses = []
        epochs = []
        accuracies = []
        for e in range(self.epochs):
            avg_loss = 0
            epochs.append(e)
            num_correct = 0
            for batch_idx, (data, target) in enumerate(self.train_dataloader):

                #Detach LSTM hidden state from previous sequence if LSTM
                if(args.model == 'LSTM'):
                    self.model.initial_states = self.model.initialize_states(len(data[1]))
                else:
                    pass

                #Wrap inputs and targets in Variable
                data, target = (Variable(data)).to(self.device), (Variable(target)).to(self.device)


                #Zero Gradients for each batch
                self.optimizer.zero_grad()

                #Forward, backward, and step of optimizer
                result = self.model.forward(data)
                loss = self.loss_function(result, target)
                avg_loss += loss
                loss.backward()
                self.optimizer.step()

                #Calculate predicted output
                value, index = torch.max(result.data, 1)
                for i in range(0, len(target.data)):
                    if index[i] == target.data[i]:
                        num_correct += 1

                #Print model statistics
                print ('Epoch: ' + str(e + 1) + "\t" + "Progress: " + str(((batch_idx + 1) * 5)) + " / " +  str(len(self.train_dataloader.dataset)) + "\t" + "Loss: " + str(loss.item()))

                #Save model state for ease of access/training later
                torch.save(self.model.state_dict(), args.save)

            print ("Total Correct:" + str(num_correct))
            print ("Accuracy: " + str(num_correct/len(self.train_dataloader.dataset)))

            avg_loss = avg_loss/len(self.train_dataloader.dataset)
            losses.append(avg_loss)

            accuracies.append(num_correct/len(self.train_dataloader.dataset))


        #Plot loss
        plt.xlabel('# of epochs')
        plt.ylabel('Train Loss')
        plt.title('Epochs vs. Loss')
        plt.grid(True)
        plt.plot(epochs, losses)
        if args.model == 'LSTM':
            plt.savefig('BBC_LSTM_loss_epoch.png')
        else:
            plt.savefig('BBC_avgembed_loss_epoch.png')
        plt.close()

        #Plot Accuracy
        plt.xlabel('# of epochs')
        plt.ylabel('Accuracy')
        plt.title('Epochs vs. Accuracy')
        plt.grid(True)
        plt.plot(epochs, accuracies)
        if args.model == 'LSTM':
            plt.savefig('BBC_LSTM_acc_epoch.png')
        else:
            plt.savefig('BBC_avgembed_acc_epoch.png')
        plt.close()

    def validation(self):
        self.model.load_state_dict(torch.load(args.save))
        self.model.eval()

        losses = []
        accuracies = []
        for e in range(self.epochs):
            avg_loss = 0
            num_correct = 0
            for data, target in enumerate(self.valid_dataloader):

                #Detach LSTM hidden state from previous sequence
                self.model.initial_states = self.model.initialize_states(1)

                #Wrap inputs and targets in Variable
                data, target = (Variable(data)).to(self.device), (Variable(target)).to(self.device)

                #Forward, backward, and step of optimizer
                result = self.model.forward(data)
                loss = self.loss_function(result, target)
                avg_loss += loss

                #Calculate predicted output
                value, index = torch.max(result.data, 1)
                for i in range(0, len(target.data)):
                    if index[i] == target.data[i]:
                        num_correct += 1

                #Print model statistics
                print ('Epoch: ' + str(e + 1) + "\t" + "Progress: " + str(((batch_idx + 1))) + " / " +  str(len(self.valid_dataloader.dataset)) + "\t" + "Loss: " + str(loss.item()))

            print ("Total Correct:" + str(num_correct))
            print ("Accuracy: " + str(num_correct/len(self.valid_dataloader.dataset)))

            avg_loss = avg_loss/len(self.valid_dataloader.dataset)
            losses.append(avg_loss)

            accuracies.append(num_correct/len(self.valid_dataloader.dataset))

        print ("Max Accuracy: " + str(max(accuracies)))
        print ("Min Loss: " + str(min(accuracies)))






if __name__ == "__main__":
     model = ModelTrainer()
     model.train()
     model.validation()
