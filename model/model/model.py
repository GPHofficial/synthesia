import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import keras
from keras.utils import to_categorical



class WordLSTM(nn.ModuleList):
    
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size, device):
        super(WordLSTM, self).__init__()
        
        # init the hyperparameters
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = device

        
        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)
        
        # second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # third layer lstm cell
        #self.lstm_3 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # dropout layer
        self.dropout = nn.Dropout(p=0.35)
        
        # fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        
    # forward pass in training   
    def forward(self, x, hc):
        """
        x: input of each batch - shape 128*149 (batch_size*vocab_size)
        hc: tuple of init hidden, cell states - each of shape 128*512 (batch_size*hidden_dim)
        """

        # create empty output seq
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.vocab_size))
        # if using gpu        
        output_seq = output_seq.to(self.device)
        
        # init hidden, cell states for lstm layers
        hc_1, hc_2, hc_3 = hc, hc, hc
        
        # for t-th word in every sequence 
        for t in range(self.sequence_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(x[t], hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3
            
            # dropout and fully connected layer
            output_seq[t] = self.fc(self.dropout(h_2))
            
        return output_seq.view((self.sequence_len * self.batch_size, -1))
          
    def init_hidden(self):
        
        # initialize hidden, cell states for training
        return (torch.zeros(self.batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.batch_size, self.hidden_dim).to(self.device))
    
    def init_hidden_generator(self):
        
        # initialize hidden, cell states for prediction of 1 sequence
        return (torch.zeros(1, self.hidden_dim).to('cpu'),
                torch.zeros(1, self.hidden_dim).to('cpu'))
    
    def predict(self, seed_seq, top_k=5, pred_len=128, int2word=[], word2int=[]):
        """
            1. seed_seq: seed string sequence for prediction (prompt)
            2. top_k: top k words to sample prediction from
            3. pred_len: number of words to generate after the seed seq
        """
        
        # set evaluation mode
        self.eval()
        
        # split string into list of words
        seed_seq = seed_seq.split()
        
        # get seed sequence length
        seed_len = len(seed_seq)
        
        # create output sequence
        out_seq = np.empty(seed_len+pred_len)
        
        # append input seq to output seq
        out_seq[:seed_len] = np.array([word2int[word] for word in seed_seq])
 
        # init hidden, cell states for generation
        hc = self.init_hidden_generator()
        hc_1, hc_2, hc_3 = hc, hc, hc
        
        # feed seed string into lstm
        # get the hidden state set up
        for word in seed_seq[:-1]:
            
            # encode starting word to one-hot encoding
            word = to_categorical(word2int[word], num_classes=self.vocab_size)

            # add batch dimension
            word = torch.from_numpy(word).unsqueeze(0)
            # if using gpu
            word = word.to('cpu') 
            
            # layer 1 lstm
            hc_1 = self.lstm_1(word, hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3            

        word = seed_seq[-1]
        
        # encode starting word to one-hot encoding
        word = to_categorical(word2int[word], num_classes=self.vocab_size)

        # add batch dimension
        word = torch.from_numpy(word).unsqueeze(0)
        # if using gpu
        word = word.to('cpu') 

        # forward pass
        for t in range(pred_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(word, hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3
            
            # fully connected layer without dropout (no need)
            output = self.fc(h_2)
            
            # software to get probabilities of output options
            output = F.softmax(output, dim=1)
            
            # get top k words and corresponding probabilities
            p, top_word = output.topk(top_k)
            # if using gpu           
            p = p.cpu()
            
            # sample from top k words to get next word
            p = p.detach().squeeze().numpy()
            top_word = torch.squeeze(top_word)
            
            word = np.random.choice(top_word, p = p/p.sum())
            
            # add word to sequence
            out_seq[seed_len+t] = word
            
            # encode predicted word to one-hot encoding for next step
            word = to_categorical(word, num_classes=self.vocab_size)
            word = torch.from_numpy(word).unsqueeze(0)
            word = word.to('cpu')
            
        return out_seq
