import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import time

import pretty_midi
from midi2audio import FluidSynth

import librosa
import numpy as np
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track


import keras
from keras.utils import to_categorical


from mido import MidiFile
import tqdm

import os

from model.model import WordLSTM

print(os.getcwd())

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available Device:', device)

def get_batches(arr, n_seqs, n_words):
    """
        create generator object that returns batches of input (x) and target (y).
        x of each batch has shape 128*128*149 (batch_size*seq_len*vocab_size).
        
        accepts 3 arguments:
        1. arr: array of words from text data
        2. n_seq: number of sequence in each batch (aka batch_size)
        3. n_word: number of words in each sequence
    """
    
    # compute total elements / dimension of each batch
    batch_total = n_seqs * n_words
    
    # compute total number of complete batches
    n_batches = arr.size//batch_total
    
    # chop array at the last full batch
    arr = arr[: n_batches* batch_total]
    
    # reshape array to matrix with rows = no. of seq in one batch
    arr = arr.reshape((n_seqs, -1))
    
    # for each n_words in every row of the dataset
    for n in range(0, arr.shape[1], n_words):
        
        # chop it vertically, to get the input sequences
        x = arr[:, n:n+n_words]
        
        # init y - target with shape same as x
        y = np.zeros_like(x)
        
        # targets obtained by shifting by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, n+n_words]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        
        # yield function is like return, but creates a generator object
        yield x, y

if __name__ == "__main__":

    sample_freq_variable = 12 
    note_range_variable = 62
    note_offset_variable = 33
    number_of_instruments = 2
    chamber_option = True

    select_training_dataset_file = "./dataset/notewise_chamber.txt"

    # replace with any text file containing full set of data
    MIDI_data = select_training_dataset_file

    with open(MIDI_data, 'r') as file:
        text = file.read()

    # get vocabulary set
    words = sorted(tuple(set(text.split())))
    n = len(words)

    # create word-integer encoder/decoder
    word2int = dict(zip(words, list(range(n))))
    int2word = dict(zip(list(range(n)), words))

    # encode all words in dataset into integers
    encoded = np.array([word2int[word] for word in text.split()])


    training_batch_size = 1024
    attention_span_in_tokens = 256
    hidden_dimension_size = 256
    test_validation_ratio = 0.1 
    learning_rate = 0.001 

    # compile the network - sequence_len, vocab_size, hidden_dim, batch_size
    net = WordLSTM(sequence_len=attention_span_in_tokens, vocab_size=len(word2int), hidden_dim=hidden_dimension_size, batch_size=training_batch_size, device=device)
    # if using gpu
    net.to(device)

    # define the loss and the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # split dataset into 90% train and 10% using index
    val_idx = int(len(encoded) * (1 - test_validation_ratio))
    train_data, val_data = encoded[:val_idx], encoded[val_idx:]

    # empty list for the validation losses
    val_losses = list()

    # empty list for the samples
    samples = list()
    number_of_training_epochs = 1 #@param {type:"slider", min:1, max:300, step:1}

    # track time
    start_time = time.time()

    # declare seed sequence
    #seed_string = "p47 p50 wait8 endp47 endp50 wait4 p47 p50 wait8 endp47 endp50"

    # finally train the model
    for epoch in tqdm.auto.tqdm(range(number_of_training_epochs)):
        
        # init the hidden and cell states to zero
        hc = net.init_hidden()
        
        # (x, y) refers to one batch with index i, where x is input, y is target
        for i, (x, y) in enumerate(get_batches(train_data, training_batch_size, hidden_dimension_size)):
            
            # get the torch tensors from the one-hot of training data
            # also transpose the axis for the training set and the targets
            x_train = torch.from_numpy(to_categorical(x, num_classes=net.vocab_size).transpose([1, 0, 2]))
            targets = torch.from_numpy(y.T).type(torch.LongTensor)  # tensor of the target
            
            # if using gpu
            x_train = x_train.to(device)
            targets = targets.to(device)
            
            # zero out the gradients
            optimizer.zero_grad()
            
            # get the output sequence from the input and the initial hidden and cell states
            # calls forward function
            output = net(x_train, hc)
        
            # calculate the loss
            # we need to calculate the loss across all batches, so we have to flat the targets tensor
            loss = criterion(output, targets.contiguous().view(training_batch_size*hidden_dimension_size))
            
            # calculate the gradients
            loss.backward()
            
            # update the parameters of the model
            optimizer.step()
            
            # track time
        
            # feedback every 100 batches
            if i % 100 == 0:
                
                # initialize the validation hidden state and cell state
                val_h, val_c = net.init_hidden()
                
                for val_x, val_y in get_batches(val_data, training_batch_size, hidden_dimension_size):
            
                    # prepare the validation inputs and targets
                    val_x = torch.from_numpy(to_categorical(val_x).transpose([1, 0, 2]))
                    val_y = torch.from_numpy(val_y.T).type(torch.LongTensor).contiguous().view(training_batch_size*hidden_dimension_size)
    
                    # if using gpu
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
            
                # track time
                duration = round(time.time() - start_time, 1)
                start_time = time.time()
        
                print("Epoch: {}, Batch: {}, Duration: {} sec, Test Loss: {}".format(epoch, i, duration, loss.item()))

    torch.save(net, '/music-generator/saved_models/trained_model.h5')