import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import pretty_midi
from midi2audio import FluidSynth
# import librosa
import numpy as np
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track
from mido import MidiFile
import tqdm
import dill as pickle
from pathlib import Path
import random
import numpy as np
import pandas as pd
from math import floor
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note
import music21
import random
import os, argparse
from model.model import WordLSTM

# model = WordLSTM(sequence_len=256, vocab_size=100, hidden_dim=256, batch_size=1024, device='cpu') #*args, **kwargs
# model.load_state_dict(torch.load('/Users/puayhiang/dev/music-generation/model/model/trained_model.h5', map_location='cpu'))
# model.eval()



def decoder(filename, time_coefficient):

    filedir = './model/output/'

    notetxt = filedir + filename

    with open(notetxt, 'r') as file:
        notestring=file.read()

    score_note = notestring.split(" ")

    # from train
    sample_freq_variable = 12 
    note_range_variable = 62
    note_offset_variable = 33
    number_of_instruments = 2
    chamber_option = True

    # define some parameters (from encoding script)
    sample_freq=sample_freq_variable
    note_range=note_range_variable
    note_offset=note_offset_variable
    chamber=chamber_option
    numInstruments=number_of_instruments

    

    # define variables and lists needed for chord decoding
    speed=time_coefficient/sample_freq
    piano_notes=[]
    violin_notes=[]
    time_offset=0

    # start decoding here
    score = score_note

    i=0

    # for outlier cases, not seen in sonat-1.txt
    # not exactly sure what scores would have "p_octave_" or "eoc" (end of chord?)
    # it seems to insert new notes to the score whenever these conditions are met
    while i<len(score):
        if score[i][:9]=="p_octave_":
            add_wait=""
            if score[i][-3:]=="eoc":
                add_wait="eoc"
                score[i]=score[i][:-3]
            this_note=score[i][9:]
            score[i]="p"+this_note
            score.insert(i+1, "p"+str(int(this_note)+12)+add_wait)
            i+=1
        i+=1


    # loop through every event in the score
    timeNote = {}
    for i in tqdm.auto.tqdm(range(len(score))):

        # if the event is a blank, space, "eos" or unknown, skip and go to next event
        if score[i] in ["", " ", "<eos>", "<unk>"]:
            continue

        # if the event starts with 'end' indicating an end of note
        elif score[i][:3]=="end":

            # if the event additionally ends with eoc, increare the time offset by 1
            if score[i][-3:]=="eoc":
                time_offset+=1
            continue

        # if the event is wait, increase the timestamp by the number after the "wait"
        elif score[i][:4]=="wait":
            time_offset+=int(score[i][4:])
            continue

        # in this block, we are looking for notes   
        else:
            # Look ahead to see if an end<noteid> was generated
            # soon after.  
            duration=1
            has_end=False
            note_string_len = len(score[i])
            for j in range(1,200):
                if i+j==len(score):
                    break
                if score[i+j][:4]=="wait":
                    duration+=int(score[i+j][4:])
                if score[i+j][:3+note_string_len]=="end"+score[i] or score[i+j][:note_string_len]==score[i]:
                    has_end=True
                    break
                if score[i+j][-3:]=="eoc":
                    duration+=1

            if not has_end:
                duration=12

            add_wait = 0
            if score[i][-3:]=="eoc":
                score[i]=score[i][:-3]
                add_wait = 1

            try: 
                if(time_offset in timeNote):
                    timeNote[time_offset].append([score[i][0],int(score[i][1:]),duration]) 
                else:
                    timeNote[time_offset] = [[score[i][0],int(score[i][1:]),duration]]

                new_note=music21.note.Note(int(score[i][1:])+note_offset)    
                new_note.duration = music21.duration.Duration(duration*speed)
                new_note.offset=time_offset*speed
                if score[i][0]=="v":
                    violin_notes.append(new_note)
                else:
                    piano_notes.append(new_note)                
            except:
                print("Unknown note: " + score[i])

            time_offset+=add_wait

    # list of all notes for each instrument should be ready at this stage

    # creating music21 instrument objects      
    
    piano=music21.instrument.fromString("Piano")
    violin=music21.instrument.fromString("Violin")

    # insert instrument object to start (0 index) of notes list
    
    piano_notes.insert(0, piano)
    violin_notes.insert(0, violin)
    # create music21 stream object for individual instruments
    
    piano_stream=music21.stream.Stream(piano_notes)
    violin_stream=music21.stream.Stream(violin_notes)
    # merge both stream objects into a single stream of 2 instruments
    note_stream = music21.stream.Stream([piano_stream, violin_stream])

    note_stream.write('midi', fp="./model/output/"+filename[:-4]+".mid")

    FluidSynth("./model/dataset/font.sf2", 16000).midi_to_audio('./model/output/output.mid', './model/output/output.wav')

    print("Done! Decoded midi file saved to 'content/'")
    return timeNote

def predictApp(seed_prompt = "p25",tokens_to_generate = 512, time_coefficient = 4, top_k_coefficient = 12):

    model = torch.load('./model/model/trained_model.h5', map_location='cpu')

    model.eval()

    select_training_dataset_file = "./model/dataset/notewise_chamber.txt"

    # replace with any text file containing full set of data
    MIDI_data = select_training_dataset_file

    with open(MIDI_data, 'r') as file:
        text = file.read()

    # get vocabulary set
    words = sorted(tuple(set(text.split())))
    n = len(words)

    word2int = dict(zip(words, list(range(n))))
    int2word = dict(zip(list(range(n)), words))

    with open("./model/output/output.txt", "w") as outfile:
        outfile.write(' '.join([int2word[int_] for int_ in model.predict(seed_seq=seed_prompt, pred_len=tokens_to_generate, top_k=top_k_coefficient, int2word=int2word, word2int=word2int)]))
    
    return decoder('output.txt', time_coefficient)

if __name__ == "__main__":

    predictApp()
