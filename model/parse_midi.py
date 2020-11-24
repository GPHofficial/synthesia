import tqdm.auto
import argparse
import random
import os
import numpy as np
from math import floor
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note
import music21
from music21 import instrument, volume
from music21 import midi as midiModule
from pathlib import Path
import glob, sys
from music21 import converter, instrument

notes=[]
InstrumentID=0
folder = '/music-generator/midis/*mid'

for file in tqdm.auto.tqdm(glob.glob(folder)):
    filename = file[-53:]
    print(filename)
    fname = filename

    mf=music21.midi.MidiFile()
    mf.open(fname)
    mf.read()
    mf.close()
    midi_stream=music21.midi.translate.midiFileToStream(mf)

    sample_freq=sample_freq_variable
    note_range=note_range_variable
    note_offset=note_offset_variable
    chamber=chamber_option
    numInstruments=number_of_instruments

    s = midi_stream

    maxTimeStep = floor(s.duration.quarterLength * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, numInstruments, note_range))

    # define two types of filters because notes and chords have different structures for storing their data
    # chord have an extra layer because it consist of multiple notes

    noteFilter=music21.stream.filters.ClassFilter('Note')
    chordFilter=music21.stream.filters.ClassFilter('Chord')
      

    # pitch.midi-note_offset: pitch is the numerical representation of a note. 
    #                         note_offset is the the pitch relative to a zero mark. eg. B-=25, C=27, A=24

    # n.offset: the timestamps of each note, relative to the start of the score
    #           by multiplying with the sample_freq, you make all the timestamps integers

    # n.duration.quarterLength: the duration of that note as a float eg. quarter note = 0.25, half note = 0.5
    #                           multiply by sample_freq to represent duration in terms of timesteps

    notes = []
    instrumentID = 0
    parts = instrument.partitionByInstrument(s)
    for i in range(len(parts.parts)): 
      instru = parts.parts[i].getInstrument()
      

    for n in s.recurse().addFilter(noteFilter):
        if chamber:
          # assign_instrument where 0 means piano-like and 1 means violin-like, and -1 means neither
          if instru.instrumentName == 'Violin':
            notes.append((n.pitch.midi-note_offset, floor(n.offset*sample_freq), 
              floor(n.duration.quarterLength*sample_freq), 1))
            
        notes.append((n.pitch.midi-note_offset, floor(n.offset*sample_freq), 
              floor(n.duration.quarterLength*sample_freq), 0))
        
    # do the same using a chord filter

    for c in s.recurse().addFilter(chordFilter):
        # unlike the noteFilter, this line of code is necessary as there are multiple notes in each chord
        # pitchesInChord is a list of notes at each chord eg. (<music21.pitch.Pitch D5>, <music21.pitch.Pitch F5>)
        pitchesInChord=c.pitches
        
        # do same as noteFilter and append all notes to the notes list
        for p in pitchesInChord:
            notes.append((p.midi-note_offset, floor(c.offset*sample_freq), 
                          floor(c.duration.quarterLength*sample_freq), 1))

        # do same as noteFilter and append all notes to the notes list
        for p in pitchesInChord:
            notes.append((p.midi-note_offset, floor(c.offset*sample_freq), 
                          floor(c.duration.quarterLength*sample_freq), 0))

    # the variable/list "notes" is a collection of all the notes in the song, not ordered in any significant way

    for n in notes:
        
        # pitch is the first variable in n, previously obtained by n.midi-note_offset
        pitch=n[0]
        
        # do some calibration for notes that fall our of note range
        # i.e. less than 0 or more than note_range
        while pitch<0:
            pitch+=12
        while pitch>=note_range:
            pitch-=12
            
        # 3rd element refers to instrument type => if instrument is violin, use different pitch calibration
        if n[3]==1:      #Violin lowest note is v22
            while pitch<22:
                pitch+=12

        # start building the 3D-tensor of shape: (796, 1, 38)
        # score_arr[0] = timestep
        # score_arr[1] = type of instrument
        # score_arr[2] = pitch/note out of the range of note eg. 38
        
        # n[0] = pitch
        # n[1] = timestep
        # n[2] = duration
        # n[3] = instrument
        #print(n[3])
        try:
          score_arr[n[1], n[3], pitch]=1                  # Strike note
          score_arr[n[1]+1:n[1]+n[2], n[3], pitch]=2      # Continue holding note
        except:
          continue

    instr={}
    instr[0]="p"
    instr[1]="v"

    score_string_arr=[]

    # loop through all timesteps
    for timestep in score_arr:
        
        # selecting the instruments: i=0 means piano and i=1 means violin
        for i in list(reversed(range(len(timestep)))):   # List violin note first, then piano note
            
            score_string_arr.append(instr[i]+''.join([str(int(note)) for note in timestep[i]]))

    modulated=[]
    # get the note range from the array
    note_range=len(score_string_arr[0])-1

    for i in range(0,12):
        for chord in score_string_arr:
            
            # minus the instrument letter eg. 'p'
            # add 6 zeros on each side of the string
            padded='000000'+chord[1:]+'000000'
            
            # add back the instrument letter eg. 'p'
            # append window of len=note_range back into 
            # eg. if we have "00012345000"
            # iteratively, we want to get "p00012", "p00123", "p01234", "p12345", "p23450", "p34500", "p45000",
            modulated.append(chord[0]+padded[i:i+note_range])

    # input of this function is a modulated string
    long_string = modulated

    translated_list=[]

    # for every timestep of the string
    for j in range(len(long_string)):
        
        # chord at timestep j eg. 'p00000000000000000000000000000000000100'
        chord=long_string[j]
        next_chord=""
        
        # range is from next_timestep to max_timestep
        for k in range(j+1, len(long_string)):
            
            # checking if instrument of next chord is same as current chord
            if long_string[k][0]==chord[0]:
                
                # if same, set next chord as next element in modulation
                # otherwise, keep going until you find a chord with the same instrument
                # when you do, set it as the next chord
                next_chord=long_string[k]
                break
        
        # set prefix as the instrument
        # set chord and next_chord to be without the instrument prefix
        # next_chord is necessary to check when notes end
        prefix=chord[0]
        chord=chord[1:]
        next_chord=next_chord[1:]
        
        # checking for non-zero notes at one particular timestep
        # i is an integer indicating the index of each note the chord
        for i in range(len(chord)):
            
            if chord[i]=="0":
                continue
            
            # set note as 2 elements: instrument and index of note
            # examples: p22, p16, p4
            #p = music21.pitch.Pitch()
            #nt = music21.note.Note(p)
            #n.volume.velocity = 20
            #nt.volume.client == nt
            #V = nt.volume.velocity
            #print(V)
            #note=prefix+str(i)+' V' + str(V)
            note=prefix+str(i)                
            
            # if note in chord is 1, then append the note eg. p22 to the list
            if chord[i]=="1":
                translated_list.append(note)
            
            # If chord[i]=="2" do nothing - we're continuing to hold the note
            
            # unless next_chord[i] is back to "0" and it's time to end the note.
            if next_chord=="" or next_chord[i]=="0":      
                translated_list.append("end"+note)

        # wait indicates end of every timestep
        if prefix=="p":
            translated_list.append("wait")

    #print(len(translated_list))
    translated_list[:10]

    # this section transforms the list of notes into a string of notes

    # initialize i as zero and empty string
    i=0
    translated_string=""


    while i<len(translated_list):
        
        # stack all the repeated waits together using an integer to indicate the no. of waits
        # eg. "wait wait" => "wait2"
        wait_count=1
        if translated_list[i]=='wait':
            while wait_count<=sample_freq*2 and i+wait_count<len(translated_list) and translated_list[i+wait_count]=='wait':
                wait_count+=1
            translated_list[i]='wait'+str(wait_count)
            
        # add next note
        translated_string+=translated_list[i]+" "
        i+=wait_count

    chordwise_folder = "../"
    notewise_folder = "../"

    # export notewise encoding
    f=open(notewise_folder+fname+"_notewise"+".txt","w+")
    f.write(translated_string)
    f.close()

folder = '/music-generator/midis/*notewise.txt'

filenames = glob.glob('/music-generator/')
with open('notewise_custom_dataset.txt', 'w') as outfile:
    for fname in glob.glob(folder)[-53:]:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)