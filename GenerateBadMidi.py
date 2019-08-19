'''
Created on 1 Apr 2019

@author: Tom
'''

import numpy as np
import os
import random
from mido import Message, MidiFile, MidiTrack

if __name__ == '__main__':
    pass

notes = np.arange(36,72,1,dtype=int)
note_lengths = [40,60,80,120,160,240,320,640,960]


mid_count = 0
folder_count = 0
while mid_count < 100000:
    if mid_count%1000 == 0:
        folder_count = folder_count + 1
        print(os.getcwd() + 'midi\\bad\\' + str(folder_count))
        if not os.path.exists(os.getcwd() + '\\midi\\bad\\' + str(folder_count)):
            os.makedirs(os.getcwd() + '\\midi\\bad\\' + str(folder_count))

        
        
    mid = MidiFile()
    track = MidiTrack()
    track.append(Message('program_change', program=12, time=0))
    mid.tracks.append(track)
    mid.type = 1
        
    while mid.length < 10.0:
        rnd_note = random.choice(notes)
        rnd_length1 = random.choice(note_lengths)
        rnd_length2 = random.choice(note_lengths)
            
        if mid.length < 9.5: 
            track.append(Message('note_on', channel=1, note=rnd_note, velocity=100, time=rnd_length1))
            track.append(Message('note_off', channel=1, note=rnd_note, velocity=64, time=rnd_length2))
        else:
            break
        

    mid.save('midi\\bad\\' + str(folder_count) + '\\bad-midi-' + str(mid_count) + '.mid')

    mid_count = mid_count + 1
        
        