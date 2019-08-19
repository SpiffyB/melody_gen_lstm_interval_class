'''
Created on 20 Mar 2019

@author: Tom
'''

import sys
import os
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

if len(sys.argv) <= 1:
    sys.exit("ERROR: Not enough input arguments")
    
for (dirpath, dirnames, filenames) in os.walk(sys.argv[1]):
    filenames = list(filter(lambda x: (not '_part_' in x) and ('.mid' in x), filenames))
    
    
    for midi_file in filenames:
        new_mid = MidiFile()
        new_track = MidiTrack()
        new_track.append(Message('program_change', program=12, time=0))
        new_mid.tracks.append(new_track)
        new_mid.type = 1
        
        try:
            mid=mido.MidiFile(dirpath + "\\" + midi_file)
            new_mid.ticks_per_beat = mid.ticks_per_beat
            os.remove(dirpath + "\\" + midi_file)
            
            if mid.type == 0:
                continue
            
        except:
            print("Deleting Invalid Midi: ", midi_file)
            os.remove(dirpath + "\\" + midi_file)
            continue
            
        tick_count = 0
        track_count = 0
        file_count = 0
        current_tempo = MetaMessage('set_tempo', tempo=500000)
        tempo_int = current_tempo.tempo
        
        for track in mid.tracks:
            for msg in track:
                if not msg.is_meta:
                    tick_count = tick_count + msg.time
                    if tick_count < mid.ticks_per_beat * 64: #16 bars in ticks
                        new_track.append(msg)
                    else:
                        tick_count = 0
                        file_count = file_count + 1
                        track_name = midi_file[:-4]
                        new_track.append(MetaMessage('end_of_track'))
                        new_mid.save(dirpath + "\\" + track_name + "_part_"+ str(file_count) + ".mid")
                        
                        new_mid = MidiFile()
                        new_mid.ticks_per_beat = mid.ticks_per_beat
                        new_track = MidiTrack()
                        new_track.append(Message('program_change', program=12, time=0))
                        new_mid.tracks.append(new_track)
                        new_mid.type = 1
                        new_track.append(current_tempo)
                        
                elif msg.type == 'set_tempo':
                    current_tempo = msg
                    new_track.append(current_tempo)

                    

            track_count = track_count + 1
            
            if track_count > 4:
                break
    










