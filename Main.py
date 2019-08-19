'''
Created on 3 Feb 2019

@author: Tom
'''

from numpy import float32
from builtins import len, str


if __name__ == '__main__':
    pass

import os
import mido
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from mido import MidiFile, Message, MidiTrack
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

is_playing = False

'''
Changes the is_playing boolean
'''
def change_is_playing():
    global is_playing
    if is_playing == False:
        is_playing = True
    else:
        is_playing = False

'''
Gets the is_playing boolean
'''
def get_is_playing():
    global is_playing
    return is_playing

'''
Gets the pitch class interval from the current and previous note
'''
def get_pc_interval(prev_note, current_note):
    interval = (abs(prev_note - current_note)) % 12
    
    if interval > 6:
        interval = 12 - interval
    return interval
    

def play_midi(midi_file):
    global is_playing
    try:
        port = mido.open_output(None, autoreset=True)
        
        if isinstance(midi_file, MidiFile):
            file = midi_file
        else:
            file = MidiFile(midi_file)
        
        for msg in file.play():
            if is_playing == True:
                port.send(msg)
            else:
                return
    except KeyboardInterrupt:
        pass


'''
imports all midi files 
midi_group:
    0 for training files
    1 for test files

return:
    list of type MidiFile
'''
def import_midi():


    directory = os.getcwd() + '\\midi' # CHANGE TO MIDI
    list_good = []
    list_bad = []
    target_good = []
    target_bad  = []
    
    #count number of good midi files
    file_count_good = 0
    for (dirpath, dirnames, filenames) in os.walk(directory + '\\good'):
        file_count_good = file_count_good + len(filenames)

    
    #count number of bad midi files
    file_count_bad = 0
    for (dirpath, dirnames, filenames) in os.walk(directory + '\\bad'):
        file_count_bad = file_count_bad + len(filenames)



    load_count = 0
    for (dirpath, dirnames, filenames) in os.walk(directory + '\\good'):
        for midi_file in filenames:
            try:
                mid=MidiFile(dirpath + "\\" + midi_file)
                list_good.append(mid)
                target_good.append([1, midi_file])
            except:
                print("Invalid Midi: ", midi_file)
            
            load_count = load_count + 1
            
            if load_count%500 == 0:
                print("Good Files Loaded: " + str(load_count) + "/" + str(file_count_good))

    
    load_count = 0
    for (dirpath, dirnames, filenames) in os.walk(directory + '\\bad'):
        for midi_file in filenames:
            try:
                mid=MidiFile(dirpath + "\\" + midi_file)
                list_bad.append(mid)
                target_bad.append([0, midi_file])
            except:
                print("Invalid Midi: ", midi_file)
            
            load_count = load_count + 1
            
            if load_count%500 == 0:
                print("Bad Files Loaded: " + str(load_count) + "/" + str(file_count_bad))

    
    
    data_list = list_good + list_bad

    target_list = target_good + target_bad
    comb = list(zip(list_good + list_bad, target_good + target_bad))
    random.shuffle(comb)

    
    data_list[:], target_list[:] = zip(*comb)
    print("Length Data List: ", len(data_list))
    print("Length Target List: ", len(target_list))

    
    return (data_list, target_list)


def convert_to_one_hot(midi_data):
    '''
    Converts a numpy array of features to one hot and extends the dimensions
    
    Parameters
    ---------
    midi_data: array
        numpy array of converted data (from convert_to_data)
        
    Return
    ------
    one_hot_data: array
        one hot representation of data
    '''
    one_hot_data = np.zeros((len(midi_data), 128, 7), dtype=float32)
    
    for i, melody in enumerate(midi_data):
        for ii, note in enumerate(melody):
            pc_interval = note[0]
            
            if pc_interval == 0.0:
                one_hot_data[i,ii,int(pc_interval)] = 1.0
            elif pc_interval == 1.0:
                one_hot_data[i,ii,int(pc_interval)] = 1.0
            elif pc_interval == 2.0:
                one_hot_data[i,ii,int(pc_interval)] = 1.0
            elif pc_interval == 3.0:
                one_hot_data[i,ii,int(pc_interval)] = 1.0
            elif pc_interval == 4.0:
                one_hot_data[i,ii,int(pc_interval)] = 1.0
            elif pc_interval == 5.0:
                one_hot_data[i,ii,int(pc_interval)] = 1.0
            else:
                one_hot_data[i,ii,int(pc_interval)] = 1.0
    
    return one_hot_data
    


'''
converts a list of midi files to a numpy array which the classifier can use
'''
def convert_to_data(midi_file_list):
    '''
    converts a list of midi files to a numpy array which the classifier can use
    '''
    midi_data = np.zeros((len(midi_file_list), 128, 3), dtype=float32)

    for i,midi_file in enumerate(midi_file_list):
        sort_list = []
        for track in midi_file.tracks:
            total_time = 0
            id_count = 0
            for msg in track:
                if not msg.is_meta:
                    if msg.type == 'note_on':
                        total_time = total_time + msg.time
                        if msg.velocity != 0:
                            msg_id = str(id_count).zfill(6)
                            id_count = id_count + 1
                            
                            sort_list.append([msg_id , msg.note, msg.time, total_time, -1, -1])
                        else:
                            matches = [msg_list for msg_list in sort_list if msg_list[1] == msg.note and msg_list[4] == -1 and msg_list[5] == -1]
                            if not matches == []:
                                msg_index = sort_list.index(matches[0])
                                sort_list[msg_index][4] = msg.time
                                sort_list[msg_index][5] = total_time
                        
                    elif msg.type == 'note_off':
                        total_time = total_time + msg.time
                        
                        matches = [msg_list for msg_list in sort_list if msg_list[1] == msg.note and msg_list[4] == -1 and msg_list[5] == -1]
                        if not matches == []:
                            msg_index = sort_list.index(matches[0])
                            sort_list[msg_index][4] = msg.time
                            sort_list[msg_index][5] = total_time

        sort_list = list(filter(lambda x: not x[5] == -1 and not x[4] == -1, sort_list))
        
        prev_note = -1
        for ii, msg in enumerate(sort_list):
            if ii < 128:
                if prev_note < 0:
                    midi_data[i,ii,0] = 0.0
                else:
                    midi_data[i,ii,0] = get_pc_interval(prev_note, msg[1])
                    
                midi_data[i,ii,1] = round(msg[2] / midi_file.ticks_per_beat, 1)
                midi_data[i,ii,2] = round((msg[5] - msg[3]) / midi_file.ticks_per_beat, 1)
                prev_note = msg[1]
        
    midi_data = convert_to_one_hot(midi_data)
    return midi_data



def train_classifier(training_data, 
                     training_target, 
                     test_data, 
                     test_target, 
                     epoch, 
                     neurons, 
                     batch,
                     num_layers,
                     dropout_amount,
                     act_func,
                     opt_func):
    
    model_name = "model-e" + str(epoch) + "-n" + str(neurons) + "-b" + str(batch) + "-lay" + str(num_layers) + "-drop" + str(dropout_amount) + "-act(" + str(act_func)+ ")-opt(" + str(opt_func)+ ")-" + str(int(time.time()))

    tf.global_variables_initializer
    data_dim = 7
    maxlen = 128
    act_func = 'sigmoid'
    opt_func = 'rmsprop'
    training_data = training_data[:,:maxlen,:]
    test_data = test_data[:,:maxlen,:]
    
    model = Sequential()
    model.add(LSTM(neurons, activation=act_func, return_sequences=False,input_shape=(maxlen, data_dim)))
    model.add(Dense(1, activation=act_func))
    
    tb_callback = TensorBoard(log_dir="logs\\" + model_name ,
                            histogram_freq=0, 
                            batch_size=32, 
                            write_graph=True, 
                            write_grads=False, 
                            write_images=False)
    
    early_stop = EarlyStopping(monitor='loss',
                               min_delta=0.0005, 
                               patience=4,
                               verbose=1,
                               mode='min',
                               baseline=0.8,
                               restore_best_weights=False)
        
    model.compile(loss='binary_crossentropy',
                  optimizer=opt_func,
                  metrics=['accuracy'])


    train_history = model.fit(training_data, 
                              training_target,
                              batch_size=batch, 
                              epochs=epoch,
                              validation_data=(test_data, test_target), 
                              verbose=1,
                              callbacks=[tb_callback])
    
    model.save("models\\" + model_name + ".h5", True, True)
    return (model,train_history)

 

def generate_melodies(key, scale, num_return):
    '''
    Generates a list of random midi files
    
    Parameters
    ----------
    key: int
        midi note number of the musical key to generate melodies in
        
    scale: String
        the scale to generate melodies in
    
    num_return: int 
        number of midi files that are returned
        
    Returns
    -------
    midi_list: list
        List of random midi files
    '''

    major = np.array([0,2,4,5,7,9,11])
    minor = np.array([0,2,3,5,7,8,10])
    
    notes = np.arange(36,72,1,dtype=int)
    
    if scale == 'Minor':
        notes = notes[np.isin(notes%12, minor)] + key
    elif scale == 'Major':
        notes = notes[np.isin(notes%12, major)] + key
        
    note_lengths = [240,480,720]
    time_between_notes = [0,240]
    
    midi_list = []
    i=0
    while i < num_return:
        mid = MidiFile()
        track = MidiTrack()
        track.append(Message('program_change', program=12, time=0))
        mid.tracks.append(track)
        mid.type = 1
        
        i=i+1
        while mid.length < 6.8571:
            rnd_note = random.choice(notes)
            rnd_length1 = random.choice(time_between_notes)
            rnd_length2 = random.choice(note_lengths)
            
            if mid.length < 6.8: 
                track.append(Message('note_on', channel=1, note=rnd_note, velocity=100, time=rnd_length1))
                track.append(Message('note_off', channel=1, note=rnd_note, velocity=64, time=rnd_length2))
            else:
                break
        
        midi_list.append(mid)
    
    return midi_list




def get_best_midi(midi_files, model, num_return):
    '''
    Returns the best midi files from a list of midi
    
    Parameters
    ----------
    midi_files: list 
        list of midi files
        
    model: Keras model
        classifier model to get the best midi
    
    num_return: int 
        number of midi files that are returned
    
    Returns
    -------
    best_midi: list
        list of best midi files
    '''
    converted_data = convert_to_data(midi_files)
    prediction = model.predict(converted_data, batch_size=1, verbose=0)
    prediction = prediction.tolist()
    zipped = zip(midi_files, prediction)
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    best_midi = list(zip(*zipped)) 

    return best_midi[0][:num_return]


def access_data(load_midi_files):
    '''
    Loads and formats the datasets for training the model
    
    Parameters
    ----------
    load_midi_files: bool
        if true, loads the data from the midi files and reformats, else, loads from numpy file without reformat
        
    Returns
    -------
    training_data: list
        list of training data
        
    training_target: list
        targets for training data
        
    test_data: list
        test data
        
    test_target: list
        targets for test data
    '''
    test_set_size = 500
    
    if load_midi_files == True:
        temp = import_midi()
        data = temp[0]
        targets = temp[1]
        
        training_data = convert_to_data(data[test_set_size:])
        test_data = convert_to_data(data[:test_set_size])
        
        training_target = targets[test_set_size:]
        training_target = np.asarray([i[0] for i in training_target])
    
        test_target = targets[:test_set_size]
        test_target = np.asarray([i[0] for i in test_target])
        
        np.save('training_data\\training_data.npy', training_data, False, False)
        np.save('training_data\\test_data.npy', test_data, False, False)
        np.save('training_data\\training_target.npy', training_target, False, False)
        np.save('training_data\\test_target.npy', test_target, False, False)
        
        
    else:
        training_data = np.load('training_data\\training_data.npy', None, False, False)
        test_data = np.load('training_data\\test_data.npy', None, False, False)
        training_target = np.load('training_data\\training_target.npy', None, False, False)
        test_target = np.load('training_data\\test_target.npy', None, False, False)
    
    return (training_data, training_target, test_data, test_target)

#Command to run tensorboard in the cmd
#python -m tensorboard.main --logdir=logs


#Code used to tune hyper parameters
#Commented out so it's not called when running the GUI
'''
data = access_data(False)


repeats = 3
n_epochs = 50
n_neurons = 512
n_batch = 128
n_layers = 0
n_dropout = 0.0
act_func = 'sigmoid'
opt_func = 'rmsprop(lr=0.001)'

i = 1
while i < repeats+1:
    model = train_classifier(data[0], data[1], data[2], data[3],n_epochs,n_neurons, n_batch, n_layers, n_dropout, act_func, opt_func)
    plt.plot(model[1].history['loss'], color='blue')
    plt.plot(model[1].history['val_loss'], color='orange') 
    i = i + 1
    n_neurons = int(n_neurons / 2)


model = load_model('models\\model-e25-n512-b128-lay3-drop0.01-act(sigmoid)-opt(adam)-1554918724.h5')
midi_files = generate_melodies(0,'Major')
best_midi = get_best_midi(midi_files, model[0], 20)

    
plt.title('Epochs: ' + str(n_epochs) + '  Neurons: ' + str(n_neurons) + '  Batch Size: ' + str(n_batch) + '  Deep Layers: ' + str(n_layers) + '  Dropout: ' + str(n_dropout))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
plt.savefig("figures\\figure-e" + str(n_epochs) + "-n" + str(n_neurons) + "-b" + str(n_batch) + "-lay" + str(n_layers) + "-drop" + str(n_dropout) + "-act(" + str(act_func)+ ")-opt(" + str(opt_func)+ ").png")

'''