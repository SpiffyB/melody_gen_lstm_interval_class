import tkinter as tk
import threading
import os
import numpy as np
from keras.models import Sequential, load_model
from melody_gen_code import Main


root = tk.Tk()
root.title("Neurody")
frame = tk.Frame(root)
frame.pack()

learn_count = 0
scale = 'Major'
key = 'C'
model = load_model('model-e50-n128-b128-lay0-drop0.1-act(sigmoid)-opt(rmsprop(lr=0.001))-1555166116.h5')

current_mid = 0
good_midi = []
bad_midi = []

display_learn_counter = tk.Label(root, fg="black")
display_learn_counter.config(text=int(0))
display_learn_counter.pack(side=tk.RIGHT)

lbl_learn_counter = tk.Label(root, fg="black", text="Learn Counter:")
lbl_learn_counter.pack(side=tk.RIGHT)

scale_choices = {'Major','Minor','None'}
scale_value = tk.StringVar(root)
scale_value.set('Major')

key_choices = {'C','C#','D','D#','E','F','F#','G','G#','A','A#','B'}
key_value = tk.StringVar(root)
key_value.set('C')



#scale_drop = tk.OptionMenu(frame, scale_values, scale_choices)
key_drop = tk.OptionMenu(frame, key_value, *key_choices)
key_drop.pack(side=tk.BOTTOM)

scale_drop = tk.OptionMenu(frame, scale_value, *scale_choices)
scale_drop.pack(side=tk.BOTTOM)


def change_key(*args):
    global key
    global generated_files
    key = key_value.get()
    generated_files = generate_files(300, 30)
    
def change_scale(*args):
    global scale
    global generated_files
    scale = scale_value.get()
    generated_files = generate_files(300, 30)

def get_midi_note(key):
    if key == 'C':
        return 0
    elif key == 'C#':
        return 1
    elif key == 'D':
        return 2
    elif key == 'D#':
        return 3
    elif key == 'E':
        return 4
    elif key == 'F':
        return 5
    elif key == 'F#':
        return 6
    elif key == 'G':
        return 7
    elif key == 'G#':
        return 8
    elif key == 'A':
        return 9
    elif key == 'A#':
        return 10
    elif key == 'B':
        return 11
    else:
        print('Error: not a key')

   
def generate_files(num_random, num_best):
    global key
    global scale
    return Main.get_best_midi(Main.generate_melodies(get_midi_note(key), scale, num_random), model, num_best)
    
    
generated_files = generate_files(300, 30)

def background_thread(func):
    thread = threading.Thread(target=func)
    thread.start()

def save_training_data():
    global good_midi
    global bad_midi
    
    directory = os.getcwd() + '\\midi\\good\\generated\\'
    
    i = 1
    while i > 0:
        if os.path.isfile(directory + 'good' + str(i) + '.mid'):
            i = i + 1
        else:  
            break
        
    for file in good_midi:
        file.save(directory + 'good' + str(i) + '.mid')
        i=i+1
    
    directory = os.getcwd() + '\\midi\\bad\\generated\\'
    
    i = 1
    while i > 0:
        if os.path.isfile(directory + 'bad' + str(i) + '.mid'):
            i = i + 1
        else:  
            break
        
    for file in bad_midi:
        file.save(directory + 'bad' + str(i) + '.mid')
        i=i+1
        
        

def next_midi():
    global key
    global scale
    global generated_files
    global current_mid
    global model
    
    if current_mid < len(generated_files) - 1:
        current_mid = current_mid + 1
    else:
        current_mid = 0
        generated_files = generate_files(300, 30)

def create_targets(good_midi, bad_midi):
    good_targets = np.zeros((len(good_midi),1)) + 1
    bad_targets = np.zeros((len(bad_midi),1))
    
    if good_targets == []:
        targets = bad_targets
    elif bad_targets == []:
        targets = good_targets
    else:
        targets = np.concatenate((good_targets,bad_targets), axis=0)

    return(targets)
    
def play():
    global generated_files
    global current_mid
    if Main.get_is_playing() == False:
        Main.change_is_playing()
        print('Start Playing')
        Main.play_midi(generated_files[current_mid])
        print('Stopped Playing')
        stop()
              
def stop():
    if Main.get_is_playing() == True:
        Main.change_is_playing()
    
def save():
    global generated_files
    global current_mid
    if not os.path.exists(os.path.expanduser('~\Music\Generated Midi Files')):
        os.makedirs(os.path.expanduser('~\Music\Generated Midi Files'))
    
    i = 1
    while i > 0:
        if os.path.isfile(os.path.expanduser('~\Music\Generated Midi Files\Melody_') + str(i) + '.mid'):
            i = i + 1
        else:  
            generated_files[current_mid].save(os.path.expanduser('~\Music\Generated Midi Files\Melody_') + str(i) + '.mid')
            i = -1

def like():
    global key
    global scale
    global learn_count
    global good_midi
    global display_learn_counter
    stop()
    learn_count = learn_count + 1
    display_learn_counter.config(text=str(learn_count))
    
    good_midi.append(generated_files[current_mid])

    next_midi()
    play()
    


    
    
def dislike():
    global key
    global scale
    global learn_count
    global display_learn_counter
    stop()
    learn_count = learn_count + 1
    display_learn_counter.config(text=str(learn_count))
    
    bad_midi.append(generated_files[current_mid])    
     
    next_midi()
    play()


    
    
def learn():
    global learn_count
    global display_learn_counter
    global good_midi
    global bad_midi
    global model
    
    if learn_count < 5:
        print("Learn Counter must be 5 or more to learn!")
        return
    
    learn_count = 0
    display_learn_counter.config(text=str(learn_count))
    save_training_data()
    targets = create_targets(good_midi, bad_midi)
    

    training_data = Main.convert_to_data(good_midi + bad_midi)
    
    split_ratio = int(round(len(training_data) / 10,0))
    if split_ratio < 1:
        split_ratio = 1
    
    test_data = training_data[:split_ratio,:,:]
    test_target = targets[:split_ratio,:]
    
    training_data = training_data[split_ratio:,:,:]
    training_target = targets[split_ratio:,:]
    
    model.fit(training_data, 
              training_target,
              batch_size=split_ratio, 
              epochs=5,
              validation_data=(test_data, test_target), 
              verbose=1)
    
    return
    


btn_play = tk.Button(frame, text="Play", fg="black", command=lambda: background_thread(play), height=3, width=6)
btn_play.pack(side=tk.LEFT, padx=1, pady=5)
    
btn_stop = tk.Button(frame, text="Stop", fg="black", command=stop, height=3, width=6)
btn_stop.pack(side=tk.LEFT, padx=1, pady=5)
      
btn_save = tk.Button(frame, text="Save", fg="black", command=save, height=3, width=6)
btn_save.pack(side=tk.LEFT, padx=1, pady=5)
    
btn_like = tk.Button(frame, text="Like", fg="green", command=like, height=3, width=6)
btn_like.pack(side=tk.LEFT, padx=1, pady=5)
      
btn_dislike = tk.Button(frame, text="Dislike", fg="red", command=dislike, height=3, width=6)
btn_dislike.pack(side=tk.LEFT, padx=1, pady=5)
    
btn_learn = tk.Button(frame, text="Learn", fg="blue", command=learn, height=3, width=6)
btn_learn.pack(side=tk.LEFT, padx=1, pady=5)
    
key_value.trace('w', change_key)
scale_value.trace('w', change_scale)

root.resizable(width=False, height=False)
root.mainloop()