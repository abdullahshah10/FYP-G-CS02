import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

#the 3 lines below remove the warning default code
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Load the model from the H5 file
model = tf.keras.models.load_model('/home/ubuntu/FYP-G-CS02/mymodel.h5')

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#DATASET_PATH = 'C:/Users/hp compaq/Downloads/Complete Dataset/Complete Dataset'
#DATASET_PATH='C:/Users/hp compaq/Downloads/Newfolder/Latestdataset'
#data_dir = pathlib.Path(DATASET_PATH)
#print(data_dir)


#commands = np.array(tf.io.gfile.listdir(str(data_dir)))
#commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
#print('Commands:', commands)
#print()


'''train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.4,
    seed=0,
    output_sequence_length=16000,
    subset='both')'''

'''label_names = np.array(train_ds.class_names)'''
#print()
#print("label names:", label_names)


#print("Shape before: ", train_ds.element_spec)
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

'''train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)'''
'''val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)'''
#print()
#print("Shape After: ", train_ds.element_spec)

'''test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)'''
#print()

'''for example_audio, example_labels in train_ds.take(1):  
  #print(example_audio.shape)
  #print(example_labels.shape)
    break'''
    
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis] #adding new dimensions to the np arrays of the spectrograms
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

'''train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)'''

train_spectrogram_ds = None
val_spectrogram_ds = None
test_spectrogram_ds = None

#print(train_spectrogram_ds )

'''for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break'''


#Signalgeneration.record_audio("recording")
x = '/home/ubuntu/FYP-G-CS02/api/test.wav'

# this will read the audio file and load into the memory as a binary string
x = tf.io.read_file(str(x))
# decodes binary string of audio file in audio tensor(array)
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
# it will remove the last axis of audio tensor if size is 1
x = tf.squeeze(x, axis=-1)
waveform = x # saving waveform in x
x = get_spectrogram(x) 
x = x[tf.newaxis,...] # add an extra dimension in spectogram tensor (array) tom make it compatible with model
prediction = model(x)
x_labels = ['001' ,'002' ,'003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018' ,'019', '020', '021', '022', '023', '024', '025']
#plt.figure(figsize=(30, 6))
# softmax is used to make the predicted values in the range from 0 to 1 and place it in y-axis of bar
#plt.bar(x_labels, tf.nn.softmax(prediction[0])) 
#plt.title(x_labels)
#plt.show()

#display.display(display.Audio(waveform, rate=16000))


# Get the index of the word with the highest probability
predicted_index = tf.argmax(prediction[0])

# Get the corresponding word label
predicted_word = x_labels[predicted_index]

# Print the predicted word with its probability
#print(f"{predicted_word} has the highest probability: {tf.nn.softmax(prediction[0])[predicted_index]:.4f}")

my_dict = {'001': 'abdullah', '002': 'brother', '003': 'do', '004': 'got', '005': 'is', '006': "let's go", '007': 'my', '008': 'nahum', '009': 'name', '010': 'of the', '011': 'parmeet', '012': 'remained', '013': 'safiullah', '014': 'school', '015': 'small', '016': 'sufyan', '017': 'the time', '018': 'what', '019': 'what', '020': 'where', '021': 'will go', '022': 'you', '023': 'your', '024': 'home', '025': 'work'}

spoken_word=(my_dict[f"{predicted_word}"])
print(spoken_word)

    
