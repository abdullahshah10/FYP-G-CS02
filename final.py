import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Load the model from the H5 file
model = tf.keras.models.load_model('/home/ubuntu/main/mymodel.h5')    #REPLACE THIS PATH

import os
import soundfile as sf
import numpy as np
import librosa

# Load the audio file
audio_file = '/home/ubuntu/main/Sentence/safiullah (4).wav'    #CHANGE PATH HERE
audio, sr = librosa.load(audio_file, sr=44100)

# Set the desired segment length and threshold
segment_length = int(sr * 1.3)  # 1.3 seconds
threshold = -20  # in decibels

# Find the segments with non-silent audio
segments = []
segment_start = 0
while segment_start < len(audio):
    segment_end = segment_start + segment_length
    segment = audio[segment_start:segment_end]
    magnitude = np.abs(librosa.stft(segment, n_fft=2048))
    segment_energy_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    if np.max(segment_energy_db) > threshold:
        segments.append(segment)
    segment_start += segment_length

'''# Save the segments to files
output_path = 'C:/Users/hp compaq/Downloads/trimming'
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i, segment in enumerate(segments):
    # Save the segment to a file
    segment_filename = os.path.join(output_path, f"segment_{i}.wav")
    sf.write(segment_filename, segment, sr)'''

# Save the segments to files
#output_path = 'C:/Users/hp compaq/Downloads/trimming'
#if not os.path.exists(output_path):
#    os.makedirs(output_path)
#else:
#    # Delete existing segment files in output_path directory
#    for file in os.listdir(output_path):
#        if file.startswith("segment_") and file.endswith(".wav"):
#            os.remove(os.path.join(output_path, file))

#for i, segment in enumerate(segments):
#    # Save the segment to a file
#    segment_filename = os.path.join(output_path, f"segment_{i}.wav")
#    sf.write(segment_filename, segment, sr)


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
model = tf.keras.models.load_model('/home/ubuntu/main/mymodel.h5') #CHANGE PATH HERE

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#DATASET_PATH = 'C:/Users/hp compaq/Downloads/Complete Dataset/Complete Dataset'
DATASET_PATH='C:/Users/hp compaq/Downloads/Newfolder/Latestdataset'
data_dir = pathlib.Path(DATASET_PATH)
#print(data_dir)


commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
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

import os

# Define the path to the folder containing the audio files
folder_path = "/home/ubuntu/main/trimming/"     #REPLACE THIS PATH

sentence_created=[]

# Loop through each file in the folder
for i in range(1000):
    file_path = os.path.join(folder_path, f"segment_{i}.wav")
    if not os.path.exists(file_path):
        # Stop the loop if the file doesn't exist
        break

    # Perform your desired operations on the file
    x = tf.io.read_file(str(file_path))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    waveform = x # saving waveform in x
    x = get_spectrogram(x) 
    x = x[tf.newaxis,...]
    prediction = model(x)
    x_labels = ['001' ,'002' ,'003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018' ,'019', '020', '021', '022', '023', '024', '025']
    predicted_index = tf.argmax(prediction[0])
    predicted_word = x_labels[predicted_index]
    my_dict = {'001': 'abdullah', '002': 'brother', '003': 'do', '004': 'go to', '005': 'is', '006': "let's go", 
               '007': 'my', '008': 'nahum', '009': 'name', '010': 'of the', '011': 'parmeet', '012': 'remained', 
               '013': 'safiullah', '014': 'school', '015': 'small', '016': 'sufyan', '017': 'time', '018': 'what',
               '019': 'what', '020': 'where', '021': 'will go', '022': 'you', '023': 'your', '024': 'home', '025': 'work'}
    my_sindhi = {'abdullah': 'عبدالله',
                'brother': 'ڀاءُ',
                'do': 'ڪري', 
                'go to': 'پيو', 
                'is': 'آهي', 
                "let's go": "وڃين", 
                'my': 'منهنجو', 
                'nahum': 'نهم', 
                'name': 'نالو', 
                'of the': 'جو', 
                'parmeet': 'پرميت', 
                'remained': 'رهيو', 
                'safiullah': 'صفي الله', 
                'school': 'اسڪول', 
                'small': 'ننڍو', 
                'sufyan': 'سفيان', 
                'time': 'وقت', 
                'what': 'ڪهڙي', 
                'what': 'ڇا', 
                'where': 'ڪيڏانهن', 
                'will go': 'ويندو', 
                'you': 'تون', 
                'your': 'تنهنجو', 
                'home': 'گهر ', 
                'work': 'ڪم'}
    
    
    spoken_word = my_dict[f"{predicted_word}"]
    #print(spoken_word,end=' ')
    
    sentence_created.append((spoken_word))
    
    
    #MERGE SENTENCE_CREATED for final sentence
    sentence_final = ' '.join(sentence_created)

#print(sentence_created)
#print(sentence_final)
#print(type(sentence_final))
  

    
    
'''from googletrans import Translator
from gtts import gTTS
from gtts.lang import tts_langs



# Create a Translator object
translator = Translator()

# Define the English sentence to be translated
sentence_new = str(sentence_final)

# Translate the sentence to Urdu
translation = translator.translate(sentence_new, lang_tgt='ur') # en for english and ur for urdu


# Print the translated sentence
print(translation.text)
    
# Convert the translated text to speech in Urdu
tts = gTTS(text=translation.text, lang='ur')

# Save the speech as an audio file
tts.save("C:/Users/hp compaq/Downloads/text_to_speech/translation.mp3")

# Play the audio file using display
display.Audio("C:/Users/hp compaq/Downloads/text_to_speech/translation.mp3")'''


from translate import Translator
from gtts import gTTS
from gtts.lang import tts_langs
from IPython.display import Audio

# Create a Translator object
translator = Translator(to_lang="en")
# Define the English sentence to be translated
sentence_new = str(sentence_final)

# Translate the sentence to Urdu
translation = translator.translate(sentence_new) # en for english and ur for urdu

# Print the translated sentence
print(translation)
    
