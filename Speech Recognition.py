import os
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

import librosa
import tensorflow as tf
from tensorflow import keras
import IPython as ipd
import numpy as np
import cv2
import IPython
import sounddevice as sd
from scipy.io.wavfile import write

'''print(librosa.__version__)
print(tf.__version__)
print(keras.__version__)
print(ipd.__version__)
print(np.__version__)
print(cv2.__version__)
print(ipd.__version__)'''


class Signals:

    def __init__(self, datadir):
        self.hop_length = 512  # The amount of samples we are shifting after each fft
        self.n_fft = 2048  # this is the number of samples in a window per fft
        self.sr = 22050  # sample rate
        self.datadir = datadir  # Drop the path of the database main folder here
        self.word_folder_list = [
            'folder1']  # name the folders here could either be words itself or range from 001 to n (condition if foldername in word_folder_list)
        self.sample_rate = 16000

    def record_audio(self,foldername):
        # Get user input to start recording
        input("Press Enter to start recording")
        print("Recording")
        # Record audio
        recording = sd.rec(frames=int(sd.query_devices('input')['default_samplerate']), samplerate=self.sample_rate,
                           channels=1)

        # Get user input to stop recording
        input("Press Enter to stop recording")

        # Read the last recorded file
        try:
            with open('last_recorded.txt', 'r') as f:
                last_recorded_num = int(f.read())
        except:
            last_recorded_num = 0

        # Increment the last recorded file
        last_recorded_num += 1
        # Save the incremented value
        with open('last_recorded.txt', 'w') as f:
            f.write(str(last_recorded_num))

        # Save recording as wav file
        write(f"C:/Users/hp compaq/Desktop/audios/{foldername}/{str(last_recorded_num)}.wav", self.sample_rate, recording)
        print(f"Recording saved as {last_recorded_num}.wav")

    def generate_waveform(self, foldername):
        for category in [foldername]:
            path = os.path.join(self.datadir, category)
            for audiofile in os.listdir(path):
                print(audiofile)
                signal, sr = librosa.load(self.datadir + '/' + foldername + '/' + audiofile, sr=22050)
                librosa.display.waveshow(signal, sr=sr)
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.show()

    def generate_melSpectrogram(self, foldername):
        imagenamer = 000
        for category in [foldername]:
            path = os.path.join(self.datadir, category)
            for audiofile in os.listdir(path):
                imagenamer = imagenamer + 1
                print(audiofile)
                signal, sr = librosa.load(self.datadir + '/' + foldername + '/' + audiofile, sr=22050)
                mel_signal = librosa.feature.melspectrogram(y=signal, sr=22050, hop_length=self.hop_length,
                                                            n_fft=self.n_fft)
                spectrogram = np.abs(mel_signal)
                power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
                plt.figure(figsize=(8, 7))
                librosa.display.specshow(power_to_db, sr=22050, cmap='magma',
                                         hop_length=self.hop_length)  # x_axis='time', y_axis='mel',
                # plt.colorbar(label='dB')
                # plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
                # plt.xlabel('Time', fontdict=dict(size=15))
                # plt.ylabel('Frequency', fontdict=dict(size=15))
                #plt.show()
                plt.savefig("C:/Users/hp compaq/Desktop/Spectrograms/folder1/" + str(imagenamer) + ".png", format="png",dpi=100, bbox_inches='tight', pad_inches=-0.15)
                #print(mel_signal)
                print(spectrogram)
                #print(power_to_db)

    def generate_frequencyhist(self, foldername):
        imagenamer = 000
        for category in [foldername]:
            path = os.path.join(self.datadir, category)
            for audiofile in os.listdir(path):
                imagenamer = imagenamer + 1
                print(audiofile)
                signal, sr = librosa.load(self.datadir + '/' + foldername + '/' + audiofile, sr=22050)
                S = librosa.stft(y=signal, n_fft=self.n_fft, hop_length=self.n_fft//2)
                S=abs(S)
                D_AVG = np.mean(S, axis=1)
                plt.figure(figsize=(8, 4))
                plt.bar(np.arange(D_AVG.shape[0]), D_AVG)
                x_ticks_positions = [n for n in range(0, self.n_fft // 2, self.n_fft // 16)]
                x_ticks_labels = [str(sr / 2048 * n) + 'Hz' for n in x_ticks_positions]
                plt.xticks(x_ticks_positions, x_ticks_labels)
                plt.xlabel('Frequency')
                plt.ylabel('dB')
                plt.show()
                

Signalgeneration = Signals("C:/Users/hp compaq/Desktop/audios")
#Signalgeneration.record_audio('folder1')
#Signalgeneration.generate_waveform('folder1')
Signalgeneration.generate_melSpectrogram('folder1')
#Signalgeneration.generate_frequencyhist('folder1')



class LoadData:

    def __init__(self, datadir):
        self.word_folder_list = ['folder1']
        self.datadir = datadir

    '''def loadimages(self,foldername):
        for category in [foldername]:
            path = os.path.join(self.datadir, category)
            for specimage in os.listdir(path):
                print('this is image',specimage)
                #specimage.show()'''

    def loadimages(self, foldername):
        for category in [foldername]:
            path = os.path.join(self.datadir, category)
            for specimage in os.listdir(path):
                #print('this is the image name: ', specimage)
                cvimage = cv2.imread("C:/Users/hp compaq/Desktop/Spectrograms/folder1/" + specimage, -1) #https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
                '''print(cvimage)'''
                #print(cvimage)
                #cv2.imshow('Mel-Spectrogram', cvimage) #panel window name and file name/file path
                #cv2.waitKey(5000) #show for 5000=5 seconds
                #cv2.destroyAllWindows() #destroys all the windows we created
                '''print(cvimage.shape) #dimensions and colors present in the image (yaxis xasix and channels)'''
                #print(cvimage.shape) #this is the shape of our spectrogram
                X=cvimage
                print(X.shape)
                #X =X.reshape(1,513,594,4) # shape of X is 4D, (1, 2, 2, 1) 
                #print(X.shape)
                return X

                


LD = LoadData("C:/Users/hp compaq/Desktop/Spectrograms/")
LD.loadimages('folder1')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class CNNlayers(LoadData):

    def __inint__(self):
        train=None
        test=None
        LD = LoadData("C:/Users/hp compaq/Desktop/Spectrograms/")
        

    def cnnmodel(self,train):        
        model= Sequential()
        model.add(Conv2D(16,(3,3), activation='relu', input_shape=(513, 594, 4)))
       # model.add(MaxPooling2D()) #added to minimize the load
        model.add(Conv2D(16, (3,3), activation='relu'))
       # model.add(MaxPooling2D()) #added to minimize the load
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        #model.summary()
        train=LD.loadimages('folder1')
        #print(train)
        test=LD.loadimages('folder1')
        
        hist=model.fit(train,test,batch_size=4, epochs=3)


cn=CNNlayers("C:/Users/hp compaq/Desktop/Spectrograms/")
#cn.cnnmodel('folder1')





#Signalgeneration.generate_waveform('folder1')
#Signalgeneration.generate_melSpectrogram('folder1')
#LD.loadimages('folder1')
#cn.cnnmodel('folder1')
