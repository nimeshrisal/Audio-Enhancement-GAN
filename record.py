from turtle import goto
import sounddevice as sd
from scipy.io.wavfile import write
import numpy

def record():
#sampling frequency
    freq = 48000

#recording duration
    duration = 10

#start recorder with given values of duration and sampling frequency
    recording = sd.rec(int(duration * freq),samplerate=freq, channels=2)

#Record audio for the given number of seconds
    sd.wait()

#this will convert the numpy array to an audio file with the given sampling frequency
    write("test_sample.wav",freq,recording.astype(numpy.float32))

#convert the numpy array to audio file
    #write('test_sample.wav',freq,)

def run():
    while(1):
        val=input('Enter "s" to start recording')
        if (val == 's'):
            record()
            break
        else:
            print('please enter s to start a 10 sec audio')