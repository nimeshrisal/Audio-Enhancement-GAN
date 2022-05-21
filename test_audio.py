import argparse
import os
from wave import open
from scipy.io import wavfile
import numpy as np
from scipy.io.wavfile import read
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from data_preprocess import slice_signal, window_size, sample_rate
from model import Generator
from utils import emphasis
from record import run



if __name__ == '__main__':

    #run()
    #parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    #parser.add_argument('--file_name', default='test_sample.wav', type=str, required=False, help='audio file name')
    #parser.add_argument('--epoch_name', default= 'generator-100.pkl', type=str, required=False, help='generator epoch name')

    #opt = parser.parse_args()
    #FILE_NAME = opt.file_name
    #EPOCH_NAME = opt.epoch_name

    FILE_NAME = './Speech-Enhancement-Measures/noisy_test_snr1.wav'
    EPOCH_NAME= 'generator-110.pkl'

    generator = Generator()
    generator.load_state_dict(torch.load('epochs/' + EPOCH_NAME, map_location='cpu'))
    if torch.cuda.is_available():
        generator.cuda()


    noisy_slices = slice_signal(FILE_NAME, window_size, 1, sample_rate)
    enhanced_speech = []
    for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
        z = nn.init.normal(torch.Tensor(1, 1024, 8))
        noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
        if torch.cuda.is_available():
            noisy_slice, z = noisy_slice.cuda(), z.cuda()
        noisy_slice, z = Variable(noisy_slice), Variable(z)
        generated_speech = generator(noisy_slice, z).data.cpu().numpy()
        generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
        generated_speech = generated_speech.reshape(-1)
        enhanced_speech.append(generated_speech)

        
    enhanced_speech = np.array(enhanced_speech).reshape(1,-1)
    file_name = os.path.join(os.path.dirname(FILE_NAME),
                             'enhanced_{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    wavfile.write(file_name, sample_rate, enhanced_speech.T.astype(np.float32))