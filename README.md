# Audio-Enhancement-GAN

This is a Audio enhancement system which uses GAN.

Data: https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/

How to run the code? 
py data_preprocess.py
  for data pre-processing
  
py main.py
  for model training

> after every epoch .pkl checkpoints of both generator and discriminator is saved to the local directory.

py test_audio.py 
for enhancing the audio 
you can record the audio up to 1 minutes(can be changed to desired time span from record.py file) 
              OR
you can feed the recorded audio into the system and also enhance.(both can be changed inside the test_audio.py file)

also can select the generator checkpoint i.e generator-100.pkl is set to default here but again you can change it in (you know where..)



ENJOY!!!!!!!

