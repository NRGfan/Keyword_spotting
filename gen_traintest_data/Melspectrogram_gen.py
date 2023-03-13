# Auto generate all MFCC in csv
import torch
import torchaudio.transforms as T
import pandas as pd
import torchaudio
import numpy as np
#import soundfile as sf


device = "cpu"
input_max_min = [0.0, 0.0]
NUM_CLASSES = 12
new_sr = 16000
num_datapoints = int(1 * new_sr)
melkwargs = {"n_fft": 256,
             "win_length": 256,
             "hop_length": 128,
             "f_min": 0,
             "f_max": new_sr,
             "n_mels": 16
             }
fft = "128"
sr = "8000"
mel = "16"
mfcc_num = "20"


# audio processing + store mfcc for all audio file
def get_melspectrogram(base_path):
  csv_data = pd.read_csv(base_path)

  for i in range(len(csv_data)):
    path = csv_data.iloc[i, 0]
    sound, sample_rate = torchaudio.load(path)


    ### resample audio to 16kHz
    new_sr = 16000
    sound = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)(sound)
    sound = torch.mean(sound, dim=0, keepdim=True)  # downmixing to mono
    num_datapoints = int(1 * new_sr)
    soundData = torch.zeros([1, num_datapoints])  # make all audio 1 secs
    if sound.numel() < num_datapoints:
      soundData[0, :sound.numel()] = sound[0, :]
    else:
      soundData[0, :num_datapoints] = sound[0, :num_datapoints]


    mel_spectrogram = T.MelSpectrogram(
        sample_rate=new_sr,
        n_fft=256,
        win_length=256,
        hop_length=128,
        f_min=0,
        f_max=new_sr,
        n_mels=16
        )
    melspec = mel_spectrogram(soundData)
    melspec = melspec[0, 1:16:, :]
    

    #mfcc = torchaudio.transforms.MFCC(sample_rate=new_sr, n_mfcc=16, dct_type=2, \
    #                                  norm="ortho", log_mels=True, melkwargs=melkwargs
    #                                  )(soundData)
    #fcc = mfcc[0, 1:16:, :]
    # mfcc = mfcc.permute(1, 0)
    print("saved: %s" % path)
    path = path.replace("speech_commands", "MFCC")
    np.save(path[:-4] + "0noise" + "melspec" + ".npy", melspec)


training_list_path = r"C:\Users\user\Desktop\Intern_Joshua\kws_training2\folder\train_clean.csv"
testing_list2_path = r"C:\Users\user\Desktop\Intern_Joshua\kws_training2\folder\test_clean.csv"
get_melspectrogram(training_list_path)
get_melspectrogram(testing_list2_path)