import torch
import pandas as pd
import torchaudio
import numpy as np

print("start")

new_sr = 16000
num_datapoints = int(1 * new_sr)
melkwargs = {"n_fft": 256,
             "win_length": 256,
             "hop_length": 128,
             "f_min": 0,
             "f_max": new_sr,
             "n_mels": 16
             }

# audio processing + store mfcc for all audio file
def get_mfcc(base_path):
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

    mfcc = torchaudio.transforms.MFCC(sample_rate=new_sr, n_mfcc=16, dct_type=2, \
                                      norm="ortho", log_mels=True, melkwargs=melkwargs)(soundData)
    mfcc = mfcc[0, 1:16:, :]
    print("saved: %s" % path)
    #saving the extracted data in a separate folder from the .wav files
    path = path.replace("speech_commands_v0.01.tar", "extracted_data")
    np.save(path[:-4] + "0noise" + "_16mfcc" + ".npy", mfcc)

training_list_path = "*\data_prep\train_clean.csv"
testing_list_path = "*\data_prep\test_clean.csv"
get_mfcc(training_list_path)
get_mfcc(testing_list_path)
print("done")

