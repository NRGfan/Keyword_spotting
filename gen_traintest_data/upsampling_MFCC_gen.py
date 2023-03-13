import torch
import pandas as pd
import torchaudio
import numpy as np
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
def get_mfcc(base_path):
  csv_data = pd.read_csv(base_path)

  for i in range(len(csv_data)):
    path = csv_data.iloc[i, 0]
    sound, sample_rate = torchaudio.load(path)

    new_sr = 32000
    sound = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)(sound)
    sound = torch.mean(sound, dim=0, keepdim=True)  # downmixing to mono
    num_datapoints = int(1 * new_sr)
    soundData = torch.zeros([1, num_datapoints])  # make all audio 1 secs
    if sound.numel() < num_datapoints:
      soundData[0, :sound.numel()] = sound[0, :]
    else:
      soundData[0, :num_datapoints] = sound[0, :num_datapoints]

    # n_mfcc=14 originally

    mfcc = torchaudio.compliance.kaldi.mfcc(waveform=soundData, sample_frequency=new_sr, num_mel_bins=16,
                                            num_ceps=13)
    mfcc = torch.clamp(mfcc, min=-32.0, max=32.0)

    mfcc = torchaudio.transforms.MFCC(sample_rate=new_sr, n_mfcc=16, dct_type=2, \
                                      norm="ortho", log_mels=True, melkwargs=melkwargs
                                      )(soundData)
    mfcc = mfcc[0, 1:16:, :]
    # mfcc = mfcc.permute(1, 0)
    path = path.replace("speech_commands", "MFCC")
    np.save(path[:-4] + "0noise" + "mfcc_sr32000" + ".npy", mfcc)
    print("saved: %s" % path)


training_list_path = r"C:\Users\user\Desktop\Intern_Joshua\kws_training2\folder\train_clean.csv"
testing_list2_path = r"C:\Users\user\Desktop\Intern_Joshua\kws_training2\folder\test_clean.csv"
get_mfcc(training_list_path)
get_mfcc(testing_list2_path)
print("done")

