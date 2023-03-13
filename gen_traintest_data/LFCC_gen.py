import torch
import pandas as pd
import torchaudio
import numpy as np
import noisereduce as nr
import torchaudio.transforms as T
NUM_CLASSES = 12
new_sr = 16000
num_datapoints = int(1 * new_sr)
melkwargs = {"n_fft": 256,
             "win_length": 256,
             "hop_length": 128,
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
    sound = nr.reduce_noise(y=sound, sr=sample_rate)
    sound = torch.from_numpy(sound)
    new_sr = 16000
    sound = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)(sound)
    sound = torch.mean(sound, dim=0, keepdim=True)  # downmixing to mono
    num_datapoints = int(1 * new_sr)
    soundData = torch.zeros([1, num_datapoints])  # make all audio 1 secs
    if sound.numel() < num_datapoints:
      soundData[0, :sound.numel()] = sound[0, :]
    else:
      soundData[0, :num_datapoints] = sound[0, :num_datapoints]
      
    lfcc = T.LFCC(
        sample_rate=16000,
        speckwargs=melkwargs
        )
    mfcc = lfcc(soundData)
    mfcc = mfcc[0, 1:16:, :]
    # mfcc = mfcc.permute(1, 0)
    path = path.replace("speech_commands", "MFCC")
    np.save(path[:-4] + "lfcc" + ".npy", mfcc)
    print("saved: %s" % path)


training_list_path = r"C:\Users\user\Desktop\Intern_Joshua\Tests\training_testing_data\train_clean.csv"
testing_list2_path = r"C:\Users\user\Desktop\Intern_Joshua\Tests\training_testing_data\test_clean.csv"
get_mfcc(training_list_path)
get_mfcc(testing_list2_path)
print("done")

