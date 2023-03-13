import torch
import pandas as pd
import torchaudio
import numpy as np
import math

new_sr = 16000
num_datapoints = int(1 * new_sr)
melkwargs = { "n_fft": 256, 
              "win_length": 256,
              "hop_length": 128,
              "f_min": 0,
              "f_max": new_sr,
              "n_mels": 16
            }
snr_db = 10 # signal to noise ratio

#audio processing + store mfcc for all audio file
def get_mfcc(base_path, file_name):
    csv_data = pd.read_csv(base_path + file_name)
    
    for i in range(len(csv_data)):
        path = csv_data.iloc[i, 0]
        sound, sample_rate = torchaudio.load(path)

    ### resample audio to 16kHz
        new_sr = 16000
        sound = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)(sound)
        sound = torch.mean(sound, dim=0, keepdim=True)  # downmixing to mono

        # Load white noise
        noise, _ = torchaudio.load("speech_commands\_back\white_noise_0.wav")
        noise = noise[:, : sound.shape[1]]
        speech_power = sound.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = math.exp(snr_db / 10)
        if sound.size() == noise.size():
            scale = snr * noise_power / speech_power
            noisy_speech = (scale * sound + noise) / 2

            num_datapoints = int(1 * new_sr)
            soundData = torch.zeros([1, num_datapoints]) # make all audio 1 secs
            if noisy_speech.numel() < num_datapoints:
                soundData[0, :noisy_speech.numel()] = noisy_speech[0,:]
            else:
                soundData[0, :num_datapoints] = noisy_speech[0, :num_datapoints]

            mfcc = torchaudio.transforms.MFCC(sample_rate=new_sr, n_mfcc=16, dct_type=2, \
                                          norm="ortho", log_mels=True, melkwargs=melkwargs)(soundData)
            mfcc = mfcc[0, 1:16:, :]
            np.save(path[:-4] + "_SNR" + str(snr_db) + "_whitenoise" + "_16mfcc" + ".npy", mfcc)
            print("saved: %s"%path )
        else:
            continue

training_list_path = "*\data_prep\train_clean.csv"
testing_list_path = "*\data_prep\test_clean.csv"
get_mfcc(training_list_path)
get_mfcc(testing_list_path)
print("done")
