## To split big background noise audios to 1-sec segments
import os
import glob
from scipy.io import wavfile

## global variables
AUDIO_LEN = 1  # sec

## iterate through the background noise and segment them
#wav_list = glob.glob("_background_noise_/*.wav")
wav_list = glob.glob("*\speech_commands\_background_noise_\*.wav")
for audio in wav_list:
    sample_rate, sound_data = wavfile.read(audio)
    _, audio_name = os.path.split(audio)
    audio_name = audio_name[:-4]
    print("Processing:", audio_name)

    num_samples = len(sound_data)
    num_segments = num_samples // (sample_rate // 2)  # count num of segments

    # write out the segmented audio
    for i in range(num_segments):
        filename = "*\speech_commands\_back\{}_{}.wav".format(audio_name, i)
        start = i * (sample_rate // 2)
        end = start + sample_rate
        wavfile.write(filename, sample_rate, \
sound_data[start:end])

    print("Done")

