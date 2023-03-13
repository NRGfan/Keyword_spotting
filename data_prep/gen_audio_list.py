import os
import glob

## create full audio list
fp = open("*\audio_list.txt", "w")

# get all wav files in a master list
folder_list = [f.path for f in os.scandir("*\speech_commands") if f.is_dir()]
fp.write("filepath\n")  # write header row
for folder in folder_list:
    wav_list = glob.glob(folder + "/*.wav")
    for wav in wav_list:
            temp = wav[2:]  # remove "./"
            fp.write(temp + "\n")
fp.close()

