import pandas as pd
import numpy as np
import torch 

class GoogleSpeechCommands_MFCC_Clean():
    def __init__(self, base_path, file_name):
        csv_data = pd.read_csv(base_path + file_name)
        self.base_path = base_path
        self.file_names = []
        self.labels = []
        self.num_samples = len(csv_data)
        for i in range(self.num_samples):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])

    def __getitem__(self, index):
        path = self.file_names[index]
        mfcc = np.load(path[:-4] + "0noise" + "_16mfcc" + ".npy")[1:12]
        mfcc = torch.Tensor(np.transpose(mfcc))
        return mfcc, self.labels[index]
    
    def __len__(self):
        return self.num_samples
    
class GoogleSpeechCommands_MFCC_Whitenoise():
    def __init__(self, base_path, file_name):
        csv_data = pd.read_csv(base_path + file_name)
        self.base_path = base_path
        self.file_names = []
        self.labels = []
        self.num_samples = len(csv_data)
        for i in range(self.num_samples):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])
    
    def __getitem__(self, index):
        path = self.file_names[index]
        snr_db = 10
        mfcc = np.load(path[:-4] + "_SNR" + str(snr_db) + "_whitenoise" + "_16mfcc" + ".npy")[1:12]
        mfcc = torch.Tensor(np.transpose(mfcc))
        return mfcc, self.labels[index]
    
    def __len__(self):
        return self.num_samples
    
class CNN_dataloader_clean_MFCC():
    def __init__(self, base_path, file_name):
        csv_data = pd.read_csv(base_path + file_name)
        #initialize lists to hold file names, labels, and folder numbers
        self.base_path = base_path
        self.file_names = []
        self.labels = []
        self.num_samples = len(csv_data)
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(self.num_samples):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_names[index]
        mfcc = np.load(path[:-4] + "0noise" + "_16mfcc" + ".npy")[1:12]
        mfcc = torch.from_numpy(mfcc)
        mfcc = torch.unsqueeze(mfcc, 0)
        return mfcc, self.labels[index]
    
    def __len__(self):
        return self.num_samples
    
class MLP_dataloader_Noisy_MFCC():
    def __init__(self, base_path, file_name):
        csv_data = pd.read_csv(base_path + file_name)
        #initialize lists to hold file names, labels, and folder numbers
        self.base_path = base_path
        self.file_names = []
        self.labels = []
        self.num_samples = len(csv_data)
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(self.num_samples):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])

    def __getitem__(self, index):
        # format the file path and load the file
        #path = self.base_path + "/" + self.file_names[index]
        path = self.file_names[index]
        snr_db = 10
        mfcc = np.load(path[:-4] + "_SNR" + str(snr_db) + "_whitenoise" + "_16mfcc" + ".npy")[1:12]
        mfcc = torch.from_numpy(mfcc)
        mfcc = torch.unsqueeze(mfcc, 0)
        return mfcc, self.labels[index]
    
    def __len__(self):
        return self.num_samples
    
class LSTM_dataloader_MFCC():
    def __init__(self, base_path, file_name):
        csv_data = pd.read_csv(base_path + file_name)
        #initialize lists to hold file names, labels, and folder numbers
        self.base_path = base_path
        self.file_names = []
        self.labels = []
        self.num_samples = len(csv_data)
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(self.num_samples):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])

    def __getitem__(self, index):
        # format the file path and load the file
        #path = self.base_path + "/" + self.file_names[index]
        path = self.file_names[index]
        mfcc = np.load(path[:-4] + "0noise" + "_16mfcc" + ".npy")[1:12]
        mfcc = torch.Tensor(np.transpose(mfcc))
        return mfcc, self.labels[index]
    
    def __len__(self):
        return self.num_samples