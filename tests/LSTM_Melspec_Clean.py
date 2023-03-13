import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

snr_db = 10

device = "cuda"
input_max_min = [0.0, 0.0]
train_acc_trace = []
valid_acc_trace = []

NUM_CLASSES = 12
new_sr = 16000
num_datapoints = int(1 * new_sr)
melkwargs = { "n_fft": 256, 
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
ver = "38_v1"


class GoogleSpeechCommands():
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
        path = path.replace("speech_commands", "MFCC")
        
        mfcc = np.load(path[:-4] + "0noise" + "melspec" + ".npy")[1:12]
        mfcc = torch.Tensor(np.transpose(mfcc))

        return mfcc, self.labels[index]
    
    def __len__(self):
        return self.num_samples

#LSTM architecture
class kws_lstm(nn.Module):
    def __init__(self):
        super(kws_lstm, self).__init__()
        self.dropout = nn.Dropout(0.4)
        self.lstm1 = nn.LSTM(input_size=11, hidden_size=64, batch_first=True, num_layers=1)
        self.fc1 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = torch.reshape(x, self.num_flat_features(x))
        #x = self.relu(x)
        x = self.fc1(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return (x.size()[0], num_features)

########
# Training and validation
########

def train(model, epoch=1):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.requires_grad_() #set requires_grad to True for training
        data, target = data.to(device), target.to(device)
        # forward network
        result = model.forward(data)
        pred = torch.argmax(result, dim=1)
        correct += pred.eq(target).cpu().sum().item()
        loss = nn.CrossEntropyLoss()(result, target)
        #loss = nn.CrossEntropyLoss(weight=torch.Tensor([13.6891747052519,13.5010570824524,13.6380138814736,13.8750678978816,13.7333333333333,13.7852131678359,13.7038626609442,13.8600108518719,13.8675352877307,13.8901576943991,0.785606643087806]).to(device))(result, target)
        loss.backward()
        optimizer.step()
        ## print status
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
             epoch, len(train_loader.dataset), len(train_loader.dataset),
             100. * batch_idx / len(train_loader), loss))
    
    print('\nTrain Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    accuracy = 100. * correct / len(train_loader.dataset)
    return accuracy

def test(model):
    model.eval()
    correct = 0
    for data, target in test_loader:
        batch = len(target)
        data, target = data.to(device), target.to(device)
        # forward network
        result = model.forward(data)
        #pred = nn.Softmax(dim=1)(result)
        pred = torch.argmax(result, dim=1)
        correct += pred.eq(target).cpu().sum().item()
        #(prediction, target, path_pred, path_target)
        #if pred != target:
        #    wrong_pred.append((pred, target))
        #for i in range(len(pred)):
        #    if pred[i] != target[i]:
        #        wrong_pred.append((pred[i], target[i]))
        #print("Correct:", correct, end="\r")
        # create confusion matrix
        ones = torch.ones([batch]).to(device)
        zeros = torch.zeros([batch]).to(device)
        for i in range(NUM_CLASSES):  # number of class
            cond2 = torch.where(target == i, ones, zeros)
            for j in range(NUM_CLASSES):
                cond1 = torch.where(pred == j, ones, zeros)
                cond = torch.logical_and(cond1, cond2)
                count = len(torch.where(cond == True)[0])
                cm[i][j] += count
    #print(wrong_pred)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def list_param(model):
    for name, param in model.named_parameters():
        max_weight = torch.max(param)
        min_weight = torch.min(param)
        plt.plot(param.cpu().detach().numpy())
        plt.show()
        nonzero = torch.nonzero(param, as_tuple=False).shape
        print(name, "\t", param.shape, "\t", max_weight.data, "\t", min_weight.data, "\t", nonzero)

## test network forward
model = kws_lstm()
model.to(device)

## load dataset
csv_path = r"C:\Users\user\Desktop\Intern_Joshua\kws_training2\folder"
train_set = GoogleSpeechCommands(csv_path, "\\train_clean.csv")
test_set = GoogleSpeechCommands(csv_path, "\\test_clean.csv")

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

## trainer and optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.003, weight_decay = 0.0001) ## weight decay before 0.0001
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
epoch = 1

cm = torch.zeros([NUM_CLASSES, NUM_CLASSES])  ## confusion matrix

for i in range(50):
    print("Started Training Epoch: %s"%i)
    cm = torch.zeros([NUM_CLASSES, NUM_CLASSES])  ## confusion matrix
    acc = train(model, i)
    train_acc_trace.append(acc)
    acc = test(model)
    valid_acc_trace.append(acc)
    #print("Confusion matrix\n", cm)
    #np.savetxt("cm.txt", cm.numpy(), fmt="%i")

print(train_acc_trace)
print(valid_acc_trace)

end_time = time.time()
time_taken = (end_time - start_time)/60
print("Time taken: %s" %time_taken)
