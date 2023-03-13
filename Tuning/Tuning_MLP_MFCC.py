from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from CustomDataSet import MLP_dataset as Load_MFCC_Noisy
import random

def load_data(data_dir="./data"):
    csv_path = r"C:\Users\user\Desktop\Intern_Joshua\kws_training2\folder"
    train_set = Load_MFCC_Noisy(csv_path, "\\train_clean.csv")
    test_set = Load_MFCC_Noisy(csv_path, "\\test_clean.csv")
    return train_set, test_set

class Net(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, l1=1000, l2=1000, l3=1000):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(1386, l1),
      nn.Dropout(p=0.5),
      nn.ReLU(),
      nn.Linear(l1, l2),
      nn.Dropout(p=0.5),
      nn.ReLU(),
      nn.Linear(l2, l3),
      nn.Dropout(p=0.5),
      nn.ReLU(),
      nn.Linear(l3, 12)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
    
def train_MLP(config, checkpoint_dir=None, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = config["lr"], weight_decay = 0.0001)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    path = r"C:\Users\user\Desktop\Intern_Joshua\Tests\final_tuning_model\MLP_MFCC_tuning.pth"
    torch.save(net, path)
    print("Finished Training")
    
def test_accuracy(net, device="cuda"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main(num_samples=8, max_num_epochs=10, gpus_per_trial=1):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    #defining search space
    config = {
        "l1": tune.sample_from(lambda _: random.sample([500, 800, 1000, 1200, 1400], 1)[0]),
        "l2": tune.sample_from(lambda _: random.sample([500, 800, 1000, 1200, 1400], 1)[0]),
        "l3": tune.sample_from(lambda _: random.sample([500, 800, 1000, 1200, 1400], 1)[0]),
        "lr": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice([16, 32, 64, 128])
    }    

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        #parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_MLP, data_dir=data_dir),
        resources_per_trial={"cpu": 16, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"], best_trial.config["l3"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=1)