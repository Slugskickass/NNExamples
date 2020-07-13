import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import time
from collections import OrderedDict
from collections import namedtuple
from itertools import product

class mynetwork(nn.Module):
    def __init__(self, middle_width, drop_out):
        super().__init__()
        self.linear = nn.Linear(11, np.int(middle_width/2))
        self.m = nn.Sigmoid()
        self.linear1 = nn.Linear(np.int(middle_width/2), middle_width)
        self.linear2 = nn.Linear(middle_width, np.int(middle_width/2))
        self.linear3 = nn.Linear(np.int(middle_width/2), 10)
        self.drop = nn.Dropout(drop_out)

    def forward(self, xb):
        xb = self.linear(xb)
        xb = self.m(xb)
        xb = self.linear1(xb)
        xb = self.drop(xb)
        xb = self.m(xb)
        xb = self.linear2(xb)
        xb = self.m(xb)
        out = self.linear3(xb)
        return out

def data_get(file_name):
    data = pd.read_csv(file_name)
    train_target = torch.tensor(data['quality'].values)
    train = torch.tensor(data.iloc[:, 0:11].values).type(torch.float)
    train_tensor = data_utils.TensorDataset(train, train_target)
    return (train_tensor, train_target)

def calculate_correct(out, labels):
    return torch.eq(out.argmax(dim=1), labels).sum().numpy()

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class RunManager():
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        characteristics, labels = next(iter(self.loader))

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def inform(self, discrete_n):
        if self.epoch_count % discrete_n == 0:
            print(self.epoch_count, ' ', self.run_count)


file_name = 'wine.csv'
train_tensor, train_target = data_get(file_name)
params = OrderedDict(lr=[.01], batch_size=[100], middle_width=[200], number_epocs=[1000], drop_out=[0.1, 0.2, 0.3, 0.4, 0.5])
m = RunManager()

for run in RunBuilder.get_runs(params):

    network = mynetwork(run.middle_width, run.drop_out)
    loader = DataLoader(train_tensor, batch_size=run.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)
    m.begin_run(run, network, loader)

    for epoch in range(run.number_epocs):
        m.begin_epoch()
        for batch in loader:
            characteristics, labels = batch
            preds = network(characteristics)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss
            optimizer.zero_grad()  # Zero Gradients
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights
            m.track_loss(loss, batch)
            m.track_num_correct(preds, labels)
        m.inform(250)
        m.end_epoch()
    m.end_run()