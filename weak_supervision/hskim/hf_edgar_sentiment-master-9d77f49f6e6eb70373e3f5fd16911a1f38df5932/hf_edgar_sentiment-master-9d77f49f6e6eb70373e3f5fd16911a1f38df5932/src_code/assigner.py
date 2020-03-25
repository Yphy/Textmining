### python ###
import random
import os
import pandas as pd
import numpy as np
import pickle

### torch ###
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

### custom ###
import discrete_cnn as model
import noise_aware_cnn as noise_aware_model

SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# Global Parameters
params = {'batch_size': 10,
          'shuffle': False,
          'num_workers': 1}

class Dataset(data.Dataset):
    def __init__(self, X, y):
        'Initialization'
        self.y = y
        self.X = X

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        # Load; data and get label
        'Generates one sample of data'
        # Select sample
        X = self.X[index]
        y = self.y[index]

        return X, y

def load_generator(test_X,test_y) :
    # Generators
    testing_set = Dataset(test_X,test_y)
    test_iter = data.DataLoader(testing_set, **params)
    return test_iter
