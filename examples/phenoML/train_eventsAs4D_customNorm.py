import sys
BIN = '/home/ppe/m/mvaskev/'
sys.path.append(BIN + 'AE-Compression-pytorch/')
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import datetime

import torch
import torch.nn as nn
import torch.utils.data

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from fastai.callbacks.tracker import SaveModelCallback

#import fastai
#from fastai import train as tr
#from fastai import basic_train, basic_data
#from fastai.callbacks import ActivationStats
#from fastai import data_block, basic_train, basic_data

from fastai import learner
from fastai.data import core
from fastai.metrics import mse
from fastai.callback import schedule

#from HEPAutoencoders.utils import plot_activations
from HEPAutoencoders.nn_utils import get_data, RMSELoss
from HEPAutoencoders.nn_utils import AE_basic, AE_bn, AE_LeakyReLU, AE_bn_LeakyReLU, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn

import matplotlib as mpl

class AE_3D_200_LeakyReLU(nn.Module):
    def __init__(self, feature_no=4):
        super(AE_3D_200_LeakyReLU, self).__init__()
        #self.tanh = nn.Tanh()
        self.enc_l1 = nn.Linear(feature_no, 200)
        self.enc_l2 = nn.Linear(200, 200)
        self.enc_l3 = nn.Linear(200, 20)
        self.latent = nn.Linear(20, 3)
        self.dec_l1 = nn.Linear(3, 20)
        self.dec_l2 = nn.Linear(20, 200)
        self.dec_l3 = nn.Linear(200, 200)
        self.output = nn.Linear(200, feature_no)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.latent(self.tanh(self.enc_l3(self.tanh(self.enc_l2(self.tanh(self.enc_l1(x)))))))

    def decode(self, x):
        return self.output(self.tanh(self.dec_l3(self.tanh(self.dec_l2(self.tanh(self.dec_l1(x)))))))

    def forward(self, x):
        return self.decode(self.encode(x))

    def describe(self):
        return 'in-200-200-20-3-20-200-200-out'

def main():
    # define variables
    bs = 8192
    loss_func = nn.MSELoss()
    one_epochs = 100
    one_lr = 1e-4
    one_wd = 1e-2
    one_pp = None
    one_module = AE_bn_LeakyReLU
    continue_training = False
    checkpoint_name = 'nn_utils_bs1024_lr1e-04_wd1e-02_ppNA'
    recorder = learner.Recorder()

    # check if GPU is available
    if torch.cuda.is_available():
        fastai.torch_core.defaults.device = 'cuda'
        print('Using GPU for training')

    # Get data from pkl file (change to point to the file location)
    global_path = '/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D.pkl'
    data = pd.read_pickle(global_path)

    # Split into training and testing datasets
    train, test = train_test_split(data, test_size=0.2, random_state=41)
    # Store split training and testing sets (change to point to your location)
    train.to_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_train.pkl')
    test.to_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_test.pkl')
    # Read the datasets into pandas DataFrame
    train = pd.read_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_train.pkl')
    test = pd.read_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_test.pkl')

    # Convert dataset items into floats
    train = train.astype('float32')
    test = test.astype('float32')

    # Custom normalize training and testing data sets
    train['E'] = train['E'] / 1000.0
    train['pt'] = train['pt'] / 1000.0
    test['E'] = test['E'] / 1000.0
    test['pt'] = test['pt'] / 1000.0

    train['eta'] = train['eta'] / 5
    train['phi'] = train['phi'] / 3
    train['E'] = np.log10(train['E']) 
    train['pt'] = np.log10(train['pt'])

    test['eta'] = test['eta'] / 5
    test['phi'] = test['phi'] / 3
    test['E'] = np.log10(test['E']) 
    test['pt'] = np.log10(test['pt'])

    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float), torch.tensor(test.values, dtype=torch.float))

    # Create DataLoaders
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    # Return DataBunch
    db = core.DataLoaders(train_dl, valid_dl)

    model = AE_3D_200_LeakyReLU()
    model.to('cpu')
    print(model)

    learn = learner.Learner(db, model=model, wd=one_wd, loss_func=loss_func, cbs=recorder)

    min_lr, steepest_lr = learn.lr_find()

    start_tr = time.perf_counter()
    learn.fit_one_cycle(one_epochs, steepest_lr)
    end_tr = time.perf_counter()
    time_tr = end_tr - start_tr
    print('Training lasted for {} seconds'.format(time_tr))

    print('MSE on test set is {}'.format(learn.validate()))

if __name__=='__main__':
    main()
