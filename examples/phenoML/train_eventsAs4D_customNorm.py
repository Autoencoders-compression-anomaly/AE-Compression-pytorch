import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import torch
import torch.nn as nn
import torch.utils.data

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from fastai import learner
from fastai.data import core
from fastai.metrics import mse
from fastai.callback import schedule
from pathlib import Path

# Need this to be able to access HEPAutoencoder library
sys.path.append(str(Path(os.getcwd()).parent.parent))

from HEPAutoencoders.nn_utils import AE_basic, AE_bn, AE_LeakyReLU, AE_bn_LeakyReLU, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn

class AE_3D_200_LeakyReLU(nn.Module):
    def __init__(self, feature_no=4):
        super(AE_3D_200_LeakyReLU, self).__init__()
        self.tanh = nn.Tanh()
        self.enc_l1 = nn.Linear(feature_no, 200)
        self.enc_l2 = nn.Linear(200, 200)
        self.enc_l3 = nn.Linear(200, 20)
        self.latent = nn.Linear(20, 3)
        self.dec_l1 = nn.Linear(3, 20)
        self.dec_l2 = nn.Linear(20, 200)
        self.dec_l3 = nn.Linear(200, 200)
        self.output = nn.Linear(200, feature_no)

    def encode(self, x):
        return self.latent(self.tanh(self.enc_l3(self.tanh(self.enc_l2(self.tanh(self.enc_l1(x)))))))

    def decode(self, x):
        return self.output(self.tanh(self.dec_l3(self.tanh(self.dec_l2(self.tanh(self.dec_l1(x)))))))

    def forward(self, x):
        return self.decode(self.encode(x))

    def describe(self):
        return 'in-200-200-20-3-20-200-200-out'

def custom_normalise(df):
    # Convert dataset items into floats
    df = df.astype('float32')

    # Custom normalize dataset
    df['E'] = df['E'] / 1000.0
    df['pt'] = df['pt'] / 1000.0

    df['eta'] = df['eta'] / 5 
    df['phi'] = df['phi'] / 3

    df['E'] = np.log10(df['E'])
    df['pt'] = np.log10(df['pt'])
    return df

def plot(data_in, data_out, col_names):
    for col in np.arange(4):
        plt.figure()
        plt.hist(data_in[:, col], label='Input', bins=200, alpha=1, histtype='step')
        plt.hist(data_out[: , col], label='Output', bins=200, alpha=0.8, histtype='step')
        plt.xlabel(str(col_names[col]))
        plt.ylabel('Number')
        plt.yscale('log')
        plt.legend()
        plt.savefig('plts/comparison_{}.png'.format(str(col_names[col])))
        plt.close()

def main():
    # define variables
    bs = 8192
    loss_func = nn.MSELoss()
    one_epochs = 30
    wd = 1e-2
    recorder = learner.Recorder()

    # check if GPU is available
    if torch.cuda.is_available():
        fastai.torch_core.defaults.device = 'cuda'
        print('Using GPU for training')

    # Get data from pkl file (change to point to the file location)
    train_path = '/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D.pkl'
    data = pd.read_pickle(train_path)
    test_path = '/nfs/atlas/mvaskev/sm/processed_4D_z_jets_10fb_events_with_only_jet_particles_4D.pkl'
    test = pd.read_pickle(test_path)

    # Split into training and validation datasets
    train, valid = train_test_split(data, test_size=0.2, random_state=41)
    # Store split training and validation sets (change to point to your location)
    train.to_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_train.pkl')
    valid.to_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_test.pkl')
    # Read the datasets into pandas DataFrame
    train = pd.read_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_train.pkl')
    valid = pd.read_pickle('/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_events_with_only_jet_particles_4D_test.pkl')

    # Custom normalise training and testing datasets
    train = custom_normalise(train)
    valid = custom_normalise(valid)
    test = custom_normalise(test)

    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(valid.values, dtype=torch.float), torch.tensor(valid.values, dtype=torch.float))

    # Create DataLoaders
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    # Return DataBunch
    db = core.DataLoaders(train_dl, valid_dl)

    model = AE_3D_200_LeakyReLU()
    model.to('cpu')

    learn = learner.Learner(db, model=model, wd=wd, loss_func=loss_func, cbs=recorder)

    min_lr, steepest_lr = learn.lr_find()

    print('Minimum loss learning rate {}'.format(min_lr))
    print('Steepest loss drop learning rate {}'.format(steepest_lr))

    start_tr = time.perf_counter()
    learn.fit_one_cycle(one_epochs, steepest_lr)
    end_tr = time.perf_counter()
    time_tr = end_tr - start_tr
    print('Training lasted for {} seconds'.format(time_tr))

    print('MSE on validation set is {}'.format(learn.validate()))

    data = torch.tensor(test.values, dtype=torch.float)
    predictions = model(data)
    data = data.detach().numpy()
    predictions = predictions.detach().numpy()

    plot(data, predictions, test.columns)

if __name__=='__main__':
    main()
