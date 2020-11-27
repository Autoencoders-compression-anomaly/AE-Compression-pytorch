import argparse
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
from sklearn.preprocessing import StandardScaler
from fastai import learner
from fastai.data import core
from fastai.metrics import mse
from fastai.callback import schedule
from pathlib import Path

# Need this to be able to access HEPAutoencoder library
sys.path.append(str(Path(os.getcwd()).parent.parent))

from HEPAutoencoders.nn_utils import AE_basic, AE_bn, AE_LeakyReLU, AE_bn_LeakyReLU, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn

# Class describing autoencoder model object
class AE_3D_200_LeakyReLU(nn.Module):
    # Initialise autoencoder object
    # Arguments:
    #     feature_no: number of input (and thus output) vector
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

    # Function encodes vector to a latent space
    # Arguments:
    #     x: indexable object representing a vector to be encoded
    def encode(self, x):
        return self.latent(self.tanh(self.enc_l3(self.tanh(self.enc_l2(self.tanh(self.enc_l1(x)))))))

    # Function decodes latent space representation to initial vector representation
    # Arguments:
    #     x: indexable object representing a latent space representation of a vector
    def decode(self, x):
        return self.output(self.tanh(self.dec_l3(self.tanh(self.dec_l2(self.tanh(self.dec_l1(x)))))))

    # Function runs full encoder-decoder cycle
    # Arguments:
    #     x: indexable object representing input vector
    def forward(self, x):
        return self.decode(self.encode(x))

    # Function describes autoencoder structure
    def describe(self):
        return 'in-200-200-20-3-20-200-200-out'

# Function for parsing command line arguments
# Returns: composite Object containing command line argument
def args_parser():
    parser = argparse.ArgumentParser(description='Run autoencoder over full train-test cycle')
    # User must provide arguments for training and testing datasets
    required = parser.add_argument_group('required named arguments')
    # If only flag is given, const value of each argument will be used
    required.add_argument('-tr', '--train', nargs='?', required=True,
                          const='/nfs/atlas/mvaskev/sm/processed_4D_ttbar_10fb_all_events_but_only_jet_particles_4D.pkl',
                          help='global path to training data file; must be in pickle (.pkl) format')
    required.add_argument('-t', '--test', nargs='?', required=True,
                          const='/nfs/atlas/mvaskev/sm/processed_4D_z_jets_10fb_all_events_but_only_jet_particles_4D.pkl',
                          help='global path to testing data file; must be in pickle (.pkl) format')
    # Optional flag if user wants to plot loss values
    # Note that some minor changes in fastaiv2 may be needed for this to work
    parser.add_argument('-p', '--plot', default=False, action='store_true',
                          help='choose to attempt saving of loss optimisation and training loss plots')
    parser.add_argument('-n', '--norm', nargs='?', default='custom', const='custom',
                        choices=['custom', 'std'], 
                        help='choose normalisation method (default: %(default)s)')
    return parser.parse_args()

# Function to custom normalise dataset of 4-vectors
# Arguments:
#     df: pandas DataFrame with one particle per row, columns being the 4 features (E, pt, eta, phi)
# Returns: DataFrame containing custom normalised dataset
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

# Function to standard normalise dataset of 4-vectors
# Arguments:
#     df: pandas DataFrame with one particle per row, columns being the 4 features (E, pt, eta, phi)
# Returns: DataFrame containing standard normalised dataset
def std_normalise(df):
    variables = df.keys()
    x = df[variables].values
    x_scaled = StandardScaler().fit_transform(x)
    df[variables] = x_scaled
    return df

# Function to plot autoencoder performance
# Arguments:
#     data_in: input numpy array with values of the 4-vectors in initial order
#     data_out: output (decoded) numpy array with values of the 4-vectors in initial order
#     col_names: indexable object with names of the 4 features in order same as data_in and data_out
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
    # Parse command line arguments
    args = args_parser()
    loss_plot = args.plot
    train_path, test_path = args.train, args.test
    # Check that training and testing datasets are formatted correctly
    train_filename, train_extension = os.path.splitext(train_path)
    if (train_extension != '.pkl' or os.path.splitext(test_path)[1] != '.pkl'):
        print('Invalid file type: Training and testing datasets must be in pickle (.pkl) format.')
        return 1
    else:
        # Define files where training set split into training and validation will be stored
        train_split_path = train_filename + '_train' + train_extension
        valid_split_path = train_filename + '_valid' + train_extension

    # Define variables
    bs = 8192
    loss_func = nn.MSELoss()
    one_epochs = 30
    wd = 1e-2
    recorder = learner.Recorder()

    # check if GPU is available
    if torch.cuda.is_available():
        fastai.torch_core.defaults.device = 'cuda'
        print('Using GPU for training')

    # Get data from pkl file
    data = pd.read_pickle(train_path)
    test = pd.read_pickle(test_path)

    # Split into training and validation datasets
    train, valid = train_test_split(data, test_size=0.2, random_state=41)
    # Store split training and validation sets (change to point to your location)
    train.to_pickle(train_split_path)
    valid.to_pickle(valid_split_path)
    # Read the datasets into pandas DataFrame
    train = pd.read_pickle(train_split_path)
    valid = pd.read_pickle(valid_split_path)

    if (args.norm == 'custom'):
        # Custom normalise training and testing datasets
        train = custom_normalise(train)
        valid = custom_normalise(valid)
        test = custom_normalise(test)
    elif (args.norm == 'std'):
       # Standard normalise training and testing datasets
       train = std_normalise(train)
       valid = std_normalise(valid)
       test = std_normalise(test)

    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(valid.values, dtype=torch.float), torch.tensor(valid.values, dtype=torch.float))

    # Create DataLoaders
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    # Return DataBunch
    db = core.DataLoaders(train_dl, valid_dl)

    # Model set-up
    model = AE_3D_200_LeakyReLU()
    model.to('cpu')
    learn = learner.Learner(db, model=model, wd=wd, loss_func=loss_func, cbs=recorder)

    # Get learning rates at minimum and steepest decrease losses
    min_lr, steepest_lr = learn.lr_find()
    if loss_plot:
        learn.recorder.plot_lr_find().savefig('plts/loss_optimisation.png')
        plt.close('all')
    print('Minimum loss learning rate {}'.format(min_lr))
    print('Steepest loss drop learning rate {}'.format(steepest_lr))

    # Fit model
    start_tr = time.perf_counter()
    learn.fit_one_cycle(one_epochs, steepest_lr)
    end_tr = time.perf_counter()
    time_tr = end_tr - start_tr
    print('Training lasted for {} seconds'.format(time_tr))

    if loss_plot:
        learn.recorder.plot_loss().savefig('plts/training_loss.png')

    # Get final validation loss
    print('MSE on validation set is {}'.format(learn.validate()))
    
    # Make predictions
    data = torch.tensor(test.values, dtype=torch.float)
    predictions = model(data)
    data = data.detach().numpy()
    predictions = predictions.detach().numpy()

    # Plot results
    plot(data, predictions, test.columns)

if __name__=='__main__':
    main()
