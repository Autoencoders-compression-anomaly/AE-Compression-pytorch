import os
import h5py
import torch
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

from fastai import learner
from scipy import stats

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
    data = parser.add_mutually_exclusive_group(required=True)
    # If only flag is given, const value of each argument will be used
    data.add_argument('-d', '--data', metavar='DATA', nargs=2,
                      help='global paths to model and testing data files in that order; must be in PTH (.pth) and HDF5 (.h5) format, respectively')
    data.add_argument('-f', '--dfile', metavar='READ_FILE', nargs=1,
                      help='global path to text file containing global paths to model and testing datasets, one per line, in that order')
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
        plt.savefig('plts/comparison_{}_1.png'.format(str(col_names[col])))
        plt.close()

def plot_residuals(data_in, data_out, col_names):
    residual_strings = [r'$(m_{out} - m_{in}) / m_{in}$',
                        r'$(p_{T,out} - p_{T,in}) / p_{T,in}$',
                        r'$(\eta_{out} - \eta_{in}) / \eta_{in}$',
                        r'$(\phi_{out} - \phi_{in}) / \phi_{in}$']
    residuals = data_out - data_in
    residuals = np.divide(residuals, data_in)
    rang = (-0.1, 0.1)
    for col in np.arange(4):
        plt.figure()
        std = np.nanstd(residuals[:, col])
        std_err = np.nanstd(residuals[:, col], ddof=0) / np.sqrt(2 * len(residuals[:, col]))
        mean = np.nanmean(residuals[:, col])
        sem = stats.sem(residuals[:, col], nan_policy='omit')
        plt.hist(residuals[:, col], alpha=0.8, bins=100, range=rang,
                 label='Residuals \n Mean = {} $\pm$ {}\n $\sigma$ = {} $\pm$ {}'.format(mean, sem, std, std_err))
        plt.title('Residuals of {}'.format(col_names[col]))
        plt.xlabel(residual_strings[col])
        plt.ylabel('Number')
        plt.legend()
        plt.savefig('plts/residual_{}_1.png'.format(col_names[col]))
        plt.close()

def plot_MSE(data_in, data_out):
    mse = np.sum(np.power(data_in - data_out, 2), axis=1) / 4
    plt.figure()
    plt.plot(mse)
    plt.title('Testing MSE Loss')
    #plt.xlabel()
    #plt.ylabel()
    plt.savefig('plts/testing_mse_loss_1.png')
    plt.close()

def get_MSE(data_in, data_out):
    l1 = np.sum(np.sum(np.power(data_in - data_out, 2), axis=1) / 4) / data_in.size
    l2 = np.sum(np.power(data_in - data_out, 2)) / (data_in.size)
    return l1, l2

# Function to read model and data file locations from file
# Arguments:
#     string containing path to the file to be read
# Returns: list of model and data file locations
def read_path_file(path):
    return open(path, 'r').read().split('\n')[:-1]

def main():
    # Parse command line arguments
    args = args_parser()
    if args.data:
        model_path, test_path = args.data
    elif args.dfile:
        model_path, test_path = read_path_file(args.dfile[0])

    # Check that model and testing datasets are formatted correctly
    model_filename, model_extension = os.path.splitext(model_path)
    _, test_extension = os.path.splitext(test_path)
    if (model_extension not in ['.pth'] or test_extension not in ['.h5']):
        print('Invalid file type: Model must be in PTH (.pth) and testing datasets in either pickle (.pkl) or in HDF5 (.h5) formats.')
        return
    
    # Create model
    model = AE_3D_200_LeakyReLU()
    model.to('cpu')
    learn = learner.load_model(model_path, model, None, with_opt=False)

    # Format testing data
    f = h5py.File(test_path, 'r')
    test = f['data'][...]
    if (args.norm == 'custom'):
        test = custom_normalise(pd.DataFrame(data=test[0:100000], columns= ['E', 'pt', 'eta', 'phi']))
    elif (args.norm == 'std'):
        test = std_normalise(pd.DataFrame(data=test[0:100000], columns= ['E', 'pt', 'eta', 'phi']))
    data = torch.tensor(test.values, dtype=torch.float)

    # Make predictions
    predictions = model(data)
    data = data.detach().numpy()
    predictions = predictions.detach().numpy()

    # Plot results
    plot(data, predictions, ['E', 'pt', 'eta', 'phi'])
    plot_residuals(data, predictions, ['E', 'pt', 'eta', 'phi'])
    plot_MSE(data, predictions)
    mse = get_MSE(data, predictions)
    print('MSE1: {}; MSE2: {}'.format(mse[0], mse[1]))

if __name__=='__main__':
    main()
