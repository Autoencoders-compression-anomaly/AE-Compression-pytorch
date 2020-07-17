import sys
BIN = '/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/'
sys.path.append(BIN)
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset
from fastai.callbacks.tracker import SaveModelCallback
#import my_matplotlib_style as ms
#from utils as ms

from fastai import basic_train, basic_data, torch_core
from fastai.callbacks import ActivationStats
from fastai import train as tr

from HEPAutoencoders.nn_utils import AE_basic, AE_bn, AE_LeakyReLU, AE_bn_LeakyReLU, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn
from HEPAutoencoders.nn_utils import get_data, RMSELoss
from HEPAutoencoders.utils import min_filter_jets, filter_jets, plot_activations, custom_normalization, normalize, custom_unnormalize
import HEPAutoencoders.utils as utils
import matplotlib as mpl

def getPred(data):
    loss_func = nn.MSELoss()
    path_to_saved_model = '/afs/cern.ch/user/s/sarobert/autoencoders/outputs/jul16-100ep-TLAfilterNorm/models/'
    modelFile =  'best_nn_utils_bs4096_lr1e-02_wd1e-02'
    bn_wd = False  # Don't use weight decay for batchnorm layers
    true_wd = True  # wd will be used for all optimizers
    bs = 4096
    wd = 1e-2
    module = AE_bn_LeakyReLU
    model = module([27, 200, 200, 200, 14, 200, 200, 200, 27])

    # Load data
    dataPath = '/afs/cern.ch/work/s/sarobert/autoencoders/processedData/'
    train = pd.read_pickle(dataPath + 'tla_jets_train.pkl')#[0:100]
    test = pd.read_pickle(dataPath + 'tla_jets_test.pkl')#[0:100]
    columns = data.columns

    #Filter and normalize data
    train = filter_jets(train)
    test = filter_jets(test)

    train, test = custom_normalization(train, test)
    #train, test = normalize(train, test)
    train_mean = train.mean()
    train_std = train.std()
    data, p = custom_normalization(data, data)
    del p

    # Create TensorDatasets
    train = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
    test = TensorDataset(torch.tensor(test.values, dtype=torch.float), torch.tensor(test.values, dtype=torch.float))
    data = torch.tensor(data.values, dtype=torch.float)
     # Create DataLoaders
    train, test = get_data(train, test, bs=bs)

    # Return DataBunch
    db = basic_data.DataBunch(train, test)

    #Load Model
    learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, wd=wd, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)
    learn.model_dir = path_to_saved_model
    learn.load(modelFile)
    learn.model.eval()

    #Make prediction, add labels, unnormalize
    pred = model(data)
    try:
        pred = pred.detach().numpy()
    except:
        pred = pred.cpu().detach().numpy()
    pred = pd.DataFrame(pred, columns=columns)
    pred = custom_unnormalize(pred)
    
    return pred

def passCleaning(row):
    if (row['HECFrac'] > 0.5 and row['HECQuality'] > .5 and row['AverageLArQF']/65535 > .8): #See Evernote for division explanation
        return False
    if (np.abs(row['NegativeE']) > 60000):
        return False
    if (row['EMFrac'] > .95 and row['LArQuality'] > .8 and row['AvgerageLArQF']/65535 > .8 and np.abs(row['eta']) < 2.8):
        return False
    if (row['EMFrac'] < .05 and np.abs(row['eta']) >= 2):
        return False
    else:
        return True

def correlationPlots():
    makeCorner = False #Make corner plots
    curr_save_folder = '/eos/user/s/sarobert/correlationPlots/'

     # Figures setup
    plt.close('all')
    unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
    variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
    line_style = ['--', '-']
    colors = ['red', 'c']
    markers = ['*', 's']

    # Histograms
    data = pd.read_pickle('/afs/cern.ch/work/s/sarobert/autoencoders/processedData/TLAJets_testing.pkl')
    data = filter_jets(data)
    data = data[0:1000]
    pred = getPred(data)
    columns = data.columns
    
    #print('Input')
    #print(data)
    #print('Output')
    #print(pred)
    dataPF = []
    predPF = []
    for index, row in data.iterrows():
        passed = int(passCleaning(row))
        dataPF.append(passed)
    
    for index, row in pred.iterrows():
        passed = int(passCleaning(row))
        predPF.append(passed)

    #print ('input', 'output')
    #print (dataPF, predPF)

    plt.scatter(dataPF, predPF)
    plt.title('reconstruction of phi comparison')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.savefig(curr_save_folder + 'phiCompare2.png')
#data = pd.read_pickle('/afs/cern.ch/work/s/sarobert/autoencoders/processedData/TLAJets.pkl')
#getPred(data)
correlationPlots()
