#Makes a prediction dataset from a trained network
import sys
BIN = '/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/'
sys.path.append(BIN)
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset

from fastai import basic_train, basic_data, torch_core
from fastai.callbacks import ActivationStats

from HEPAutoencoders.nn_utils import AE_basic, AE_bn, AE_LeakyReLU, AE_bn_LeakyReLU, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn, get_data, RMSELoss
from HEPAutoencoders.utils import min_filter_jets, filter_jets, plot_activations, custom_normalization, normalize, custom_unnormalize
import HEPAutoencoders.utils as utils

def getPred(data_str):
    saveDir = '/afs/cern.ch/work/s/sarobert/autoencoders/processedData/'
    loss_func = nn.MSELoss()
    path_to_saved_model = '/afs/cern.ch/user/s/sarobert/autoencoders/outputs/jul21-100ep-TLANorm/models/'
    modelFile =  'best_nn_utils_bs4096_lr1e-02_wd1e-02'
    bn_wd = False  # Don't use weight decay for batchnorm layers
    true_wd = True  # wd will be used for all optimizers
    bs = 4096
    wd = 1e-2
    module = AE_bn_LeakyReLU
    model = module([27, 200, 200, 200, 14, 200, 200, 200, 27])

    # Load data
    data = pd.read_pickle(data_str)
    train = pd.read_pickle(BIN + 'process_data/tla_jets_train.pkl')
    test = pd.read_pickle(BIN + 'process_data/tla_jets_test.pkl')
    columns = data.columns

    #Filter and normalize data
    train = filter_jets(train)
    test = filter_jets(test)
    data = filter_jets(data)

    data.to_pickle(saveDir + 'TLAJets_test_filter.pkl') # To make sure that the jets are in the same order
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

    pred.to_pickle(saveDir + 'pred_TLAJets_testing.pkl')

pathToData = '/afs/cern.ch/work/s/sarobert/autoencoders/processedData/'
data_str = pathToData + 'TLAJets_testing.pkl'
getPred(data_str)
