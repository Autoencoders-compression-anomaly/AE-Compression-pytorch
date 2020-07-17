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
# import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset
from fastai.callbacks.tracker import SaveModelCallback
# import my_matplotlib_style as ms
# from utils as ms

from fastai import basic_train, basic_data, torch_core
from fastai.callbacks import ActivationStats
from fastai import train as tr

from HEPAutoencoders.nn_utils import AE_basic, AE_bn, AE_LeakyReLU, AE_bn_LeakyReLU, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn
from HEPAutoencoders.nn_utils import get_data, RMSELoss
from HEPAutoencoders.utils import min_filter_jets, filter_jets, plot_activations, custom_normalization, normalize, custom_unnormalize 
import HEPAutoencoders.utils as utils
import matplotlib as mpl
import seaborn as sns
from corner import corner

def makePlots():
    curr_save_folder = '/eos/user/s/sarobert/correlationPlots/'
    # Load data
    train = pd.read_pickle(BIN + 'process_data/all_jets_partial_train.pkl')
    test = pd.read_pickle(BIN + 'process_data/all_jets_partial_test.pkl')
    
    #Filter and normalize data
    train = filter_jets(train)
    test = filter_jets(test)
    
    train, test = custom_normalization(train, test)
    #train, test = normalize(train, test)
    
    train_mean = train.mean()
    train_std = train.std()

    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float), torch.tensor(test.values, dtype=torch.float))
    
    # Create DataLoaders
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
    
    # Return DataBunch
    db = basic_data.DataBunch(train_dl, valid_dl)

    loss_func = nn.MSELoss()
    path_to_saved_model = '/afs/cern.ch/user/s/sarobert/autoencoders/outputs/jul6-100ep-filterNorm/models/'
    modelFile =  'best_nn_utils_bs4096_lr1e-02_wd1e-02' 
    bn_wd = False  # Don't use weight decay for batchnorm layers
    true_wd = True  # wd will be used for all optimizers
    bs = 4096 
    wd = 1e-2
    module = AE_bn_LeakyReLU
    model = module([27, 200, 200, 200, 14, 200, 200, 200, 27])
    learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, wd=wd, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)
    learn.model_dir = path_to_saved_model
    learn.load(modelFile)
    learn.model.eval()

    # Figures setup
    plt.close('all')
    unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
    variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
    line_style = ['--', '-']
    colors = ['red', 'c']
    markers = ['*', 's']

     # Histograms
    #idxs = (0, 100000)  # Pick events to compare
    data = torch.tensor(test.values, dtype=torch.float)
#    unnormalized_pred_df, unnormalized_data_df = get_unnormalized_reconstructions(learn.model, df=data_df, train_mean=train_mean, train_std=train_std, idxs=idxs)
    pred = learn.model(data).cpu().detach().numpy()
    data = data.cpu().detach().numpy()

    data_df = pd.DataFrame(data, columns=test.columns)
    pred_df = pd.DataFrame(pred, columns=test.columns)

     # Unnormalize
    unnormalized_data_df = custom_unnormalize(data_df)
    unnormalized_pred_df = custom_unnormalize(pred_df)

    # Handle variables with discrete distributions
    unnormalized_pred_df['N90Constituents'] = unnormalized_pred_df['N90Constituents'].round()
    uniques = unnormalized_data_df['ActiveArea'].unique()
    utils.round_to_input(unnormalized_pred_df, uniques, 'ActiveArea')

    data = unnormalized_data_df
    pred = unnormalized_pred_df
    #data = data_df
    #pred = pred_df
    residuals = (pred - data) / data

    #idxs = (0, 100000)  # Choose events to compare
    alph = 0.8
    n_bins = 80
    labels = train.columns
    data = data.to_numpy() #convert to numpy to feed into pyplot
    pred = pred.to_numpy()
    for kk in np.arange(27):
        plt.close('all')
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
        plt.suptitle(labels[kk])
        plt.legend(loc="upper right")
        # plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        plt.xlabel(labels[kk])
        plt.ylabel('Number of events')
        plt.yscale('log')
        fig_name = 'hist_%s' % train.columns[kk]
        plt.savefig(curr_save_folder + fig_name)

    resNump = residuals.to_numpy()
    rangeDict = { #Hardcoding residual plot ranges
        'm' : (-.5, .5),
        'LArQuality' : (-1,1),
        'HECQuality' : (-1, 1),
        'WidthPhi' : (-1, 1.0100613135335297),
        'HECFrac' : (-.5, .5),
        'OotFracClusters5' : (-1, 1),
        'OotFracClusters10' : (-1, 1),
        'AverageLArQF' : (-1, 1),
        'NegativeE' : (-.5, .5),
        'LeadingClusterCenterLambda' : (-.5, .5),
        'LeadingClusterSecondLambda' : (-1, 1),
        'LeadingClusterSecondR' : (-1.5, 3),
        'Timing' : (-10, 10),
        'Width' : (-.5, .5),
        'phi' : (-.5, .5),
        'N90Constituents' : (-.25, .25),
         'eta' : (-.5, .5),
        'EMFrac' : (-.25, .25),
        'DetectorEta' : (-.5, .5),
        'CentroidR' : (-.2, .2),
        'ActiveArea4vec_phi' : (-.5, .5),
        'ActiveArea4vec_eta' : (-.5, .5)
    }
    n_bins = 150
    for kk in np.arange(27):
        plt.close('all')
        print ('making residual plot for ' + str(residuals.columns[kk]))
        plt.figure()
        if (residuals.columns[kk] in rangeDict):
            ranges = rangeDict[residuals.columns[kk]]
        else:
            ranges = None
        n_hist_data, bin_edges, _ = plt.hist(resNump[:, kk], range=ranges, color=colors[1], label='Residuals', alpha=1, bins=n_bins)
        plt.suptitle(residuals.columns[kk])
        plt.legend(loc="upper right")
        plt.xlabel(residuals.columns[kk])
        plt.ylabel('Number of events')
        fig_name = 'residualHist_%s' % residuals.columns[kk]
        plt.savefig(curr_save_folder + fig_name)

makePlots()
