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
import seaborn as sns
from corner import corner

def evaluateNetwork():
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
    train = pd.read_pickle(BIN + 'process_data/tla_jets_train.pkl')
    test = pd.read_pickle(BIN + 'process_data/tla_jets_test.pkl')

    #Filter and normalize data
    train = filter_jets(train)
    test = filter_jets(test)

    train, test = custom_normalization(train, test)
    #train, test = normalize(train, test)
    data = pd.concat([train, test])
    train_mean = train.mean()
    train_std = train.std()

    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float), torch.tensor(test.values, dtype=torch.float))

     # Create DataLoaders
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)

    # Return DataBunch
    db = basic_data.DataBunch(train_dl, valid_dl)

    learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, wd=wd, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)
    learn.model_dir = path_to_saved_model
    learn.load(modelFile)
    learn.model.eval()

    data = torch.tensor(test.values, dtype=torch.float)
    #unnormalized_pred_df, unnormalized_data_df = get_unnormalized_reconstructions(learn.model, df=data_df, train_mean=train_mean, train_std=train_std, idxs=idxs)
    pred = learn.model(data).cpu().detach().numpy()

    return pred

def correlationPlots():
    makeCorner = False #Make corner plots
    useNetwork = False #If using a network to make predictions
    curr_save_folder = '/eos/user/s/sarobert/TLAcorrelationPlots/'
    pathToData = '/afs/cern.ch/work/s/sarobert/autoencoders/processedData/'

     # Figures setup
    plt.close('all')

    # Histograms
    if useNetwork:
        pred = evaluateNetwork()
        data = pd.read_pickle(BIN + 'process_data/TLAJets.pkl')
        columns = data.columns
        #data_df = pd.DataFrame(data, columns=test.columns)
        pred_df = pd.DataFrame(pred, columns=columns)
    
        # Unnormalize
        #unnormalized_data_df = custom_unnormalize(data)
        unnormalized_pred_df = custom_unnormalize(pred_df)
        
        # Handle variables with discrete distributions
        unnormalized_pred_df['N90Constituents'] = unnormalized_pred_df['N90Constituents'].round()
        uniques = data['ActiveArea'].unique()
        utils.round_to_input(unnormalized_pred_df, uniques, 'ActiveArea')

        #data = unnormalized_data_df
        pred = unnormalized_pred_df
        residuals = (pred - data) / data
        res_df = pd.DataFrame(residuals, columns=columns)

#        data = data.to_numpy()
#        pred = pred.to_numpy()

    else:
        data = pd.read_pickle(pathToData + 'TLAJets.pkl')
        columns = data.columns

    data = filter_jets(data)
    data, p = custom_normalization(data, data)
    del p

    alph = 0.8
    n_bins = 80
#    labels = data.columns
    

    plt.close('all')
    # Compute correlations
    corr = data.corr()
    print (corr)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap = 'RdBu'
    # Plot heatmap
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.subplots_adjust(left=.23, bottom=.30, top=.99, right=.99)

    fig_name = 'TLAnormCorrelations.png'
    plt.savefig(curr_save_folder + fig_name)
    
    if makeCorner:
        corner_groups = [
            ['pt', 'eta', 'phi', 'm'],
            ['pt', 'eta', 'ActiveArea', 'ActiveArea4vec_eta', 'ActiveArea4vec_phi', 'ActiveArea4vec_pt', 'ActiveArea4vec_m'],
            ['pt', 'eta', 'AverageLArQF', 'NegativeE'],
            ['pt', 'eta', 'HECQuality', 'LArQuality'],
            ['pt', 'eta', 'Width', 'WidthPhi', 'N90Constituents'],
            ['pt', 'eta', 'CentroidR', 'DetectorEta'],
            ['pt', 'eta', 'LeadingClusterPt', 'LeadingClusterCenterLambda', 'LeadingClusterSecondLambda', 'LeadingClusterSecondR'],
            ['pt', 'eta', 'EMFrac', 'HECFrac'],
            ['pt', 'eta', 'Timing', 'OotFracClusters5', 'OotFracClusters10'],
        ]

        #Make Correlation corner plots
        for i_group, group in enumerate(corner_groups):
            plt.close('all')
            group_df = data[group]
            #plt.figure()
            # Compute correlations
            corr = group_df.corr()
        
            # Generate a mask for the upper triangle
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(10, 220, as_cmap=True)
            #cmap = 'RdBu'
            norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            # Plot heatmap
            mpl.rcParams['xtick.labelsize'] = 12
            mpl.rcParams['ytick.labelsize'] = 12

            mpl.rcParams['ytick.labelsize'] = 12
            #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,
            #            square=True, linewidths=.5, cbar_kws={"shrink": .5})
            #plt.subplots_adjust(left=.23, bottom=.30, top=.99, right=.99)
            #mpl.rc_file(BIN + 'my_matplotlib_rcparams')

            #     fig_name = 'corr_%d_group%d.png' % (latent_dim, i_group)
        #     plt.savefig(curr_save_folder + fig_name)

            label_kwargs = {'fontsize': 12, 'rotation': -60, 'ha': 'right'}
            title_kwargs = {"fontsize": 9}
            mpl.rcParams['lines.linewidth'] = 1
            mpl.rcParams['xtick.labelsize'] = 12
            mpl.rcParams['ytick.labelsize'] = 12
            group_arr = group_df.values
            qs = np.quantile(group_arr, q=[.0025, .9925], axis=0)
            ndim = qs.shape[1]
            if (i_group == 0):
                ranges = [(-0.09871116682884362, 0.08086916175124935), (-0.4287975852936506, 0.25447835482657233), (-0.4813213293999434, 0.16533099643886573), (-.5, .5)]
            if (i_group == 3):
                ranges = [(-0.09871116682884362, 0.08086916175124935), (-0.4287975852936506, 0.25447835482657233), (-1, 1), (-1, 1)]
            if (i_group == 4):
                ranges = [(-0.09871116682884362, 0.08086916175124935), (-0.4287975852936506, 0.25447835482657233), (-0.4473390866829308, 0.33586700443955386), (-2, 1.0100613135335297), (-0.2000000298023224, 0.1111111119389534)]
            if (i_group == 7):
                ranges = [(-0.09871116682884362, 0.08086916175124935), (-0.4287975852936506, 0.25447835482657233), (-0.08020495220005382, 0.06399791303144527), (-.5, .5)]
            if (i_group == 8):
                ranges = [(-0.09871116682884362, 0.08086916175124935), (-0.4287975852936506, 0.25447835482657233), (-37.25050435066223, 11.922051482200635), (-1, 1), (-1, 1)]
            if (i_group in [1,2,5,6]):
                ranges = [tuple(qs[:, kk]) for kk in np.arange(ndim)]
            #print (i_group)
            #print (ranges)
            figure = corner(group_arr, range=ranges, plot_density=True, plot_contours=True, no_fill_contours=False, #range=[range for i in np.arange(ndim)],
                        bins=50, labels=group, label_kwargs=label_kwargs, #truths=[0 for kk in np.arange(qs.shape[1])],
                        show_titles=True, title_kwargs=title_kwargs, quantiles=(0.16, 0.84),
                        # levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.2e')
                        levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.1e')

            # # Extract the axes
            axes = np.array(figure.axes).reshape((ndim, ndim))
            # Loop over the diagonal
            linecol = 'r'
            linstyl = 'dashed'

            # for i in range(ndim):
            #     ax = axes[i, i]
            #     ax.axvline(0, color=linecol, linestyle=linstyl)
            #     ax.axvline(0, color=linecol, linestyle=linstyl)
            for xi in range(ndim):
                ax = axes[0, xi]
                # Set xlabel coords
                ax.xaxis.set_label_coords(.5, -.8)

            for yi in range(ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    # Set face color according to correlation
                    ax.set_facecolor(color=mappable.to_rgba(corr.values[yi, xi]))
            cax = figure.add_axes([.87, .4, .04, 0.55])
            cbar = plt.colorbar(mappable, cax=cax, format='%.1f', ticks=np.arange(-1., 1.1, 0.2))
            cbar.ax.set_ylabel('Correlation', fontsize=20)

            if i_group == 6:
                plt.subplots_adjust(left=0.13, bottom=0.21, right=.82)
            else:
                plt.subplots_adjust(left=0.13, bottom=0.20, right=.83)

            fig_name = 'slide_corner_%d_group%d' % (14, i_group)
            plt.savefig(curr_save_folder + fig_name)

correlationPlots()
