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
# mpl.rc_file(BIN + 'my_matplotlib_rcparams')

# print('torch.cuda.is_available(): ' + str(torch.cuda.is_available()))

if torch.cuda.is_available():
    torch_core.defaults.device = 'cuda'
    print('Using GPU for training')

lr = 1e-3
wds = 1e-5
pp = 0

save_dict = {}

# Load data
train = pd.read_pickle(BIN + 'process_data/all_jets_partial_train.pkl')
test = pd.read_pickle(BIN + 'process_data/all_jets_partial_test.pkl')

#Filter and normalize data
#train = filter_jets(train)
#test = filter_jets(test)

train = min_filter_jets(train)
test = min_filter_jets(test)

#train, test = custom_normalization(train, test)
#train, test = normalize(train, test)

train_mean = train.mean()
train_std = train.std()

bs = 4096 #Batch Size
# Create TensorDatasets
train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float), torch.tensor(test.values, dtype=torch.float))
# Create DataLoaders
train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
# Return DataBunch
db = basic_data.DataBunch(train_dl, valid_dl)

# loss_func = RMSELoss()
loss_func = nn.MSELoss()

bn_wd = False  # Don't use weight decay for batchnorm layers
true_wd = True  # wd will be used for all optimizers

# Figures setup
plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
line_style = ['--', '-']
colors = ['red', 'c']
markers = ['*', 's']


def get_unnormalized_reconstructions(model, df, train_mean, train_std, idxs=None):
    if idxs is not None:
        data = torch.tensor(df[idxs[0]:idxs[1]].values)
    else:
        data = torch.tensor(df.values)
    pred = model(data).detach().numpy()
    pred = np.multiply(pred, train_std.values)
    pred = np.add(pred, train_mean.values)
    data = np.multiply(data, train_std.values)
    data = np.add(data, train_mean.values)
    return pred, data


def train_model(model, epochs, lr, wd, module_string, ct, path):
    plt.close('all')
    learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, wd=wd, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)
    start = time.perf_counter()
    if ct:
        learn.load(path)
        print('Model loaded: ', path)
    learn.fit_one_cycle(epochs, max_lr=lr, wd=wd, callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='best_%s_bs%s_lr%.0e_wd%.0e' % (module_string, bs, lr, wd))])
    end = time.perf_counter()
    delta_t = end - start
    return learn, delta_t


def get_mod_folder(module_string, lr, pp, wd):
    if pp is None:
        curr_mod_folder = '%s_bs%d_lr%.0e_wd%.0e_ppNA/' % (module_string, bs, lr, wd)
    else:
        curr_mod_folder = '%s_bs%d_lr%.0e_wd%.0e_p%.0e/' % (module_string, bs, lr, wd, pp)
    return curr_mod_folder


def save_plots(learn, module_string, lr, wd, pp):
    # Make and save figures
    curr_mod_folder = get_mod_folder(module_string, lr, pp, wd)
    curr_save_folder = curr_mod_folder
    if not os.path.exists(curr_save_folder):
        os.mkdir(curr_save_folder)

    # Weight activation stats
    #plot_activations(learn, save=curr_save_folder + 'weight_activation')

    # Plot losses
    batches = len(learn.recorder.losses)
    epos = len(learn.recorder.val_losses)
    val_iter = (batches / epos) * np.arange(1, epos + 1, 1)
    loss_name = str(loss_func).split("(")[0]
    plt.figure()
    plt.plot(learn.recorder.losses, label='Train')
    plt.plot(val_iter, learn.recorder.val_losses, label='Validation', color='orange')
    plt.yscale(value='log')
    plt.legend()
    plt.ylabel(loss_name)
    plt.xlabel('Batches processed')
    fig_name = 'losses'
    plt.savefig(curr_save_folder + fig_name)
    plt.figure()
    plt.plot(learn.recorder.val_losses, label='Validation', color='orange')
    plt.title('Validation loss')
    plt.legend()
    plt.ylabel(loss_name)
    plt.yscale('log')
    plt.xlabel('Epoch')
    # for i_val, val in enumerate(learn.recorder.val_losses):
    #     plt.text(i_val, val, str(val), horizontalalignment='center')
    fig_name = 'losses_val'
    plt.savefig(curr_save_folder + fig_name + '.png')
    with open(curr_save_folder + 'losses.txt', 'w') as f:
        for i_val, val in enumerate(learn.recorder.val_losses):
            f.write('Epoch %d    Validation %s: %e    Training %s: %e\n' % (i_val, loss_name, val, loss_name, learn.recorder.losses[(i_val + 1) * (int(batches / epos - 1))]))

    # Uncomment this in order to plot histograms and residuals at the end of training
    # Histograms
    idxs = (0, 100000)  # Pick events to compare
    data = torch.tensor(test[idxs[0]:idxs[1]].values, dtype=torch.float, device=torch.device('cuda'))
#    unnormalized_pred_df, unnormalized_data_df = get_unnormalized_reconstructions(learn.model, df=data_df, train_mean=train_mean, train_std=train_std, idxs=idxs)
    pred = learn.model(data).cpu().detach().numpy()
    data = data.cpu().detach().numpy()

    data_df = pd.DataFrame(data, columns=test.columns)
    pred_df = pd.DataFrame(pred, columns=test.columns)

    # Unnormalize
    #unnormalized_data_df = custom_unnormalize(data_df)
    #unnormalized_pred_df = custom_unnormalize(pred_df)

    # Handle variables with discrete distributions
    unnormalized_pred_df['N90Constituents'] = unnormalized_pred_df['N90Constituents'].round()
    uniques = unnormalized_data_df['ActiveArea'].unique()
    utils.round_to_input(unnormalized_pred_df, uniques, 'ActiveArea')

    #data = unnormalized_data_df
    #pred = unnormalized_pred_df
    data = data_df
    pred = pred_df
    residuals = (pred - data) / data

    idxs = (0, 100000)  # Choose events to compare
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

    #Make correlation matrix plot

    # diff = (pred - data)

    diff_list = ['ActiveArea',
                 'ActiveArea4vec_phi',
                 'ActiveArea4vec_eta',
                 'ActiveArea4vec_pt',
                 'ActiveArea4vec_m',
                 'N90Constituents',
                 'NegativeE',
                 'OotFracClusters5',
                 'OotFracClusters10',
                 'Timing',
                 'Width',
                 'WidthPhi',
                 'LeadingClusterCenterLambda',
                 'LeadingClusterSecondLambda',
                 'CentroidR',
                 'LeadingClusterSecondR',
                 'LArQuality',
                 'HECQuality',
                 'HECFrac',
                 'EMFrac',
                 'AverageLArQF',
                 'phi',
                 'eta',
                 'DetectorEta',
             ]
    
    res_df = pd.DataFrame(residuals, columns=test.columns)

    lab_dict = {
        'pt': '$(p_{T,out} - p_{T,in}) / p_{T,in}$',
        'eta': '$\eta_{out} - \eta_{in}$ [rad]',
        'phi': '$\phi_{out} - \phi_{in}$ [rad]',
        'm': '$(m_{out} - m_{in}) / m_{in}$',
        'ActiveArea': 'Difference in ActiveArea',
        'ActiveArea4vec_eta': 'Difference in ActiveArea4vec_eta',
        'ActiveArea4vec_m': 'Difference in ActiveArea4vec_m',
        'ActiveArea4vec_phi': 'Difference in ActiveArea4vec_phi',
        'ActiveArea4vec_pt': 'Difference in ActiveArea4vec_pt',
        'AverageLArQF': 'Difference in AverageLArQF',
        'NegativeE': 'Difference in NegativeE',
        'HECQuality': 'Difference in HECQuality',
        'LArQuality': 'Difference in LArQuality',
        'Width': 'Difference in Width',
        'WidthPhi': 'Difference in WidthPhi',
        'CentroidR': 'Difference in CentroidR',
        'DetectorEta': 'Difference in DetectorEta',
        'LeadingClusterCenterLambda': 'Difference in LeadingClusterCenterLambda',
        'LeadingClusterPt': 'Difference in LeadingClusterPt',
        'LeadingClusterSecondLambda': 'Difference in LeadingClusterSecondLambda',
        'LeadingClusterSecondR': 'Difference in LeadingClusterSecondR',
        'N90Constituents': 'Difference in N90Constituents',
        'EMFrac': 'Difference in EMFrac',
        'HECFrac': 'Difference in HECFrac',
        'Timing': 'Difference in Timing',
        'OotFracClusters10': 'Difference in OotFracClusters10',
        'OotFracClusters5': 'Difference in OotFracClusters5',
    }
    
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
    
    plt.close('all')
    # Compute correlations
    corr = res_df.corr()

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

    fig_name = 'corr_16.png'
    plt.savefig(curr_save_folder + fig_name)

    #Make Correlation corner plots
    for i_group, group in enumerate(corner_groups):
        plt.close('all')
        group_df = residuals[group]
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
        # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,
        #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
        # plt.subplots_adjust(left=.23, bottom=.30, top=.99, right=.99)
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
            ranges = [(-0.09871116682884362, 0.08086916175124935), (-0.4287975852936506, 0.25447835482657233), (-0.4473390866829308, 0.33586700443955386), (-2, 1.0100613135335297), (-0.20000000298023224, 0.1111111119389534)]
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

    return curr_mod_folder


def train_and_save(model, epochs, lr, wd, pp, module_string, save_dict, ct, path):
    if pp is None:
        curr_param_string = 'bs%d_lr%.0e_wd%.0e_ppNA' % (bs, lr, wd)
    else:
        curr_param_string = 'bs%d_lr%.0e_wd%.0e_pp%.0e' % (bs, lr, wd, pp)

    learn, delta_t = train_model(model, epochs=epochs, lr=lr, wd=wd, module_string=module_string, ct=ct, path=path)
    time_string = str(datetime.timedelta(seconds=delta_t))
    curr_mod_folder = save_plots(learn, module_string, lr, wd, pp)

    val_losses = learn.recorder.val_losses
    train_losses = learn.recorder.losses
    min_val_loss = np.min(val_losses)
    min_epoch = np.argmin(val_losses)

    save_dict[module_string].update({curr_param_string: {}})
    save_dict[module_string][curr_param_string].update({'val_losses': val_losses, 'train_losses': train_losses, 'hyper_parameter_names': [
        'bs', 'lr', 'wd', 'pp'], 'hyper_parameters': [bs, lr, wd, pp], 'training_time_seconds': delta_t})
    curr_save_folder = get_mod_folder(module_string, lr, pp, wd)
    with open(curr_save_folder + 'save_dict%s.pkl' % curr_param_string, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    learn.save(curr_mod_folder.split('/')[0])
    with open(curr_save_folder + 'summary.txt', 'w') as f:
        f.write('%s Minimum validation loss: %e epoch: %d lr: %.1e wd: %.1e p: %s Training time: %s\n' % (module_string, min_val_loss, min_epoch, lr, wd, pp, time_string))


#<<<<<<< HEAD
#one_epochs = 100
#one_lr = 1e-4
#=======
one_epochs = 100
one_lr = 1e-2
one_wd = 1e-2
one_pp = None
one_module = AE_bn_LeakyReLU
continue_training = True
checkpoint_name = 'nn_utils_bs1024_lr1e-04_wd1e-02_ppNA' # Want to format string like: % (bs, one_lr, one_wd)

def one_run(module, epochs, lr, wd, pp, ct):
    module_string = str(module).split("'")[1].split(".")[1]
    save_dict[module_string] = {}
    if pp is not None:
        print('Training %s with lr=%.1e, p=%.1e, wd=%.1e ...' % (module_string, lr, pp, wd))
        curr_model_p = module(dropout=pp)
        train_and_save(curr_model_p, epochs, lr, wd, pp, module_string, save_dict, ct, checkpoint_name)
        print('...done')
    else:
        print('Training %s with lr=%.1e, p=None, wd=%.1e ...' % (module_string, lr, wd))
        curr_model = module([27, 200, 200, 200, 14, 200, 200, 200, 27])
        train_and_save(curr_model, epochs, lr, wd, pp, module_string, save_dict, ct, checkpoint_name)
        print('...done')

tic = time.time()
one_run(module=one_module, epochs=one_epochs, lr=one_lr, wd=one_wd, pp=one_pp, ct=continue_training)
toc = time.time()

time_taken = (toc-tic)/60.0

print('Total time taken: ', time_taken , 'minutes')
print('Time taken per epoch: ', time_taken / one_epochs, ' minutes')
