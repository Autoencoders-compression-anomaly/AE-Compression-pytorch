import sys
BIN = '/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/'
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HEPAutoencoders.utils import normalize, filter_jets, custom_normalization

def varDistHists():
    curr_save_folder = '/eos/user/s/sarobert/variableDists/' #Where you want the plots
    
    #Set up figures
    unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
    variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
    line_style = ['--', '-']
    colors = ['orange', 'c']
    markers = ['*', 's']
    n_bins = 80
    train = pd.read_pickle(BIN + 'process_data/all_jets_partial_train.pkl')
    test = pd.read_pickle(BIN + 'process_data/all_jets_partial_test.pkl')
    data = pd.concat([train, test])
    labels = data.columns
    data = filter_jets(data)
    data = data.to_numpy()
    limitsDict = {
        'EMFrac' : [-2, 2],
        'OotFracClusters5' : [-.1, -.1], #Only one cut applied, can come up with a better method
        'OotFracClusters10' : [-.1, -.1],
        #'Width' : [-5, 5]
        'WidthPhi' : [-5, 5],
        'Timing' : [-125, 125],
        'LArQuality' : [4, 4],
        'HECQuality' : [-2.5, 2.5],
        'NegativeE' : [-300, 300]
    }

    for kk in np.arange(27):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        try:
            plt.axvline(x=limitsDict[labels[kk]][0])
            plt.axvline(x=limitsDict[labels[kk]][1])
        except:
            pass
        plt.suptitle(labels[kk])
        plt.xlabel(labels[kk])
        plt.ylabel('Number of events')
        plt.yscale('log')
        fig_name = 'hist_%s' % kk
        plt.savefig(curr_save_folder + fig_name)
        plt.close('all')

    print ('Saved plots to ' + curr_save_folder)
 
varDistHists()
