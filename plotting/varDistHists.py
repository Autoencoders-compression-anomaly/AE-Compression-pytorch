import sys
BIN = '/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/'
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HEPAutoencoders.utils import min_filter_jets, normalize, filter_jets, custom_normalization

def varDistHists():
    curr_save_folder = '/eos/user/s/sarobert/TLAvariableDists/' #Where you want the plots
    #curr_save_folder = ''

    #Which plots to make
    makeRaw = False
    makeFilter = False
    makeMin = False
    makeNorm = True
    makeMinNorm = False

    #Set up figures
    unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
    variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
    line_style = ['--', '-']
    colors = ['orange', 'c']
    markers = ['*', 's']
    n_bins = 80

    pathToData = '/afs/cern.ch/work/s/sarobert/autoencoders/processedData/'
    train = pd.read_pickle(BIN + 'process_data/tla_jets_train.pkl')
    test = pd.read_pickle(BIN + 'process_data/tla_jets_test.pkl')
    data = pd.concat([train, test])
    labels = data.columns

    if (makeFilter or makeNorm):
        filterTrain = filter_jets(train)
        filterTest = filter_jets(test)
        dataFilter = pd.concat([filterTrain, filterTest]).to_numpy()

    if (makeMin or makeMinNorm):
        minTrain = min_filter_jets(train)
        minTest = min_filter_jets (test)
        dataMin = pd.concat([minTrain, minTest]).to_numpy()

    dataRaw = data.to_numpy()
    
    if makeNorm:
        normTrain = pd.DataFrame()
        normTest = pd.DataFrame()
        normTrain, normTest = custom_normalization(filterTrain, filterTest)
        dataNorm = pd.concat([normTrain, normTest]).to_numpy()
    
    if makeMinNorm:
        minNormTrain = pd.DataFrame()
        minNormTest = pd.DataFrame()
        minNormTrain, minNormTest = custom_normalization(minTrain, minTest)
        dataMinNorm = pd.concat([minNormTrain, minNormTest]).to_numpy()

    limitsDict = { #Cut thresholds for filtering scheme
            'EMFrac' : [-5, 5],
            'OotFracClusters5' : [-.1],
            'OotFracClusters10' : [-.1],
            'Width' : [-5, 5],
            'WidthPhi' : [-5, 5],
            'Timing' : [-125, 125],
            'LArQuality' : [4],
            'HECQuality' : [-2.5, 2.5],
            'NegativeE' : [-300000]
        }
    
    #RawDistributions 
    if makeRaw:
        for kk in np.arange(27):
            plt.figure()
            n_hist_data, bin_edges, _ = plt.hist(dataRaw[:, kk], color='deepskyblue', label='Input', alpha=1, bins=n_bins)
            plt.suptitle(labels[kk])
            #if labels[kk] == 'eta':
            #    plt.axvline(x = 2, color = 'darkorange')
            #    plt.arrow(2, 1000, dx = .15, dy = 0, width = .1, color='darkorange')
                
            #    plt.axvline(x = 2.8, color = 'red')
            #    plt.arrow(2.8, 1000, dx = .15, dy = 0, width = .1, color='red')
                
            #    plt.axvline(x = -2, color = 'darkorange')
            #    plt.arrow(-2, 1000, dx = -.15, dy = 0, width = .1, color='darkorange')
                
            #    plt.axvline(x = -2.8, color = 'red')
            #    plt.arrow(-2.8, 1000, dx = -.15, dy = 0, width = .1, color='red')
            #if labels[kk] == 'HECFrac':
            #    plt.axvline(x = .5, color = 'forestgreen')
            #    plt.arrow(.5, 100000, dx = .001, dy = 0, width = .1, color='forestgreen')
            #if labels[kk] == 'HECQuality':
            #    plt.axvline(x = .5, color = 'forestgreen')
            #    plt.arrow(.5, 100000, dx = .15, dy = 0, width = .1, color='forestgreen')
                
            #    plt.axvline(x = -.5, color = 'forestgreen')
            #    plt.arrow(-.5, 100000, dx = -.15, dy = 0, width = .1, color='forestgreen')
            #if labels[kk] == 'AverageLArQF':
            #    plt.axvline(x = .8 * 65535, color = 'k')
            #    plt.arrow(.8 * 65535, 100000, dx = 2000, dy = 0, width = .1, color='k')
            #if labels[kk] == 'NegativeE':
            #    plt.axvline(x = -60000, color = 'k')
            #    plt.arrow(-60000, 100000, dx = -1000, dy = 0, width = .1, color='k')
            #if labels[kk] == 'EMFrac':
            #    plt.axvline(x=.95, color = 'red')
            #    plt.arrow(.95, 100000, dx = .01, dy = 0, width = .1, color='red')
                
            #    plt.axvline(x=.05, color = 'darkorange')
            #    plt.arrow(.05, 100000, dx = -.01, dy = 0, width = .1, color='darkorange')
            #if labels[kk] == 'LArQuality':
            #    plt.axvline(x=.8, color='red')
            #    plt.arrow(.8, 100000, dx = .15, dy = 0, width = .1, color='red')
            try:
                try:
                    plt.axvline(x=limitsDict[labels[kk]][0], color = 'black')
                    plt.axvline(x=limitsDict[labels[kk]][1], color = 'black')
                except:
                    plt.axvline(x=limitsDict[labels[kk]][0], color = 'black')
            except:
                pass
            plt.xlabel(labels[kk])
            plt.ylabel('Number of events')
            plt.yscale('log')
            fig_name = 'raw_%s' % labels[kk]
            plt.savefig(curr_save_folder + fig_name)
            plt.close('all')

    if makeFilter:    
        #Distributions after filtering
        for kk in np.arange(27):
            plt.figure()
            n_hist_data, bin_edges, _ = plt.hist(dataFilter[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
            try:
                try:
                    plt.axvline(x=limitsDict[labels[kk]][0], color = 'black')
                    plt.axvline(x=limitsDict[labels[kk]][1], color = 'black')
                except:
                    plt.axvline(x=limitsDict[labels[kk]][0], color = 'black')
            except:
                pass
            plt.suptitle(labels[kk])
            plt.xlabel(labels[kk])
            plt.ylabel('Number of events')
            plt.yscale('log')
            fig_name = 'filter_%s' % labels[kk]
            plt.savefig(curr_save_folder + fig_name)
            plt.close('all')

    if makeMin:
        minLimitsDict = {
            'OotFracClusters5' : [-.1],
            'OotFracClusters10' : [-.1],
            'LeadingClusterPt' : [0]
        }

        for kk in np.arange(27):
            plt.figure()
            n_hist_data, bin_edges, _ = plt.hist(dataMin[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
            try:
                try:
                    plt.axvline(x=minLimitsDict[labels[kk]][0], color = 'black')
                    plt.axvline(x=minLimitsDict[labels[kk]][1], color = 'black')
                except:
                    plt.axvline(x=minLimitsDict[labels[kk]][0], color = 'black')
            except:
                pass
            plt.suptitle(labels[kk])
            plt.xlabel(labels[kk])
            plt.ylabel('Number of events')
            plt.yscale('log')
            fig_name = 'min_filter_%s' % labels[kk]
            plt.savefig(curr_save_folder + fig_name)
            plt.close('all')

    if makeNorm:
        normLimitsDict = {
            'EMFrac' : [-1.2, 1.2],
            'OotFracClusters5' : [-1./3.],
            'OotFracClusters10' : [-1./3.],
            'Width' : [-5, 5],
            'WidthPhi' : [-5/.6, 5/.6],
            'Timing' : [-25/8, 25/8],
            'LArQuality' : [1.8],
            'HECQuality' : [-2.5, 2.5],
            'NegativeE' : [np.log(301) / 1.6]# 300]
        }

        #Filtered and Normalized
        for kk in np.arange(27):
            plt.figure()
            n_hist_data, bin_edges, _ = plt.hist(dataNorm[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
            try:
                try:
                    plt.axvline(x=normLimitsDict[labels[kk]][0], color = 'black')
                    plt.axvline(x=normLimitsDict[labels[kk]][1], color = 'black')
                except:
                    plt.axvline(x=normLimitsDict[labels[kk]][0], color = 'black')
            except:
                pass
            plt.suptitle(labels[kk])
            plt.xlabel(labels[kk])
            plt.ylabel('Number of events')
            plt.yscale('log')
            fig_name = 'filterNorm_%s' % labels[kk]
            plt.savefig(curr_save_folder + fig_name)
            plt.close('all')

    if makeMinNorm:
        minNormLimitsDict = {
            'OotFracClusters5' : [-1./3.],
            'OotFracClusters10' : [-1./3.]
            #'LeadingClusterPt' : [0]
        }
        #Minimal filtering and custom normalization
        for kk in np.arange(27):
            plt.figure()
            n_hist_data, bin_edges, _ = plt.hist(dataMinNorm[:, kk], color='red', label='Input', alpha=1, bins=n_bins)
            try:
                try:
                    plt.axvline(x=minNormLimitsDict[labels[kk]][0], color = 'black')
                    plt.axvline(x=minNormLimitsDict[labels[kk]][1], color = 'black')
                except:
                    plt.axvline(x=minNormLimitsDict[labels[kk]][0], color = 'black')
            except:
                pass
            plt.suptitle(labels[kk])
            plt.xlabel(labels[kk])
            plt.ylabel('Number of events')
            plt.yscale('log')
            fig_name = 'min_filterNorm_%s' % labels[kk]
            plt.savefig(curr_save_folder + fig_name)
            plt.close('all')

    print ('Saved plots to ' + curr_save_folder)
 
varDistHists()
