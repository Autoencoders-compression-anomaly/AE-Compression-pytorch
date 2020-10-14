import sys
BIN = '/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/'
sys.path.append(BIN)
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def passClean(row): #All Cleaning Criteria
    if (row['HECFrac'] > 0.5 and np.abs(row['HECQuality']) > .5 and row['AverageLArQF']/65535 > .8): #See Evernote for division explanation
        return False
    if (np.abs(row['NegativeE']) > 60):
        return False
    if (row['EMFrac'] > .95 and row['LArQuality'] > .8 and row['AverageLArQF']/65535 > .8 and np.abs(row['eta']) < 2.8):
        return False
    if (row['EMFrac'] < .05 and np.abs(row['eta']) >= 2):
        return False
    else:
        return True

def passVar(row, Cut, var):
    if np.abs(row[var]) < Cut:
        return False
    else:
        return True

def thresholdMigration(data_str, pred_str):
    curr_save_folder = '/eos/user/s/sarobert/thresholds2/'
    save = True
    plt.close('all')

    #Load Data
    data = pd.read_pickle(data_str)
    pred = pd.read_pickle(pred_str)
    print ('Loaded Data')
    columns = data.columns
    
    dataPF = []
    predPF = []

    var = 'EMFrac' #Which variable to be studied
    cut = .05 #The cut value

    for index, row in data.iterrows():
        #passed = int(passClean(row))
        passed = int(passVar(row, cut, var))
        dataPF.append(passed)
    
    for index, row in pred.iterrows():
        #passed = int(passClean(row))
        passed = int(passVar(row, cut, var))
        predPF.append(passed)
    
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    labels = [var + ' < ' + str(cut), var + ' >' + str(cut)] #Doesn't work for badJet threshold
    ax.set_xticks([.25, .75])
    ax.set_yticks([.25, .75])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.rcParams.update({'figure.autolayout': True})

    counts, xedges, yedge, plot = plt.hist2d(dataPF, predPF, bins=2, range=[[0, 1], [0, 1]], cmap=plt.cm.jet, norm=mpl.colors.LogNorm(), data=None)
    plt.colorbar()
    plt.title('Reconstruction Migration around ' + var + ' = ' + str(cut))
    plt.xlabel('Input')
    plt.ylabel('AE Reconstruction')

    for i, row in enumerate(counts):
        for j, binCount in enumerate(counts[i]):
            plt.text(float(i)/2 + 1/4, float(j)/2 + 1/4, str(counts[i][j]), color='lightgray', horizontalalignment='center', verticalalignment='center')
    if save:
        plt.savefig(curr_save_folder + var + str(cut) + '.png')

pathToData = '/afs/cern.ch/work/s/sarobert/autoencoders/processedData/'
data = pathToData + 'TLAJets_testing2_minFilter.pkl'
pred = pathToData + 'pred_TLAJets_testing2_minFilter.pkl'
thresholdMigration(data, pred)
