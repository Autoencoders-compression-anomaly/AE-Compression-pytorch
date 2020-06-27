import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

path_to_data = '/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/'

# Load a ROOT file
folder = 'data18_13TeV.00364292.calibration_DataScouting_05_Jets.deriv.DAOD_TRIG6.r10657_p3592_p3754/'
fname = 'DAOD_TRIG6.16825104._000263.pool.root.1'
filePath = path_to_data + folder + fname
#ttree = uproot.open(filePath)['outTree']['nominal']
tree = uproot.open(filePath)['CollectionTree']

print(tree.keys())

n_jets = sum(tree.array('HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.pt').counts)
print(n_jets)

branchnames = [
    # 4-momentum
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.pt',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.eta',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.phi',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.m',
    # Energy deposition in each calorimeter layer
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.EnergyPerSampling',
    # Area of jet,used for pile-up suppression (4-vector)
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_eta',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_m',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_phi',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_pt',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.JetGhostArea',
    # Variables related to quality of jet
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.AverageLArQF',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.BchCorrCell',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.NegativeE',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.HECQuality',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LArQuality',
    # Shape and position, most energetic cluster
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.Width',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.WidthPhi',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.CentroidR',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.DetectorEta',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterCenterLambda',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterPt',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterSecondLambda',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterSecondR',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.N90Constituents',
    # Energy released in each calorimeter
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.EMFrac',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.HECFrac',
    # Variables related to the time of arrival of a jet
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.Timing',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.OotFracClusters10',
    'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.OotFracClusters5',
]

print(len(branchnames))

EnergyPerSampling = tree.array(branchnames[4])
n_events = len(EnergyPerSampling)
counts = EnergyPerSampling.counts
print(n_events)

prefix = 'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn'
#prefix = 'HLT_xAOD__JetContainer_a4tcemsubjesISFSAuxDyn'
branchnames = [
    # 4-momentum
    prefix + '.pt',
    prefix + '.eta',
    prefix + '.phi',
    prefix + '.m',
    # Energy deposition in each calorimeter layer
    # prefix + '.EnergyPerSampling',
    # Area of jet,used for pile-up suppression (4-vector)
    prefix + '.ActiveArea',
    prefix + '.ActiveArea4vec_eta',
    prefix + '.ActiveArea4vec_m',
    prefix + '.ActiveArea4vec_phi',
    prefix + '.ActiveArea4vec_pt',
    # prefix + '.JetGhostArea',
    # Variables related to quality of jet
    prefix + '.AverageLArQF',
    # prefix + '.BchCorrCell',
    prefix + '.NegativeE',
    prefix + '.HECQuality',
    prefix + '.LArQuality',
    # Shape and position, most energetic cluster
    prefix + '.Width',
    prefix + '.WidthPhi',
    prefix + '.CentroidR',
    prefix + '.DetectorEta',
    prefix + '.LeadingClusterCenterLambda',
    prefix + '.LeadingClusterPt',
    prefix + '.LeadingClusterSecondLambda',
    prefix + '.LeadingClusterSecondR',
    prefix + '.N90Constituents',
    # Energy released in each calorimeter
    prefix + '.EMFrac',
    prefix + '.HECFrac',
    # Variables related to the time of arrival of a jet
    prefix + '.Timing',
    prefix + '.OotFracClusters10',
    prefix + '.OotFracClusters5',
]

print(len(branchnames))

df_dict = {}
for pp, branchname in enumerate(branchnames):
    if 'EnergyPerSampling' in branchname:
        pass
    else:
        variable = branchname.split('.')[1]
        df_dict[variable] = []
        jaggedX = tree.array(branchname)
        for ii, arr in enumerate(jaggedX):
            for kk, val in enumerate(arr):
                df_dict[variable].append(val)
    if pp % 3 == 0:
        print((pp * 100) // len(branchnames), '%')
print('100%')
print('Creating DataFrame...')
partial_df = pd.DataFrame(data=df_dict)
print('done.')

del df_dict

print(partial_df.head)

print(partial_df.columns)

partial_train, partial_test = train_test_split(partial_df, test_size=0.2, random_state=41)

partial_train.to_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/all_jets_partial_train_263.pkl')
partial_test.to_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/all_jets_partial_test_263.pkl')


