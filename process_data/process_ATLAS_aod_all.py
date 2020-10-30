#!/bin/python

#Authors: Eric Wulff, Erik Wallin, Honey Gupta, Caterina Doglioni
#This script processes TLA ATLAS data into pickle files. The number of variables kept is currently 27.
#If you want more variables, check the original root file and add to branchnames
#If you want fewer variables, remove what you don't need from branchnames

import numpy as np
import pandas as pd
import uproot
import awkward
import ROOT

from sklearn.model_selection import train_test_split

#change this to where your fileis
path_to_data = '/Users/urania277/Work/20192020_Autoencoders/2020_HoneyProject/ATLAS_datasets/'

# Load a ROOT input file and its tree with uproot
fname = 'DAOD_TRIG6.16825104._000263.pool.root.1'

#Branches in the initial trees
prefix = 'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn'
#another possible prefix found in other files:
#prefix = 'HLT_xAOD__JetContainer_a4tcemsubjesISFSAuxDyn'

branchnames = [
    # 4-momentum
    prefix + '.pt',
    prefix + '.eta',
    prefix + '.phi',
    prefix + '.m',
    ]

#this is if you only want the leading jet, or all jets - the example works with leading jet only...
leading_only = True

extraBranchNames = [
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

print("Number of variables considered: ", len(branchnames))

#one problem we will have here is that when we want to call uproot's functions, the variable names have too many "."s in the names.
#the solution is to set an alias, but this needs to be done on the ROOT version (rather than on the uproot version that we will use later)

#f=ROOT.TFile(path_to_data+fname)
#root_tree = f.Get("CollectionTree")

#for branchName in branchnames :
#    root_tree.SetAlias(prefix+"_"+branchName,prefix+"."+branchName)

#You can print every variable (branch) in the original tree here
#print(tree.keys())
filePath = path_to_data + fname
tree = uproot.open(filePath)['CollectionTree']

#find out how many jets there are in this tree
#Note that one event has more jets, but we ignore those boundaries
n_jets = sum(tree.array('HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.pt').counts)
print("Number of jets in this dataset: ", n_jets)

#dictionary containing the dataframes, one per variable
df_dict = {}
cache = {}
#loop over all the variables
for branchname in branchnames:
    #for the time being, do not consider vector variables (there are many calorimeter samplings in a single jet)
    if 'EnergyPerSampling' in branchname:
        pass
    else:
        print("Working on branch: ", branchname)
        #create an array per branch and flatten it into a mega-array
        branch_array = tree.array(branchname, cache=cache)
        #only get the branch name without the prefix to clean things up a bit in the pandas later
        variable = branchname.split('.')[1]
        df_dict[variable] = []
        #this is not super efficient/fast but it's simple - we want a dictionary that has the variable as key and an array as value, and each jet-entry as one of the entries in the array
        #...for better solutions, see uproot.iterate
        for event_entry in branch_array :
            for jet_entry in event_entry :
                df_dict[variable].append(jet_entry)
                if (leading_only) : break

#make a dataFrame out of the dictionary for easier splitting into train/test
print('Creating DataFrame...')
partial_df = pd.DataFrame(data=df_dict)
print('done.')

partial_train, partial_test = train_test_split(partial_df, test_size=0.2, train_size=0.8, random_state=41)

print('Creating DataFrame...')

print(partial_df.head)
print(partial_df.columns)

partial_train.to_pickle(path_to_data+'TLA_leadingJet_train_80.pkl')
partial_test.to_pickle(path_to_data+'TLA_leadingJet_test_20.pkl')
