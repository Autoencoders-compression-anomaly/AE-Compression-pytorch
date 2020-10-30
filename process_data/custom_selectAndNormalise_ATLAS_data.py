import sys
import numpy as np
import pandas as pd

#If this doesn't work, make sure the directory where the directory HEPAutoencoders is found is in your PYTHONPATH
#Something like this in bash, but use your own directory:
#export PYTHONPATH=/Users/urania277/Work/20192020_Autoencoders/2020_HoneyProject/Code/AE-Compression-pytorch:$PYTHONPATH
import utils.processing_ATLASData_utils as utils

#Here you should have the files that you produced using the process_ATLAS_aod_all.py script
train = pd.read_pickle('../../../ATLAS_datasets/TLA_leadingJet_train_80.pkl')
original_train_shape = train.shape
print('Original train.shape:', original_train_shape)
test = pd.read_pickle('../../../ATLAS_datasets/TLA_leadingJet_test_20.pkl')

# Remove extreme/bad jets, if any (the 4D version just removes massless jets)
train = utils.filter_unitconvert_jets_4D(train)
test = utils.filter_unitconvert_jets_4D(test)

print(train.shape)

print('Number of jets excluded:', original_train_shape[0] - train.shape[0])
print('Equivalent to:', 100*(original_train_shape[0] - train.shape[0]) / train.shape[0], '%')

print('New train.head() after cleaning:')
print(train.head())

print('New test.head() after cleaning:')
print(test.head())

#Use ad-hoc normalization
custom_normalized_train, custom_normalized_test = utils.custom_normalization_4D(train, test)

print('New train.head() after normalization:')
print(custom_normalized_train.head())

print('New test.head() after normalization:')
print(custom_normalized_test.head())

custom_normalized_train.to_pickle('../../../ATLAS_datasets/TLA_leadingJet_custom_normalized_train_80.pkl')
custom_normalized_test.to_pickle('../../../ATLAS_datasets/TLA_leadingJet_custom_normalized_train_20.pkl')
