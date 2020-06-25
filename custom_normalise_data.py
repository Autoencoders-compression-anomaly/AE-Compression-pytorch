import sys
import numpy as np
import pandas as pd

import HEPAutoencoders.utils as utils

train = pd.read_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/processed_data/aod/all_jets_partial_train.pkl')
original_train_shape = train.shape
print('Original train.shape:', original_train_shape)
test = pd.read_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/processed_data/aod/all_jets_partial_test.pkl')

# Remove extreme/bad jets
train = utils.filter_jets(train)
test = utils.filter_jets(test)

print(train.shape)

print('Number of jets excluded:')
print(original_train_shape[0] - train.shape[0])

print((original_train_shape[0] - train.shape[0]) / train.shape[0])

print('Number of samples with m=0', np.sum(train['m']==0))

print(train.head())

custom_normalized_train, custom_normalized_test = utils.custom_normalization(train, test)
unnormalized_test = utils.custom_unnormalize(custom_normalized_test)

print((np.abs(test - unnormalized_test) < 1e-10).all())

custom_normalized_train.to_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/processed_data/aod/custom_normalized_train.pkl')
custom_normalized_test.to_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/processed_data/aod/custom_normalized_test.pkl')

