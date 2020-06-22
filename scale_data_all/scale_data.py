import sys
# BIN = '../../'
# sys.path.append(BIN)
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pickle


#import my_matplotlib_style as ms
#import matplotlib as mpl
# mpl.rc_file(BIN + 'my_matplotlib_rcparams')

train = pd.read_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/processed_data/aod/all_jets_partial_train.pkl')
test = pd.read_pickle('/afs/cern.ch/work/h/hgupta/public/AE-Compression-pytorch/datasets/processed_data/aod/all_jets_partial_test.pkl')

train['pt'] = train['pt'] / 1000.  # Convert to GeV
test['pt'] = test['pt'] / 1000.  # Convert to GeV
train['m'] = train['m'] / 1000.  # Convert to GeV
test['m'] = test['m'] / 1000.  # Convert to GeV

train['LeadingClusterPt'] = train['LeadingClusterPt'] / 1000.  # Convert to GeV
test['LeadingClusterPt'] = test['LeadingClusterPt'] / 1000.  # Convert to GeV

train['LeadingClusterSecondR'] = train['LeadingClusterSecondR'] / 1000.  # Convert to GeV
test['LeadingClusterSecondR'] = test['LeadingClusterSecondR'] / 1000.  # Convert to GeV

train['LeadingClusterSecondLambda'] = train['LeadingClusterSecondLambda'] / 1000.  # Convert to GeV
test['LeadingClusterSecondLambda'] = test['LeadingClusterSecondLambda'] / 1000.  # Convert to GeV

# Remove all jets with EMFrac outside (-2, 2)
train = train[(train['EMFrac'] < 2) & (train['EMFrac'] > -2)]
test = test[(test['EMFrac'] < 2) & (test['EMFrac'] > -2)]

for key in test.keys():
    print(key)

print(len(train.keys()))

print(train.head())

from FunctionScaler import FunctionScaler
fs = FunctionScaler.FunctionScaler(FunctionScaler.TransformedFunction_Uniform(-1,1), downplay_outofbounds_lower_n_range=None, downplay_outofbounds_upper_n_range=None, downplay_outofbounds_lower_set_point=None, downplay_outofbounds_upper_set_point= None) # calling the normal function by name

test_data = test.values
train_data = train.values
fs.fit(train_data[:len(train_data)//100])

scaled_train_data = fs.transform(train_data)
scaled_test_data = fs.transform(test_data)

scaled_train_df = pd.DataFrame(scaled_train_data, columns=train.columns)
scaled_test_df = pd.DataFrame(scaled_test_data, columns=test.columns)

scaled_train_df.to_pickle('scaled_all_jets_partial_train.pkl')
scaled_test_df.to_pickle('scaled_all_jets_partial_test.pkl')

