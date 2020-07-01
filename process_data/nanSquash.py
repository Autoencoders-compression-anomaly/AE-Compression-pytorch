import sys
BIN = '/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/'
import pandas as pd
import numpy as np
from math import isnan, isinf
from HEPAutoencoders.utils import min_filter_jets, custom_normalization

def nanSquash():
    train = pd.read_pickle('all_jets_partial_train.pkl')
    test = pd.read_pickle('all_jets_partial_test.pkl')
    train = min_filter_jets(train)
    test = min_filter_jets(test)
    train, test = custom_normalization(train, test)
    data = pd.concat([train, test])
    badVar = []
    for var in data:
        for entry in data[var]:
            if ((isnan(entry) or isinf(entry)) and (var not in badVar)):
                print (var)
                badVar.append(var)
                
nanSquash()
