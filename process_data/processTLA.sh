#!/bin/bash
python -m virtualenv -p python3 myvenv
source myvenv/bin/activate
pip3 install pandas
#pip3 install /usr/lib64/python3.6/site-packages/ROOT.py
#pip install fastai
#pip install corner
#pip install uproot
#pip install scikit-learn
python3 processTLA.py
#mv all_jets_partial_train.pkl ~/autoencoders/AE-Compression-pytorch/process_data
#mv all_jets_partial_test.pkl ~/autoencoders/AE-Compression-pytorch/process_data
