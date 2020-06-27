#!/bin/bash
python -m virtualenv -p python3 myvenv
source myvenv/bin/activate
pip install pandas
pip install fastai
pip install corner
pip install uproot
pip install scikit-learn
python process_aod_all.py
mv all_jets_partial_train.pkl ~/autoencoders/AE-Compression-pytorch/process_data
mv all_jets_partial_test.pkl ~/autoencoders/AE-Compression-pytorch/process_data
