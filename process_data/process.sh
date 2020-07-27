#!/bin/bash
#Setup Environment
mkdir venv
cd venv
python3 -m venv .
source bin/activate
cd ..

#Install necessary packages
pathToAE="/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/"
modelDir="examples/27D/models/"
pip3 install -r ${pathToAE}"requirements.txt"
pip3 install $pathToAE 

python makePreds.py
#mv all_jets_partial_train.pkl ~/autoencoders/AE-Compression-pytorch/process_data
#mv all_jets_partial_test.pkl ~/autoencoders/AE-Compression-pytorch/process_data
