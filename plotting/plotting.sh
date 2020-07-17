#!/usr/bin/bash
#When making plots uses too much memory for lxplus

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

#Make plots
python3 migrationThreshold.py
