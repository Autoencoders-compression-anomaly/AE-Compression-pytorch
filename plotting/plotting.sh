#!/usr/bin/bash
#Wrapper script for 27D_train.py

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

#outdir="correlationPlots" #Where all outputs are stored
#savedir="/afs/cern.ch/user/s/sarobert/autoencoders/outputs/" #Where the output is moved to after completion

#Transfer input files
#cp -r $pathToAE$modelDir .

#Train the network
python3 correlationPlots.py

#Move outputs
#cp nn_utils*/*png $eosdir
#mv models/ $outdir"/"
#mv nn_utils* $outdir"/"

