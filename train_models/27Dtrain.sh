#!/usr/bin/bash
#Wrapper script for 27D_train.py

#Setup Environment
mkdir venv
cd venv
python3 -m venv .
source bin/activate
cd ..

pip install --upgrade pip
#Install necessary packages
pathToAE="/afs/cern.ch/user/s/sarobert/autoencoders/AE-Compression-pytorch/"
modelDir="examples/27D/models/"
pip3 install --no-cache-dir -r ${pathToAE}"requirements.txt"
pip3 install $pathToAE

outdir="dec7-100ep-reparam-noNorm" #Where all outputs are stored
savedir="/afs/cern.ch/user/s/sarobert/autoencoders/outputs/" #Where the output is moved to after completion
eosdir="/eos/user/s/sarobert/"${outdir}"_plots/" #Where copy of plots are stored
mkdir $outdir
mkdir $eosdir

#Transfer input files
cp -r $pathToAE$modelDir .

#Train the network
python3 27D_train_reparam.py

#Move outputs
cp nn_utils*/*png $eosdir
mv models/ $outdir"/"
mv nn_utils* $outdir"/"
mv $outdir"/" $savedir
