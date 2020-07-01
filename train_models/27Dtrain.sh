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
req="requirements.txt" #My bash scripting isn't good enough to not make this a variable
modelDir="examples/27D/models/"
pip3 install -r $pathToAE$req
pip3 install $pathToAE

outdir="jul1-100ep-filter/" #Where all outputs are stored
savedir="/afs/cern.ch/user/s/sarobert/autoencoders/outputs/" #Where the output is moved to after completion
mkdir $outdir

#Transfer input files
cp -r $pathToAE$modelDir .

#Train the network
python3 27D_train.py

#Move outputs
mv models/ $outdir
mv nn_utils* $outdir
mv $outdir $savedir
