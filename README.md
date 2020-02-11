# HEPAutoencoders
Autoencoder compression of leading jet events using PyTorch and fastai.

It strives to be easy to experiment with, parallelizable and GPU-friendly.

Builds on the work of Eric Wulff at https://github.com/erwulff/lth_thesis_project 

## Usage
Extract data from the ROOT file-format in the scripts named process_

nn_utils and utils hold various helpful functions

An example of running a 4-dimensional an already trained network is 4D/fastai_AE_3D_200_no1cycle_analysis.ipynb

## Structure
The folders named 4D/, 25D/ and 27D/ simply holds training analysis scripts for that amount of dimensions. 

All processed data will be placed in processed_data/ after extraction from the ROOT-files.
