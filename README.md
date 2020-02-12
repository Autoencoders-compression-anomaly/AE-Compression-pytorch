# HEPAutoencoders
Autoencoder compression of leading jet events using PyTorch and fastai.

It strives to be easy to experiment with, parallelizable and GPU-friendly.

Builds on the work of Eric Wulff at https://github.com/erwulff/lth_thesis_project 

## Usage
Extract data from the ROOT file-format in the scripts named process_.
The data should be stored in data/

nn_utils.py holds various heplful for networks structures and training functions.
utils holds amongst many, functions for normalization and event filtering.

An example of running a 4-dimensional already trained network is 4D/fastai_AE_3D_200_no1cycle_analysis.ipynb
For an example of analysing a 27-D network is 27D/002_analysis_example.py

## Structure
The folders named 4D/, 25D/ and 27D/ simply holds training analysis scripts for that amount of dimensions. 

All processed data will be placed in processed_data/ after extraction from the ROOT-files.
