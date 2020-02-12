# HEPAutoencoders
Autoencoder compression of ATLAS jet events (seen as single-jets) using PyTorch and fastai.

It strives to be easy to experiment with, but also parallelizable and GPU-friendly in order to aid hyperparameters scans on clusters.

Builds directly on the work of Eric Wulff at https://github.com/erwulff/lth_thesis_project 

## Usage
Extract data from the ROOT file-format in the scripts named process_*
This comes in two types: The 4D data and the 27D data. Although, the original events holds 29 values, only 27 of them are easy to work with.
The ROOT-data should be stored in data/
All processed data will be placed in processed_data/ after extraction from the ROOT-files.

nn_utils.py holds various heplful for networks structures and training functions.
utils holds amongst many, functions for normalization and event filtering.

An example of running a 4-dimensional already trained network is 4D/fastai_AE_3D_200_no1cycle_analysis.ipynb
For an example of analysing a 27-D network is 27D/002_analysis_example.py

The folders named 4D/, 25D/ and 27D/ simply holds training analysis scripts for that amount of dimensions. 

TODO:
Major refactoring is being being done. 

Analysis scripts for CPU/GPU and memory usage when evaluating the networks. 

Adding clearer examples and assuring a more stable workflow when building the project from scratch.

Adding more robust scripts for extraction from the raw ROOT data, i.e. actual scripts and not jupyter-notebooks.
