# HEPAutoencoders
Autoencoder compression of ATLAS jet events using PyTorch and fastai.

It strives to be easy to experiment with, but also parallelizable and GPU-friendly in order to aid hyperparameters scans on clusters.

Builds directly on the work of Eric Wulff at https://github.com/erwulff/lth_thesis_project 

## Quick guide
Extract data from the ROOT file-format in the scripts named process_*
This comes in two types: The 4D data and the 27D data. Although, the original events holds 29 values, only 27 of them are easy to work with.
The ROOT-data should be stored in data/
All processed data will be placed in processed_data/ after extraction from the ROOT-files.

nn_utils.py holds various heplful for networks structures and training functions.
utils holds amongst many, functions for normalization and event filtering.

An example of running a 4-dimensional already trained network is 4D/fastai_AE_3D_200_no1cycle_analysis.ipynb
For an example of analysing a 27-D network is 27D/002_analysis_example.py

The folders named 4D/, 25D/ and 27D/ simply holds training analysis scripts for that amount of dimensions. 

## Data extraction
The raw DAODs can be processed into a 4-dimensional dataset with process_ROOT_4D.ipynb, where the data is pickled into a 4D pandas Dataframe. (Eventually for equivalents for 25 and 27 dimensions).
Since pickled python objects are very version incompatible, it is recommended to process the raw ROOT DAODs instead of providing the pickled processed data. 

For ease of use, put raw data in data/ and put processed data in processed_data/

## Training
WIP, there will probably be some technical details of the normalization and training specifics here etc. 

## Analysis
fastai saves trained models in the folder models/ relative to the training script, with the .pth file extension. 

## TODO:
Major refactoring is being done. (50% done)

Analysis scripts for CPU/GPU and memory usage when evaluating the networks. (5% done)

Adding clearer examples and assuring a more stable workflow when building the project from scratch, try if there is a suitable docker container where everything will run. Add a proper list of dependencies. (0% done)

Adding more robust scripts for extraction from the raw ROOT data, i.e. actual scripts and not jupyter-notebooks, for 4, 25 and 27 dimensions. And make them as quick as possible.(25% done)
