# HEPAutoencoders
Autoencoder compression of ATLAS jet events using PyTorch and fastai.

It strives to be easy to experiment with, but also parallelizable and GPU-friendly in order to aid hyperparameters scans on clusters.

Builds directly on the work of Eric Wulff at https://github.com/erwulff/lth_thesis_project 

## Quick guide
Pre-processing:. Extract data from the ROOT file-format in the scripts named process_*
This comes in two types: The 4D data and the 27D data. Although, the original events holds 29 values, only 27 of them are easy to work with.
The ROOT-data should be stored in data/ (by default)
All processed data will be placed in processed_data/ after extraction from the ROOT-files (by default of course). No normalization or other ML-related pre-processing is done in this step. 

Training: An (uncommented) example of training a 4D-network is fastai_AE_3D_200_no1cycle.ipynb and looks very much like every other training script in this project.

Analysis and plots: An example of running a 4-dimensional already trained network is 4D/fastai_AE_3D_200_no1cycle_analysis.ipynb
For an example of analysing a 27-D network is 27D/002_analysis_example.py

The folders named 4D/, 25D/ and 27D/ simply holds training analysis scripts for that amount of dimensions. 

nn_utils.py holds various heplful for networks structures and training functions.
utils holds amongst many, functions for normalization and event filtering.

## Data extraction
The raw DAODs can be processed into a 4-dimensional dataset with process_ROOT_4D.ipynb, where the data is pickled into a 4D pandas Dataframe, with process_ROOT_27D.ipynb for the 27-dimensional data.
Since pickled python objects are very version incompatible, it is recommended to process the raw ROOT DAODs instead of providing the pickled processed data. 

For ease of use, put raw data in data/ and put processed data in processed_data/

The 27-variables in question are:

|Value|
|:---|
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.pt
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.eta
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.phi
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.m
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_eta
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_m
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_phi
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.ActiveArea4vec_pt
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.AverageLArQF
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.NegativeE
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.HECQuality
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LArQuality
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.Width
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.WidthPhi
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.CentroidR
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.DetectorEta
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterCenterLambda
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterPt
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterSecondLambda
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.LeadingClusterSecondR
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.N90Constituents
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.EMFrac
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.HECFrac
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.Timing
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.OotFracClusters10
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.OotFracClusters5 

These values are nice to work with since they are not lists of variable length, which suits our networks with constant input sizes. Worth noting is that ActiveArea and N90Constituents are discrete values. 

## Training
WIP, there will probably be some technical details of the normalization and training specifics here etc. 

## Analysis
fastai saves trained models in the folder models/ relative to the training script, with the .pth file extension. 

## TODO:
Major refactoring is being done. (75% done)

Analysis scripts for CPU/GPU and memory usage when evaluating the networks. (20% done)
Using the hwcounter module seems to give results for CPU cycle usage (only dependent on architecture I suppose)

Adding clearer examples and assuring a more stable workflow when building the project from scratch, try if there is a suitable docker container where everything will run. Add a proper list of dependencies. (0% done)

Adding more robust scripts for extraction from the raw ROOT data, i.e. actual scripts and not jupyter-notebooks, for 4, 25 and 27 dimensions. And make them as quick as possible.(66% done)
