# HEPAutoencoders
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Gitter](https://badges.gitter.im/HEPAutoencoders/community.svg)](https://gitter.im/HEPAutoencoders/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


ML-compression of ATLAS trigger jet events using autoencoders, with the PyTorch and fastai python libraries.

It strives to be easy to experiment with, but also parallelizable and GPU-friendly in order to aid hyperparameters scans on clusters.

This repository is developed by Erik Wallin, as a bachelor project at Lund University. Builds directly on the master thesis [project](https://github.com/erwulff/lth_thesis_project) of Eric Wulff. Technical explanations can be found in his [thesis](https://lup.lub.lu.se/student-papers/search/publication/9004751). Erik Wallin's thesis holds further details (link will be here soon).

[Setup](#setup)

[Quick guide](#quick-guide)

[Data extraction](#data-extraction)

[Training](#training)

[Analysis](#analysis)

[Saving back to ROOT](#saving-back-to-root)

[TODO and ideas](#todo-and-ideas)

## Setup:
#### Running the container:
Pull the docker container containing useful libraries:
`docker pull atlasml/ml-base`

Run an interactive bash shell in the container, allowing the hostmachine to open jupyter-notebooks running in the container. The port 8899 can be changed if it is already taken.
`docker run -it -p 8899:8888 atlasml/ml-base`

#### To install the project:
Create directories for the repo (and alternatively one for your virtualenv)
```
mkdir HEPAutoencoders
mkdir venv
```
(To enter the virtualenv:)
```
cd venv
python3 -m venv .
source bin/activate
cd ..
```

Now, to fetch the latest version of the project:
```
cd HEPAutoencoders
git init
git pull https://github.com/Skelpdar/HEPAutoencoders
```
Install dependencies:
```
pip3 -r requirements.txt
```
Lastly the HEPAutoencoders package can be installed (run from the directory that holds setup.py):
```
pip3 install .
```
Alternatively, if you want to easily edit and run the contents of the package without manually re-installing it, instead run:
```
pip3 install -e .
```

With a jupyter-notebook running inside the container, one can access it on the hostmachine from the URL localhost:8899

## Quick guide
**Pre-processing:** Extract data from the ROOT DxAOD file-format in the scripts named `process_*`

The data comes in two types: 4-dim data and the 27-dim data. (Although the original events holds 29 values, only 27 of them are of constant size.)

The ROOT-data should be stored in `data/` (by default)

All processed data will be placed in `processed_data/` after extraction (by default). No normalization or other ML-related pre-processing is done in this step. 

**Training:** An (uncommented) example of training a 4D-network is `examples/4D/4D_training.ipynb` and looks very much like every other training script in this project. If the data you have looks any different it will need to be retrained. See Wulff's [report](https://lup.lub.lu.se/student-papers/search/publication/9004751) for previous searches of optimal network sizes. 

**Analysis and plots:** An example of running a 4-dimensional already trained network is `examples/4D/TLA_analysis.ipynb`
For an example of analysing a 27-D network is `examples/27D/27D_analysis.py`.

**Code structure:** 
`nn_utils.py` holds various heplful for networks structures and training functions.

`utils.py` holds functions for normalization and event filtering, amongst others.

## Data extraction
The raw DxAODs can be processed into a 4-dimensional dataset with `process_root()`, which returns pandas dataframes. These can be pickled for ease of use. Since pickled python objects are very version incompatible, it is recommended to process the raw ROOT xAODs instead of providing the pickled processed data. 

## Training
ML details of the training process is in Wulff's [thesis](https://lup.lub.lu.se/student-papers/search/publication/9004751). A well-functioning example is examples/4D/4D_training.ipynb

## Analysis
fastai saves trained models in the folder `models/` relative to the training script, with the .pth file extension. 

In `27D/27D_analysis.py` there is analysis of a network with a 18D latent space (i.e. a 27/18 compression ratio), with histogram comparisons of the different values and residual plots. Special attention might be given to these residuals as they tell a lot about the performance of the network.

For a more detailed plots of residuals and correlations between them, see the last part of `4D/fastai_AE_3D_200_no1cycle_analysis.ipynb`  

## Saving back to ROOT
To save a 27-dim multi-dimensional array of decoded data back into a ROOT TTree for analysis once again, the script `ndarray_to_ROOT.py` is available. (Soon for other dimensions as well) You'll have to run Athena yourself to turn this into a proper xAOD.

## TODO and ideas:
Analysis scripts for CPU/GPU and memory usage when evaluating the networks.

Adding more robust scripts for extraction from the raw ROOT data, i.e. actual scripts and not jupyter-notebooks, for 4, 25 and 27 dimensions. (And optimize them.)

Chain networks with other compression techniques. 

Train on whole events and not only on individual jets. 


