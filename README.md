# HEPAutoencoders
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Gitter](https://badges.gitter.im/HEPAutoencoders/community.svg)](https://gitter.im/HEPAutoencoders/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


ML-compression of HEP data events using deep autoencoders with the PyTorch and fastai python libraries. The scripts in the repository were used to perform compression and evaluate the performance of autoencoders on three different datasets: 
1. Internal data generated from ATLAS event generator
2. [PhenoML dataset](https://zenodo.org/record/3685861#.Xz7aAJZS-kA)
3. [Dataset from a hackathon related with the darkmachines unsupervised challenge project](https://zenodo.org/record/3961917#.Xz7ZnJZS-kB)

The repository is developed by Honey Gupta, as a part of the Google Summer of Code project, before which, it was built by Eric Wallin and Eric Wulff as a part of their bachelors' and master's project at Lund University. Technical explanations can be found in Eric Wulff's [thesis](https://lup.lub.lu.se/student-papers/search/publication/9004751). 

A  summary of the experiments performed and the results obtained as a part of [Google Summer of Code 2020 Project](https://summerofcode.withgoogle.com/projects/#5677663735250944) can be found in this [report](). 

## Jump to:
* [Setup](#setup)

* [Repository Structure](#repository-structure)

* [Process data](#process-data)

* [Training](#training)

* [Testing and Analysis](#testing-and-analysis)


## Setup:
---

### Using virtual environment (recommended for server machines, such as [LXPLUS](http://information-technology.web.cern.ch/services/lxplus-service) )

1. Fetch the latest version of the project:
    ```
    git clone https://github.com/Autoencoders-compression-anomaly/AE-Compression-pytorch.git
    cd AE-Compression-pytorch
    ```

2. #### Predefined virtual env creation and package installation (Simpler):

    ```
    chmod +x install_libs.sh
    ./install_libs.sh

    ```
    _Note that all the required packages/libraries for running the scripts in this repo have been added to the bash file. You can add others if needed._
    
    OR


2. #### Manual installation
    Create directory for your virtualenv
    ```
    mkdir venv
    cd venv
    python -m virtualenv -p python3 venv
    source bin/activate
    cd ..
    ```

    Now to install dependencies:
    ```
    pip -r requirements.txt
    ```
    
---

### Using a Docker container:

1. Pull the docker container containing useful libraries:
    ```
    docker pull atlasml/ml-base
    ```

    Run an interactive bash shell in the container, allowing the hostmachine to open jupyter-notebooks running in the container. The port 8899 can be changed if it is already taken.
    ```
    docker run -it -p 8899:8888 atlasml/ml-base
    ```

    Check the container's name and attach it:

    ```
    docker ps
    docker attach <name>
    ```

2. Install the package

    Lastly the AE-Compression-pytorch package can be installed (run from the directory that holds setup.py):
    ```
    pip install .
    ```
    Alternatively, if you want to easily edit and run the contents of the package without manually re-installing it, instead run:
    ```
    pip install -e .
    ```

    With a jupyter-notebook running inside the container, one can access it on the hostmachine from the URL localhost:8899

---

## Repository Structure

* ### HEPAutoencoders
    This folder contains utility python scripts needed by the main python scripts.

    1. `pre-processing.py`: extracts data from the ATLAS (D)xAOD file-format (ROOT files) using the functions named `prep_processing.process_*()`

        The experiments for this dataset was done with two types of data: 4-dim data and the 27-dim data. (Although the original events holds 29 values, only 27 of them are of constant size.) 

        These dataframe are converted into pandas Dataframes, which in turned may be pickled for further use. 

    2. `nn_utils.py`: holds various helpful methods for building the networks. It also contains some methods for training.

    3. `utils.py`: holds functions for normalization and event filtering, amongst others.

    4. `postprocessing.py`: holds various functions for saving data back into the ROOT fileformat.

* ### process_data
    This repository contains python scripts that can be used to create the training and testing datasets from ATLAS, PhenoML and DarkMachines datasets. 
    
    All the python script have very similar codes and functions, except for some small variations depending on the experiments for which they were used. The names of the python files should be self explanatory to define the task and the kind of dataset they create. 
    
    This repository also contains two Jupyter notebooks: 
    1. `plot_particle_distribution.ipynb`: contains the functions to plot the particle distribution for a particular process (from the PhenoML dataset). It also contains the scripts to create data distribution plots for different process (.csv) files.
    2. `process_data_as_4D.ipynb`: gives a visual intuition about different parts of the processing scripts and their functions.

* ### scale_data_all
    This folder contains a script that scales (or normalizes) the data generated by the `processing` scripts. The script uses `FunctionScaler` scaler to normalize the data. This was used in the experiments mentioned in Eric Wulff's thesis. 
    
    However, during our experiments, we shifted to standard normalization and mainly to custom normalization, which is implemented as a part of the training and testing scripts. 

    _Keeping this script for the sake of completeness._

* ### train_models
    Throughout the project, the experiments were run using a batch service at CERN called HTCondor. This folder contains the scripts that were used for submitting different training jobs during the experiments. 

    Have included these to ensure reproducability and for making Knowledge Transfer easier.

* ### examples
    This is the folder that contains the training, testing and analysis scripts for all the experiments for the abovementioned three datasets, with standard and custom normalization. 
    1. `phenoML`
        
        This folder contains backbone training scripts and testing notebooks. 
        
        a. `train_eventsAs4D*` can be used to train an autoencoder model on 4D data extracted from the event-level data present in the PhenoML dataset. 
        
        b. `test_4D(*)` can be used to test the trained models and create residual anc correlation plots.

        c. `test_4D_customNorm_differentParticles_stackedPlots.ipynb` contains the script to test a model trained with 4D custom normalized data and contains the methods to create stacked or overlapped residual or error plots for performing analysis.

        The model for this experiment was trained on jet (j and b) particles and tested on different particles such as electrons (e<sup>-</sup>), positrons (e<sup>+</sup>), muons(&mu;<sup>-</sup>), antimuons (&mu;<sup>+</sup>) and photons (&gamma;) 

        d. `4D_customNorm` contains the scripts and the models used for training and analysing the 4D data from PhenoML dataset. The model in this is the one used to create the analysis plot for the related experiment

        e. `4D_stdNorm`: similar to custom norm folder, this contains the training script, trained model and the testing jupyter notebook for experiments performed with standard normalization

        f. `half_data`: this folder has the same structure as above, just that the models in this folder were trained with half the training data used in the above experiments.

    2. `darkmachines`

        `4D_customNorm/` contains the training script to train an autoencoder model on data belonging to `chan2a` type of the DarkMachine challenge dataset, which mostly contains other particles (e<sup>-</sup>, e<sup>+</sup>, &mu;<sup>-</sup>, &mu;<sup>+</sup>, &gamma;) mixed with jets but with a lesser percentage of jets. 
        
        The analysis was done on `chan3` data, that mostly contains jets (j and b).


    3. ATLAS data

        a. `4D`: An example of training a 4D-network can be found in `examples/4D/4D_training.ipynb`. 
        
        Fastai saves trained models in the folder `models/` relative to the training script, with the .pth file extension. An example of running or testing a 4-dimensional already trained network can be found in `TLA_analysis.ipynb`

        b. `27D`: Most of the code here was take from the repository by Eric Wulff. `27D_train.py` contains the script to train the network on a 27D data.
        For an example of analysing a 27-D network, you can refer to `27D_analysis.py`.

---

## Process data

The scripts to process data can be found in the `process_data` folder. The `process.sub` and `process.sh` files can be used to submit a batch job on `HTCondor`.

A description of the scripts can be found in the previous section.

---

## Training

The training for different experiments can be found in the `examples` folder according to their dataset and nomalization type. 

Again, a description of the folder and scripts can be found in the previous section.

---

## Testing and Analysis

Each experiment's folder in the `examples` folder contains a Jupyter notebook to load the testing dataset, load the pre-trained model, run the models on the testing datasets and create residual and correlation plots. 

Instructions to understand the methods in the Notebook can be found inside the notebooks.

