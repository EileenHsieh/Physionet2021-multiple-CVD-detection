# Python code for the PhysioNet/Computing in Cardiology Challenge 2021.
## Introduction
The code contained in this repo is submitted to the PhysioNet/Computing in Cardiology Challenge 2021, where the goal is to detect multiple cardiovascular diseases. Please check out this paper for more details

## Installation
To build the programming environment, the docker image is provided.
```
# git clone this repo
cd physionet2021 
docker build -t physionet2021 . 
```

## Using the scripts
* Step 1: Build docker image
    ```
    git clone this repo
    cd physionet2021 
    docker build -t physionet2021 . 
    ```
* Step 2: Download dataset from NAS
    
    The datasets.zip structure is `datasets/raw/WFDB_*`, where `WFDB_*` are `WFDB_CPSC2018/`, `WFDB_CPSC2018_2/`, `WFDB_ChapmanShaoxing/`, `WFDB_Ga/`, `WFDB_Ningbo/`, `WFDB_PTB/`, `WFDB_PTBXL/` and `WFDB_StPetersburg/`. These 8 datasets are sourced from China, USA, Germany and Georgia. For more details, please check the data description on the [website](https://moody-challenge.physionet.org/2021/).

    NAS path: `/AIC-shared1/Physionet-data/datasets.zip `

    Unzip it under ./physionet2021/: 
    ```
    unzip datasets.zip
    ```

* Step 3: Setup folders from trained model, test data and test outputs  

    ```
    mkdir model test_data test_outputs 
    ```

* Step 3: Build docker container

    You can mount your local physionet2021 repo to the container, so that when there is any change in your local repo, it'll be sync in the container also.

    ```
    docker run --gpus=all --shm-size=65g --name=physionet -p 9180-9185:9180-9185 -it -v [LOCAL PATH TO REPO]:/workspace/physionet physionet2021 bash 
    ```

* Step 4: Train the models for 12 leads, 6 leads, 4 leads, 3 leads and 2 leads ECG in sequence.

    Given the datasets and the empty folder for saving the trained model, the `train_model.py` script will follow the config file to train the 12/6/4/3/2 lead models one by one and output the models to the `model` folder. The config files are in `config` folder, where there are 10 entries for different modeling pipeline. The entry 1, 5, 9 is the SERsn+M, SERsn+M, SERsn+M mentioned in the [paper](https://arxiv.org/pdf/2204.13917.pdf), respectively. 

    ```
    python train_model.py datasets model
    ```

* Step 5: Inference stage

    The inference stage makes used of the trained model that is saved in `model` and test on the given test data in `test_data` folder. The inference is doing sample by sample, besides, it will detect the number of leads of each test samples and apply the corresponding model. The output of the inference code is saved in `test_outputs` folder created before. Each of the test samples will be saved individually as a csv file, with three information inside: classes, predicted label and predicted probability.

    To test if the inference script can run properly, first, create your own test set. Try to copy some raw data to test_data folder that you created before. For example,

    ```
    cp -r  datasets/raw/WFDB_CPSC2018/A0017* test_data/
    cp -r  datasets/raw/WFDB_PTBXL/HR00206* test_data/
    ```

    Run the inference script:
    ```
    python test_model.py model test_data test_outputs
    ```

## Description for main scripts
Since the challenge organization wants to maintain the codebase, they provided some predefined scripts that we are not allowed to modify. In the following, I'll introduce some important scripts and mention which are predefined, and which are self-defined. 

* **train_model.py**: Predefined code come with the challenge. The main training code that calls functions from the `team_code.py` script to run your training code on the training data.
* **test_model.py**: Predefined code come with the challenge. The main inference code that calls your trained models to run on the test data. 
* **helper_code.py**: Predefined code come with the challenge. All the functions you need to load the data.
* **./lib/evaluation_2021**: Predefined code come with the challenge. Includes all the evaluation related information, such as labels distribution and evaluation metric codes. There are more details in this [repo](https://github.com/physionetchallenges/evaluation-2021).
    * `dx_mapping_scored.csv`: Information of labels focused in this challenge. Contains the label mapping to the SNOMEDCTCode, Abbreviation, and the distribution in 8 datasets.
    * `dx_mapping_unscored.csv`: Information of other labels that's not cared in this challenge. Contains the label mapping to the SNOMEDCTCode, Abbreviation, and the distribution in 8 datasets.
    * `evaluate_model.py`:  Evaluates the output of your classifier using the evaluation metric. We call the function defined in this scripts to compute AUROC, AUPRC and challenge matric.
    * `weights.csv`:  describes a table that defines the Challenge evaluation metric. The purpose is to allow the misclassification between classes since they might have the same outcome or the same treatment. The rows and columns of this table correspond to classes or diagnoses that are scored for the Challenge, and the entries define the value or weight give to each classifier output for each label. Some rows or columns contain multiple classes that are separated with a pipe (|), and these classes are treated equivalently by the evaluation metric.

* **team_code.py**: The code related to self-defined training function, ways to dump/load/run the trained model. **The definition of leads as well as the config files are specified in this script.**
    * `training_code`: self-defined function that loads the config file and builds up the solver, besides logs the results in mlflow server. 
    * `run_model`: self-defined function that inferences the given sample with the trained model. The given sample is raw includeing header and recording. The given model is loaded.
    * `load_model`: self-defined function that saves the trained model given the model folder and the filename info including leads, classes. imputer and classifier.
* **helper_code.py**: Predefined code come with the challenge. All the functions you need to load the data.
* **./src/engine**: Self-defined codes including loaders, models, losses and solver.py.
    * `./loaders`: The folder contains waveform loader (`raw_loader.py` and `raw_testloader` for inference stage), and feature loader (`hrv_loader`, not used).
    * `./losses`: The folder contains asymmetric loss.
    * `./models`: The folder contains all the models.
    * `solver_raw.py`: The file to build data loader, model and training.
    * `utils.py`: contains functions such as mlflow related, probability to binary, tune binary threshold, data proprocessing method used in data loader.  
* **./src/dataset_generator**: generate the handcrafted ECG feature as training set or for the rule-based models.
    * `extract_nk2_single_model.py`: generate the features for low QRS voltage, prolong QT, bundle branch block, and qwaves abnormal. 
* **./config**: contains 12/6/4/3/2 lead configs for 10 experiments.


## Useful links
Please see the [PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

* [The PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/)
* [MATLAB example code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/matlab-classifier-2021)
* [Evaluation code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/evaluation-2021) 
* [2021 Challenge Frequently Asked Questions (FAQ)](https://physionetchallenges.org/2021/faq/) 
* [Frequently Asked Questions (FAQ)](https://physionetchallenges.org/faq/) 
