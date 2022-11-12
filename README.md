# Steps for running this code
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

# Description for main scripts
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
  







# Python example code for the PhysioNet/Computing in Cardiology Challenge 2021

## What's in this repository?

We implemented a random forest classifier that uses age, sex, and the root mean square of the ECG lead signals as features. This simple example illustrates how to format your Python entry for the Challenge, and it should finish running on any of the Challenge training datasets in a minute or two on a personal computer. However, it is **not** designed to score well (or, more accurately, it is designed not to score well), so you should not use it as a baseline for your model's performance.

This code uses four main scripts, as described below, to train and test your model for the 2021 Challenge.

## How do I run these scripts?

You can run this classifier code by installing the requirements

    pip install requirements.txt

and running

    python train_model.py training_data model
    python test_model.py model test_data test_outputs

where `training_data` is a folder of training data files, `model` is a folder for saving your models, `test_data` is a folder of test data files (you can use the training data locally for debugging and cross-validation), and `test_outputs` is a folder for saving your models' outputs. The [PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/) provides training databases with data files and a description of the contents and structure of these files.

After training your model and obtaining test outputs with above commands, you can evaluate the scores of your models using the [PhysioNet/CinC Challenge 2021 evaluation code](https://github.com/physionetchallenges/evaluation-2021) by running

    python evaluate_model.py labels outputs scores.csv class_scores.csv

where `labels` is a folder containing files with one or more labels for each ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a folder containing files with outputs produced by your models for those recordings; `scores.csv` (optional) is a collection of scores for your models; and `class_scores.csv` (optional) is a collection of per-class scores for your models.

## Which scripts I can edit?

We will run the `train_model.py` and `test_model.py` scripts to run your training code and testing code, so please check these scripts and the functions that they call.
Our example code uses four main scripts to train and test your model for the 2021 Challenge:

Please edit the following script to add your training and testing code:

* `team_code.py` is a script with functions for training your model and running your trained models.

Please do **not** edit the following scripts. We will use the unedited versions of these scripts.

* `train_model.py` is a script for calling your training code on the training data.
* `test_model.py` is a script for calling your trained models on the test data.
* `helper_code.py` is a script with helper variables and functions that we used for our code. You are welcome to use them in your code.

These four scripts must remain in the root path of your repository, but you can put other scripts and other files elsewhere in your repository.

## How do I train, save, load, and run my model?

To train and save your models, please edit the `training_code` function in the `team_code.py` script. Please do not edit the input arguments or output arguments of the `training_code` function.

To load and run your trained model, please edit the `load_twelve_lead_model`, `load_six_lead_model`, `load_three_lead_model`, and `load_two_lead_model` functions as well as the `run_twelve_lead_model`, `run_six_lead_model`, `run_three_lead_model` and `run_two_lead_model` functions in the `team_code.py` script, which takes an ECG recording as an input and returns the class labels and probabilities for the ECG recording as outputs. Please do not edit the input or output arguments of the functions for loading or running your models.

## What else is in this repository?

This README has instructions for running the example code and writing and running your own code.

We also included a script, `extract_leads_wfdb.py`, for extracting reduced-lead sets from the training data. You can use this script to produce reduced-lead data that you can use with your code. You can run this script using the following commands:

    python extract_leads_wfdb.py -i twelve_lead_directory -o two_lead_directory -l II V5 
    python extract_leads_wfdb.py -i twelve_lead_directory -o three_lead_directory -l I II V2 
    python extract_leads_wfdb.py -i twelve_lead_directory -o six_lead_directory -l I II III aVL aVR aVF 

Here, the `-i` argument gives the input folder, the `-o` argument gives the output folder, and the `-l` argument gives the leads.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies that you can run reliably in other computing environments and operating systems.

To guarantee that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a subset of the training data.

If you have trouble running your code, then please try the follow steps to run the example code, which is known to work.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2021/#data-access). Put some of the training data in `training_data` and `test_data`. You can use some of the training data to check your code (and should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-classifier-2021.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  python-classifier-2021  test_data  test_outputs  training_data

        user@computer:~/example$ ls training_data/
        A0001.hea  A0001.mat  A0002.hea  A0002.mat  A0003.hea  ...

        user@computer:~/example$ cd python-classifier-2021/

        user@computer:~/example/python-classifier-2021$ docker build -t image .

        Sending build context to Docker daemon  30.21kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-classifier-2021$ docker run -it -v ~/example/model:/physionet/model -v ~/example/test_data:/physionet/test_data -v ~/example/test_outputs:/physionet/test_outputs -v ~/example/training_data:/physionet/training_data image bash

        root@[...]:/physionet# ls
            Dockerfile             model             test_data      train_model.py
            extract_leads_wfdb.py  README.md         test_model.py
            helper_code.py         requirements.txt  test_outputs
            LICENSE                team_code.py      training_data

        root@[...]:/physionet# python train_model.py training_data model

        root@[...]:/physionet# python test_model.py model test_data test_outputs

        root@[...]:/physionet# exit
        Exit

        user@computer:~/example/python-classifier-2021$ cd ..

        user@computer:~/example$ ls test_outputs/
        A0006.csv  A0007.csv  A0008.csv  A0009.csv  A0010.csv  ...

## How do I learn more?

Please see the [PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

## Useful links

* [The PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/)
* [MATLAB example code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/matlab-classifier-2021)
* [Evaluation code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/evaluation-2021) 
* [2021 Challenge Frequently Asked Questions (FAQ)](https://physionetchallenges.org/2021/faq/) 
* [Frequently Asked Questions (FAQ)](https://physionetchallenges.org/faq/) 
