# bacteria-ID

This repo contains demonstrations of using convolutional neural networks (CNNs) to identify the Raman spectra of pathogenic bacteria. This repository is adapted from the codebase used to produce the results in the paper "Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning."

The best place to start is with the Jupyter notebooks which are documented and commented, and should run out-of-the-box. We have provided pre-trained models that have been trained on the reference datasets. 

## Requirements

The code in this repo has been tested with the following software versions:
- Python 3.7.0
- PyTorch 0.4.1
- Scikit-Learn 0.20.0
- Numpy 1.15.1
- Jupyter 5.0.0
- Seaborn 0.9.0
- Matplotlib 3.0.0

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package. Once Anaconda is installed, it should take no more than 10 minutes to install all of the requirements on a standard laptop or desktop computer.

## Data

The data for the experiments can be downloaded [here](https://www.dropbox.com/sh/gmgduvzyl5tken6/AABtSWXWPjoUBkKyC2e7Ag6Da?dl=0) and should be saved in the `data` subdirectory, which should look like:

```
data/X_finetune.npy
data/y_finetune.npy
data/X_test.npy
data/y_test.npy
data/X_2018clinical.npy
data/y_2018clinical.npy
data/X_2019clinical.npy
data/y_2019clinical.npy
```

## Files

This repo should contain the following files:
- 1_reference_finetuning.ipynb - demonstrates fine-tuning a pre-trained CNN on the 30-isolate classification task
- 2_prediction.ipynb - demonstrates making predictions with a fine-tuned CNN
- 3_clinical_finetuning.ipynb - demonstrates fine-tuning a pre-trained CNN using clinical data and making predictions for individual patients
- config.py - contains information about the provided dataset
- datasets.py - contains code for setting up datasets and dataloaders for spectral data
- resnet.py - contains ResNet CNN model class
- training.py - contains code for training CNN and making predictions
- reference_model.ckpt - saved parameters for pre-trained CNN to be used for notebooks 1 and 2
- clinical_model.ckpt - saved parameters for pre-trained CNN to be used for demo 3

## Running the notebooks

Experiment times reported in the notebooks were achieved on a 2018 Macbook Pro. If you find any bugs or have questions, please contact `csho@stanford.edu`
