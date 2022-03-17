# ECG MAR Model

## Overview
This repository presents an implementation of a ECG classification model based on multivariate autoregressive (MAR) coefficients.  Interpretability of the trained models (LightGBM) on these features is then achieved using Shap.

## Data
Data used for this research have been uploaded [here](https://sandbox.zenodo.org/record/1036220#.YjMBknrMJGM). This dataset is a denoised version of the CPSC dataset presented in *Classification of 12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 20201.* The denoising consists in removing low and high frequency artifacts. More details on the method used are available upon request.

## Usage

### Requirements
To install requirements:
 
```setup
pip install -r requirements.txt
```

### Results replication
The results can be replicated using the `main_notebook.ipynb` notebook. This notebook includes the steps to download the raw data, preprocess them as well as the code used to train the models and compute the Shapley values.  

The notebook allows users to specify which disease they wish to train the model on as well as specifying correlation between given leads to generate plots with Shapley values.