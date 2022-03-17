#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Main to perform data processing steps
    Written by H.Turbé, March 2022.
    
"""
import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

import utils.utils_data as utils_data
import utils.utils_statistics as utils_statistics
import utils.utils_visualisation as utils_visualisation

FILEPATH = os.path.dirname(os.path.realpath(__file__))

dict_snomed = {
    426783006: "normal",  # sinus rhythm
    59118001: "rbbb",  # right bundle branch block
    164909002: "lbbb",  # left bundle branch block
}

path_data = os.path.join(FILEPATH, "data")
path_results = os.path.join(FILEPATH, "results")
path_petastorm = os.path.join(
    FILEPATH, "data", "PhysioNetChallenge2020_Training_CPSC", "dataParquet_filtered"
)

def extract_signal_disease():
    """
    Extract signal from dataset formatted in Petastorm for the disease of interest
    and save them as numpy arrays in data/formatted_data/

    """

    for snomed_code in dict_snomed.keys():
        disease_name = dict_snomed[snomed_code]
        print(f"Extracting signal for {disease_name} samples")

        path_save = os.path.join(path_data, "formatted_data")

        if not os.path.exists(path_save):
            os.makedirs(path_save)

        utils_data.extract_signal_disease(
            path_petastorm, path_save, snomed_code, disease_name
        )


def extract_correlation_matrix():
    """
    Extract MAR coeff Matrix. The matrices are saved in data/coeff_matrices/
    """
  
    for path_array in glob(os.path.join(path_data, "formatted_data", "*.npy")):
        utils_statistics.matrix_corr_coeff(path_array, path_data)


def compute_permutation_test(disease_name):
    """
    Test for significance between two groups using permutation test. CSV files with
    p values are saved in results/permutation_test/
    """
    path_coeff = os.path.join(path_data,"coeff_matrices")
    coeff_normal = np.load(os.path.join(path_coeff, "coeff_normal.npy"))
    coeff_disease = np.load(os.path.join(path_coeff, f"coeff_{disease_name}.npy"))
  
    p = utils_statistics.permutation_test_matrix(
        coeff_normal, coeff_disease, n_permutation=1500
    )
    path_save = os.path.join(path_results, "permutation_test")
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    pd.DataFrame(p).to_csv(os.path.join(path_save, f"test_normal_{disease_name}.csv"))


def plot_matrix_test(disease_name, signal_names):
    """
    Plot the results of the permutation test. The figures are saved in results/figures/
    """
    path_df = os.path.join(path_results, "permutation_test")
    df = pd.read_csv(os.path.join(path_df, f"test_normal_{disease_name}.csv"), index_col=0)
    utils_visualisation.plot_results_test(
        df, signal_names, disease_name
    )

def parse_args(args):
    parser = argparse.ArgumentParser(description="Process files.")
    parser.add_argument(
        "--operations",
        help="Specified required operations. Should be included in [extract_signal, compute_permutation]",
        default=["extract_signal", "compute_permutation"],
        nargs="+",
    ) 

 
    return parser.parse_args(args)

def main(args= None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    requested_operations = args.operations

    if "extract_signal" in requested_operations:
        print("Extracting signal and computing MAR coefficients")
        extract_signal_disease()
        extract_correlation_matrix()

    if "compute_permutation" in requested_operations:
        print("Performing permutation test")
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        signal_names = utils_data.extract_signal_names(path_petastorm)
        for disease_name in dict_snomed.values():
            if disease_name == "normal":
                continue
            else:
                compute_permutation_test(disease_name)
                plot_matrix_test(disease_name, signal_names)

if __name__ == "__main__":
    main()
    
   
