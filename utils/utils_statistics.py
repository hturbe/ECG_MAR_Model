#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Utils to perform to compute MAR coefficients and perform statistical tests
    Written by H.Turbé, March 2022.
    
"""
import copy
import multiprocessing as mp
import os
import random
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def autoregressive_model(signal):
    """
    Returns the first order autoregressive model of the signal.
    Input:
        signal: numpy array - Signal with shape (timestep, nb_channels)
    Output:
        A: numpy array - MAR Coefficients of the model with shape (nb_channels, nb_channels)
        residual: numpy array - Residual between initial and reconstructed signal using MAR coefficients
    """

    scaler = StandardScaler()
    scaler.fit(signal)
    transformed_signal = scaler.transform(signal)
    transformed_signal = signal.T

    Z = transformed_signal[:, 0:-1]
    Y = transformed_signal[:, 1:]

    A = Y @ Z.T @ np.linalg.inv(Z @ Z.T)

    residual = Y - A @ Z  # Residuals of the model
    return A, residual


def test_stationarity(signal):
    """
    Tests the stationarity of the signal.
    """

    result = adfuller(signal)
    if result[1] > 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")


def permutation_test(data_1, data_2, n_permutation=1000, name_fig=None):
    """
    Permutation test for two data sets
    Input:
        data_1: numpy array
        data_2: numpy array
        n_permutation: int
        name_fig: str
    Output:
        p_value: float

    """

    gT = np.abs(np.average(data_1) - np.average(data_2))  # — np.average(data_2)
    pV = np.append(data_1, data_2)

    # Copy pooled distribution:
    pS = copy.copy(pV)
    # Initialize permutation:
    pD = []
    # Define p (number of permutations):
    # Permutation loop:
    """
    Parallel(n_jobs=-1)delayed(self._generate_series)(name, i) for i in range(nb_simulation)
            )
    """
    for i in range(0, n_permutation):
        # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled distributions and store it in pD:
        pD.append(
            np.abs(
                np.average(pS[0 : int(len(pS) / 2)])
                - np.average(pS[int(len(pS) / 2) :])
            )
        )

    p_val = len(np.where(pD >= gT)[0]) / n_permutation
    return p_val


def permutation_test_matrix(np_coeff1, np_coeff2, n_permutation=1000):
    """
    Permutation test for two matrices of coefficients of the autoregressive model.
    Input:
        np_coeff1: numpy array
        np_coeff2: numpy array
        n_permutation: int
    Output:
        p_value: float
    """
    assert (
        np_coeff1.shape[1:] == np_coeff2.shape[1:]
    ), "The two coefficients matrices must have the same nb of coefficients per sample"
    p_val = np.empty(np_coeff1.shape[1:])
    for i in tqdm(
        range(np_coeff1.shape[1]),
        desc="Computing significance for each index of the matrix",
    ):
        for j in range(np_coeff1.shape[2]):
            p_val[i, j] = permutation_test(
                np_coeff1[:, i, j], np_coeff2[:, i, j], n_permutation
            )

    return p_val


def matrix_corr_coeff(path_array, path_results):
    """
    Save and returns the matrix of coefficients of the autoregressive model.
    Input:
        path_array: str - path to the array of signals
        path_results: str - path to the results folder

    """
    name_array = os.path.split(path_array)[-1]
    pat = r"(?<=_).+?(?=.npy)"
    name_save = re.search(pat, name_array).group(0)
    np_signal = np.load(path_array)
    np_coeff = np.empty([np_signal.shape[0], 12, 12])

    for idx in tqdm(
        range(np_signal.shape[0]),
        desc=f"Computing MAR coefficients matrix for {name_array} array",
    ):
        try:
            np_coeff[idx, :, :], _ = autoregressive_model(np_signal[idx, :, :])
        except:
            np_coeff[idx, :, :] = np.nan
    np_coeff_normal = np_coeff[~np.isnan(np_coeff).any(axis=1).any(axis=1), :, :]
    path_save = os.path.join(path_results, "coeff_matrices")
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    np.save(os.path.join(path_save, f"coeff_{name_save}.npy"), np_coeff_normal)

    return True
