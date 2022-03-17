#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Utils to perform data visualization
    Written by H.Turbé, March 2022.
    
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FILEPATH = os.path.dirname(os.path.realpath(__file__))




def plot_results_test(df, signal_names, diseasse_name):
    """
    Plot Heatmap of p-values using permutation test
    Input:
        df: dataframe with p-values for each channel combination
        signal_names: list of signal names
        disease_name: name of the disease
    """
    df.index = signal_names
    df.columns = signal_names
    fig = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(df, vmax=0.05)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=15)
    path_save_fig = os.path.join(FILEPATH, "..", "results", "permutation_test","figures")
    if not os.path.exists(path_save_fig):
        os.makedirs(path_save_fig)
    plt.savefig(os.path.join(path_save_fig, f"significance_matrix_{diseasse_name}.png"))
    plt.close()
