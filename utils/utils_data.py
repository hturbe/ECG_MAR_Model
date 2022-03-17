#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Utils to extract and organise the data
    Written by H.Turbé, March 2022.
    
"""

import os
import sys
from glob import glob

import numpy as np
from petastorm import make_reader
from petastorm.predicates import in_lambda

python_interpeter = sys.executable
os.environ["PYSPARK_PYTHON"] = python_interpeter
os.environ["PYSPARK_DRIVER_PYTHON"] = python_interpeter

FILEPATH = os.path.dirname(os.path.realpath(__file__))


def extract_signal_names(path_data):
    """
    Extracts the signal names (leads names) from the schema.
    Input:
        path_data: str - path to the petastorm data
    Output:
        list_signal_names: list
    """

    path_petastorm = f"file://{path_data}"
    with make_reader(path_petastorm, schema_fields=["leads_name"]) as reader:
        sample = reader.next()
        signal_names = sample[0]

    signal_names = [name.decode("utf-8") for name in signal_names]
    return signal_names


def extract_signal_disease(path_data, path_save, snomed_code, disease_name):
    """
    Extract the signal for a specific disease given its snomed code
    Input:
        path_data: str - path to the petastorm data
        path_save: str - path to the directory where the data will be saved
        snomed_code: int - snomed code of the disease
        disease_name: str - name of the disease

    """

    predicate_expr = in_lambda(
        ["Dx"], lambda target: any(np.isin(target.astype("int"), int(snomed_code)))
    )
    path_petastorm = f"file://{path_data}"

    sample_extracted = np.empty([1, 5000, 12])
    with make_reader(
        path_petastorm, schema_fields=["signal", "Dx"], predicate=predicate_expr
    ) as reader:
        for idx, sample in enumerate(reader):
            if sample.signal.shape[1] == 12:
                if sample.signal.shape[0] >= 5000:
                    signal = sample.signal[np.newaxis, -5000:, :]
                    sample_extracted = np.concatenate(
                        [sample_extracted, signal], axis=0
                    )

    sample_extracted = sample_extracted[1:, :, :]
    np.save(os.path.join(path_save, f"signal_{disease_name}.npy"), sample_extracted)

    return True
