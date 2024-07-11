#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-04-01
# email: samuel DOT louviot AT nki DOT rfmh DOT org
#        sam DOT louviot AT gmail.com
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================================
"""Module to preprocess the EEG dataset.

It uses the argparse default CLI while waiting for our specific CLI module to 
be documented
"""

import argparse
import datetime
import functools
import os
from typing import Callable, ParamSpec, TypeVar

import asrpy as asr
import mne
import numpy as np
import pandas as pd
import pyprep as prep

from eeg_research.preprocessing.tools import utils

parser = argparse.ArgumentParser(
                    prog='eeg_preprocessing_pipeline',
                    description="""
    This code uses different preprocessing methods/techniques to prepare EEG
    data. 
                    """,
                    epilog="""
    The preprocessing methods has to be sepcified by by calling --methods
    when calling the script. It can be several methods, names have to be
    separated by a comma. Beware the order matters
    """)

parser.add_argument('reading_filename')
parser.add_argument('saving_filename')
parser.add_argument('--methods',
                    nargs='*')
kwargs_namespace = parser.parse_args()

ParamType = ParamSpec('ParamType')
ReturnType = TypeVar('ReturnType')

def trackcalls(func: Callable[ParamType, ReturnType]
               ) -> Callable[ParamType, ReturnType]:
    """Decorator to track if a method have been called.

    Args:
        func (Callable[ParamType, ReturnType]): The function to decorate

    Returns:
        Callable[ParamType, ReturnType]: The function decorated
    """
    @functools.wraps(func)
    def wrapper(*args: ParamType.args, 
                **kwargs: ParamType.kwargs) -> ReturnType:
        setattr(wrapper, "has_been_called", True)
        return func(*args, **kwargs)
    setattr(wrapper, "has_been_called", False)
    return wrapper

class MissingStepError(Exception):
    """Custom exception telling a step is missing when calling a method.

    Args:
        Exception (_type_): _description_
    """
    def __init__(self, message: str) -> None:  # noqa: D107
        self.message = message
    def __str__(self) -> str:  # noqa: D105
        return self.message

class EEGpreprocessing:
    """Class that wrap several preprocessing techniques.

    This class is a wrapper around the pyprep and asrpy pipelines and other
    preprocessing steps.
    """

    def __init__(
        self, 
        eeg_filename: str | os.PathLike 
    ) -> None:
        """Constructor for the EEGpreprocessing object.

        Args:
            eeg_filename (str or os.PathLike): the path to the EEG file
        """
        self.eeg_filename = eeg_filename
        self.raw = mne.io.read_raw(eeg_filename, preload=True)
        channels_map = utils.map_channel_type(self.raw)
        self.raw = utils.set_channel_types(self.raw, channels_map)

    def set_annotations_to_raw(
        self,
        events_filename: str | os.PathLike
        ) -> "EEGpreprocessing":
        """Automatically set the annotations on the raw object.

        It takes care of the subtelties of the CST dataset. It handles correctly
        the timestamps and the timezone set.

        Args:
            events_filename (str of os.PathLike): the path to the events file 
                                                  saved in a tsv or csv format

        Returns:
            self
        """
        self.events = pd.read_csv(events_filename)
        events_renamed = self.events.copy()
        events_renamed.loc[
            events_renamed["StimMarkers_alpha"].str.contains("Crash"),
            "StimMarkers_alpha",
        ] = "Crash"

        timestamp = [
            datetime.datetime.fromtimestamp(t) for t in events_renamed["timestamps"]
        ]

        events_renamed["timestamps"] = [
            t.replace(tzinfo=datetime.timezone.utc) for t in timestamp
        ]

        description = events_renamed["StimMarkers_alpha"].values

        onsets = [
            t.seconds for t in events_renamed["timestamps"] - self.raw.info["meas_date"]
        ]

        self.annotations = self.raw.annotations.append(
            onset=onsets, duration=np.zeros((len(onsets))), description=description
        )

        self.raw.set_annotations(self.annotations)
        return self

    @trackcalls
    def set_montage(self,
                    montage: str = "easycap-M1") -> "EEGpreprocessing":
        """Wrapper around mne.channels.make_standard_montage('easycap-M1').

        The montage is hardcoded to 'easycap-M1' because it is the one used in
        the CST dataset.
        
        Args:
            montage (str): The montage name. Has to be one among the known
                           standard montage in MNE. It is possible to get all
                           possible values by running: 
                           `mne.channels.get_builtin_montages()`

        Returns:
             EEGpreprocessing object
        """
        self.montage = mne.channels.make_standard_montage(montage)
        self.raw.set_montage(self.montage)
        return self

    def run_prep(self) -> "EEGpreprocessing":
        """Run the pyprep pipeline on the raw object.

        Returns:
            EEGpreprocessing object
        """
        if self.set_montage.has_been_called:
            prep_params = {
                "ref_chs": "eeg",
                "reref_chs": "eeg",
                "line_freqs": np.arange(60, self.raw.info["sfreq"] / 2, 60),
            }
            prep_obj = prep.PrepPipeline(
                self.raw, montage=self.montage, prep_params=prep_params
            )
            prep_obj.fit()
            self.raw = prep_obj.raw_eeg
            return self
        
        else:
            raise MissingStepError(
                "You must set a montage before. Please run `set_montage`"
                )

    def run_asr(self) -> "EEGpreprocessing":
        """Run the asrpy pipeline on the raw object.

        Returns:
            EEGpreprocessing object
        """
        if self.set_montage.has_been_called:
            asr_obj = asr.ASR(sfreq=self.raw.info["sfreq"], cutoff=10)
            asr_obj.fit(self.raw)
            self.raw = asr_obj.transform(self.raw)
            return self
        else:
            raise MissingStepError(
                "You must set a montage before. Please run `set_montage`"
            )

    def save(self, filename: str | os.PathLike) -> "EEGpreprocessing":
        """Save the data.

        Args:
            filename: the name of the file to save
        """
        mne.export.export_raw(filename, self.raw)
        return self

def main(reading_filename: str | os.PathLike, 
         saving_filename: str | os.PathLike,
         methods: list[str]) -> None:
    """Wrapper to access to the preprocess from CLI.
    
    Args:
        reading_filename (str | PathLike): The EEG filename (full path)
        saving_filename (str | PathLike): Where to save the preprocessed data
        methods (list of str): The methods to use in the pipeline
    
    Return:
        None
    """
    preprocess = EEGpreprocessing(reading_filename)
    existing_method = [cl_method for cl_method in dir(preprocess)
                    if not cl_method.startswith('__')]
    methods.insert(0,'set_montage')
    for method in methods:
        if method in existing_method:
            getattr(preprocess, method)()
        else:
            raise AttributeError(f"""`
Wrong input method name. Should be one of the following:
{"\n".join(existing_method)}
""")
    preprocess.save(saving_filename)

if __name__ == "__main__":
    main(**kwargs_namespace.__dict__)