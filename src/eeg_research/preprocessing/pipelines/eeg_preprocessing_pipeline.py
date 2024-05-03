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
"""Module to preprocess the CST dataset."""

import datetime
import os

import asrpy as asr
import mne
import numpy as np
import pandas as pd
import pyprep as prep


class CSTpreprocessing:
    """Class to preprocess the CST dataset.

    This class is a wrapper around the pyprep and asrpy pipelines and other
    preprocessing steps. It is designed to be used with the CST dataset with its
    specificities.
    """

    def __init__(
        self, eeg_filename: str | os.PathLike, events_filename: str | os.PathLike
    ) -> None:
        """Constructor for the CSTpreprocessing object.

        Args:
            eeg_filename (str or os.PathLike): the path to the EEG file
            events_filename (str or os.PathLike): the path to the events file
        """
        self.eeg_filename = eeg_filename
        self.events_filename = events_filename
        self.raw = mne.io.read_raw(eeg_filename, preload=True)
        self.events = pd.read_csv(events_filename)

    def set_annotations_to_raw(self) -> "CSTpreprocessing":
        """Automatically set the annotations on the raw object.

        It takes care of the subtelties of the CST dataset. It handles correctly
        the timestamps and the timezone set.

        Returns:
            self
        """
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

    def set_montage(self) -> "CSTpreprocessing":
        """Wrapper around mne.channels.make_standard_montage('easycap-M1').

        The montage is hardcoded to 'easycap-M1' because it is the one used in
        the CST dataset.

        Returns:
             CSTpreprocessing object
        """
        self.montage = mne.channels.make_standard_montage("easycap-M1")
        self.raw.set_montage(self.montage)
        return self

    def run_prep(self) -> "CSTpreprocessing":
        """Run the pyprep pipeline on the raw object.

        Returns:
            CSTpreprocessing object
        """
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

    def run_asr(self) -> "CSTpreprocessing":
        """Run the asrpy pipeline on the raw object.

        Returns:
            CSTpreprocessing object
        """
        asr_obj = asr.ASR(sfreq=self.raw.info["sfreq"], cutoff=10)
        asr_obj.fit(self.raw)
        self.raw = asr_obj.transform(self.raw)
        self.raw.set_annotations(self.annotations)
        return self

    def save(self, filename: str | os.PathLike) -> "CSTpreprocessing":
        """Save the data.

        Args:
            filename: the name of the file to save
        """
        mne.export.export_raw(filename, self.raw)
        return self
