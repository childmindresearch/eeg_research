#!/opt/anaconda3/envs/mne/bin/python3 -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-04-04
# email: samuel DOT louviot AT nki DOT rfmh DOT org
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

"""Cleaning pipelines for EEG data.

This module contains several methods to clean the EEG data recorded during fMRI.
The different methods consist of cleaning the gradient and BCG artifacts with
the homemade pipelines (called here CBIN-CLEANER). Once this first step is done,
the data can be further cleaned by either using ASR and/or PyPrep algorithms.
"""

import os
import shutil
from pathlib import Path

import asrpy
import bids
import mne
import numpy as np
import pyprep

from eeg_research.preprocessing.pipelines.bcg_cleaning_pipeline import clean_bcg
from eeg_research.preprocessing.pipelines.gradient_cleaning_pipeline import (
    clean_gradient,
)
from eeg_research.preprocessing.tools.utils import read_raw_eeg
from eeg_research.simulators.decorators import pipe
from eeg_research.simulators.simulate_data import simulate_eeg_data


class CleanerPipelines:
    """Class to clean the EEG data using different algorithms."""

    def __init__(self, BIDSFile: bids.layout.BIDSFile) -> None:  # noqa: D107
        self.BIDSFile = BIDSFile
        self.entities = BIDSFile.get_entities()
        self.rawdata_path = Path(BIDSFile.path)
        self.process_history: list[str] = list()
        self._make_derivatives_path()

    def _task_is(self, task_name: str) -> bool:
        return self.BIDSFile.get_entities()["task"] == task_name

    def read_raw(self: "CleanerPipelines") -> "CleanerPipelines":
        """Read the raw EEG data using MNE."""
        try:
            self.raw = read_raw_eeg(self.BIDSFile.path)
        except Exception as e:
            print(f"Error while reading the raw data: {e}")
        return self

    def _make_derivatives_path(self: "CleanerPipelines") -> "CleanerPipelines":
        """Create the path to save the cleaned files in the BIDS format.

        It is a file specific path that is generated based on the BIDSFile
        object.
        """
        path_parts = list(self.rawdata_path.parts)
        rawdata_dirname = [name for name in path_parts if "raw" in name.lower()][0]
        path_parts[path_parts.index(rawdata_dirname)] = "DERIVATIVES"
        derivatives_path_parts = path_parts[: path_parts.index("DERIVATIVES") + 1]
        self.derivatives_path = Path(*derivatives_path_parts)
        self.derivatives_path.mkdir(parents=True, exist_ok=True)
        return self

    def _make_process_path(self: "CleanerPipelines") -> "CleanerPipelines":
        """Create the path to save the cleaned files in the BIDS format.

        It is a file specific path that is generated based on the BIDSFile
        object.

        Args:
            added_folder (str, optional): The folder to be added after the
                                          derivatives one.
        """
        if len(self.process_history) > 1:
            added_folder = "_".join(self.process_history)
        else:
            added_folder = self.process_history[0]

        self.process_path = self.derivatives_path.joinpath(added_folder)
        self.process_path.mkdir(parents=True, exist_ok=True)
        return self

    def _make_subject_session_path(self: "CleanerPipelines") -> "CleanerPipelines":
        """Create the path to save the cleaned files in the BIDS format.

        It is a file specific path that is generated based on the BIDSFile
        object.

        Args:
            added_folder (str, optional): The folder to be added after the
                                          derivatives one.
        """
        if self.process_path:
            self.subject_session_path = self.process_path.joinpath(
                f"sub-{self.entities['subject']}", f"ses-{self.entities['session']}"
            )
            self.subject_session_path.mkdir(parents=True, exist_ok=True)
            return self
        else:
            raise ValueError(
                """The process path is not defined.
                Run the method _make_process_path first."""
            )

    def _make_modality_path(
        self: "CleanerPipelines", modality: str = "eeg"
    ) -> "CleanerPipelines":
        """Create the path to save the cleaned files in the BIDS format.

        It is a file specific path that is generated based on the BIDSFile
        object.

        Args:
            modality (str, optional): The modality used (eeg, mri, etc.)
        """
        if self.subject_session_path:
            self.modality_path = self.subject_session_path.joinpath(modality)
            self.modality_path.mkdir(parents=True, exist_ok=True)
            return self
        else:
            raise ValueError(
                """The process path is not defined.
                Run the method _make_subject_session_path first."""
            )

    def _copy_sidecar(self: "CleanerPipelines") -> None:
        """Copy the sidecar file to the derivative folder.

        Args:
            BIDSFile (bids.layout.models.BIDSFile): The BIDSFile object.
            where_to_copy (str | os.PathLike): The folder to copy the sidecar file.
        """
        base_filename, _ = os.path.splitext(self.BIDSFile.filename)
        source_sidecar_path = self.rawdata_path.parent
        source_json_filename = source_sidecar_path.joinpath(base_filename).with_suffix(
            ".json"
        )
        print(source_json_filename)

        destination_json_filename = self.modality_path.joinpath(
            base_filename
        ).with_suffix(".json")

        if source_json_filename.is_file():
            shutil.copyfile(source_json_filename, destination_json_filename)
        else:
            message = f"""The sidecar file {source_json_filename} does not exist."""
            print(message)

    def _save_raw(self: "CleanerPipelines") -> "CleanerPipelines":
        """Save the cleaned raw EEG data in the BIDS format."""
        base_filename, _ = os.path.splitext(self.BIDSFile.filename)
        saving_filename = base_filename + ".fif"
        destination_filename = self.modality_path.joinpath(saving_filename)
        self.raw.save(destination_filename, overwrite=True)
        return self

    @pipe
    def run_clean_gradient_and_bcg(self: "CleanerPipelines") -> "CleanerPipelines":
        """Clean the gradient and BCG artifacts from the EEG data."""
        self.raw = clean_gradient(self.raw)
        self.raw = clean_bcg(self.raw)
        self.process_history += ["GRAD", "BCG"]
        return self

    @pipe
    def run_clean_gradient(self: "CleanerPipelines") -> "CleanerPipelines":
        """Clean the gradient artifacts from the EEG data."""
        self.raw = clean_gradient(self.raw)
        self.process_history.append("GRAD")
        return self

    @pipe
    def run_clean_bcg(self: "CleanerPipelines") -> "CleanerPipelines":
        """Clean the BCG artifacts from the EEG data."""
        self.raw = clean_bcg(self.raw)
        self.process_history.append("BCG")
        return self

    @pipe
    def run_pyprep(
        self: "CleanerPipelines", montage_name: str = "easycap-M1"
    ) -> "CleanerPipelines":
        """Run the PyPrep pipeline on the EEG data.

        Args:
            montage_name (str, optional): The name of the montage to use.
                                           Defaults to "easycap-M1".

        Returns:
            CleanerPipelines: The cleaned data.
        """
        montage = mne.channels.make_standard_montage(montage_name)
        self.raw.set_montage(montage, on_missing="ignore")
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(60, self.raw.info["sfreq"] / 2, 60),
        }
        # Montages are set like twice there is a lot of redundancy, I need to fix
        # it.
        prep = pyprep.PrepPipeline(self.raw, prep_params, montage, channel_wise=True)
        # Pyprep doesn't like emg channels. I will need to submit an issue to
        # see if we can add the montage parameters in the pyprep configuration.

        prep.fit()
        self.raw = prep.raw
        self.process_history.append("PREP")
        return self

    @pipe
    def run_asr(self) -> mne.io.Raw:
        """Clean the EEG data using the ASR algorithm.

        Args:
            raw (mne.io.Raw): The raw EEG data.

        Returns:
            mne.io.Raw: The cleaned EEG data.
        """
        asr = asrpy.ASR(sfreq=self.raw.info["sfreq"])
        asr.fit(self.raw)
        self.raw = asr.transform(self.raw)
        self.process_history.append("ASR")
        return self

    @pipe
    def function_testing_decorator(self) -> None:
        """A function to test the decorator."""
        print("This is a test function.")
        self.raw = simulate_eeg_data()
        self.process_history.append("TEST_PIPE")

    def write_report(self, message: str) -> None:
        """Append a message to a txt file.

        Args:
            message (str): The message to append.
        """
        filename = self.derivatives_path.joinpath("report.txt")
        with open(filename, "a") as f:
            f.write(message)
            f.write("\n")
