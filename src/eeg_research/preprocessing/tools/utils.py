#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
#         Dr. Alp Erkent, MD, MA
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-02-27
# email: samuel DOT louviot AT nki DOT rfmh DOT org
#        alp DOT erkent AT childmind DOT org
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

"""GENERAL DOCUMENTATION HERE."""

import os

import mne
import numpy as np


def read_raw_eeg(filename: str, preload: bool = False) -> mne.io.Raw:
    """Read raw EEG data from a file.

    Wrapper function around mne.io.read_raw_* functions
    to chose the right reading method based on the file extension.

    Format  of the fileallowed are:
    - egi (.mff, .RAW)
    - bdf (.bdf)
    - edf (.edf)
    - fif (.fif)
    - eeglab (.set)
    - brainvision (.eeg)

    Args:
        filename (str): path to the file
        preload (bool, optional): if True, the data will be preloaded into memory.
            Defaults to False.

    Raises:
        FileNotFoundError: if the specified file does not exist.

    Returns:
        raw (mne.io.Raw): MNE raw object
    """
    if os.path.exists(filename):
        extension = os.path.splitext(filename)[1]
        if extension == ".mff" or extension == ".RAW":
            method = "read_raw_egi"
        elif extension == ".bdf":
            method = "read_raw_bdf"
        elif extension == ".edf":
            method = "read_raw_edf"
        elif extension == ".fif":
            method = "read_raw_fif"
        elif extension == ".set":
            method = "read_raw_eeglab"
        elif extension == ".vhdr":
            method = "read_raw_brainvision"

        reader = getattr(mne.io, method)
        try:
            raw = reader(filename, preload=preload)
            return raw

        except mne.io.ReadingFileError:
            print(
                f"File {filename} is corrupted or "
                f"extension {extension} is not recognized"
            )

    else:
        raise FileNotFoundError(f"File {filename} does not exist")


def save_clean_eeg(raw: mne.io.Raw, file: str, scripts: list[str]) -> None:
    """Save the cleaned raw EEG data in the BIDS format.

    Args:
        raw (mne.io.Raw): MNE raw object containing the cleaned EEG data.
        file (str): path to the original file.
        scripts (list[str]): list of scripts used for cleaning the data.
    """
    base_filepath, _ = os.path.splitext(file)
    destination_path = (
        base_filepath.split("sub-", 1)[0]
        + "derivatives/sub-"
        + base_filepath.split("sub-", 1)[1]
    )
    scripts_str = "_".join(scripts) + "_clean_eeg"
    destination_path = destination_path.replace(
        "/eeg/sub-", "/" + scripts_str + "/sub-"
    )
    saving_filename = destination_path[:-3] + scripts_str + ".fif"
    os.makedirs(os.path.dirname(saving_filename), exist_ok=True)
    raw.save(saving_filename, overwrite=True)


def find_real_channel_name(raw: mne.io.Raw, name: str = "ecg") -> list:
    """Find the name as it is in the raw object.

    Channel names vary across different EEG systems and manufacturers. It varies
    in terms of capitalization, spacing, and special characters. This function
    finds the real name of the channel in the raw object.

    Args:
        raw (mne.io.Raw): The mne Raw object
        name (str): The name of the channel to find in lower case.

    Returns:
        str: The real name of the channel in the raw object.
    """
    channel_found = list()
    for ch_name in raw.info["ch_names"]:
        if name.lower() in ch_name.lower():
            channel_found.append(ch_name)
    return channel_found


def map_channel_type(raw: mne.io.Raw) -> dict:
    """Find and map into MNE type the ECG and EOG channels.

    Args:
        raw (mne.io.Raw): MNE raw object

    Returns:
        dict: dictionary of channel type to map into `raw.set_channel_types` method
    """
    channels_mapping = dict()
    for ch_type in ["ecg", "eog"]:
        ch_name_in_raw = find_real_channel_name(raw, ch_type)
        if ch_name_in_raw:
            if len(ch_name_in_raw) == 1:
                channels_mapping.update({ch_name_in_raw[0]: ch_type})
            elif len(ch_name_in_raw) > 1:
                for name in ch_name_in_raw:
                    channels_mapping.update({name: ch_type})
        else:
            print(f"No {ch_type.upper()} channel found.")
            if ch_type == "eog":
                print("Fp1 and Fp2 will be used for EOG signal detection")

    return channels_mapping


def set_channel_types(raw: mne.io.Raw, channel_map: dict) -> mne.io.Raw:
    """Set the channel types of the raw object.

    Args:
        raw (mne.io.Raw): MNE raw object
        channel_map (dict): dictionary of channel type to map into
        `raw.set_channel_types` method

    Returns:
        mne.io.Raw: MNE raw object
    """
    raw.set_channel_types(channel_map)
    return raw


def extract_gradient_trigger_name(
    raw: mne.io.Raw, desired_trigger_name: str = "R128"
) -> str:
    """Extract the name of the trigger for gradient artifact removal.

    Name of the gradient trigger can change across different paradigm,
    acquisition etc.

    Args:
        raw (mne.io.Raw): The raw object containing the EEG data.
        desired_trigger_name (str, optional): The theoretical name of the
                                            trigger or a substring.
                                            Defaults to "R128".

    Returns:
        str: The gradient trigger name as it is in the raw object

    Raises:
        Exception: No gradient trigger found.
    """
    annotations_names = np.unique(raw.annotations.description)
    for annotation_name in annotations_names:
        if desired_trigger_name.lower() in annotation_name.lower():
            return annotation_name

    raise Exception("No gradient trigger found. Check the desired trigger name.")
