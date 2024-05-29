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

# standard-library imports
import os

# third-party imports (and comments indicating how to install them)
# python -m conda install -c conda-forge mne       or    python -m pip install mne
import mne

# python -m conda install -c conda-forge numpy     or    python -m pip install numpy
import numpy as np

# python -m conda install -c conda-forge scipy     or    python -m pip install scipy
import scipy


class NoSubjectFoundError(Exception):  # noqa: D101
    pass


class NoSessionFoundError(Exception):  # noqa: D101
    pass


class NoDataTypeError(Exception):  # noqa: D101
    pass


class ReadingFileError(Exception):  # noqa: D101
    pass


def get_numbers(
    kwargs: dict, key: str, path: str | os.PathLike, prefix: str
) -> list[str]:
    """Retrieve existing numbers and return a list of desired numbers.

    Args:
        kwargs (dict): Dictionary of keyword arguments.
        key (str): Key to retrieve desired numbers from kwargs.
        path (str or PathLike): Path to data directory.
        prefix (str): Prefix for desired numbers.

    Raises:
        Exception: An error occurred while retrieving existing numbers.
        Exception: An error occurred while retrieving desired numbers.

    Returns:
        list[str]: List of desired numbers.
    """
    try:
        existing_numbers = set(numerical_explorer(path, prefix))
    except Exception as e:
        raise Exception(
            "An error occurred while retrieving existing " f"{key} numbers. {e}"
        )
    try:
        desired_numbers = set(
            input_interpreter(
                str(kwargs.get(key)), prefix, max_value=max(existing_numbers)
            )
        )
    except Exception as e:
        raise Exception(
            "An error occurred while retrieving desired " f"{key} numbers. {e}"
        )

    numbers = list(existing_numbers.intersection(desired_numbers))
    return [f"{number:02d}" for number in numbers]

def extract_gradient_trigger_name(
    raw: mne.io.Raw,
    desired_trigger_name: str = "R128"
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

def find_real_channel_name(raw: mne.io.Raw, name: str = "ecg") -> str:
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
    for ch_name in raw.info['ch_names']:
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
                channels_mapping.update({ch_name_in_raw[0]:ch_type})
            elif len(ch_name_in_raw) > 1:
                for name in ch_name_in_raw:
                    channels_mapping.update({name:ch_type})
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
    #for ch_names, ch_type in channel_map.items():
    raw.set_channel_types(channel_map)
    return raw


def input_interpreter(
    input_string: str, input_param: str, max_value: int = 1000
) -> list[int]:
    """Interpret input string as a list of integers.

    The input string can contain:
        - a list of integers separated by commas (e.g. "1,3,5,7")
        - a range of integers separated by a hyphen (e.g. "1-5") which will be
          interpreted as "1,2,3,4,5"
        - a range of integers separated by a hyphen with an asterisk
          (e.g. "1-*") which will be interpreted as "1 to the maximum value"
          which can be specified by the max_value argument (max number of
          subjects for example).

    The input can be a combination of the above (e.g. "1,3-5,7-*") which will
    be interpreted as "1,3,4,5,7 to the maximum value"


    Args:
        input_string (str): input string
        input_param (str): input parameter name
        max_value (int, optional): maximum value for the asterisk. Defaults to 1000.

    Returns:
        list[int]: list of integers
    """
    elements = input_string.split(",")
    desired_subject_numbers: list[int] = []
    for element in elements:
        if "-" in element:
            start, stop = element.split("-")
            start = start.replace("*", "0")
            stop = stop.replace("*", str(max_value))
            start = start.strip()
            stop = stop.strip()
            if start.isnumeric() and stop.isnumeric():
                desired_subject_numbers.extend(range(int(start), int(stop) + 1))
            else:
                raise ValueError(
                    f"Please make sure that '{input_param}'='{input_string}' is "
                    "correctly formatted. See help for more information."
                )
                break
        else:
            if element.strip().isnumeric():
                desired_subject_numbers.append(int(element))
            elif element.strip() == "all":
                desired_subject_numbers.extend(range(1, max_value + 1))
            else:
                raise ValueError(
                    f"Please make sure that '{input_param}'='{input_string}' is "
                    "correctly formatted. See help for more information."
                )
                break
    return desired_subject_numbers


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

        except mne.io.reading.ReadingFileError:
            print(
                f"File {filename} is corrupted or "
                f"extension {extension} is not recognized"
            )

    else:
        raise FileNotFoundError(f"File {filename} does not exist")


def numerical_explorer(directory: str | os.PathLike, prefix: str) -> list[int]:
    """Give the existing numerical elements based on the prefix.

    Args:
        directory (str of PathLike): directory to explore
        prefix (str): prefix to filter the elements (e.g., "sub", "ses", "run")

    Returns:
        list[int]: List of existing numerical elements based on the prefix
    """
    if os.path.isdir(directory):
        if prefix == "run":
            elements = [
                int(element.split("_run-")[-1][:2])
                for element in os.listdir(directory)
                if prefix in element
            ]
        else:
            elements = [
                int(element.split("-")[-1])
                for element in os.listdir(directory)
                if prefix in element
            ]
        if elements:
            return elements
        else:
            raise ValueError(f"No element with prefix '{prefix}' found in {directory}")
    else:
        raise NotADirectoryError(f"{directory} is not a directory")
