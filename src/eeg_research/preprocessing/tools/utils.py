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

import mne


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
    real_name = [ch_name for ch_name in raw.ch_names if name.lower() in ch_name.lower()]
    return real_name[0]


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
            channels_mapping.update({ch_type: ch_name_in_raw})
        else:
            print(f"No {ch_type.upper()} channel found.")
            if ch_type == "eog":
                print("Fp1 and Fp2 will be used for EOG signal detection")
                channels_mapping.update({"eog": "will be Fp1 and Fp2"})

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
    for ch_type, ch_name in channel_map.items():
        raw.set_channel_types({ch_name: ch_type})
    return raw
