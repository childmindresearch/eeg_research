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

from eeg_research.preprocessing.tools.utils import map_channel_type, set_channel_types


def clean_bcg(raw: mne.io.Raw) -> mne.io.Raw:
    """Pre-clean EEG data from BCG and EOG artifacts.

    Args:
        raw (mne.io.Raw): MNE raw object

    Returns:
        raw (mne.io.Raw): MNE raw object
    """
    raw.filter(1, 50).resample(250)

    # ===============================================================================
    # ECG AND EOG CHANNELS DETECTION
    # ===============================================================================

    channel_map = map_channel_type(raw)
    raw = set_channel_types(raw, channel_map)
    raw.set_montage("easycap-M10", on_missing="warn")

    # ===============================================================================
    # COMPUTE SSP ON RAW
    # ===============================================================================

    raw_projs = mne.compute_proj_raw(raw, n_eeg=2)
    raw.add_proj(raw_projs)
    raw.apply_proj()

    # ================================================================================
    # COMPUTE SSP ON ECG AND EOG
    # ================================================================================

    projections, _ = mne.preprocessing.compute_proj_ecg(
        raw, ch_name=channel_map["ecg"], n_eeg=2, reject=None
    )
    raw.add_proj(projections)
    if not channel_map:
        print(
            "No electrophysiological projections computed "
            "due to lack of ECG and EOG channels."
        )
    return raw
