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
import numpy as np

from eeg_research.preprocessing.tools.gradient_remover import GradientRemover


def extract_gradient_trigger_name(
    raw: mne.io.Raw, desired_trigger_name: str = "R128"
) -> str:
    """Extract the name of the trigger for gradient artifact removal.

    Name of the gradient trigger can change across different paradigm,
    acquisition etc.

    Args:
        raw (mne.io.Raw): MNE raw object
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

    raise Exception("No gradient trigger found.")


def clean_gradient(raw: mne.io.Raw) -> mne.io.Raw:
    """Pre-clean EEG data from gradient artifacts.

    Args:
        raw (mne.io.Raw): MNE raw object

    Returns:
        mne.io.Raw: MNE raw object
    """
    raw.filter(1, None)
    gradient_trigger_name = extract_gradient_trigger_name(
        raw, desired_trigger_name="R128"
    )

    gradient_trigger = mne.events_from_annotations(
        raw, event_id={gradient_trigger_name: 0}
    )

    gradient_remover = GradientRemover(raw.get_data(), gradient_trigger[0])
    raw_corrected = gradient_remover.correct()
    raw = mne.io.RawArray(raw_corrected, raw.info)
    return raw
