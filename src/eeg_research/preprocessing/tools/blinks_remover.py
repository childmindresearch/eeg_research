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


class BlinksRemover:
    """Instance for removing blinks from EEG data."""

    def __init__(self, raw: mne.io.Raw,   # noqa: ANN204
                 channels: list[str] = ['Fp1', 'Fp2']):
        """Initialize BlinksRemover instance.

        Args:
            raw (mne.io.Raw): The eeg signal
            channels (list[str], optional): The channel name on which to base
                                            the automatic detection. 
                                            Defaults to ['Fp1', 'Fp2'].
        """
        self.raw = raw
        self.channels = channels
    
    def _find_blinks(self) -> "BlinksRemover":
        """Helper for automatically finding blinks using mne functions.

        Returns:
            BlinksRemover: _description_
        """
        self.eog_evoked = mne.preprocessing.create_eog_epochs(
            self.raw, ch_name = self.channels
            ).average()
        self.eog_evoked.apply_baseline((None, None))
        return self
    
    def plot_removal_results(self, 
                             saving_filename: str | os.PathLike
                             ) ->"BlinksRemover":
        """Plot the result after removing the blinks.

        Args:
            saving_filename (str | os.PathLike): Where to save the figure

        Returns:
            BlinksRemover instance
        """
        figure = mne.viz.plot_projs_joint(self.eog_projs, self.eog_evoked)
        figure.suptitle("EOG projectors")
        if saving_filename:
            figure.savefig(saving_filename)
        return figure
    
    def plot_blinks_found(self, 
                          saving_filename: str | os.PathLike
                          ) ->"BlinksRemover":
        """Plot the blink automatically found.

        Args:
            saving_filename (str | os.PathLike): Where to save teh figure

        Returns:
            BlinksRemover instance
        """
        self._find_blinks()
        figure = self.eog_evoked.plot_joint(times = 0)
        if saving_filename:
            figure.savefig(saving_filename)
        return figure
    
    def remove_blinks(self) -> mne.io.Raw:
        """Remove the EOG artifacts from the raw data.

        Args:
            raw (mne.io.Raw): The raw data from which the EOG artifacts will be removed.

        Returns:
            mne.io.Raw: The raw data without the EOG artifacts.
        """
        self.eog_projs, _ = mne.preprocessing.compute_proj_eog(
            self.raw, 
            n_eeg=1,
            reject=None,
            no_proj=True,
            ch_name = self.channels
        )
        self.blink_removed_raw = self.raw.copy()
        self.blink_removed_raw.add_proj(self.eog_projs).apply_proj()
        return self