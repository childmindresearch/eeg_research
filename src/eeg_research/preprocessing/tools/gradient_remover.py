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

import numpy as np
from scipy.signal import detrend


class GradientRemover:
    """A class to remove gradients from EEG data using a template approach."""

    def __init__(
        self,
        eeg_data: np.ndarray,
        tr_events: np.ndarray,
        window: int | tuple[int, int] = (4, 4),
    ) -> None:
        """Constructor for the gradient remover.

        Args:
            eeg_data (np.ndarray): The raw EEG data to perform gradient correction on.
                Expected in shape (channels, time_points).
            tr_events (np.ndarray): The sample numbers for when TRs begin. The array
                may be a subset of an mne find_events that is a shape (N, 3), or a
                1-dimensional array of sample numbers. TRs must be perfectly
                spaced in time.
            window (int or tuple[int, int]): The window to use for templates.
                Must be either an even integer to indicate the total size,
                with an even number of TRs templated before and after,
                OR a tuple containing the number of TRs to use in the template
                before the current TR and then the number of TRs to use in the template
                after the current TR. For example, (4, 0) would use 4 TRs before
                the current and 0 after. Default is (4, 4).

        Raises:
            ValueError: if any inputs are invalid.
        """
        self._window = GradientRemover._valid_window(self, window)
        self._tr_events = GradientRemover._valid_tr_events(self, tr_events)
        if self._tr_events[-1] > eeg_data.shape[1]:
            raise ValueError(
                f"Last TR event is sample {self._tr_events[-1]} but "
                f"eeg data only contains {eeg_data.shape[1]} samples. "
                "Please check your tr event markers."
            )
        self._data = eeg_data
        self._raw = eeg_data  # placeholder for raw data
        # Get weights for template
        window_total = self.window[0] + self.window[1]
        self._weight_before = self.window[0] / window_total
        self._weight_after = self.window[1] / window_total
        # Lazy evaluation
        self._corrected = eeg_data

    @property
    def corrected(self) -> np.ndarray:
        """The gradient-corrected data."""
        if self._corrected:
            return self._corrected
        else:
            return self.correct()

    @property
    def raw(self) -> np.ndarray:
        """The raw data."""
        return self._raw

    @property
    def window(self) -> tuple[int, int]:
        """The window to use for templates."""
        return self._window

    @property
    def tr_spacing(self) -> int:
        """The time between TRs in samples."""
        return self._tr_events[1] - self._tr_events[0]

    @property
    def n_tr(self) -> int:
        """The number of TRs in the data."""
        return len(self._tr_events)

    @property
    def n_channels(self) -> int:
        """The number of channels in the data."""
        return len(self._data)

    def get_tr(self, n: int) -> np.ndarray:
        """Get the uncorrected data at a given TR.

        Args:
            n (int): The TR to get the uncorrected data at (0-indexed).

        Returns:
            np.ndarray: The uncorrected data at the given TR,
                with shape (channels, tr_timepoints).

        Raises:
            ValueError: If an invalid TR index is supplied.
        """
        this_start, this_end = self._tr_bounds(n)
        return self._data[:, this_start:this_end]

    def get_tr_detrended(self, n: int) -> np.ndarray:
        """Get the detrended data at a given TR.

        Args:
            n (int): The TR to get the detrended data at (0-indexed).

        Returns:
            np.ndarray: The detrended data at the given TR,
                with shape (channels, tr_timepoints).

        Raises:
            ValueError: If an invalid TR index is supplied.
        """
        return detrend(self.get_tr(n))

    def get_tr_template(self, n: int) -> np.ndarray:
        """Get the gradient template data at a given TR.

        Args:
            n (int): The TR to get the template data at (0-indexed).

        Returns:
            np.ndarray: The template data at the given TR,
                represented as an array of shape (channels, tr_timepoints).

        Raises:
            ValueError: If an invalid TR index is supplied.
        """
        self._check_valid_tr(n)
        if n < self.window[0] or n > (self.n_tr - self.window[1]):
            return np.zeros((self.n_channels, self.tr_spacing))
        if self.window[0]:
            before = self._get_tr_template_part(n - self.window[0], n)
        else:
            before = np.zeros((self.n_channels, self.tr_spacing))
        if self.window[1]:
            after = self._get_tr_template_part(n + 1, n + self.window[1] - 1)
        else:
            after = np.zeros((self.n_channels, self.tr_spacing))
        return self._weight_before * before + self._weight_after * after

    def _get_tr_template_part(self, start: int, stop: int) -> np.ndarray:
        """Returns the template part of data between the given start and stop indices.

        Args:
            start (int): The starting index of the time series data.
            stop (int): The ending index of the time series data.

        Returns:
            np.ndarray: The template part of the time series data.
        """
        return np.mean(
            np.asarray([self.get_tr_detrended(tr) for tr in range(start, stop)]), axis=0
        )

    def get_tr_corrected(self, n: int) -> np.ndarray:
        """Get the gradient-corrected data at a given TR.

        Args:
            n (int): The TR to get the corrected data at (0-indexed).

        Returns:
            np.ndarray: The template data at the given TR,
                with shape (channels, tr_timepoints).

        Raises:
            ValueError: If an invalid TR index is supplied.
        """
        detrended = self.get_tr_detrended(n)
        template = self.get_tr_template(n)
        return detrended - template

    def correct(self) -> np.ndarray:
        """Generate the gradient-corrected data."""
        corrected = self._data.copy()
        for tr in range(self.n_tr):
            this_start, this_end = self._tr_bounds(tr)
            corrected[:, this_start:this_end] = self.get_tr_corrected(tr)
        self._corrected = corrected
        return corrected

    def _valid_window(self, window: int | tuple[int, int]) -> tuple[int, int]:
        """Validates the window parameter for the GradientRemover class.

        Args:
            window (int or tuple): The window size or range.

        Returns:
            tuple: The validated window size or range.

        Raises:
            ValueError: If the window is not a positive, even integer or
                a tuple of size 2 containing positive integers.
            TypeError: If the window is not an integer or a tuple.
        """
        if isinstance(window, int):
            if not window % 2 == 0:
                raise ValueError(f"Integer windows must be even (received {window}).")
            window = (window // 2, window // 2)
        elif isinstance(window, tuple):
            if not len(window) == 2:
                raise ValueError(
                    "Tuple windows must contain 2 elements " f"(received {window})."
                )
        else:
            raise TypeError(
                "Window must be a positive, even integer or a tuple of "
                "size 2 containing a positive integer."
                "(Received {window})."
            )
        if window[0] < 0 or window[1] < 0:
            raise ValueError(
                "Window must contain a positive integer. " f"(Received {window})."
            )
        if window[0] == 0 and window[1] == 0:
            raise ValueError(
                "Window must contain a positive integer. " f"(Received {window})."
            )
        return window

    def _valid_tr_events(self, tr_events: np.ndarray) -> np.ndarray:
        # Check to make sure TRs are evenly spaced
        if len(tr_events.shape) == 2:
            if tr_events.shape[1] == 3:
                tr_events = tr_events[:, 0]
            else:
                raise ValueError(
                    "TRs must be a 1D array or a (N, 3) ndarray from mne. "
                    f"Received array of shape {tr_events.shape}."
                )
        elif len(tr_events.shape) != 1:
            raise ValueError(
                "TRs must be a 1D array or a (N, 3) ndarray from mne. "
                f"Received array of shape {tr_events.shape}."
            )
        unique = np.unique(np.diff(tr_events))
        if len(unique) != 1:
            raise ValueError(
                "TR spacings are not consistent; the following unique "
                f"distances were present: {unique}."
            )
        return tr_events

    def _check_valid_tr(self, n: int) -> None:
        if n < 0 or n >= self.n_tr:
            raise ValueError(f"Index {n} not in TR range [0, {self.n_tr - 1}]")

    def _tr_bounds(self, n: int) -> tuple[int, int]:
        self._check_valid_tr(n)
        offset = self._tr_events[0]
        length = self.tr_spacing
        this_start = offset + n * length
        this_end = offset + (n + 1) * length
        return (this_start, this_end)
