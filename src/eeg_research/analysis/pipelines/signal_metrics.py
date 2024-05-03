#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-03-06
# email: samuel DOT louviot AT nki DOT rfmh DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
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
"""MODULE DESCRIPTION HERE."""

# third-party imports (and comments indicating how to install them)
# python -m conda install -c conda-forge mne or python -m pip install mne
import mne

# python -m conda install -c conda-forge numpy or python -m pip install numpy
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from eeg_research.analysis.tools import freq_analysis, time_analysis


class SignalMetrics:
    """Class to store the signal metrics of an EEG signal."""

    def __init__(self, mne_object: mne.io.Raw | mne.Epochs) -> None:  # noqa: D107
        self.mne_object = mne_object
        self.info = mne_object.info

    def calculate_frequency_metrics(
        self, frequency_range: tuple = (17, 20), frequency_type: str = "noise"
    ) -> "SignalMetrics":
        """Calculate the frequency metrics of the EEG signal.

        Args:
            frequency_range (tuple, optional): The range of the frequency
                to look for the peak artifact. Defaults to (17,20).

            frequency_type (str, optional): The type of frequency to look for
                if it is a noise or an electrophysiological signal.
                Defaults to 'noise'.

        Returns:
            SignalMetrics:  The signal metrics object
        """
        spectrum_object = freq_analysis.Spectrum()
        # Here is a strange way to check the instance. But using isinstance
        # within the 'if' statement makes everything crash because when it
        # is not an eeglab raw instance, it throw an error saying
        # that mne doesn't have eeglab attribute. They coded in a dynamic
        # way that prevent me to use isinstance.
        raw_type = "raw" in str(type(self.mne_object)).lower()
        if raw_type:
            spectrum_object.calculate_fft(self.mne_object)
            amplitude = spectrum_object.copy().calculate_amplitude()
            amplitude.get_peak_magnitude(frequency_range)
            amplitude._set_frequency_of_interest(amplitude.peak_frequency_Hz)
            zscore = amplitude.copy().calculate_zscore()

            if frequency_type.lower() == "noise":
                power = -1
            elif frequency_type.lower() == "signal":
                power = 1

            snr = amplitude.copy().calculate_snr()
            np.power(snr.spectrum, power)

            for spectrum, name in zip(
                [amplitude, zscore, snr], ["amplitude", "zscore", "snr"]
            ):
                spectrum.get_peak_magnitude(frequency_range)
                setattr(self, name, spectrum.spectrum)
                setattr(self, name + "_peak_magnitude", spectrum.peak_magnitude)
                setattr(self, name + "_peak_frequency", spectrum.peak_frequency_Hz)
        else:
            raise ValueError("Frequency analysis is only available for Raw object")
        return self

    def calculate_time_metrics(
        self, sliding_time_window: float = 1, overlap: float = 0.5
    ) -> "SignalMetrics":
        """Calculate the signal metrics of the EEG signal.

        Args:
            sliding_time_window (float, optional): The time in second of the
                sliding window for analysis. Defaults to 1.

            overlap (float, optional): The overlapping ratio of 2 consecutive
                time window from 0.99 wich overlap at 99% the previous window
                to 0 which overlap at 0%. Defaults to 0.5.

            frequency_range (tuple, optional): The range of the frequency
                to look for the peak artifact. Defaults to (17,20).

        Returns:
            SignalMetrics:  The signal metrics object
        """
        data = self.mne_object.get_data()

        raw_type = "raw" in str(type(self.mne_object)).lower()
        if raw_type:
            window_nb_samples = int(
                (sliding_time_window * self.mne_object.info["sfreq"])
            )
            step = int(window_nb_samples - window_nb_samples * overlap)
            window_view = sliding_window_view(
                data, window_shape=window_nb_samples, axis=1
            )[:, ::step, :]
            data_epoched = np.moveaxis(window_view, 0, 1)

        elif isinstance(self.mne_object, mne.Epochs):
            data_epoched = data
            self.snr_epochs = time_analysis.snr_epoch(data_epoched)

        metric_names = [
            "average_rms",
            "max_gradient",
            "zero_crossing_rate",
            "hjorth_mobility",
            "hjorth_complexity",
            "kurtosis",
            "skewness",
            "variance",
            "signal_range",
            "signal_iqr",
        ]

        for metric_name in metric_names:
            setattr(
                self,
                metric_name,
                getattr(time_analysis, metric_name)(data_epoched, axis=0),
            )
        return self
