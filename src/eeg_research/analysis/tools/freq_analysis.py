#!/usr/bin/env -S  python  #
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
"""This module handle frequency spectrum obtained from an FFT.

The different calculations perfomed on the spectrum
follow a methodology used in Fast Periodic Visual Stimulation (FPVS) to
quantify with a high Signal Noise Ratio some cognitive processes at a specific
frequency (e.g. the frequency of an Oddball). It is inspired from the Rossion
et al. 2014 and Jonas et al. 2016.

Note:
    The articles mentionned perform a Steady State Evoked Related Potential
    (SSVEP) paradigm which is called Fast Periodic Visual Stimulation
    (Rossion et al. 2014). The paradigm involves a visual stimulation (images)
    at a base frequency (6 Hz). Every 5 images, an oddball is pressented
    (A face among non face images, or a known face among uknown faces etc.). The
    odball frequency is 6/5 = 1.2 Hz. In order to quantify the
    electrophysiological response of the cognitive process exhibited by the
    oddball, the authors perform a frequency tagging analysis of the signal.
    They calculate the fast fourier transform (FFT) of the signal and then
    perform different operations and claculations on the spectrum.
    The different operations are the correction of the baseline, the zscore
    (to quantify the significance of the amplitude of the frequency of interest
    from the "noise"), and the signal to noise ratio. To do so, they consider a
    specific zone of 50 surrounding bins around the frequency of interest
    (25 bins before and 25 bins after the frequency of interest).
    They leave out one bin before and one bin after the frequency
    of interest that lead to 24 bins each side of the frequency of interest.
    This zone is consider as the "surrounding noise" of the brain activity and
    the amplitude of the frequency of interest is considered as the "signal".
    Therefore, the zscore is calculated as the amplitude of the signal divided
    by the standard deviation of the surrounding noise. The signal to noise
    ratio is calculated as the amplitude of the signal divided by the mean
    amplitude of the surrounding noise.
    Their methdology was implemented for a signal sampled at 512 Hz. Therefore
    1 bin = 0.0135 Hz. In this module the approach used is in frequency step
    rather than frequency bin, in order to have the same frequency
    values as in the articles still offering flexibility in term of input
    sampling frequency and frequency resolution.

    This whole approach is interesting to quantify also noise that are at a
    known frequency (e.g. 60 Hz noise from the electrical grid, gradient
    artifacts for EEG-fMRI).

    .. _Rossion et al. 2014: https://pubmed.ncbi.nlm.nih.gov/24728131/
    .. _Jonas et al. 2016: https://pubmed.ncbi.nlm.nih.gov/27354526/

"""

# standard library imports
import copy

# third-party imports (and comments indicating how to install them)
# python -m conda install -c conda-forge mne or python -m pip install mne
import mne

# python -m conda install -c conda-forge numpy or python -m pip install numpy
import numpy as np
import scipy


class Spectrum:
    """A class to store and manipulate the Fourier Spectrum of a signal."""

    def __init__(self) -> None:
        """Initialize."""
        self.info: dict = dict()
        self.info["process_history"] = list()

    def calculate_fft(self, raw: mne.io.Raw) -> "Spectrum":
        """Initialize the Spectrum object.

        Args:
            raw (mne.io.Raw): The raw signal to be analyzed
        """
        self.sampling_rate = raw.info["sfreq"]
        self.signal = raw.get_data()
        self._adjust_signal_length()
        self.signal_length = np.shape(self.signal)[1]
        self.spectrum = scipy.fft.rfft(self.signal) * 2 / self.signal_length
        self.frequencies = scipy.fft.fftfreq(
            self.spectrum.shape[1], 1 / self.sampling_rate
        )

        self.frequency_resolution = self.frequencies[1] - self.frequencies[0]
        self.info["units"] = "V"
        self.info["sfreq"] = self.sampling_rate
        self.info["frequency_resolution"] = self.frequency_resolution
        self.info["n_channels"] = self.signal.shape[0]
        self.info["ch_names"] = raw.info["ch_names"]

        return self

    def _adjust_signal_length(self) -> "Spectrum":
        """Adjust the signal length to be a power of 2.

        This is done in order to have a faster computation of the fft.

        Args:
        signal (numpy.ndarray): A 2D numpy array (channels, time).

        Returns:
        Self: the Spectrum object
        """
        current_length = self.signal.shape[1]

        if current_length & (current_length - 1) != 0:
            new_length = 2 ** int(np.log2(current_length))
            self.signal = np.resize(self.signal, (self.signal.shape[0], new_length))

        self.info["process_history"].append("Signal length adjusted to be a power of 2")

        return self

    def _set_frequency_of_interest(
        self, frequency_of_interest: float = 12
    ) -> "Spectrum":
        """Set the frequency of interest.

        This is following the methodology of Rossion et al. 2014 and Jonas et al. 2016
        in order to get the amplitude of the surrounding bins.

        Args:
            frequency_of_interest (float): The frequency of interest.

        Returns:
            typing.Self: The Spectrum object
        """
        self.frequency_of_interest = frequency_of_interest
        self.info["process_history"].append(
            f"Frequency of interest set to {frequency_of_interest}"
        )
        return self

    def _frequency_of_interest_exists(self) -> bool:
        """Check if the frequency of interest has been set.

        Returns:
            bool: True if the frequency of interest has been set, False otherwise.
        """
        return hasattr(self, "frequency_of_interest")

    def _get_frequency_index(self, frequency: np.ndarray | float) -> np.ndarray:
        """Get the index of the frequency of interest in the spectrum.

        Args:
            frequency (np.ndarray | float): The frequency of interest.

        Returns:
            int: The index (position) in the spectrum
        """
        if isinstance(frequency, np.ndarray):
            freq_per_electrode = np.broadcast_to(
                self.frequencies, (frequency.shape[0], self.frequencies.shape[0])
            )

            frequencies_index = np.argmin(
                np.abs(
                    np.subtract(
                        frequency.reshape(freq_per_electrode.shape[0], 1),
                        freq_per_electrode,
                    )
                ),
                axis=1,
            )

        elif isinstance(frequency, (int, float)):
            frequencies_index = np.argmin(
                np.abs(self.frequencies - frequency),
            )

        else:
            raise TypeError("The frequency has to be a float or an array of floats")

        return frequencies_index.astype(int)

    def _get_amplitude_surounding_bins(
        self,
        desired_frequency_step: float = 0.0135,
        nb_steps: int = 25,
    ) -> np.ndarray:
        """Get the amplitude of the surrounding bins of the frequency of interest.

        Args:
            frequency_of_interest (float ): The frequency around
            desired_frequency_step (float, optional): _description_. Defaults to 0.0135
                                                      to reproduce the articles
                                                      methodology.
            nb_steps (int, optional): _description_. Defaults to 25.

        Returns:
            np.ndarray: _description_
        """
        frequency_index = self._get_frequency_index(self.frequency_of_interest)

        nb_bins = int(desired_frequency_step / self.frequency_resolution)
        indices_left = np.linspace(
            frequency_index - nb_steps * nb_bins,
            frequency_index - nb_bins,
            axis=1,
        )

        indices_right = np.linspace(
            frequency_index + nb_bins,
            frequency_index + nb_steps * nb_bins,
            axis=1,
        )

        amplitude_surrounding_left_bins = self.spectrum[
            np.arange(frequency_index.shape[0]), indices_left.astype(int).T
        ]

        amplitude_surrounding_right_bins = self.spectrum[
            np.arange(frequency_index.shape[0]), indices_right.astype(int).T
        ]

        self.info["process_history"].append(
            f"""Amplitude of the {(nb_steps-1)*2} surrounding bins of the
            frequency of interest calculated with a frequency step of
            {desired_frequency_step} Hz"""
        )

        return np.concatenate(
            (amplitude_surrounding_left_bins.T, amplitude_surrounding_right_bins.T),
            axis=1,
        )

    def _get_baseline(self) -> float:
        """Get the baseline of the spectrum around the frequency of interest.

        The baseline is the mean amplitude of the surrounding bins of the
        frequency of interest.

        Returns:
            float : The baseline value

        """
        if not self._frequency_of_interest_exists():
            raise ValueError("The frequency of interest has to be set first.")
        amplitude_surrounding_bins = self._get_amplitude_surounding_bins()
        return np.mean(amplitude_surrounding_bins, axis=1).reshape(-1, 1)

    def correct_baseline(self) -> "Spectrum":
        """Remove the baseline of the spectrum.

        It removes the baseline on the entire spectrum. What is considered
        as the baseline is the mean amplitude of the surrounding bins of the
        frequency of interest (see _get_baseline() method for more details).
        WARNING: This method is not reversible and will modify the spectrum.
        Consider using the copy() method before using this method if you want
        to keep the original spectrum.

        Args:
            frequency_of_interest (float, optional): The frequency around which
                                                     the baseline, snr and
                                                     zscore will be calculated.
                                                     Defaults to 12.

        Returns:
            self : The modified Spectrum object.
        """
        baseline = self._get_baseline()
        self.info["process_history"].append("Baseline corrected")
        self.spectrum = np.subtract(self.spectrum, baseline)
        return self

    def _baseline_corrected(self) -> bool:
        """Check if the baseline has been corrected.

        Returns:
            bool: True if the baseline has been corrected, False otherwise.
        """
        return "basline corrected" in self.info["process_history"]

    def average_across_channel(self: "Spectrum") -> "Spectrum":
        """Calculate the average across channels."""
        self.spectrum = np.mean(self.spectrum, axis=0, keepdims=True)
        self.spectrum_std = np.std(self.spectrum, axis=0, keepdims=True)

        return self

    def pick_channels(self, channel_names: list) -> "Spectrum":
        """Pick the desired channel.

        The spectrum contain only the desired channels

        Args:
            channel_names (list, optional): Defaults to None.

        Returns:
            Spectrum: The spectrum object
        """
        channel_truth_table = [
            channel_name in channel_names for channel_name in self.info["ch_names"]
        ]
        self.spectrum = self.spectrum[channel_truth_table, :]
        return self

    def calculate_amplitude(self) -> "Spectrum":
        """Calculate the amplitude of the spectrum.

        Args:
            spectrum (Spectrum): The Spectrum object
                                        to be converted.

        Returns:
            Spectrum: The converted Spectrum object.
        """
        self.spectrum = np.abs(self.spectrum)
        self.info["units"] = "V"
        return self

    def calculate_zscore(self) -> "Spectrum":
        """Calculate the zscore of the spectrum.

        The zscore is calculated as the amplitude of the frequency of interest
        divided by the standard deviation of the amplitude of the surrounding
        bins.

        Args:
            spectrum (Spectrum): The Spectrum object to be
                                        converted.
            frequency_of_interest (int, optional): The frequency around
                                                   which the standard
                                                   deviation is calculated.
                                                   Defaults to 12.

        Raises:
            TypeError: The input has to be an AmplitudeSpectrum to calculate
                       the zscore.

        Returns:
            Spectrum: The converted Spectrum object.
        """
        if not self._frequency_of_interest_exists():
            raise ValueError("The frequency of interest has to be set first.")

        if not self._baseline_corrected():
            self.correct_baseline()

        amplitude_surrounding_bins = self._get_amplitude_surounding_bins()

        surrounding_bin_std = np.std(amplitude_surrounding_bins, axis=1).reshape(-1, 1)

        self.spectrum = np.divide(self.spectrum, surrounding_bin_std)
        self.info["process_history"].append("Zscore calculated")
        self.info["units"] = "A.U"

        return self

    def calculate_snr(self) -> "Spectrum":
        """Calculate the signal to noise ratio of the spectrum.

        The signal to noise ratio is calculated as the baseline corrected
        amplitude of the frequency of interest divided by the baseline
        amplitude.

        Args:
            spectrum (Spectrum): The Spectrum object to be
                                        converted.
            frequency_of_interest (int, optional): The frequency around
                                                   which the baseline and
                                                   snr will be calculated.
                                                   Defaults to 12.

        Raises:
            TypeError: The input has to be an AmplitudeSpectrum to calculate
                       the snr.

        Returns:
            Spectrum: The converted Spectrum object.
        """
        if not self._frequency_of_interest_exists():
            raise ValueError("The frequency of interest has to be set first.")

        baseline = self._get_baseline()
        self.spectrum = 20 * np.log10(np.divide(self.spectrum, baseline))
        self.info["process_history"].append("Snr calculated")
        self.info["units"] = "dB"
        return self

    def calculate_phase(self) -> "Spectrum":
        """Calculate the phase of the spectrum.

        Returns:
            Self: The modified Spectrum object.
        """
        self.phase = np.angle(self.spectrum)
        self.info["process_history"].append("Phase calculated")
        self.info["units"] = "rad"
        return self

    def copy(self) -> "Spectrum":
        """Copy the object following the same philosophy as mne objects.

        Returns:
            self: A copy of the instance
        """
        self.info["process_history"].append("Object copied")
        self = copy.copy(self)
        return self

    def get_peak_magnitude(self, frequency_window: tuple = (17, 20)) -> "Spectrum":
        """Get the peak magnitude and frequency within a specific window.

        Args:
            frequency_window (tuple, optional): The chosen frequency window
                                                to get the max amplitude

        Returns:
            dict: A dictionary containing the window, the peak magnitude and
                  the peak frequency.
        """
        if isinstance(frequency_window, tuple):
            self.index_window = [
                self._get_frequency_index(frequency) for frequency in frequency_window
            ]

        elif isinstance(frequency_window, int) or isinstance(frequency_window, float):
            self.index_window = [
                self._get_frequency_index(frequency_window) - 1,
                self._get_frequency_index(frequency_window) + 1,
            ]

        self.peak_magnitude = np.max(
            self.spectrum[:, self.index_window[0] : self.index_window[1]], axis=1
        )

        self.index_max = (
            np.argmax(
                self.spectrum[:, self.index_window[0] : self.index_window[1]], axis=1
            )
            + self.index_window[0]
        )

        self.peak_frequency_Hz = self.frequencies[self.index_max]

        self.info["process_history"].append(
            f"Peak magnitude and frequency calculated in the window {frequency_window}"
        )

        return self


# TODO
# - Elaborate more the info
