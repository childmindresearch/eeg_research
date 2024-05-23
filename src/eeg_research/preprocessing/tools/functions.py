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


def avg_fft_calculation(signal: np.ndarray) -> list[float]:
    """Calculate the average Fast Fourier Transform (FFT) for each channel of the input.

    Args:
        signal (np.ndarray): input signal with shape (num_channels, num_samples)

    Returns:
        list[float]: list of average FFT values for each channel
    """
    avg_fft = []
    for channel in range(signal.shape[0]):
        fft = abs(scipy.fft.fft(np.squeeze(signal[channel, :])))
        fft = fft[: len(fft) // 2]
        avg_fft.append(np.mean(fft))

    return avg_fft


def rms_calculation(signal: np.ndarray) -> list[float]:
    """Calculates the root mean square (RMS) of each channel in the given signal.

    Args:
        signal (np.ndarray): input signal with shape (num_channels, num_samples)

    Returns:
        list[float]: list of RMS values, one for each channel in the signal
    """
    rms = []
    for channel in range(signal.shape[0]):
        rms.append(np.sqrt(np.mean(signal[channel, :] ** 2)))
    return rms


def max_gradient_calculation(
    signal: np.ndarray, sampling_rate: int = 1000
) -> list[float]:
    """Calculate the maximum gradient of the signal.

    Returns the largest difference between samples within the signal.
    Threshold is usually 10uV/ms.

    Args:
        signal (np.ndarray): input signal for which the maximum gradient needs to be
            calculated. The signal should be a 2-dimensional numpy array,
            where each row represents a channel and each column
            represents a sample.
        sampling_rate (int, optional): sampling rate of the signal. Defaults to 1000.

    Returns:
        list[float]: list of maximum gradients for each channel in the input signal.
    """
    if sampling_rate <= 1000:
        samples = 1
    else:
        samples = int(sampling_rate / 1000)
    max_gradient = []
    for channel in range(signal.shape[0]):
        max_gradient.append(np.max(np.diff(signal[channel, :], n=samples)))
    return max_gradient


def kurtosis_calculation(signal: np.ndarray) -> list[float]:
    """Calculate the kurtosis of each channel in the EEG signal.

    Kurtosis is a statistical measure that describes the shape of a distribution.
    In the context of EEG signal quality assessment, kurtosis is important because
    it provides information about the presence of outliers or extreme values
    in the signal. High kurtosis values indicate heavy tails and a higher likelihood
    of extreme values, which can be indicative of artifacts or abnormal brain activity.
    Therefore, calculating the kurtosis of each channel in the EEG signal can help in
    identifying channels with potential quality issues or abnormalities.

    Parameters:
    signal (np.ndarray): The EEG signal with shape (num_channels, num_samples).

    Returns:
    list[float]: list of kurtosis values, where each value corresponds to
        a channel in the signal.

    References:
    - scipy.stats.kurtosis: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
    """
    kurtosis = []
    for channel in range(signal.shape[0]):
        kurtosis.append(scipy.stats.kurtosis(signal[channel, :]))
    return kurtosis


def zero_crossing_calculation(signal: np.ndarray) -> list[int]:
    """Calculates the number of zero crossings in each channel of the given EEG signal.

    Zero crossing refers to the point at which the signal changes its polarity, i.e.,
    when it crosses the zero axis. In EEG analysis, zero crossing calculation can
    provide insights into the frequency content and dynamics of the signal. It is
    often used as a feature to characterize the temporal properties of the EEG waveform.

    Parameters:
    signal (numpy.ndarray): The input EEG signal with shape (num_channels, num_samples).

    Returns:
    list[int]: list containing the number of zero crossings for each channel
        in the signal.
    """
    zero_crossing = []
    for channel in range(signal.shape[0]):
        zero_crossing.append(np.sum(np.abs(np.diff(np.sign(signal[channel, :]))) == 2))
    return zero_crossing


def hjorth_parameters(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Hjorth parameters.

    This function calculates the Hjorth parameters for a given signal.
    The Hjorth parameters are measures of activity, mobility, and complexity
    of the signal.

    Args:
        signal (numpy.ndarray): The input signal for which to calculate the Hjorth
            parameters. The signal should be a 2D array, where each row represents
            a different channel and each column represents a different time point.

    Returns:
        tuple: A tuple containing the calculated Hjorth parameters.
            - activity (numpy.ndarray): The activity parameter for each channel.
            - mobility (numpy.ndarray): The mobility parameter for each channel.
            - complexity (numpy.ndarray): The complexity parameter for each channel.
    """
    activity = np.var(signal, axis=1)
    mobility = np.sqrt(np.var(np.diff(signal, axis=1), axis=1) / activity)
    complexity = np.sqrt(
        np.var(np.diff(np.diff(signal, axis=1), axis=1), axis=1)
        / np.var(np.diff(signal, axis=1), axis=1)
    )
    return activity, mobility, complexity
