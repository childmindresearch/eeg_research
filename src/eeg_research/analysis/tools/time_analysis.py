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
# python -m conda install -c conda-forge numpy or python -m pip install numpy
import numpy as np

# python -m conda install -c conda-forge scipy or python -m pip install scipy
import scipy


def average_rms(signal: np.ndarray, axis: int = 1) -> float:
    """Calculate the average root mean square of the signal.

    The RMS of a signal is regarded as the magnitude of it.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array
        axis (int): the axis along which the RMS is calculated

    Returns:
        float: the average root mean square of the signal
    """
    return np.sqrt(np.mean(signal**2, axis=axis))


def max_gradient(signal: np.ndarray, axis: int = 1) -> float:
    """Calculate the maximum gradient of the signal.

    The maximum gradient is the maximum absolute value
    between 2 consecutive values of the signal for 2 consecutive
    time samples. It is usefull to detect high amplitude,
    high frequency artifacts.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array
        axis (int): the axis along which the gradient is calculated

    Returns:
        float: the maximum gradient of the signal
    """
    return np.max(np.abs(np.diff(signal, axis=axis)), axis=axis)


def zero_crossing_rate(signal: np.ndarray, axis: int = 1) -> float:
    """Calculate the zero crossing rate of the signal.

    It is the rate at which the signal cross the 0 line.
    High frequency signal will have a high rate,
    and low frequency/drifting signal will have a low rate.

    Args:
        signal (numpy.ndarray): the signal to be analyzed
                                has to be a 1D array
        axis (int): the axis along which the zero crossing rate is calculated

    Returns:
        float: the zero crossing rate of the signal
    """
    return np.mean(np.diff(np.sign(signal) != 0, axis=axis), axis=axis)


def hjorth_mobility(signal: np.ndarray, axis: int = 1) -> float:
    """Calculate the mobility from the Hjorth parameters.

    The mobility is a measure of the signal's frequency content.
    It is the ratio of the variance of the derivative of the signal
    to the variance of the signal.

    Args:
        signal (np.ndarrary): the signal to be analyzed
                              has to be a 1D array.
        axis (int): the axis along which the mobility is calculated

    Returns:
        float: the mobility score of the signal
    """
    derived_signal_variance = np.var(np.diff(signal, axis=axis), axis=axis)
    signal_variance = np.var(signal, axis=axis)
    return np.sqrt(derived_signal_variance / signal_variance)


def hjorth_complexity(signal: np.ndarray, axis: int = 1) -> float:
    """Calculate the complexity from the Hjorth parameters.

    The complexity, as it is indicated by the name, is a measure of the
    complexity of the signal. It is the ratio of the mobility of the derivative
    of the signal to the mobility of the signal.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.
        axis (int): the axis along which the complexity is calculated

    Returns:
        float: the complexity score of the signal
    """
    derived_signal_mobility = hjorth_mobility(np.diff(signal, axis=axis), axis=axis)
    signal_mobility = hjorth_mobility(signal, axis=axis)
    return derived_signal_mobility / signal_mobility


def signal_range(signal: np.ndarray, axis: int = 1) -> float:
    """Calculate the range of the signal.

    Range of the signal is the difference between the maximum and the minimum
    values.

    Args:
        signal (np.ndarray): the signal to be analyzed
                              has to be a 1D array.
        axis (int): the axis along which the range is calculated

    Returns:
        float: the range of the signal
    """
    return np.subtract(np.max(signal, axis=axis), np.min(signal, axis=axis))


def variance(signal: np.ndarray, axis: int = 1) -> np.ndarray:
    """Calculate the variance of an array.

    Calculate the variance of the signal.

    Args:
        signal (np.ndarray): The signal to be analyzed
        axis (int, optional): The axis along which the calculation
                              will be performed. Defaults to 1.

    Returns:
        np.ndarray: The variance of the signal with a shape of (channels,time)

    Note:
        I could just have used np.var but because sometime it's numpy and sometime
        it's scipy I prefere unifying everything in order to be able to call
        the function in serie
    """
    return np.var(signal, axis=axis)


def skewness(signal: np.ndarray, axis: int = 1) -> np.ndarray:
    """Calculate the skewness.

    Skewness is a measure of the asymmetry of the probability distribution
    of a real-valued random variable about its mean. The skewness value can be
    positive or negative, or even undefined.

    Args:
        signal (np.ndarray): The signal to be analyzed
        axis (int, optional): The axis along which the calculation
                              will be performed. Defaults to 1.

    Returns:
        np.ndarray: The skewness of the signal with a shape of (channels, time)
    """
    return scipy.stats.skew(signal, axis=axis)


def kurtosis(signal: np.ndarray, axis: int = 1) -> np.ndarray:
    """Calculate the kurtosis of the signal.

    The kurtosis is a measure of the "tailedness" of the probability
    distribution of a real-valued random variable. In a similar way to the
    skewness, the kurtosis value can be positive or negative, or even undefined.

    Args:
        signal (np.ndarray): The signal to be analyzed
        axis (int, optional): The axis along which the calculation
                              will be performed. Defaults to 1.

    Returns:
        np.ndarray: The kurtosis of the signal with a shape of (channels, time)
    """
    return scipy.stats.kurtosis(signal, axis=axis)


def signal_iqr(signal: np.ndarray, axis: int = 1) -> np.ndarray:
    """Calculate the interquartile range of the signal.

    The interquartile range is a measure of statistical dispersion,
    or how scattered, the values in a dataset are. It is the difference
    between the third quartile and the first quartile.

    Args:
        signal (np.ndarray): The signal to be analyzed
        axis (int, optional): The axis along which the calculation
                              will be performed. Defaults to 1.

    Returns:
        np.ndarray: The interquartile range of the signal with a shape of
                    (channels, time)
    """
    return scipy.stats.iqr(signal, axis=axis)


# No time window needed. I can deal with mne object now
# TO FINISH
def snr_epoch(signal: np.ndarray) -> np.ndarray:
    """Calculate the signal to noise ratio of an Evoked Related Potential.

    What is considered here as the signal is the ERP (the average signal
    across epochs). The noise is the standard deviation across epochs.

    Args:
        signal (np.ndarray): The signal to be analyzed

    Returns:
        np.ndarray: The signal to noise ratio of the epochs
    """
    erp_signal = signal.mean(axis=0)
    erp_noise = signal.std(axis=0)
    snr = np.divide(erp_signal**2, erp_noise**2)
    snr_decibel = 10 * np.log10(snr)
    return snr_decibel


# TODO
# - Make a class object to store the steps of the process
# in order to keep a history of what has been done.
# - Think about epoching the gradient peak.
