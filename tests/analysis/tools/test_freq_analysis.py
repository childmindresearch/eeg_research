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

"""Tests for the freq_analysis module."""

from typing import Generator

import numpy as np
import pytest
from mne.io import RawArray

import eeg_research.analysis.tools.freq_analysis as script
from eeg_research.simulators.eeg_simulator import simulate_light_eeg_data


@pytest.fixture
def raw() -> Generator[RawArray, None, None]:
    """Create a raw object from simulate_light_eeg_data."""
    raw = simulate_light_eeg_data(
        n_channels=64,
        duration=2,
        sampling_frequency=256,
    )
    yield raw


@pytest.fixture
def fft_spectrum(raw: RawArray) -> script.Spectrum:
    """Create a Spectrum object from simulate_light_eeg_data."""
    r = raw
    fft_object = script.Spectrum()
    fft = fft_object.calculate_fft(r)
    return fft


@pytest.fixture
def amplitude_spectrum(
    fft_spectrum: script.Spectrum,
) -> script.Spectrum:
    """Calculate the amplitude of the spectrum."""
    return fft_spectrum.calculate_amplitude()


@pytest.fixture
def zscore_spectrum(
    fft_spectrum: script.Spectrum,
) -> script.Spectrum:
    """Calculate the zscore of the spectrum."""
    fft_spectrum._set_frequency_of_interest(18)
    return fft_spectrum.copy().calculate_zscore()


@pytest.fixture
def snr_spectrum(
    fft_spectrum: script.Spectrum,
) -> script.Spectrum:
    """Calculate the signal to noise ratio of the spectrum."""
    fft_spectrum._set_frequency_of_interest(18)
    return fft_spectrum.copy().calculate_snr()


def test_shape_fft(raw: RawArray, fft_spectrum: script.Spectrum) -> None:
    """Test that fft() returns the correct shape for valid input."""
    raw_data = raw.get_data()
    n_channels = raw_data.shape[0]
    n_times = raw_data.shape[1]
    # divide by 2, round down to nearest power of 2, add 1
    n_times_rfft = 2 ** np.floor(np.log2(n_times // 2)) + 1
    raw_data_shape = (n_channels, n_times_rfft)
    fft_spectrum_shape = fft_spectrum.spectrum.shape
    assert fft_spectrum_shape == raw_data_shape


def test_type_fft(fft_spectrum: script.Spectrum) -> None:
    """Test that fft() returns the correct type for valid input."""
    assert fft_spectrum.spectrum.dtype == "complex128"


def test_shape_frequencies(fft_spectrum: script.Spectrum) -> None:
    """Test that frequencies array has the correct shape."""
    assert fft_spectrum.frequencies.shape[0] == fft_spectrum.spectrum.shape[1]


def test_value_frequency(fft_spectrum: script.Spectrum) -> None:
    """Test that diff. btw. actual and desired freq. is less than freq. resolution."""
    fft_spectrum._set_frequency_of_interest(18)
    desired_frequency = fft_spectrum.frequency_of_interest
    frequency_index = fft_spectrum._get_frequency_index(18)
    actual_frequency = fft_spectrum.frequencies[frequency_index]
    spectrum_frequency_resolution = fft_spectrum.frequency_resolution
    freq_difference = np.abs(actual_frequency - desired_frequency)
    assert freq_difference < spectrum_frequency_resolution


def test_size_max_amplitude(amplitude_spectrum: script.Spectrum) -> None:
    """Test that peak magnitude array length matches number of channels."""
    frequency_window = (17, 20)
    magnitudes = amplitude_spectrum.get_peak_magnitude(frequency_window)
    assert magnitudes.peak_magnitude.shape[0] == 64


def test_values_frequency_window(amplitude_spectrum: script.Spectrum) -> None:
    """Test that the peak frequency is within the frequency window."""
    frequency_window = (17, 20)
    magnitudes = amplitude_spectrum.get_peak_magnitude(frequency_window)
    assert frequency_window[0] <= np.max(magnitudes.peak_frequency_Hz)
    assert frequency_window[1] >= np.max(magnitudes.peak_frequency_Hz)
