"""Tests for simulated_data.py."""

import os
from pathlib import Path

import mne
import pandas as pd
import pytest

import eeg_research.simulators.simulate_data as script


@pytest.fixture
def raw_data() -> mne.io.RawArray:
    """Fixture to create a simulated EEG data."""
    return script.simulate_eeg_data()


@pytest.fixture
def testing_path() -> Path:
    """Fixture to create an output directory for testing purposes."""
    cwd = Path.cwd()
    output_dir = cwd.joinpath("data", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_simulate_light_eeg_data() -> None:
    """Test that the function returns a RawArray object."""
    result = script.simulate_light_eeg_data()
    assert isinstance(result, mne.io.RawArray)


def test_simulate_heavy_eeg_data() -> None:
    """Test that the function returns a RawArray object with annotations."""
    result = script.simulate_eeg_data(
        n_channels=16,
        duration=3,
        misc_channels=["ecg", "emg"],
        sampling_frequency=256,
        events_kwargs=dict(name="testing_event", number=2, start=0.5, stop=2.5),
    )
    assert isinstance(result, mne.io.RawArray)
    assert result.annotations.description[0] == "testing_event"
    assert len(result.annotations.onset) == 2


# The function is called with n_channels = 0.
def test_called_with_n_channels_zero() -> None:
    """Test that the function raises a ValueError when n_channels is 0."""
    with pytest.raises(ValueError):
        script.simulate_eeg_data(n_channels=0)
