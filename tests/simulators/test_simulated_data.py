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


def test_dummy_dataset_called_with_zeros() -> None:
    """Test that a ValueError is raised when n_subjects, n_sessions, or n_runs is 0."""
    with pytest.raises(ValueError):
        script.DummyDataset(n_subjects=0, n_sessions=0, n_runs=0)


def test_participant_metadata() -> None:
    """Test that the function returns a DataFrame with the participant metadata."""
    dataset = script.DummyDataset(n_subjects=5)
    dataset._create_participant_metadata()
    assert isinstance(dataset.participant_metadata, pd.DataFrame)
    assert dataset.participant_metadata.shape[0] == 5
    nan_mask = dataset.participant_metadata.isna()
    for column in dataset.participant_metadata.columns:
        assert not any(nan_mask[column].values)


def test_add_participant_metadata() -> None:
    """Test that the function adds a new participant to the participant metadata."""
    dataset = script.DummyDataset(n_subjects=5)
    dataset._create_participant_metadata()
    dataset._add_participant_metadata(
        participant_id="sub-06", age=26, sex="M", handedness="R"
    )
    assert isinstance(dataset.participant_metadata, pd.DataFrame)
    assert dataset.participant_metadata.shape[0] == 6
    nan_mask = dataset.participant_metadata.isna()
    for column in dataset.participant_metadata.columns:
        assert not any(nan_mask[column].values)


def test_generate_label(testing_path: Path) -> None:
    """Test that the function generates the correct label."""
    dataset = script.DummyDataset(root=testing_path)
    for i in range(1, 12):
        labels = dataset._generate_label("subject", i, "TEST")
        assert labels == f"sub-TEST{i:03d}"
    labels = dataset._generate_label("subject", 1)
    assert labels == "sub-001"
    labels = dataset._generate_label("session", 1)
    assert labels == "ses-001"
    labels = dataset._generate_label("run", 1)
    assert labels == "run-001"


def test_create_modality_agnostic_dir(testing_path: Path) -> None:
    """Test that the function creates a modality-agnostic directory."""
    dataset = script.DummyDataset(root=testing_path)
    path = dataset.create_modality_agnostic_dir()
    for content in testing_path.iterdir():
        if "temporary_directory_generated_" in content.name:
            temporary_directory = content
            break
    asserting_path = temporary_directory.joinpath("RAW", "sub-001", "ses-001")
    assert isinstance(path[0], Path)
    assert str(path[0]) == str(asserting_path)


def test_extract_entities_from_path(testing_path: Path) -> None:
    """Test that the function extracts the entities from a path."""
    dataset = script.DummyDataset(root=testing_path)
    asserting_path = testing_path.joinpath("RAW", "sub-001", "ses-001")
    entities = dataset._extract_entities_from_path(asserting_path)
    assert entities == {"subject": "sub-001", "session": "ses-001"}


def test_create_sidecar_json(testing_path: Path) -> None:
    """Test that the function creates a sidecar JSON file."""
    dataset = script.DummyDataset(root=testing_path)
    for content in testing_path.iterdir():
        if "temporary_directory_generated_" in content.name:
            temporary_directory = content
            break
    eeg_filename = "sub-001_ses-001_task-test_run-001_eeg.vhdr"
    base_eeg_filename, _ = os.path.splitext(eeg_filename)
    eeg_path = temporary_directory.joinpath("RAW", "sub-001", "ses-001", "eeg")
    eeg_path.mkdir(parents=True, exist_ok=True)
    eeg_full_path = eeg_path.joinpath(eeg_filename)
    dataset._create_sidecar_json(eeg_full_path)
    asserting_path = eeg_path.joinpath(base_eeg_filename + ".json")
    assert asserting_path.exists()


def test_method_create_eeg_dataset(testing_path: Path) -> None:
    """Test that the method creates an EEG dataset."""
    dataset = script.DummyDataset(root=testing_path)
    dataset.create_eeg_dataset(light=True)
    for content in testing_path.iterdir():
        if "temporary_directory_generated_" in content.name:
            temporary_directory = content
            break
    asserting_path = temporary_directory.joinpath("RAW", "sub-001", "ses-001", "eeg")
    eeg_filenames = [
        "sub-001_ses-001_task-test_run-001_eeg.vhdr",
        "sub-001_ses-001_task-test_run-001_eeg.vmrk",
        "sub-001_ses-001_task-test_run-001_eeg.eeg",
        "sub-001_ses-001_task-test_run-001_eeg.json",
    ]
    assert asserting_path.is_dir()
    for filename in eeg_filenames:
        eeg_path = asserting_path.joinpath(filename)
        assert eeg_path.exists()


def test_method_create_eeg_dataset_annotations(testing_path: Path) -> None:
    """Test that the method creates an EEG dataset with annotations."""
    dataset = script.DummyDataset(root=testing_path)
    kwargs: dict[str, int | list | dict] = {
        "duration": 10,
        "events_kwargs": dict(name="testing_event", number=3, start=2, stop=8),
    }
    dataset.create_eeg_dataset(fmt="eeglab", light=False, **kwargs)

    for content in testing_path.iterdir():
        if "temporary_directory_generated_" in content.name:
            temporary_directory = content
            break

    testing_eeg_name = "sub-001_ses-001_task-test_run-001_eeg.set"
    filename = temporary_directory.joinpath(
        "RAW", "sub-001", "ses-001", "eeg", testing_eeg_name
    )
    raw = mne.io.read_raw_eeglab(filename)
    annotations = raw.annotations
    assert len(annotations.onset) == 3
    assert annotations.description[0] == "testing_event"


def test_populate_label(testing_path: Path) -> None:
    """Test that the method populates the labels."""
    dataset = script.DummyDataset(
        n_subjects=2, n_sessions=3, n_runs=4, root=testing_path
    )
    dataset._populate_labels()
    asserting_subject = ["sub-001", "sub-002"]
    asserting_session = ["ses-001", "ses-002", "ses-003"]
    asserting_run = ["run-001", "run-002", "run-003", "run-004"]
    assertion_list = [asserting_subject, asserting_session, asserting_run]
    attributes_list = ["subjects", "sessions", "runs"]
    for attribute, assertion in zip(attributes_list, assertion_list):
        attribute_values = getattr(dataset, attribute)
        print(attribute_values)
        for i, asserting_label in enumerate(assertion):
            assert attribute_values[i] == asserting_label
