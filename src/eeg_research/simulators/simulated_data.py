"""Simulate EEG data for testing purposes."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar, Union

import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from mne import create_info
from mne.io import RawArray

from eeg_research.simulators.path_handler import DirectoryTree

# TODO:
#   - refactor the eeg dataset generation with the newly populate labels method
#   - add the simulation of:
#       - EOG
#       - gradient artifacts
#       - BCG artifacts

FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])


def simulate_light_eeg_data(
    n_channels: int = 16,
    duration: int = 2,
    sampling_frequency: int = 256,
) -> RawArray:
    """Simulate EEG data that have low impact on memory.

    When events and realistic EEG data are not needed, this function
    provides a light version of the simulate_eeg_data function.

    Args:
        n_channels (int): The number of EEG channels.
        duration (int): The duration of the EEG data in seconds.
        sampling_frequency (int): The sampling frequency of the EEG data.

    Returns:
        RawArray: The simulated EEG data.
    """
    if n_channels <= 0:
        raise ValueError("The number of channels must be greater than 0.")

    if duration <= 0:
        raise ValueError("The duration must be greater than 0.")

    eeg_data = np.random.randn(n_channels, duration * sampling_frequency)
    channel_names = [str(i) for i in range(n_channels)]
    info = create_info(channel_names, sampling_frequency, ch_types="eeg")
    raw = RawArray(eeg_data, info)

    return raw


def simulate_eeg_data(
    n_channels: int = 16,
    duration: int = 2,
    misc_channels: list = ["ecg"],
    sampling_frequency: int = 256,
    events_kwargs: dict = dict(name="R128", number=1, start=1, stop=5),
) -> RawArray:
    """Simulate EEG data.

    This function generates simulated EEG data.

    Args:
        n_channels (int, optional): The number of EEG channels.
            Defaults to 16.
        duration (int, optional): The duration of the EEG data in seconds.
            Defaults to 2.
        misc_channels (list, optional): The list of miscellaneous channels to simulate.
            Defaults to ["ecg"].
        sampling_frequency (int, optional): The sampling frequency of the EEG data.
            Defaults to 256.
        events_kwargs (dict, optional): The parameters to generate events in the EEG
            data. Defaults to dict(name="R128", number=1, start=1, stop=5).

    Returns:
        RawArray: The simulated EEG data.
    """
    if n_channels <= 0:
        raise ValueError("The number of channels must be greater than 0.")

    if duration <= 0:
        raise ValueError("The duration must be greater than 0.")

    eeg_data = np.zeros((n_channels, duration * sampling_frequency))
    for channel in range(n_channels):
        # !!!!!!
        # Careful it is sensitive to the duration. Somewhat it doesn't have
        # the same shape as the premade eeg_data array so numpy throws an error
        # saying it couldn't broadcast the array.
        # I will need to take care of that
        eeg_data[channel, :] = nk.eeg_simulate(
            duration=duration, sampling_rate=sampling_frequency, noise=0.1
        )

    channel_names = [str(i) for i in range(n_channels)]
    montage = mne.channels.make_standard_montage("biosemi16")
    ch_names = montage.ch_names
    channel_mapping = {str(i): ch_name for i, ch_name in enumerate(ch_names)}

    if misc_channels:
        misc_channels_object_list = list()
        if "ecg" in misc_channels:
            ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_frequency)

            eeg_data[ch_names.index("T8"), :] *= (ecg * 2) * 1e-6
            eeg_data[ch_names.index("T7"), :] *= -(ecg * 2) * 1e-6
            ecg = np.expand_dims(ecg, axis=0)
            raw_ecg = RawArray(
                ecg, create_info(["ecg"], sampling_frequency, ch_types="ecg")
            )
            misc_channels_object_list.append(raw_ecg)

        if "emg" in misc_channels:
            emg = nk.emg_simulate(duration=duration, sampling_rate=sampling_frequency)
            emg = np.expand_dims(emg, axis=0)
            raw_emg = RawArray(
                emg, create_info(["emg"], sampling_frequency, ch_types="emg")
            )
            misc_channels_object_list.append(raw_emg)

    info = create_info(channel_names, sampling_frequency, ch_types="eeg")
    raw = RawArray(eeg_data, info)
    raw.rename_channels(channel_mapping)
    raw.set_montage(montage, on_missing="ignore")
    if misc_channels:
        raw.add_channels(misc_channels_object_list)

    if events_kwargs:
        events_index = np.linspace(
            events_kwargs["start"] * sampling_frequency,
            events_kwargs["stop"] * sampling_frequency,
            num=events_kwargs["number"],
            endpoint=False,
        )
        print(len(events_index))
        events_name = [events_kwargs["name"]] * events_kwargs["number"]
        annotations = mne.Annotations(
            onset=events_index / sampling_frequency, duration=0, description=events_name
        )
        raw.set_annotations(annotations)

    return raw


class DummyDataset:
    """A class to create a dummy BIDS dataset for EEG data.

    This class creates a dummy BIDS dataset in order to test pipelines.
    The dataset is generated in the temporary folder of the computer.
    Once tests are done on the dataset, it is possible to remove it from memory
    using the flush method.
    """

    def __init__(
        self,
        n_subjects: int = 1,
        n_sessions: int = 1,
        n_runs: int = 1,
        task: Optional[str] = "test",
        sessions_label_str: Optional[str] = None,
        subjects_label_str: Optional[str] = None,
        data_folder: str = "RAW",
        root: Optional[Union[str, os.PathLike]] = None,
        flush: bool = True,
    ) -> None:
        """Initialize the DummyDataset object.

        Args:
            n_subjects (int, optional): The number of subjects to simulate.
                Defaults to 1.
            n_sessions (int, optional): The number of sessions to simulate.
                Defaults to 1.
            n_runs (int, optional): The number of runs to simulate.
                Defaults to 1.
            task (str, optional): The task to simulate. Defaults to 'test'.
            sessions_label_str (str, optional): The string identifier to add to
                the session label. Defaults to None.
            subjects_label_str (str, optional): The string identifier to add to
                the subject label. Defaults to None.
            data_folder (str, optional): The location of the data
                source (rawdata, derivatives). Defaults to "RAW".
            root (str | os.PathLike, optional): The root directory to create
                the temporary dataset. If None, the dataset is created in the
                temporary directory of the system. Defaults to None.
            flush (bool, optional): Whether to remove the temporary directory
                when the object is deleted. Defaults to True.
        """
        arguments_to_check = [n_subjects, n_sessions, n_runs]
        arguments_name = ["subjects", "sessions", "runs"]
        print(arguments_name)
        conditions = [
            not isinstance(argument, int) or argument < 1
            for argument in arguments_to_check
        ]
        if any(conditions):
            error_message = "The number of "
            if sum(conditions) == 1:
                error_message += arguments_name[conditions.index(True)]
            else:
                error_message += " and ".join(
                    [
                        arguments_name[i]
                        for i, condition in enumerate(conditions)
                        if condition
                    ]
                )
            error_message += " must be an integer greater than 0."
            raise ValueError(error_message)

        self.n_subjects = n_subjects
        self.n_sessions = n_sessions
        self.n_runs = n_runs
        self.data_folder = data_folder
        self.sessions_label_str = sessions_label_str
        self.subjects_label_str = subjects_label_str
        self.task = task
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="temporary_directory_generated_",
            dir=root,
            ignore_cleanup_errors=False,
            delete=flush,
        )
        self.root = Path(self.temporary_directory.name)
        self.bids_path = self.root.joinpath(self.data_folder)

    def _add_participant_metadata(
        self, participant_id: str, age: int, sex: str, handedness: str
    ) -> None:
        """Add participant metadata to the dataset.

        Args:
            participant_id (str): The participant ID.
            age (int): The age of the participant.
            sex (str): The sex of the participant.
            handedness (str): The handedness of the participant.
        """
        if not hasattr(self, "participant_metadata"):
            self._create_participants_metadata()

        temp_df = pd.DataFrame(
            {
                "participant_id": participant_id,
                "age": age,
                "sex": sex,
                "handedness": handedness,
            },
            index=[0],
        )

        self.participant_metadata = pd.concat(
            [self.participant_metadata, temp_df], ignore_index=True
        )

    def _populate_labels(self) -> "DummyDataset":
        """Populate the labels for the dataset.

        Returns:
            DummyDataset: The DummyDataset object.
        """
        labels_and_attributes_mapping = {
            "subjects": self.n_subjects,
            "sessions": self.n_sessions,
            "runs": self.n_runs,
        }
        for lab_type, lab_tot_number in labels_and_attributes_mapping.items():
            label_list = list()
            for number in range(1, lab_tot_number + 1):
                label_list.append(self._generate_label(lab_type, number))
            setattr(self, lab_type, label_list)

        return self

    def create_participants_metadata(self) -> "DummyDataset":
        """Create participant metadata for the dataset.

        Returns:
            DummyDataset: The DummyDataset object.
        """
        holder = {"participant_id": [], "sex": [], "age": [], "handedness": []}
        for subject_number in range(1, self.n_subjects + 1):
            holder["age"].append(np.random.randint(18, 60))
            holder["sex"].append(np.random.choice(["M", "F"]))
            holder["handedness"].append(
                np.random.choice(["right", "left", "ambidextrous"])
            )
            holder["participant_id"].append(
                self._generate_label(
                    "subjects",
                    label_number=subject_number,
                    label_str_id=self.subjects_label_str,
                )
            )

        self.participant_metadata = pd.DataFrame(holder)
        self.subjects = self.participant_metadata["participant_id"].tolist()
        return self

    def _save_participant_metadata(self) -> None:
        """Save the participant metadata to a file."""
        saving_filename = self.bids_path.joinpath("participants.tsv")
        self.participant_metadata.to_csv(saving_filename, sep="\t", index=False)

    def _generate_label(
        self: "DummyDataset",
        label_type: str = "subjects",
        label_number: int = 1,
        label_str_id: Optional[str] = None,
    ) -> str:
        """Generate a BIDS compliant label.

        The BIDS standard requires a specific format for the labels of the
        subject, session, and run folders. This method generates the label
        based on the label_type, label_number, and label_str_id parameters.

        Args:
            label_type (str, optional): The type of label to generate. It can be
                'subject', 'session', or 'run'. Defaults to 'subject'.
            label_number (int, optional): The number of the label. Defaults to 1.
            label_str_id (str, optional): The string identifier to add to the label.
                Defaults to None.

        Returns:
            str: The generated label.
        """
        label_prefix = label_type[:3] + "-"
        if not label_str_id:
            label_str_id = ""
        label = f"{label_prefix}{label_str_id}{label_number:03d}"
        return label

    def create_modality_agnostic_dir(self: "DummyDataset") -> List[Path]:
        """Create multiple BIDS compliant folders.

        The BIDS structure requires the structure to be an iterative
        succession of subject/session folders. This method creates the
        necessary folders (subject or session defined in 'folder_type')
        and returns the last one created.

        Returns:
            List[Path]: The paths of the created folders.
        """
        path_list = list()

        for subject_number in range(1, self.n_subjects + 1):
            subject_folder_label = self._generate_label(
                label_type="subjects",
                label_number=subject_number,
                label_str_id=self.subjects_label_str,
            )

            for session_number in range(1, self.n_sessions + 1):
                session_folder_label = self._generate_label(
                    label_type="sessions",
                    label_number=session_number,
                    label_str_id=self.sessions_label_str,
                )
                path = self.bids_path.joinpath(
                    subject_folder_label, session_folder_label
                )
                path_list.append(path)
                path.mkdir(parents=True, exist_ok=True)

        return path_list

    def _extract_entities_from_path(self, path: Union[str, os.PathLike]) -> str:
        """Extract the entities from a path.

        Args:
            path (str | os.PathLike): The path to extract the label from.

        Returns:
            str: The extracted label.
        """
        path = Path(path)
        parts = path.parts
        entities = {
            "subject": [part for part in parts if "sub-" in part][0],
            "session": [part for part in parts if "ses-" in part][0],
        }

        return entities

    def _create_sidecar_json(self, eeg_filename: Union[str, os.PathLike]) -> None:
        """Create a sidecar JSON file for the EEG data.

        Args:
            eeg_filename (str | os.PathLike): The EEG data file name.
        """
        json_filename = Path(os.path.splitext(eeg_filename)[0])
        json_filename = json_filename.with_suffix(".json")

        json_content = {
            "SamplingFrequency": 2400,
            "Manufacturer": "Brain Products",
            "ManufacturersModelName": "BrainAmp DC",
            "CapManufacturer": "EasyCap",
            "CapManufacturersModelName": "M1-ext",
            "PowerLineFrequency": 50,
            "EEGReference": "single electrode placed on FCz",
            "EEGGround": "placed on AFz",
            "SoftwareFilters": {
                "Anti-aliasing filter": {
                    "half-amplitude cutoff (Hz)": 500,
                    "Roll-off": "6dB/Octave",
                }
            },
            "HardwareFilters": {
                "ADC's decimation filter (hardware bandwidth limit)": {
                    "-3dB cutoff point (Hz)": 480,
                    "Filter order sinc response": 5,
                }
            },
        }

        with open(json_filename, "w") as json_file:
            json.dump(json_content, json_file, indent=4)

    def _create_dataset_description(self) -> None:
        """Create the dataset_description.json file."""
        self.dataset_description = {
            "Name": "THIS IS A DUMMY DATASET",
            "BIDSVersion": "1.9.0",
            "License": "CC0",
            "Authors": ["Jane Doe", "John Doe"],
        }

        with open(
            os.path.join(self.bids_path, "dataset_description.json"), "w"
        ) as desc_file:
            json.dump(self.dataset_description, desc_file, indent=4)

    def flush(self, check: bool = True) -> None:
        """Remove the temporary directory from memory.

        Args:
            check (bool, optional): Whether to check the directory before removal.
                Defaults to True.
        """
        tree = DirectoryTree(self.root)
        if check:
            print("The following directory will be removed:")
            tree.print_tree()
        else:
            print("Removing the temporary directory...")
            print("Content being removed:")
            tree.print_tree()

            shutil.rmtree(self.root, ignore_errors=True, onerror=None)
            post_removal_checker = os.path.exists(self.root)
            if post_removal_checker:
                print("The tree was not removed.")
            else:
                print("The tree was successfully removed.")

    def create_eeg_dataset(
        self, fmt: str = "brainvision", light: bool = False, **kwargs: dict
    ) -> str:
        """Create temporary BIDS dataset.

        Create a dummy BIDS dataset for EEG data with multiple subjects, sessions,
        and runs.

        Args:
            fmt (str, optional): The format of the EEG data to simulate.
                Defaults to 'brainvision'.
            light (bool, optional): Whether to simulate light EEG data.
                Defaults to False.
            **kwargs (dict): The parameters to pass to the EEG data simulation function.

        Returns:
            str: The path of the temporary BIDS dataset.
        """
        path_list = self.create_modality_agnostic_dir()
        self._create_dataset_description()
        self.create_participants_metadata()

        for path in path_list:
            for run_number in range(1, self.n_runs + 1):
                run_label = self._generate_label("runs", run_number)
                eeg_directory = path.joinpath("eeg")
                eeg_directory.mkdir(parents=True, exist_ok=True)

                # Define file names for EEG data files
                if fmt == "brainvision":
                    extension = ".vhdr"
                elif fmt == "edf":
                    extension = ".edf"
                elif fmt == "eeglab":
                    extension = ".set"
                elif fmt == "fif":
                    extension = ".fif"

                entities = self._extract_entities_from_path(path)

                eeg_filename = "_".join(
                    [
                        entities["subject"],
                        entities["session"],
                        f"task-{self.task}",
                        run_label,
                        "eeg",
                    ]
                )

                eeg_filename += extension
                eeg_absolute_filename = eeg_directory.joinpath(eeg_filename)

                if light:
                    raw = simulate_light_eeg_data(**kwargs)
                else:
                    raw = simulate_eeg_data(**kwargs)

                mne.export.export_raw(
                    fname=eeg_absolute_filename, raw=raw, fmt=fmt, overwrite=True
                )

                # Create sidecar JSON file
                self._create_sidecar_json(eeg_absolute_filename)

        self._save_participant_metadata()
        print(f"Temporary BIDS EEG dataset created at {self.bids_path}")
        self.print_bids_tree()
        return self

    def print_bids_tree(self) -> None:
        """Print the BIDS dataset tree."""
        tree = DirectoryTree(self.bids_path)
        tree.print_tree()
