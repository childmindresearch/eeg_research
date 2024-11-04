"""Create a temporary BIDS dataset to test data handling scripts."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Any

import mne
import numpy as np
import pandas as pd

from eeg_research.simulators.eeg_simulator import (
    simulate_eeg_data,
    simulate_light_eeg_data,
)


class DirectoryTree:
    """The class to handle the directory tree."""

    def __init__(self, root_dir: str | os.PathLike) -> None:
        """Initialize the class with the root directory."""
        self.root_dir = os.path.abspath(root_dir)
        self.current_dir = self.root_dir

    def generate_tree(
        self: "DirectoryTree",
    ) -> Generator[tuple[str, list[str], list[str]], None, None]:
        """A generator method to walk through the directory tree.

        It starts from the root directory. Yields tuples of (path, dirs, files)
        for each directory in the tree.
        """
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            yield dirpath, dirnames, filenames

    def list_directory_contents(
        self: "DirectoryTree",
    ) -> tuple[list[str] | str, list[str]]:
        """List the contents of the current directory."""
        try:
            with os.scandir(self.current_dir) as it:
                dirs = [entry.name for entry in it if entry.is_dir()]
                files = [entry.name for entry in it if entry.is_file()]
            return dirs, files
        except PermissionError as e:
            return f"Permission denied: {e}", []

    def change_directory(self, target_dir: str | os.PathLike) -> str | os.PathLike:
        """Change the current directory to target_dir.

        Change the current directory if it's a subdirectory
        of current_dir or a valid path.

        Args:
            target_dir (str | os.PathLike): The target directory path.
        """
        new_dir = os.path.abspath(os.path.join(self.current_dir, target_dir))
        if os.path.commonpath([self.root_dir, new_dir]) != self.root_dir:
            return f"Error: {new_dir} is not a subdirectory of the root directory."
        if os.path.isdir(new_dir):
            self.current_dir = new_dir
            return f"Changed current directory to {self.current_dir}"
        else:
            return f"Error: Directory {new_dir} does not exist."

    def print_tree(
        self: "DirectoryTree",
        start_path: str | os.PathLike = "",
        prefix: str | os.PathLike = "",
    ) -> None:
        """Print the tree in a formated way.

        Recursively prints the directory tree starting f
        rom start_path with proper formatting.

        Args:
            start_path (str | os.PathLike, optional): The starting path.
                                                      Defaults to ''.
            prefix (str | os.PathLike, optional): The prefix to be added.
                                                  Defaults to ''.
        """
        if start_path == "":
            start_path = self.root_dir
            print(start_path + "/")

        entries = [entry for entry in os.scandir(str(start_path))]
        entries = sorted(entries, key=lambda e: (e.is_file(), e.name))
        last_index = len(entries) - 1

        for index, entry in enumerate(entries):
            connector = "├──" if index != last_index else "└──"
            if entry.is_dir():
                print(f"{prefix}{connector} {entry.name}/")
                extension = "│   " if index != last_index else "    "
                self.print_tree(
                    os.path.join(start_path, entry.name), str(prefix) + str(extension)
                )
            else:
                print(f"{prefix}{connector} {entry.name}")

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
        task: str = "test",
        sessions_label_str: str | None = None,
        subjects_label_str: str | None = None,
        data_folder: str = "RAW",
        root: str | Path | None = None,
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
            root (str | Path, optional): The root directory to create
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

    def _create_participant_metadata(self) -> "DummyDataset":
        """Create participant metadata for the dataset.

        Returns:
            DummyDataset: The DummyDataset object.
        """
        holder: dict[str, list[Any]] = {
            "participant_id": [],
            "sex": [],
            "age": [],
            "handedness": [],
        }
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
            self._create_participant_metadata()

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

    def _save_participant_metadata(self) -> None:
        """Save the participant metadata to a file."""
        saving_filename = self.bids_path.joinpath("participants.tsv")
        self.participant_metadata.to_csv(saving_filename, sep="\t", index=False)

    def _generate_label(
        self: "DummyDataset",
        label_type: str = "subjects",
        label_number: int = 1,
        label_str_id: str | None = None,
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

    def create_modality_agnostic_dir(self: "DummyDataset") -> list[Path]:
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

    def _extract_entities_from_path(self, path: str | Path) -> dict[str, str]:
        """Extract the entities from a path.

        Args:
            path (str | Path): The path to extract the label from.

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

    def _create_sidecar_json(self, eeg_filename: str | Path) -> None:
        """Create a sidecar JSON file for the EEG data.

        Args:
            eeg_filename (str | Path): The EEG data file name.
        """
        json_filename = Path(eeg_filename).with_suffix("")
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

        with open(self.bids_path / "dataset_description.json", "w") as desc_file:
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

            shutil.rmtree(str(self.root))
            post_removal_checker = self.root.exists()
            if post_removal_checker:
                print("The tree was not removed.")
            else:
                print("The tree was successfully removed.")

    def create_eeg_dataset(
        self, fmt: str = "brainvision", light: bool = False, **kwargs: int | list | dict
    ) -> "DummyDataset":
        """Create temporary BIDS dataset.

        Create a dummy BIDS dataset for EEG data with multiple subjects, sessions,
        and runs.

        Args:
            fmt (str, optional): The format of the EEG data to simulate.
                Defaults to 'brainvision'.
            light (bool, optional): Whether to simulate light EEG data.
                Defaults to False.
            kwargs (int | list | dict): The parameters to pass to the EEG data

        Returns:
            DummyDataset: : The temporary DummyDataset object.
        """
        path_list = self.create_modality_agnostic_dir()
        self._create_dataset_description()
        self._create_participant_metadata()

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
                    raw = simulate_light_eeg_data(**kwargs)  # type: ignore
                else:
                    raw = simulate_eeg_data(**kwargs)  # type: ignore

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