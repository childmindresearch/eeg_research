"""Tests for main_cleaner_pipelines.py."""

# TODO: Fix the tests

# from pathlib import Path

# import bids
# import pytest

# import eeg_research.simulators.cleaner_pipelines as cp
# import eeg_research.simulators.main_cleaner_pipelines as mcp
# import eeg_research.simulators.simulated_data as simulated_data


# @pytest.fixture(scope="class")
# def dataset() -> simulated_data.DummyDataset:
#     """Fixture to create a simulated dataset with EEG data."""
#     cwd = Path.cwd()
#     test_output_path = cwd.joinpath("data", "outputs")
#     data = simulated_data.DummyDataset(
#         n_subjects=2,
#         n_sessions=2,
#         n_runs=1,
#         task="checker",
#         root=test_output_path,
#         flush=False,
#     )

#     data._populate_labels()
#     data.create_eeg_dataset(
#         fmt="eeglab",
#         n_channels=16,
#         duration=25,
#         sampling_frequency=5000,
#         misc_channels=["ecg"],
#         events_kwargs={"name": "R128", "number": 10, "start": 5, "stop": 30},
#     )
#     return data


# class TestFunctions:
#     """Test functions in the module."""

#     def test_run_cbin_cleaner(self, dataset: simulated_data.DummyDataset) -> None:
#         """Test the run_cbin_cleaner function."""
#         layout = bids.layout.BIDSLayout(dataset.bids_path)
#         files = layout.get(extension=".set")

#         for file in files:
#             cleaner = cp.CleanerPipelines(file)
#             mcp.run_cbin_cleaner(cleaner)

#         cwd = Path.cwd()
#         test_output_path = cwd.joinpath("data", "outputs")
#         for content in test_output_path.iterdir():
#             if "temporary_directory_generated_" in content.name:
#                 temporary_directory = content
#                 break

#         files_to_check = list()
#         for subject in layout.get_subjects():
#             for session in layout.get_sessions():
#                 for run in layout.get_runs():
#                     files_to_check.append(
#                         temporary_directory.joinpath(
#                             "DERIVATIVES",
#                             "GRAD_BCG",
#                             subject,
#                             session,
#                             f"{subject}_{session}_task-checker_run-{run}_eeg.fif",
#                         )
#                     )
#         for file in files_to_check:
#             print(file)
#             assert file.exists()


# class TestMain:
#     """Test the main function."""

#     def test_main_raw_path_integrity(
#         self, dataset: simulated_data.DummyDataset
#     ) -> None:
#         """Test the integrity of the raw file paths."""
#         cwd = Path.cwd()
#         test_output_path = cwd.joinpath("data", "outputs")
#         for content in test_output_path.iterdir():
#             if "temporary_directory_generated_" in content.name:
#                 temporary_directory = content
#                 break

#         layout = bids.layout.BIDSLayout(dataset.bids_path)
#         files_to_check = list()
#         for subject in layout.get_subjects():
#             for session in layout.get_sessions():
#                 for run in layout.get_runs():
#                     files_to_check.append(
#                         temporary_directory.joinpath(
#                             "RAW",
#                             subject,
#                             session,
#                             f"{subject}_{session}_task-test_run-{run}_eeg.set",
#                         )
#                     )
#         mcp.main(dataset.bids_path)
#         for file in files_to_check:
#             print(file)
#             assert file.exists()

#     def test_main_derivatives_path_integrity(
#         self, dataset: simulated_data.DummyDataset
#     ) -> None:
#         """Test the integrity of the derivatives file paths."""
#         cwd = Path.cwd()
#         test_output_path = cwd.joinpath("data", "outputs")
#         for content in test_output_path.iterdir():
#             if "temporary_directory_generated_" in content.name:
#                 temporary_directory = content
#                 break

#         layout = bids.layout.BIDSLayout(dataset.bids_path)
#         files_to_check = list()
#         additional_folders = ["GRAD", "GRAD_BCG", "GRAD_BCG_ASR"]
#         for folder in additional_folders:
#             for subject in layout.get_subjects():
#                 for session in layout.get_sessions():
#                     for run in layout.get_runs():
#                         files_to_check.append(
#                             temporary_directory.joinpath(
#                                 "DERIVATIVES",
#                                 folder,
#                                 subject,
#                                 session,
#                                 f"{subject}_{session}_task-test_run-{run}_eeg.fif",
#                             )
#                         )
#         mcp.main(dataset.bids_path)
#         for file in files_to_check:
#             print(file)
#             assert file.exists()

#     def test_main_report_exists(self, dataset: simulated_data.DummyDataset) -> None:
#         """Test the existence of the report file."""
#         cwd = Path.cwd()
#         test_output_path = cwd.joinpath("data", "outputs")
#         for content in test_output_path.iterdir():
#             if "temporary_directory_generated_" in content.name:
#                 temporary_directory = content
#                 break

#         report_path = temporary_directory.joinpath("DERIVATIVES", "report.txt")
#         mcp.main(dataset.bids_path)
#         assert report_path.exists()
