"""Tests for eeg_research.cli.pipelines.eeg_fmri_cleaning.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

import eeg_research.cli.pipelines.eeg_fmri_cleaning as script


def test_main_no_datafolder() -> None:
    """Test the main function when no datafolder is provided."""
    with (
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.parse_arguments"
        ) as mock_parse_arguments,
        patch("bids.BIDSLayout") as mock_BIDSLayout,
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.create_entities"
        ) as mock_create_entities,
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.select_scripts_interactively"
        ) as mock_select_scripts_interactively,
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.run_interactive_menu"
        ) as mock_run_interactive_menu,
    ):
        mock_parse_arguments.return_value = MagicMock(
            datafolder=None,
            interactive=True,
            gradient=False,
            bcg=False,
            qc=False,
            root="/root/path",
        )
        mock_BIDSLayout.return_value.get.return_value = ["file1", "file2"]
        mock_create_entities.return_value = {}
        mock_select_scripts_interactively.return_value = ["gradient_cleaning.py"]
        mock_run_interactive_menu.return_value = ["file1", "file2"]

        script.main()

        mock_BIDSLayout.assert_called_once_with("/root/path", validate=True)
        mock_create_entities.assert_called_once()
        mock_select_scripts_interactively.assert_called_once()
        mock_run_interactive_menu.assert_called_once_with(
            "/root/path", mock_BIDSLayout.return_value, {}
        )


def test_main_with_datafolder() -> None:
    """Test the main function when a datafolder is provided."""
    with (
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.parse_arguments"
        ) as mock_parse_arguments,
        patch("bids.BIDSLayout") as mock_BIDSLayout,
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.create_entities"
        ) as mock_create_entities,
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.select_scripts_interactively"
        ) as mock_select_scripts_interactively,
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.run_interactive_menu"
        ) as mock_run_interactive_menu,
    ):
        mock_parse_arguments.return_value = MagicMock(
            datafolder="derivatives",
            interactive=True,
            gradient=False,
            bcg=False,
            qc=False,
            root="/root/path",
        )
        mock_BIDSLayout.return_value.get.return_value = ["file1", "file2"]
        mock_create_entities.return_value = {}
        mock_select_scripts_interactively.return_value = ["gradient_cleaning.py"]
        mock_run_interactive_menu.return_value = ["file1", "file2"]

        script.main()

        mock_BIDSLayout.assert_called_once_with(
            os.path.join("/root/path", "derivatives"),
            validate=False,
            is_derivative=True,
        )
        mock_create_entities.assert_called_once()
        mock_select_scripts_interactively.assert_called_once()
        mock_run_interactive_menu.assert_called_once_with(
            os.path.join("/root/path", "derivatives"), mock_BIDSLayout.return_value, {}
        )


def test_main_no_files_found() -> None:
    """Test the main function when no files are found."""
    with (
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.parse_arguments"
        ) as mock_parse_arguments,
        patch("bids.BIDSLayout") as mock_BIDSLayout,
        patch(
            "eeg_research.cli.pipelines.eeg_fmri_cleaning.create_entities"
        ) as mock_create_entities,
    ):
        mock_parse_arguments.return_value = MagicMock(
            datafolder=None,
            interactive=False,
            gradient=False,
            bcg=False,
            qc=False,
            root="/root/path",
        )
        mock_BIDSLayout.return_value.get.return_value = []
        mock_create_entities.return_value = {}

        with pytest.raises(FileNotFoundError):
            script.main()

        mock_BIDSLayout.assert_called_once_with("/root/path", validate=True)
        mock_create_entities.assert_called_once()
