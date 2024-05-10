"""Tests for eeg_research.cli.toolbox.parser.py."""

import sys
from unittest.mock import MagicMock

import pytest
from bids import BIDSLayout

import eeg_research.cli.tools.parser as script


def run_parse_arguments_test(argv: list[str], expected: dict) -> None:
    """Helper function to run parse_arguments tests."""
    sys.argv = argv
    args = script.parse_arguments()
    for key, value in expected.items():
        assert getattr(args, key) == value


def run_test(option: str, value: str, expected: dict) -> None:
    """Helper function to run tests for an option with a given value."""
    run_parse_arguments_test(
        ["prog", "--root", "/root/path", f"--{option}", value, "--gradient"],
        expected,
    )


def test_parse_arguments_root_required() -> None:
    """Test that the root argument is required."""
    with pytest.raises(SystemExit):
        run_parse_arguments_test(["prog"], {})


def test_parse_arguments_default_values() -> None:
    """Test that the default values are set correctly."""
    run_parse_arguments_test(
        ["prog", "--root", "/root/path", "--gradient"],
        {
            "root": "/root/path",
            "datafolder": None,
            "subject": None,
            "session": None,
            "run": None,
            "task": None,
            "extension": None,
            "datatype": "eeg",
            "suffix": "eeg",
            "description": None,
            "interactive": False,
            "gradient": True,
            "bcg": False,
            "qc": False,
        },
    )


def test_parse_arguments_all_values() -> None:
    """Test that all values are parsed correctly."""
    run_parse_arguments_test(
        [
            "prog",
            "--root",
            "/root/path",
            "--datafolder",
            "source",
            "--subject",
            "1",
            "--session",
            "1",
            "--run",
            "1",
            "--task",
            "a",
            "--extension",
            "a",
            "--datatype",
            "a",
            "--suffix",
            "a",
            "--description",
            "desc",
            "--interactive",
            "--gradient",
            "--bcg",
            "--qc",
        ],
        {
            "root": "/root/path",
            "datafolder": "source",
            "subject": "1",
            "session": "1",
            "run": "1",
            "task": "a",
            "extension": "a",
            "datatype": "a",
            "suffix": "a",
            "description": "desc",
            "interactive": True,
            "gradient": True,
            "bcg": True,
            "qc": True,
        },
    )


def test_parse_arguments_no_flags() -> None:
    """Test case for the parse_arguments function when no flags are provided."""
    with pytest.raises(SystemExit):
        run_parse_arguments_test(["prog", "--root", "/root/path"], {})


def test_parse_arguments_options() -> None:
    """Test that the options are parsed correctly."""
    options = ["subject", "session", "run", "task", "extension", "datatype", "suffix"]
    for option in options:
        run_test(option, "*", {option: "*"})
        if option not in ["task", "extension", "datatype", "suffix"]:
            run_test(option, "1-3", {option: "1-3"})
            run_test(option, "1-*", {option: "1-*"})
            run_test(option, "*-3", {option: "*-3"})
        if option in ["task", "extension", "datatype", "suffix"]:
            run_test(option, "a", {option: "a"})


def test_parse_arguments_description() -> None:
    """Test that the description is parsed correctly."""
    run_parse_arguments_test(
        ["prog", "--root", "/root/path", "--description", "desc", "--gradient"],
        {"description": "desc"},
    )


def test_parse_arguments_flags() -> None:
    """Test that the flags are parsed correctly."""
    flags = ["interactive", "gradient", "bcg", "qc"]
    for flag in flags:
        run_parse_arguments_test(
            ["prog", "--root", "/root/path", f"--{flag}"], {flag: True}
        )


@pytest.fixture
def mock_layout() -> BIDSLayout:
    """Fixture to create a mock BIDSLayout."""
    layout = MagicMock(spec=BIDSLayout)
    layout.get = MagicMock()
    return layout


def test_parse_range_arg_wildcard(mock_layout: BIDSLayout) -> None:
    """Test parse_range_arg function with wildcard value."""
    mock_layout.get.return_value = ["1", "2", "3"]
    assert script.parse_range_arguments(mock_layout, "entity", "*") == ["1", "2", "3"]


def test_parse_range_arg_range(mock_layout: BIDSLayout) -> None:
    """Test parse_range_arg function with range value."""
    mock_layout.get.return_value = ["1", "2", "3", "4", "5"]
    assert script.parse_range_arguments(mock_layout, "entity", "2-4") == ["2", "3", "4"]


def test_parse_range_arg_non_range(mock_layout: BIDSLayout) -> None:
    """Test parse_range_arg function with non-range value."""
    assert script.parse_range_arguments(mock_layout, "entity", "value") == "value"


def test_parse_range_arg_invalid_range(mock_layout: BIDSLayout) -> None:
    """Test parse_range_arg function with invalid range value."""
    mock_layout.get.return_value = ["1", "2", "3", "4", "5"]
    with pytest.raises(ValueError):
        script.parse_range_arguments(mock_layout, "entity", "5-2")


def test_parse_range_arg_non_integer_ids(mock_layout: BIDSLayout) -> None:
    """Test parse_range_arg function with non-integer ids."""
    mock_layout.get.return_value = ["a", "b", "c"]
    with pytest.raises(ValueError):
        script.parse_range_arguments(mock_layout, "entity", "1-2")


def test_parse_range_arg_no_ids_in_range(mock_layout: BIDSLayout) -> None:
    """Test parse_range_arg function with no ids in range."""
    mock_layout.get.return_value = ["1", "2", "3", "4", "5"]
    with pytest.raises(ValueError):
        script.parse_range_arguments(mock_layout, "entity", "6-7")
