"""Tests for eeg_research.cli.toolbox.bids_parser.py.

TODO: Add tests for _set_layout, _set_entities and update_layout.
"""

import sys
from pathlib import Path

import pytest
from pyparsing import Any

import eeg_research.cli.tools.bids_parser as script


def test_bids_parser_init(mocker: Any) -> None:
    """Test the initialization of the BIDSParser class."""
    mock_parse_arguments = mocker.patch.object(script.BIDSParser, "_parse_arguments")
    mock_set_reading_root = mocker.patch.object(script.BIDSParser, "_set_reading_root")
    mock_set_layout = mocker.patch.object(script.BIDSParser, "_set_layout")
    mock_set_entities = mocker.patch.object(script.BIDSParser, "_set_entities")

    _ = script.BIDSParser()

    assert mock_parse_arguments.called
    assert mock_set_reading_root.called
    assert mock_set_layout.called
    assert mock_set_entities.called


def test_bids_parser_set_reading_root(mocker: Any) -> None:
    """Test the _set_reading_root method of the BIDSParser class."""
    args = mocker.MagicMock()
    args.root = "data/bids-examples/eeg_ds000117"
    args.datafolder = ""

    mocker.patch.object(script.BIDSParser, "_parse_arguments", return_value=args)
    mocker.patch("bids.BIDSLayout", return_value=mocker.MagicMock())

    parser = script.BIDSParser()

    result = parser._set_reading_root()

    assert result == Path("data/bids-examples/eeg_ds000117/")


def run_bids_parser_parse_arguments_test(argv: list[str], expected: dict) -> None:
    """Helper function to run parse_arguments tests."""
    sys.argv = argv
    parser = script.BIDSParser()
    args = parser._parse_arguments()
    for key, value in expected.items():
        assert getattr(args, key) == value


def run_test(option: str, value: str, expected: dict) -> None:
    """Helper function to run tests for an option with a given value."""
    run_bids_parser_parse_arguments_test(
        [
            "prog",
            "--root",
            "data/bids-examples/eeg_ds000117",
            f"--{option}",
            value,
            "--gradient",
        ],
        expected,
    )


def test_bids_parser_parse_arguments_root_required() -> None:
    """Test that the root argument is required."""
    with pytest.raises(SystemExit):
        run_bids_parser_parse_arguments_test(["prog"], {})


def test_bids_parser_parse_arguments_default_values() -> None:
    """Test that the default values are set correctly."""
    run_bids_parser_parse_arguments_test(
        ["prog", "--root", "data/bids-examples/eeg_ds000117", "--gradient"],
        {
            "root": "data/bids-examples/eeg_ds000117",
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


def test_bids_parser_parse_arguments_all_values() -> None:
    """Test that all values are parsed correctly."""
    run_bids_parser_parse_arguments_test(
        [
            "prog",
            "--root",
            "data/bids-examples/eeg_rest_fmri",
            "--datafolder",
            "derivatives",
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
            "root": "data/bids-examples/eeg_rest_fmri",
            "datafolder": "derivatives",
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


def test_bids_parser_parse_arguments_no_flags() -> None:
    """Test case for the parse_arguments function when no flags are provided."""
    with pytest.raises(SystemExit):
        run_bids_parser_parse_arguments_test(
            ["prog", "--root", "data/bids-examples/eeg_ds000117"], {}
        )


def test_bids_parser_parse_arguments_options() -> None:
    """Test that the options are parsed correctly."""
    options = ["subject", "run", "task", "extension", "datatype", "suffix"]
    for option in options:
        run_test(option, "*", {option: "*"})
        if option not in ["task", "extension", "datatype", "suffix"]:
            run_test(option, "1-3", {option: "1-3"})
            run_test(option, "1-*", {option: "1-*"})
            run_test(option, "*-3", {option: "*-3"})
        if option == "task":
            run_test(option, "facerecognition", {option: "facerecognition"})
        if option == "extension":
            run_test(option, "set", {option: "set"})
        if option in ["datatype", "suffix"]:
            run_test(option, "eeg", {option: "eeg"})


def test_bids_parser_parse_arguments_description() -> None:
    """Test that the description is parsed correctly."""
    run_bids_parser_parse_arguments_test(
        [
            "prog",
            "--root",
            "data/bids-examples/eeg_ds000117",
            "--description",
            "desc",
            "--gradient",
        ],
        {"description": "desc"},
    )


def test_bids_parser_parse_arguments_flags() -> None:
    """Test that the flags are parsed correctly."""
    flags = ["interactive", "gradient", "bcg", "qc"]
    for flag in flags:
        run_bids_parser_parse_arguments_test(
            ["prog", "--root", "data/bids-examples/eeg_ds000117", f"--{flag}"],
            {flag: True},
        )


@pytest.fixture
def mock_parser(mocker: Any) -> script.BIDSParser:
    """Fixture to create a mock BIDSLayout."""
    args = mocker.MagicMock()
    args.root = "data/bids-examples/eeg_ds000117"
    args.datafolder = ""
    mocker.patch.object(script.BIDSParser, "_parse_arguments", return_value=args)
    mocker.patch("bids.BIDSLayout", return_value=mocker.MagicMock())
    parser = script.BIDSParser()
    parser.layout = mocker.MagicMock()
    return parser


def test_bids_parser_parse_range_args_wildcard(mock_parser: script.BIDSParser) -> None:
    """Test parse_range_arg function with wildcard value."""
    mock_parser.layout.get.return_value = ["1", "2", "3"]
    assert mock_parser._parse_range_args("entity", "*") == ["1", "2", "3"]


def test_bids_parser_parse_range_args_range(mock_parser: script.BIDSParser) -> None:
    """Test parse_range_arg function with range value."""
    mock_parser.layout.get.return_value = ["1", "2", "3", "4", "5"]
    assert mock_parser._parse_range_args("entity", "2-4") == ["2", "3", "4"]


def test_bids_parser_parse_range_args_non_range(mock_parser: script.BIDSParser) -> None:
    """Test parse_range_arg function with non-range value."""
    assert mock_parser._parse_range_args("entity", "value") == "value"


def test_bids_parser_parse_range_args_invalid_range(
    mock_parser: script.BIDSParser,
) -> None:
    """Test parse_range_arg function with invalid range value."""
    mock_parser.layout.get.return_value = ["1", "2", "3", "4", "5"]
    with pytest.raises(ValueError):
        mock_parser._parse_range_args("entity", "5-2")


def test_bids_parser_parse_range_args_non_integer_ids(
    mock_parser: script.BIDSParser,
) -> None:
    """Test parse_range_arg function with non-integer ids."""
    mock_parser.layout.get.return_value = ["a", "b", "c"]
    with pytest.raises(ValueError):
        mock_parser._parse_range_args("entity", "1-2")


def test_bids_parser_parse_range_args_no_ids_in_range(
    mock_parser: script.BIDSParser,
) -> None:
    """Test parse_range_arg function with no ids in range."""
    mock_parser.layout.get.return_value = ["1", "2", "3", "4", "5"]
    with pytest.raises(ValueError):
        mock_parser._parse_range_args("entity", "6-7")
