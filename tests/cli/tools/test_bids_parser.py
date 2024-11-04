"""Tests for eeg_research.cli.toolbox.bids_parser.py."""

import sys
from pathlib import Path

import pytest
from pyparsing import Any

import eeg_research.cli.tools.bids_parser as script


@pytest.fixture
def mock_parser(mocker: Any, tmp_path: Path) -> script.BIDSCreator:
    """Fixture to create a mock BIDSLayout."""
    args = mocker.MagicMock()
    args.root = tmp_path
    args.datafolder = None
    mocker.patch("bids.BIDSLayout", return_value=mocker.MagicMock())
    parser = script.BIDSCreator(**vars(args))
    parser.layout = mocker.MagicMock()
    return parser


def run_bids_parser_parse_arguments_test(argv: list[str], expected: dict) -> None:
    """Helper function to run parse_arguments tests."""
    sys.argv = argv
    args = script.bids_args_parser()
    for key, value in expected.items():
        assert args[key] == value


def run_test(option: str, value: str, expected: dict) -> None:
    """Helper function to run tests for an option with a given value."""
    run_bids_parser_parse_arguments_test(
        [
            "prog",
            "--root",
            "bids-examples/eeg_ds000117",
            f"--{option}",
            value,
            "--gradient",
        ],
        expected,
    )


def test_bids_parser_init(mocker: Any) -> None:
    """Test the initialization of the BIDSCreator class."""
    mock_set_reading_root = mocker.patch.object(script.BIDSCreator, "_set_reading_root")
    mock_set_layout = mocker.patch.object(script.BIDSCreator, "_set_layout")
    mock_set_entities = mocker.patch.object(script.BIDSCreator, "_set_entities")

    _ = script.BIDSCreator()

    assert mock_set_reading_root.called
    assert mock_set_layout.called
    assert mock_set_entities.called


def test_bids_parser_set_reading_root(mock_parser: script.BIDSCreator) -> None:
    """Test the _set_reading_root method of the BIDSCreatorclass."""
    mock_parser.root = "mock_root"
    mock_parser.datafolder = None
    result = mock_parser._set_reading_root()
    assert result == Path("mock_root")

    mock_parser.root = "mock_root"
    mock_parser.datafolder = "test"
    result = mock_parser._set_reading_root()
    assert result == Path("mock_root/test")


@pytest.mark.parametrize(
    "argv",
    [
        ["prog"],
        ["prog", "--root", "bids-examples/eeg_ds000117"],
    ],
)
def test_bids_parser_parse_arguments_required_flags(argv: list[str]) -> None:
    """Test that the error is raised when the required flags are not provided."""
    with pytest.raises(SystemExit):
        run_bids_parser_parse_arguments_test(argv, {})


@pytest.mark.parametrize(
    "argv, expected",
    [
        (
            ["prog", "--root", "bids-examples/eeg_ds000117", "--gradient"],
            {
                "root": "bids-examples/eeg_ds000117",
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
        ),
        (
            [
                "prog",
                "--root",
                "bids-examples/eeg_rest_fmri",
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
                "root": "bids-examples/eeg_rest_fmri",
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
        ),
    ],
)
def test_bids_parser_parse_arguments_default_and_all_values(
    argv: list[str], expected: dict
) -> None:
    """Test that the default and all values are parsed correctly."""
    run_bids_parser_parse_arguments_test(argv, expected)


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


@pytest.mark.parametrize(
    "return_value, arg, expected",
    [
        (["1", "2", "3"], "*", ["1", "2", "3"]),
        (["1", "2", "3", "4", "5"], "2-4", ["2", "3", "4"]),
        (None, "value", "value"),
    ],
)
def test_bids_parser_parse_range_args_valid(
    mock_parser: script.BIDSCreator,
    return_value: list[str] | None,
    arg: str,
    expected: list[str] | str,
) -> None:
    """Test parse_range_arg function with valid values."""
    mock_parser.layout.get.return_value = return_value
    assert mock_parser._parse_range_args("entity", arg) == expected


@pytest.mark.parametrize(
    "return_value, arg",
    [
        (["1", "2", "3", "4", "5"], "5-2"),
        (["a", "b", "c"], "1-2"),
        (["1", "2", "3", "4", "5"], "6-7"),
        (["1", "2", "3", "4", "5"], "-6"),
        (["1", "2", "3", "4", "5"], "6-"),
    ],
)
def test_bids_parser_parse_range_args_invalid(
    mock_parser: script.BIDSCreator, return_value: list[str], arg: str
) -> None:
    """Test parse_range_arg function with invalid values."""
    mock_parser.layout.get.return_value = return_value
    with pytest.raises(ValueError):
        mock_parser._parse_range_args("entity", arg)


def test_bids_parser_update_layout(mock_parser: script.BIDSCreator, mocker: Any) -> None:
    """Test the update_layout method of the BIDSCreator class."""
    mock_indexer = mocker.patch(
        "bids.BIDSLayoutIndexer", return_value=mocker.MagicMock()
    )

    mock_layout = mocker.patch.object(mock_parser, "layout")
    mock_layout.get.side_effect = [["file1", "file2", "file3"], ["file2", "file3"]]

    entities: dict[str, str | None] = {"subject": "1", "session": "1", "run": "1"}

    updated_layout = mock_parser.update_layout(entities)

    mock_layout.get.assert_called_with(return_type="file", **entities)
    assert isinstance(updated_layout, mocker.MagicMock)
    mock_indexer.assert_called_with(ignore=mocker.ANY)
    ignore_arg = mock_indexer.call_args[1]["ignore"]
    assert "file1" in ignore_arg
    assert "file2" and "file3" not in ignore_arg


def test_bids_parser_set_layout(mock_parser: script.BIDSCreator, mocker: Any) -> None:
    """Test the _set_layout method of the BIDSCreator class."""
    mock_indexer = mocker.MagicMock()
    mock_layout = mocker.patch("bids.BIDSLayout", return_value=mocker.MagicMock())

    # Test when args.datafolder is None or doesn't contain "derivatives"
    result = mock_parser._set_layout(mock_indexer)
    mock_layout.assert_called_with(root=mock_parser.reading_root, indexer=mock_indexer)
    assert result == mock_layout.return_value

    # Test when args.datafolder contains "derivatives"
    mock_parser.datafolder = "derivatives"
    result = mock_parser._set_layout(mock_indexer)
    mock_layout.assert_called_with(
        root=mock_parser.reading_root,
        validate=False,
        is_derivative=True,
        indexer=mock_indexer,
    )
    assert result == mock_layout.return_value


def test_bids_parser_set_entities(mock_parser: script.BIDSCreator, mocker: Any) -> None:
    #"""Test the _set_entities method of the BIDSCreator class."""
    #mock_parse_range_args = mocker.patch.object(
    #    mock_parser, "_parse_range_args", return_value=mocker.MagicMock()
    #)
    
    #result = mock_parser._set_entities()
    #result_description = result.pop("description")

    #entity_names = [
    #    "subject",
    #    "session",
    #    "run",
    #    "task",
    #    "extension",
    #    "datatype",
    #    "suffix",
    #]

    #for name in entity_names:
    #    if getattr(mock_parser, name, False):
    #        mock_parse_range_args.assert_any_call(name, getattr(mock_parser, name))

    #assert result == {name: mock_parse_range_args.return_value for name in entity_names}
    #assert result_description == mock_parser.args.description
