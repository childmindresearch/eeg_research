"""Tests for eeg_research.cli.toolbox.layout.py."""

from unittest.mock import MagicMock

import eeg_research.cli.tools.layout as script


def test_create_entities() -> None:
    """Test the create_entities function."""
    # Mock the args and layout object
    args = MagicMock()
    layout = MagicMock()

    # Set the expected output
    expected_output = {
        "subject": "1",
        "session": "1",
        "run": "1",
        "task": "a",
        "extension": "a",
        "datatype": "a",
        "suffix": "a",
        "description": "a",
    }

    # Set the return values of the args object
    args.subject = "1"
    args.session = "1"
    args.run = "1"
    args.task = "a"
    args.extension = "a"
    args.datatype = "a"
    args.suffix = "a"
    args.description = "a"

    # Call the function with the mocked objects
    result = script.create_entities(args, layout)

    # Assert that the result is as expected
    assert result == expected_output
