"""This module provides the command line argument parser."""

import argparse

import bids


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        A dictionary of arguments parsed from the command line.
    """
    # Create the parser with RawTextHelpFormatter so that newlines are preserved
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--root",
        help="Root folder.",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--datafolder",
        help="Data folder to search for files. "
        "Options are 'source', 'rawdata' or 'derivatives'.",
        choices=["source", "rawdata", "derivatives"],
        default=None,
    )

    parser.add_argument(
        "--subject",
        help="Input options for subject IDs are: \n"
        "- '*' for all subjects \n"
        "- 'x' for subject x \n"
        "- 'x-y' for subjects x to y \n"
        "- 'x-*' for subjects x to the last \n"
        "- '*-y' for subjects from the first to y",
        default=None,
    )

    parser.add_argument(
        "--session",
        help="Input options for session IDs are: \n"
        "- '*' for all sessions \n"
        "- 'x' for session x \n"
        "- 'x-y' for sessions x to y \n"
        "- 'x-*' for sessions x to the last \n"
        "- '*-y' for sessions from the first to y",
        default=None,
    )

    parser.add_argument(
        "--run",
        help="Input options for run IDs are: \n"
        "- '*' for all runs \n"
        "- 'x' for run x \n"
        "- 'x-y' for runs x to y \n"
        "- 'x-*' for runs x to the last \n"
        "- '*-y' for runs from the first to y",
        default=None,
    )

    parser.add_argument(
        "--task",
        help="Input options for task IDs are: \n"
        "- '*' for all tasks \n"
        "- 'a' for task a",
        default=None,
    )

    parser.add_argument(
        "--extension",
        help="Input options for file extensions are: \n"
        "- '*' for all extensions \n"
        "- 'a' for extension a",
        default=None,
    )

    parser.add_argument(
        "--datatype",
        help="Input options for datatypes are: \n"
        "- '*' for all datatypes \n"
        "- 'a' for datatype a",
        default="eeg",
    )

    parser.add_argument(
        "--suffix",
        help="Input options for suffixes are: \n"
        "- '*' for all suffixes \n"
        "- 'a' for suffix a",
        default="eeg",
    )

    parser.add_argument(
        "--description",
        help="Description is only applicable to derivative data.",
        default=None,
    )
    parser.add_argument(
        "--interactive",
        help="Run the interactive menu",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--gradient",
        help="Clean the gradient artifacts",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--bcg",
        help="Clean the BCG artifacts",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--qc",
        help="Run the quality control script",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Throw an error if none of the --interactive, --gradient, --bcg, or --qc
    # flags are provided
    if not any([args.interactive, args.gradient, args.bcg, args.qc]):
        parser.error(
            "Please provide at least one of the following arguments: "
            "--interactive, --gradient, --bcg, --qc"
        )

    return args


def parse_range_arguments(
    layout: bids.BIDSLayout, entity: str, value: str | None
) -> list[int] | str | None:
    """Parse range argument.

    Args:
        layout: The BIDSLayout object.
        entity: The entity to get from the layout.
        value: The value to parse.

    Returns:
        A list of IDs or a string.

    Raises:
        ValueError: If the entity contains non-integers and a range is provided.
        IndexError: If the start or end index is out of range.
    """
    # If the value is a wildcard
    if value == "*":
        # Return all IDs for the argument
        return layout.get(target=entity, return_type="id")
    # If the value is a range
    elif value is not None and "-" in value:
        # Split the range into start and end
        start, end = map(lambda x: None if x == "*" else int(x), value.split("-"))

        # Get all IDs for the argument
        ids_str = layout.get(target=entity, return_type="id")

        # Check if all IDs are integers
        try:
            ids_int = [int(id) for id in ids_str]
        # If not, raise a ValueError
        except ValueError:
            raise ValueError(
                f"Range not valid for '{entity}' as it contains non-integers. "
                "Please use the interactive menu to select IDs."
            )

        # Check if start and end are positive integers and end > start
        # and raise a ValueError if not
        if start is not None and start < 1:
            raise ValueError(
                f"Start value {start} for entity {entity} is not a positive integer."
            )
        if end is not None and end < 1:
            raise ValueError(
                f"End value {end} for entity {entity} is not a positive integer."
            )
        if start is not None and end is not None and end <= start:
            raise ValueError(
                f"End value {end} for entity {entity} is not greater than "
                f"start value {start}. Please provide a valid range."
            )

        # Check if there are any IDs between start and end
        ids_in_range = [
            ids_str[i]
            for i, id in enumerate(ids_int)
            if (start is None or id >= start) and (end is None or id <= end)
        ]

        # If no IDs are found in the range, raise a ValueError
        if not ids_in_range:
            raise ValueError(
                f"No IDs found for entity {entity} between start value {start} "
                f"and end value {end}."
            )

        # Return the IDs in the range
        return ids_in_range
    else:
        # Return the value as is
        return value
