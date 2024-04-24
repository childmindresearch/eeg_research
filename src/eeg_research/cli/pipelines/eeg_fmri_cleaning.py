#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
#         Dr. Alp Erkent, MD, MA
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-02-27
# email: samuel DOT louviot AT nki DOT rfmh DOT org
#        alp DOT erkent AT childmind DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================================

"""GENERAL DOCUMENTATION HERE."""

import argparse
import os

import bids

from eeg_research.cli.tools.interactive_menu import (
    run_interactive_menu,
    select_scripts_interactively,
)


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


def parse_range_arg(
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


def create_entities(args: argparse.Namespace, layout: bids.BIDSLayout) -> dict:
    """Create entities dictionary from arguments.

    Args:
        args: The parsed command line arguments.
        layout: The BIDSLayout object.

    Returns:
        A dictionary of entities.
    """
    # List of entity names that need to be parsed
    entity_names = [
        "subject",
        "session",
        "run",
        "task",
        "extension",
        "datatype",
        "suffix",
    ]

    # Create entities dictionary
    entities = {
        name: parse_range_arg(layout, name, getattr(args, name))
        for name in entity_names
    }

    # Add other entities
    entities.update({"description": args.description})

    return entities


def main() -> None:
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()

    # If the datafolder is not provided
    if args.datafolder is None:
        # Set the reading root to the root
        reading_root = args.root
        layout = bids.BIDSLayout(reading_root, validate=True)
    # If the data folder is provided
    else:
        # Set the reading root to the root/datafolder
        reading_root = os.path.join(args.root, args.datafolder)

        # If the datafolder is derivatives
        if "derivatives" in args.datafolder:
            # Set the layout to be a derivative layout
            layout = bids.BIDSLayout(reading_root, validate=False, is_derivative=True)
        # If the datafolder is source or rawdata
        else:
            # Set the layout to be a regular layout
            layout = bids.BIDSLayout(reading_root, validate=True)

    # Create entities dictionary
    entities = create_entities(args, layout)

    # Define the script files
    scripts = {
        "gradient": "gradient_cleaning.py",
        "bcg": "bcg_cleaning.py",
        "qc": "quality_control.py",
    }

    # If the user wants to run the interactive menu with no script flags
    if args.interactive and not (args.gradient or args.bcg or args.qc):
        # Select the scripts via the interactive menu
        selected_scripts = select_scripts_interactively()
    # If the user does not want to run the interactive menu
    elif not args.interactive:
        # Select the scripts based on the flags
        selected_scripts = [
            scripts[script] for script in scripts if getattr(args, script)
        ]
    # If the user wants to run the interactive menu with script flags
    else:
        # Preselect the scripts based on the flags
        preselection = [
            i for i, arg in enumerate([args.gradient, args.bcg, args.qc]) if arg
        ]

        # Run the interactive menu with the preselected scripts
        selected_scripts = select_scripts_interactively(preselection)

    # If the user wants to run the interactive menu
    if args.interactive:
        # Run the interactive menu
        files = run_interactive_menu(reading_root, layout, entities)
    # If the user does not want to run the interactive menu
    else:
        # Remove None values from the entities dictionary
        entities = {k: v for k, v in entities.items() if v is not None}

        # Get the files based on the flags
        files = layout.get(return_type="file", **entities)

    # If no files are found
    if not files:
        # Raise a FileNotFoundError
        raise FileNotFoundError("No valid files found with the given arguments.")
    # If files are found
    else:
        # Print the number of files found
        print(f"Found {len(files)} valid files.")

        # Print the selected scripts
        print(f"Running {selected_scripts} scripts on these files:")

    # Loop through the files
    for file in files:
        print(f"Current file: {file}")

        # Read the raw file
        # raw = read_raw_eeg(file, preload=True)

        # Clean the raw file based on the selected scripts
        # if "gradient" in selected_scripts:
        #     raw = clean_gradient(raw)
        # if "bcg" in selected_scripts:
        #     raw = clean_bcg(raw)

        # Save the cleaned file

    print("Processing complete.")


if __name__ == "__main__":
    main()
