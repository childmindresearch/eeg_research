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

"""CLI for processing and cleaning EEG data in BIDS format.

It provides functionalities for:
- Parsing command-line arguments for data directory, data folder, and cleaning scripts.
- Creating a BIDS layout object for structured interaction with BIDS data.
- Creating an "entities" dictionary mapping entity names to their respective values.
- Providing an interactive menu for users to select cleaning scripts.
- Selecting files based on the entities dictionary or through the interactive menu.
- Processing the selected files by reading, cleaning, and saving the EEG data.

TODO: The actual reading, cleaning, and saving of the EEG data are commented out.
"""

import os

import bids

from eeg_research.cli.tools.interactive_menu import (
    run_interactive_menu,
    select_scripts_interactively,
)
from eeg_research.cli.tools.layout import create_entities
from eeg_research.cli.tools.parser import parse_arguments


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
