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

TODO: The actual reading, cleaning, and saving of the EEG data are commented out.
"""

from eeg_research.cli.tools.bids_parser import BIDSParser
from eeg_research.cli.tools.interactive_menu import InteractiveMenu


def main() -> None:
    """Main function."""
    parser = BIDSParser()

    # Define the script files
    scripts = {
        "gradient": "gradient_cleaning.py",
        "bcg": "bcg_cleaning.py",
        "qc": "quality_control.py",
    }

    # If the user wants to run the interactive menu with no script flags
    if parser.args.interactive and not (
        parser.args.gradient or parser.args.bcg or parser.args.qc
    ):
        # Select the scripts via the interactive menu
        menu_entries = [
            script.replace(".py", "").replace("_", " ").title()
            for script in scripts.values()
        ]
        menu = InteractiveMenu(
            menu_entries=menu_entries,
            entity="script",
            title="Select the scripts you want to run:",
        )
        selected_scripts = menu.get_selected_items()
    # If the user does not want to run the interactive menu
    elif not parser.args.interactive:
        # Select the scripts based on the flags
        selected_scripts = [
            scripts[script] for script in scripts if getattr(parser.args, script)
        ]
    # If the user wants to run the interactive menu with script flags
    else:
        # Preselect the scripts based on the flags
        preselection = [
            i
            for i, arg in enumerate(
                [parser.args.gradient, parser.args.bcg, parser.args.qc]
            )
            if arg
        ]

        # Run the interactive menu with the preselected scripts
        menu = InteractiveMenu(
            menu_entries=list(scripts.keys()),
            entity="script",
            title="Select the scripts you want to run:",
            preselection=preselection,
        )
        selected_scripts = menu.get_selected_items()

    # If the user wants to run the interactive menu
    if parser.args.interactive:
        # Create a BIDSLayout object for the data folder with given entities
        layout = parser.update_layout(parser.entities)

        # Get all entities associated with the data folder
        available_entities = layout.get_entities()

        # For each entity, get the available options and ask the user to select some
        for entity in parser.entities.keys():
            # Skip if the entity is not available or already selected
            if (
                entity not in available_entities.keys()
                or parser.entities[entity] is not None
            ):
                continue
            # Get the available options for the entity
            menu_entries = getattr(layout, f"get_{entity}s")()
            # If there is only one option, select it automatically
            if len(menu_entries) == 1:
                parser.entities[entity] = menu_entries[0]
            # If there are multiple options, ask the user to select some
            elif len(menu_entries) > 1:
                menu = InteractiveMenu(
                    menu_entries=menu_entries,
                    entity=entity,
                    title=f"Select the {entity}s you want to include:",
                )
                parser.entities[entity] = menu.get_selected_items()
                # Update the BIDSLayout object to only include selected entities
                layout = parser.update_layout(parser.entities)

        # Remove None values from the selected entities
        selected_entities = {k: v for k, v in parser.entities.items() if v is not None}
        # Get the files based on the selected entities
        files = parser.layout.get(return_type="file", **selected_entities)

    # If the user does not want to run the interactive menu
    else:
        # Remove None values from the entities dictionary
        selected_entities = {k: v for k, v in parser.entities.items() if v is not None}

        # Get the files based on the flags
        files = parser.layout.get(return_type="file", **selected_entities)

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
