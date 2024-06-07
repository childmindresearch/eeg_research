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

"""CLI for processing and cleaning EEG data in BIDS format."""

from eeg_research.cli.tools.bids_parser import BIDSParser
from eeg_research.cli.tools.interactive_menu import InteractiveMenu
from eeg_research.preprocessing.pipelines.bcg_cleaning_pipeline import clean_bcg
from eeg_research.preprocessing.pipelines.gradient_cleaning_pipeline import (
    clean_gradient,
)
from eeg_research.preprocessing.tools.utils import read_raw_eeg, save_clean_eeg


def main() -> None:
    """Main function."""
    parser = BIDSParser()

    scripts = {
        "gradient": "Gradient Cleaning",
        "bcg": "BCG Cleaning",
        "qc": "Quality Control",
    }

    # If the user wants to run the interactive menu
    if parser.args.interactive:
        # If script flags are provided, preselect the scripts based on the flags
        preselection = [
            i
            for i, arg in enumerate(
                [parser.args.gradient, parser.args.bcg, parser.args.qc]
            )
            if arg
        ]
        # Run the interactive menu with the preselected scripts
        menu = InteractiveMenu(
            menu_entries=[script for script in scripts.values()],
            entity="script",
            title="Select the scripts you want to run:",
            preselection=preselection,
        )
        selected_scripts = menu.get_selected_items()

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
        # Select the scripts based on the flags
        selected_scripts = [
            scripts[script] for script in scripts if getattr(parser.args, script)
        ]
        # Remove None values from the entities dictionary
        selected_entities = {k: v for k, v in parser.entities.items() if v is not None}

        # Get the files based on the flags
        files = parser.layout.get(return_type="file", **selected_entities)

    if not files:
        raise FileNotFoundError("No valid files found with the given arguments.")
    else:
        print(f"Found {len(files)} valid files.")

        print(f"Running {selected_scripts} scripts on these files:")

    for file in files:
        print(f"Current file: {file}")

        raw = read_raw_eeg(file, preload=True)

        selected_scripts_keys = [k for k, v in scripts.items() if v in selected_scripts]

        if "gradient" in selected_scripts_keys:
            raw = clean_gradient(raw)
        if "bcg" in selected_scripts_keys:
            raw = clean_bcg(raw)
        if "qc" in selected_scripts_keys:
            # implement quality control
            pass

        save_clean_eeg(raw, file, selected_scripts_keys)

    print("Processing complete.")


if __name__ == "__main__":
    main()
