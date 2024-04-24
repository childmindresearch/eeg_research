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

# Import necessary libraries
import re

import bids
from simple_term_menu import TerminalMenu


def create_menu(menu_items: list[str], entity: str) -> tuple[TerminalMenu, list[str]]:
    """Create a terminal menu for selecting items.

    Args:
        menu_items (list[str]): A list of items to display in the menu.
        entity (str): The entity that the items belong to.

    Returns:
        tuple[TerminalMenu, list[str]]: A tuple containing the TerminalMenu object and
            the updated list of menu items.
    """
    # Set the title of the menu
    title = f"Select the {entity}s:"

    # Add options to the menu items if there are more than one
    if len(menu_items) > 1:
        # Add an option for user input to the menu items if the entity has integers
        if entity in ["subject", "session", "run"]:
            menu_items = ["Enter range"] + menu_items
        # Add an option for selecting all items to the menu items
        menu_items = ["Select all"] + menu_items

    # If entity is run, convert ints to strings
    # In pybids, only run entities are integers
    # and terminal menu only accepts strings
    if entity == "run":
        menu_items = [str(item) for item in menu_items]

    # Create a terminal menu with the given items and title
    menu = TerminalMenu(
        menu_items, title=title, multi_select=True, show_multi_select_hint=True
    )

    return menu, menu_items


def handle_user_input(menu_items: list[str], selected_indices: list[int]) -> list[str]:
    """Handle user input for the terminal menu.

    Args:
        menu_items (list[str]): A list of items to display in the menu.
        selected_indices (list[int]): A list of indices of the selected items.

    Returns:
        list[str]: A list of the selected items.
    """
    # If the user selected the "Select all" option
    if 0 in selected_indices and "Select all" in menu_items:
        # Return all items except "Select all" and "Enter range"
        selected_items = [
            item for item in menu_items if item not in ["Select all", "Enter range"]
        ]
    # If the user selected the "Enter range" option
    elif 1 in selected_indices and "Enter range" in menu_items:
        # Ask the user to enter a range
        start = int(input("Enter the start of the range: "))
        end = int(input("Enter the end of the range: "))

        # Return the items in the selected range except 'Select all' and 'Enter range'
        selected_items = [item for item in menu_items[2:] if start <= int(item) <= end]
    # If the user selected specific items
    else:
        # Return the selected items
        selected_items = [menu_items[i] for i in selected_indices]

    return selected_items


def get_selected_items(menu_items: list[str], entity: str) -> list[str]:
    """Create a terminal menu and return the items selected by the user.

    Args:
        menu_items (list[str]): A list of items to display in the menu.
        entity (str): The entity that the items belong to.

    Returns:
        list[str]: A list of the selected items.
    """
    # Create a terminal menu for the given items and entity
    menu, menu_items = create_menu(menu_items, entity)

    # Show the menu and get the indices of the selected items
    selected_indices = menu.show()

    # Handle the user input and return the selected items
    selected_items = handle_user_input(menu_items, selected_indices)

    return selected_items


def build_layout(reading_root: str, indexer: bids.BIDSLayoutIndexer) -> bids.BIDSLayout:
    """Create and return a BIDSLayout object based on given reading_root and indexer.

    Args:
        reading_root (str): The root directory of the BIDS dataset.
        indexer (bids.BIDSLayoutIndexer): The indexer object used for filtering files.

    Returns:
        bids.BIDSLayout: The BIDSLayout object representing the BIDS dataset.
    """
    # If the reading_root contains "derivatives", set is_derivative to True
    # and validate to False so that a dataset_description.json file is not required
    if "derivatives" in reading_root:
        layout = bids.BIDSLayout(
            reading_root, validate=False, is_derivative=True, indexer=indexer
        )
    # Otherwise, create a BIDSLayout object with the given reading_root and indexer
    else:
        layout = bids.BIDSLayout(reading_root, indexer=indexer)
    return layout


def get_layout(
    reading_root: str, entities: dict[str, str | list[str]]
) -> bids.BIDSLayout:
    """Create and return a BIDSLayout object based on given reading_root and entities.

    As of April 2024, BIDSLayoutIndexer's **filters argument is not working as expected.
    Therefore, a workaround is implemented to filter out files that are not indexed.

    Args:
        reading_root (str): The root directory of the BIDS dataset.
        entities (dict[str, str]): A dictionary of BIDS entities used for indexing the
            dataset.

    Returns:
        bids.BIDSLayout: The BIDSLayout object representing the BIDS dataset.
    """
    # Remove None values from the entities dictionary
    entities = {k: v for k, v in entities.items() if v is not None}

    # Define the default ignore patterns
    default_ignore = list(
        {
            re.compile(r"^/(code|models|sourcedata|stimuli)"),
            re.compile(r"/\."),
        }
    )

    # Create a BIDSLayoutIndexer object that ignores default patterns
    indexer = bids.BIDSLayoutIndexer(ignore=default_ignore)

    # Create a BIDSLayout object with the given reading_root and default indexer
    layout = build_layout(reading_root, indexer)

    # Get all files and filtered files
    all_files = layout.get(return_type="file")
    filtered_files = layout.get(return_type="file", **entities)

    # Get the files to ignore
    ignored_files = list(set(all_files) - set(filtered_files))

    # Create a new BIDSLayoutIndexer object to also ignored these files
    indexer = bids.BIDSLayoutIndexer(ignore=default_ignore + ignored_files)

    # Create a new BIDSLayout object with the given reading_root and new indexer
    layout = build_layout(reading_root, indexer)

    return layout


def select_scripts_interactively(preselection: list[int] = []) -> list[str]:
    """Select scripts from a predefined list.

    Args:
        preselection (list[int]): A list of indices to preselect in the menu.

    Returns:
        list[str]: A list of selected scripts.
    """
    # Define the available scripts
    scripts = {
        "Gradient Cleaning": "gradient_cleaning.py",
        "BCG Cleaning": "bcg_cleaning.py",
        "Quality Control": "quality_control.py",
    }

    # Get the names of the scripts
    script_names = list(scripts.keys())

    # Create a menu for the scripts
    scripts_menu = TerminalMenu(
        script_names,
        title="Select the scripts you want to run:",
        multi_select=True,
        show_multi_select_hint=True,
        preselected_entries=preselection,
    )

    # Show the menu and get the index of the selected scripts
    selected_indices = scripts_menu.show()

    # Get the selected script
    selected_scripts = [scripts[script_names[i]] for i in selected_indices]

    return selected_scripts


# Function to run the interactive menu
def run_interactive_menu(
    reading_root: str,
    layout: bids.BIDSLayout,
    selected_entities: dict[str, str | list[str]],
) -> tuple[list[str], list[str]]:
    """Runs an interactive menu for selecting entities.

    Args:
        reading_root (str): The path to the data folder.
        layout (bids.BIDSLayout): The BIDSLayout object representing the data folder.
        selected_entities (dict[str, str]): The selected entities for indexing.

    Returns:
        tuple[list[str], list[str]]: A tuple containing the selected files and scripts.
    """
    # Create a BIDSLayout object for the data folder with given entities
    layout = get_layout(reading_root, selected_entities)

    # Get all entities associated with the data folder
    available_entities = layout.get_entities()

    # For each entity, get the available options and ask the user to select some
    for entity in selected_entities.keys():
        # Skip if the entity is not available or already selected
        if (
            entity not in available_entities.keys()
            or selected_entities[entity] is not None
        ):
            continue
        # Get the available options for the entity
        menu_items = getattr(layout, f"get_{entity}s")()
        # If there is only one option, select it automatically
        if len(menu_items) == 1:
            selected_entities[entity] = menu_items[0]
        # If there are multiple options, ask the user to select some
        elif len(menu_items) > 1:
            selected_entities[entity] = get_selected_items(menu_items, entity)
            # Update the BIDSLayout object to only include selected entities
            layout = get_layout(reading_root, selected_entities)

    # Remove None values from the selected entities
    selected_entities = {k: v for k, v in selected_entities.items() if v is not None}
    # Get the files based on the selected entities
    files = layout.get(return_type="file", **selected_entities)

    if len(files) == 0:
        print("No files found for the selected entities.")

    return files
