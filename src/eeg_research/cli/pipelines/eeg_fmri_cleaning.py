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

"""EEG-fMRI Cleaning Pipeline.

This pipeline is designed to preprocess and clean EEG data that has been recorded
simultaneously with fMRI scans. The presence of the fMRI scanner introduces significant
artifacts into the EEG data, which can obscure the underlying neural signals. This
script implements a series of steps to mitigate these artifacts and enhance the quality
of the EEG data for further analysis.

Key Features:
- Gradient Artifact Removal: Implements ---FILL IN HERE--- to minimize the impact of
gradient artifacts caused by the MR scanner.
- Ballistocardiogram (BCG) Artifact Correction: Utilizes ---FILL IN HERE--- to identify
and remove artifacts related to cardiac activity.
- Quality Control: Offers a series of checks and visualizations to assess the
effectiveness of artifact removal and the overall quality of the EEG data
post-cleaning.
- Interactive Menu: Provides an interactive menu for users to select specific cleaning
steps and configure parameters, making the pipeline more intuitive and user-friendly to
navigate.
- Automated Workflow: Supports command-line arguments for batch processing and
automation.

Usage:
The script operates within a BIDS (Brain Imaging Data Structure) environment, requiring
the EEG data to be organized according to BIDS standards. Users can run the script in
an interactive mode, which presents a menu for selecting the desired cleaning
procedures and specifying any relevant parameters. Alternatively, the script can be
executed with predefined arguments for a more automated workflow, suitable for batch
processing or integration into larger data processing pipelines.

Output:
The script outputs cleaned EEG data files, ready for further analysis or visualization.
Additionally, it prints logs and reports detailing the cleaning process, which can
be used for quality assurance and documentation purposes.

Intended Users:
This script is intended for neuroscientists, clinicians, and researchers working with
simultaneous EEG-fMRI data. It requires a basic understanding of EEG data processing
and familiarity with Python programming.

Note:
The effectiveness of the artifact removal and data cleaning procedures can vary
depending on the specific characteristics of the EEG and fMRI data. Users are
encouraged to visually inspect the data at various stages of the preprocessing pipeline
and adjust the cleaning parameters as necessary to achieve optimal results.
"""

from eeg_research.cli.tools.bids_parser import BIDSCreator, bids_args_parser
from eeg_research.cli.tools.interactive_menu import InteractiveMenu
from eeg_research.preprocessing.pipelines.bcg_cleaning_pipeline import clean_bcg
from eeg_research.preprocessing.pipelines.gradient_cleaning_pipeline import (
    clean_gradient,
)
from eeg_research.preprocessing.tools.utils import read_raw_eeg, save_clean_eeg


def main() -> None:
    """Main function."""
    parser = bids_args_parser()
    bids_dataset = BIDSCreator(**parser)

    scripts = {
        "gradient": "Gradient Cleaning",
        "bcg": "BCG Cleaning",
        "qc": "Quality Control",
    }


    # If the user wants to run the interactive menu
    if parser['interactive']:
        # If script flags are provided, preselect the scripts based on the flags
        preselection = [
            i
            for i, arg in enumerate(
                [parser['gradient'], parser['bcg'], parser['qc']]
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
        layout = bids_dataset.update_layout(bids_dataset.entities)

        available_entities = layout.get_entities()
        # For each entity, get the available options and ask the user to select some
        for entity in bids_dataset.entities.keys():
            # Skip if the entity is not available or already selected
            if (
                entity not in available_entities.keys()
                or bids_dataset.entities[entity] is not None
            ):
                continue
            # Get the available options for the entity
            menu_entries = getattr(layout, f"get_{entity}s")()
            # If there is only one option, select it automatically
            if len(menu_entries) == 1:
                 bids_dataset.entities[entity] = menu_entries[0]
            # If there are multiple options, ask the user to select some
            elif len(menu_entries) > 1:
                menu = InteractiveMenu(
                    menu_entries=menu_entries,
                    entity=entity,
                    title=f"Select the {entity}s you want to include:",
                )
                 bids_dataset.entities[entity] = menu.get_selected_items()
                # Update the BIDSLayout object to only include selected entities
                layout = bids_dataset.update_layout( bids_dataset.entities)

        # Remove None values from the selected entities
        selected_entities = {k: v for k, v in bids_dataset.entities.items() 
                             if v is not None}
        # Get the files based on the selected entities
        files = bids_dataset.layout.get(return_type="file", **selected_entities)

    else:
        # Select the scripts based on the flags
        selected_scripts = [
            scripts[script] for script in scripts if parser.get(script, False)
        ]
        # Remove None values from the entities dictionary
        selected_entities = {k: v for k, v in bids_dataset.entities.items() 
                             if v is not None}

        # Get the files based on the flags
        files = bids_dataset.layout.get(return_type="file", **selected_entities)

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
