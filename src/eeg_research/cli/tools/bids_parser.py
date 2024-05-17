"""This module contains the BIDSParser class."""

import argparse
import re
from pathlib import Path

import bids


class BIDSParser:
    """A class to parse BIDS entities."""

    def __init__(self) -> None:
        """Initialize the BIDSParser object.

        It parses command-line arguments, sets the reading root, indexer, layout,
        and entities.
        """
        self.args = self._parse_arguments()
        self.reading_root = self._set_reading_root()
        self.indexer = bids.BIDSLayoutIndexer()
        self.layout = self._set_layout(self.indexer)
        self.entities = self._set_entities()

    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
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

        if not any([args.interactive, args.gradient, args.bcg, args.qc]):
            parser.error(
                "Please provide at least one of the following arguments: "
                "--interactive, --gradient, --bcg, --qc"
            )

        return args

    def _set_reading_root(self) -> Path:
        """Set the reading root based on the provided arguments."""
        if self.args.datafolder is None:
            return Path(self.args.root)
        else:
            return Path(self.args.root) / self.args.datafolder

    def _set_layout(self, indexer: bids.BIDSLayoutIndexer) -> bids.BIDSLayout:
        """Set the BIDS layout with the given indexer based on args.datafolder."""
        if self.args.datafolder is None or "derivatives" not in self.args.datafolder:
            return bids.BIDSLayout(root=self.reading_root, indexer=indexer)
        else:
            return bids.BIDSLayout(
                root=self.reading_root,
                validate=False,
                is_derivative=True,
                indexer=indexer,
            )

    def _parse_range_args(
        self, entity: str, value: str | None
    ) -> list[int] | str | None:
        """Parse range argument.

        Args:
            entity: The entity to get from the layout.
            value: The value to parse.

        Returns:
            A list of IDs or a string.

        Raises:
            ValueError: If the entity contains non-integers and a range is provided.
            IndexError: If the start or end index is out of range.
        """
        if value == "*":
            return self.layout.get(target=entity, return_type="id")
        elif value is not None and "-" in value:
            start, end = map(lambda x: None if x == "*" else int(x), value.split("-"))

            ids_str = self.layout.get(target=entity, return_type="id")

            try:
                ids_int = [int(id) for id in ids_str]
            except ValueError:
                raise ValueError(
                    f"Range not valid for '{entity}' as it contains non-integers. "
                    "Please use the interactive menu to select IDs."
                )

            if start is not None and start < 1:
                raise ValueError(
                    f"Start value {start} for entity {entity} is not a positive "
                    "integer."
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

            ids_in_range = [
                ids_str[i]
                for i, id in enumerate(ids_int)
                if (start is None or id >= start) and (end is None or id <= end)
            ]

            if not ids_in_range:
                raise ValueError(
                    f"No IDs found for entity {entity} between start value {start} "
                    f"and end value {end}."
                )

            return ids_in_range
        else:
            return value

    def _set_entities(self) -> dict:
        """Set the entities dictionary based on the provided arguments.

        Returns:
            A dictionary of entities.
        """
        entity_names = [
            "subject",
            "session",
            "run",
            "task",
            "extension",
            "datatype",
            "suffix",
        ]

        entities = {
            name: self._parse_range_args(name, getattr(self.args, name))
            for name in entity_names
        }

        entities.update({"description": self.args.description})

        return entities

    def update_layout(self, entities: dict[str, str | None]) -> bids.BIDSLayout:
        """Update the BIDSLayout to only include given entities.

        As of April 2024, BIDSLayoutIndexer's **filters argument does not work.
        Therefore, a workaround is implemented to filter out files that are not indexed.
        """
        # Remove None values from the entities dictionary
        entities = {k: v for k, v in entities.items() if v is not None}

        # Get all files and filtered files
        all_files = self.layout.get(return_type="file")
        filtered_files = self.layout.get(return_type="file", **entities)

        # Get the files to ignore
        ignored_files = list(set(all_files) - set(filtered_files))

        # Define the default ignore patterns
        default_ignore = [
            re.compile(r"^/(code|models|sourcedata|stimuli)"),
            re.compile(r"/\."),
        ]

        # Create a new BIDSLayoutIndexer object to also ignored these files
        indexer = bids.BIDSLayoutIndexer(ignore=default_ignore + ignored_files)

        # Create a new BIDSLayout object with the new indexer
        layout = self._set_layout(indexer)

        return layout
