"""Module that get a selection of bids file from a regexp-like input.



"""

import os
import argparse
import re
from pathlib import Path
from dataclasses import dataclass
import copy
from typing import Any

import bids

def cli_bids_arg_parser() -> dict:
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

    return vars(args)

@dataclass
class BIDSselector:
    """A class for flexible selection of BIDS files based on customizable criteria.

    This class allows users to select files from a BIDS dataset by specifying 
    various BIDS entities such as subject, session, task, run, and more. Users 
    can input ranges, individual IDs, or lists to customize file selection 
    without manually specifying each file. This is especially useful for large 
    datasets where the starting and ending IDs may be unknown or numerous.

    Args:
        root (str | os.PathLike): Root directory of the BIDS dataset.
        subject (str | list | None, optional): Subject(s) to select. Can be 
            a single ID, a range (e.g., "10-100"), or '*' for all subjects.
        session (str | list | None, optional): Session(s) to select. Similar 
            input options as `subject`.
        task (str | list | None, optional): Task(s) to select.
        run (str | list | None, optional): Run(s) to select.
        datatype (str | list | None, optional): Data type(s) to select.
        suffix (str | list | None, optional): File suffix(es) to select.
        extension (str | list | None, optional): File extension(s) to select.

    Example:
        To select all subjects from "10" onward, specify `subject='10-*'`. 
        To select up to subject "20" only, use `subject='*-20'`. Alternatively, 
        provide a list of specific subject IDs (e.g., `subject=['10', '15', '30']`).

    Attributes:
        user_input (dict): Dictionary of user-provided selections for various 
            BIDS entities.
        _original_layout (bids.BIDSLayout): The original BIDS layout for 
            querying files.
        layout (list): The BIDS layout, filtered by the current selection 
            criteria.

    Methods:
        __post_init__: Completes initialization by regularizing attributes 
            and setting BIDS attributes based on user input.
        __str__: Returns a string representation of the current selection 
            criteria and the number of matching files.
        __add__: Merges selections with another BIDSselector instance or 
            dictionary.
        to_dict: Returns the current BIDS entity attributes as a dictionary.
        set_bids_attributes: Updates the BIDS attributes for selection.
        
    Internal Helper Methods:
        _regularize_input_str: Removes prefixes from entity values for 
            consistent selection.
        _regularize_attributes: Regularizes prefixes across all BIDS 
            entities in the object.
        _set_layout: Initializes a new BIDS layout with a specified indexer.
        _convert_input_to_list: Parses range or list arguments for entities.
    """

    root: str | os.PathLike
    subject: str | list | None = None
    session: str | list | None = None
    task: str | list | None = None
    run: str | list | None = None
    datatype: str | list | None = None
    suffix: str | list | None = None
    extension: str | list | None = None

    def __post_init__(self) -> None:
        """Finishing some initializations."""
        self._original_layout= bids.layout.BIDSLayout(root=self.root)
        self.user_input = self.to_dict()
        self._regularize_attributes()
        self.set_bids_attributes()

    def __str__(self) -> str:
        """Public string representation when calling print().

        Returns:
            str: String representation
        """
        str_list = list()
        for attribute, value in self.to_dict().items():
            if value is None:
                all_existing_values = self._original_layout.get(
                    return_type="id", target=attribute
                )

                all_existing_values = [str(val) for val in all_existing_values]

                if len(all_existing_values) <= 4:
                    specification_str = f"({', '.join(all_existing_values)})"
                else:
                    specification_str = (
                        f"({all_existing_values[0]} ... {all_existing_values[-1]})"
                    )

                value_length = "All"

            elif isinstance(value, str):
                specification_str = f"({value})"
                value_length = "1"

            elif isinstance(value, list):
                if len(value) <= 4:
                    specification_str = f"({', '.join(value)})"
                else:
                    specification_str = f"({value[0]} ... {value[-1]})"
                value_length = str(len(value))

            str_list.append(
                f"{str(attribute).capitalize()}s: {value_length} "
                f"{specification_str}"
            )

        files = self.layout
        all_str = f"{'\n'.join(str_list)}\nFiles: {len(files)}"
        return all_str

    def __add__(self, other: Any):
        if isinstance(other, self.__class__):
            if other.root == self.root:
                class_dict = other.to_dict()
            else:
                raise NotImplementedError(f"The two {self.__class__.__name__}"\
                    " instances are not pointing to the same"\
                    " directory. Set the root argument to point to the same"\
                    " directory for both instances")

        elif isinstance(other, dict):
            class_dict = other
        else:
            return NotImplemented

        for dict_key, dict_value in class_dict.items():
            val = getattr(self, dict_key)

            if val is None:
                setattr(self, dict_key, dict_value)
                continue
            if dict_value is None:
                continue

            instance_cond_self = isinstance(val, str) or isinstance(val, list)
            instance_cond_other = isinstance(val, str) or isinstance(val, list)

            if not instance_cond_self and instance_cond_other:
                raise TypeError("Only string type or list type are allowed")

            if isinstance(val, str):
                val = [val]

            if isinstance(dict_value, str):
                dict_value = [dict_value]

            setattr(self, dict_key, list(set(val).union(set(dict_value))))

    def to_dict(self) -> dict:
        """When inputs are needed.

        Returns:
            dict: The object attributes in a dictionary.
        """
        return {
            "subject": self.subject,
            "session": self.session,
            "task": self.task,
            "run": self.run,
            "datatype": self.datatype,
            "suffix": self.suffix,
            "extension": self.extension,
        }

    def _regularize_input_str(self, value: str) -> str:
        """Regularize the input string for consistency.

        Sometime we can think about a list of subject being ['sub-01','sub-02']
        while the authorized input is ['01', '02'] or '01-*'. This function
        remove the prefix from the input string for consistency inside of the
        class and also allow more flexibility on the user side.

        Args:
            value (str): The value to convert

        Returns:
            str: The converted value
        """
        prefix_list = [
            "sub-",
            "ses-",
            "task-",
            "run-",
            "desc-",
        ]
        for prefix in prefix_list:
            if prefix in value.lower():
                return value.replace(prefix, "")

        return value

    def _regularize_attributes(self) -> "BIDSselector":
        """Remove eventual prefixes for all the arguments of the object.

        Returns:
             BIDSselector: The instance modified
        """
        for attribute, value in self.to_dict().items():
            if isinstance(value, str):
                regularized_value = self._regularize_input_str(value)

            elif isinstance(value, list):
                regularized_value = [self._regularize_input_str(val) for val in value]

            else:
                regularized_value = value

            setattr(self, attribute, regularized_value)

        return self

    def _set_layout(self, indexer: bids.BIDSLayoutIndexer) -> bids.BIDSLayout:
        """Set the BIDS layout with the given indexer based on args.datafolder."""
        return bids.BIDSLayout(
            root=self.root,
            validate="derivative" not in str(self.root).lower(),
            is_derivative="derivative" in str(self.root).lower(),
            indexer=indexer,
        )

    def _convert_input_to_list(self, entity: str, value: str | None) -> list:
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
        if not value:
            return None

        existing_values = self._original_layout.get(target=entity, return_type="id")
        if value == "*":
            selection = existing_values

        elif "," in value:
            if "[" in value:
                value = value[1:-1]
            from_input = set(value.replace(" ", "").split(","))
            from_existing = set(existing_values)
            selection = list(from_input.intersection(from_existing))

        elif "-" in value:
            start, end = value.split("-")
            if start == "*":
                selection = existing_values[: existing_values.index(end) + 1]
            elif end == "*":
                selection = existing_values[existing_values.index(start) :]
            else:
                selection = existing_values[
                    existing_values.index(start) : existing_values.index(end)
                ]

        else:
            selection = value

        return selection

    def set_bids_attributes(self) -> "BIDSselector":
        """Set the converted values to all bids attributes.

        Returns:
            dict: The attribute/value pairs
        """
        for attribute, value in self.to_dict().items():
            if value is None:
                continue

            new_value = self._convert_input_to_list(attribute, getattr(self, attribute))
            setattr(self, attribute, new_value)

        return self

    @property
    def layout(self) -> bids.BIDSLayout:
        """Update the BIDSLayout to only include given entities.

        As of April 2024, BIDSLayoutIndexer's **filters argument does not work.
        Therefore, a workaround is implemented to filter out files that are not indexed.
        """
        self.set_bids_attributes()
        return self._original_layout.get(
            **{key: val 
               for key, val in self.to_dict().items() 
               if val is not None},
            return_type='filename'
            )

