"""Module that host the BIDSselector class.

The pybids package (here mentioned in import bids) is a little bit too strict
when it comes to deal with datatype that are not BIDS standardized. Often
in derivatives we have this kind of non-standardized datatype such as
'eyetracking' or 'brainstate' or 'respiration' that are important to separate
as individual modality. The workaround is to either do a monkey patching to the
package by forcing it to allow such datatype or write from scratch a separate
class that would be less strict.
The monkey patching is not very pythonic and not recommended in the dev community
Because of the side effect that can induce. Therefore I am on the road to 
write another 'bids-like' data handler (also simplier) from scratch.
This will be useful for everybody dealing with pseudo BIDS layout with the 
possibility to customize the queries.
"""


import os
import re
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

import bids

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
        subject (str | None, optional): Subject(s) to select. Can be
            a single ID, a range (e.g., "10-100"), or '*' for all subjects.
        session (str | None, optional): Session(s) to select. Similar
            input options as `subject`.
        task (str | None, optional): Task(s) to select.
        run (str | None, optional): Run(s) to select.
        datatype (str | None, optional): Data type(s) to select.
        suffix (str | None, optional): File suffix(es) to select.
        extension (str | None, optional): File extension(s) to select.

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

    root: Path | str | os.PathLike
    subject: str | None = None
    session: str | None = None
    datatype: str | None = None
    task: str | None = None
    run: str | None = None
    acquisition: str | None = None
    description: str | None = None
    suffix: str | None = None
    extension: str | None = None

    def __post_init__(self) -> None:  # noqa: D105
        self.root = Path(self.root)
        self.data = self._get_layout()
        
    def __str__(self) -> str:
        """Public string representation when calling print().

        Returns:
            str: String representation
        """
        str_list = list()
        for attribute, value in self.file_system.items():
            if value is None:
                all_existing_values = [str(val)
                                       for val in self.data[attribute].unique()]

                if len(all_existing_values) <= 4:
                    specification_str = (
                        f"({ ', '.join(list(map(lambda s: str(s), 
                        all_existing_values))) })"
                    )
                else:
                    specification_str = (
                        f"({str(all_existing_values[0])} ... {str(all_existing_values[-1])})"
                    )

                value_length = "All"

            elif isinstance(value, str):
                specification_str = f"({value})"
                value_length = "1"

            elif isinstance(value, list):
                if len(value) <= 4:
                    specification_str = (
                        f"({ ', '.join(list(map(lambda s: str(s),value)))})"
                    )
                else:
                    specification_str = f"({str(value[0])} ... {str(value[-1])})"
                value_length = str(len(value))

            str_list.append(
                f"{str(attribute).capitalize()}s: {value_length} "
                f"{specification_str}"
            )

        files = self.layout
        all_str = f"{'\n'.join(str_list)}\nFiles: {len(files)}"
        return all_str

    def __add__(self, other):  # noqa: ANN204, D105, ANN001
        if isinstance(other, self.__class__):
            if other.root == self.root:
                class_dict = other.to_dict()
            else:
                raise NotImplementedError(
                    f"The two {self.__class__.__name__}"
                    " instances are not pointing to the same"
                    " directory. Set the root argument to point to the same"
                    " directory for both instances"
                )

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

    def _construct_path(self):
        
        input_path_prefix = ['sub-','ses-','']
        input_path_elements = [self.subject, self.session, self.datatype]

        path_element_list = [f'{"".join([prefix,path_element])}'
                        if path_element is not None else '*'
                        for prefix, path_element in zip(
                          input_path_prefix,
                          input_path_elements
                      ) 
        ]
        return Path(*path_element_list)
    
    def _construct_file(self):

        attributes = [
            ("sub-", self.subject),
            ("ses-", self.session),
            ("task-",self.task),
            ("run-", self.run),
            ("acq-", self.acquisition),
            ("desc-",self.description),
        ]
            
        if all([attribute is None for _, attribute in attributes]):
            return '*'
        
        first_none_replaced = False
        parts = []

        for key, value in attributes:
            if value is not None:
                parts.append(f"{key}{value}")
                first_none_replaced = False
            elif not first_none_replaced:
                parts.append("*")
                first_none_replaced = True

        filename = "_".join(parts)
        
        if self.suffix is not None:
            filename = filename + f"_{self.suffix}"
        if self.extension is not None:
            if not first_none_replaced and self.suffix is None:
                filename = filename + f"*.{self.extension}"
            else:
                filename = filename + f".{self.extension}"
        elif self.extension is None:
            filename = filename + '*'

        filename = filename.replace("_*_","*")
        return filename

    def _construct_glob_query(self):
        return os.fspath(self._construct_path() / self._construct_file())

    def _get_layout(self) -> pd.DataFrame:
        """Finishing some."""
        
        files_iterator = self.root.rglob(self._construct_glob_query())
        self.file_system = {
            "root": [],
            "subject": [],
            "session": [],
            "datatype": [],
            "task": [],
            "run": [],
            "description": [],
            "acquisition": [],
            "suffix": [],
            "extension": [],
            "filename": [],
        }

        for file in files_iterator:
            self.file_system['root'].append(self.root)
            path_parts = file.relative_to(self.root).parts
            self.file_system['subject'].append(path_parts[0].split('-')[1])
            self.file_system['session'].append(path_parts[1].split('-')[1])
            self.file_system['datatype'].append(path_parts[2])
            desired_keys = ['task','run','desc','acq']
            file_parts = file.stem.split('_')
            

            for desired_key in desired_keys:
                
                if desired_key in file.stem:
                    value = [
                        part.split('-')[1] 
                        for part in file_parts 
                        if desired_key in part
                    ][0]
                else:
                    value = None

                if desired_key == 'desc':
                    desired_key = 'description'
                
                elif desired_key == 'acq':
                    desired_key = 'acquisition'
                
                self.file_system[desired_key].append(value)
                    
            self.file_system['suffix'].append(file_parts[-1])
            self.file_system['extension'].append(file.suffix)
            self.file_system['filename'].append(file)

        return pd.DataFrame(self.file_system)

    def _standardize_input_str(self, value: str) -> str:
        """Standardize the input string for consistency.

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

    def _convert_input_to_list(self, entity: str, value: str | None) -> list | None:
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
        existing_values = self._original_layout.get(target=entity, return_type="id")

        if value == "*" or value is None:
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

    def select(self):
        pass
        