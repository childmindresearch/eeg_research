"""Module that host the BIDS architecture and selector classes.

The pybids or mne-bids packages are a little bit too strict
when it comes to deal with datatype that are not BIDS standardized. Often
in derivatives we have this kind of non-standardized datatype such as
'eyetracking' or 'brainstate' or 'respiration' that are important to separate
as individual modality. The workaround is to either do a monkey patching to the
packages by forcing it to allow such datatype or write from scratch a separate
class that would be less strict.
The monkey patching is not very pythonic and it is usually a bad practice
because of the side effect that can induce. Therefore here is a 'bids-like' data 
handler (also simplier) which will be useful for everybody dealing with pseudo 
BIDS layout with the possibility to customize the queries.

The main class to use is `BidsArchitecture` which give the architecture
(the layout, the structure) of the desired dataset for a specific query.
"""

import copy
import os
from dataclasses import dataclass
from functools import cached_property, reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from warnings import warn

import pandas as pd

class BidsValidationError(Exception):
    pass

def contain_standard_name(key):
    standard_names = [
    "sub",
    "ses",
    "task",
    "acq",
    "run",
    "record",
    "desc",
    ]
    
    
    return any([name in key for name in standard_names])

def get_full_name(key):
    mapping = {
    "sub":"subject",
    "ses": "sessions",
    "task":"task",
    "acq":"acquisition",
    "run": "run",
    "recording":"recording",
    "desc": "description",
    }
    return mapping.get(key)

def _validate_bids_name(elements: list) -> tuple[list, list]:
    """Helper function to validate BIDS naming convention.
    
    Args:
        elements: List of path/filename elements to validate
        
    Returns:
        tuple containing:
        - list of indicator strings for error messages
        - list of boolean validation results
    """
    indicators = []
    element_ok = []
    
    for element in elements:
        if "-" in element and has_standard_name(element.split("-")[0]):
            indicators.append(" " * (len(element)+1))
            element_ok.append(True)
        else:
            indicators.append("^" * (len(element)))
            element_ok.append(False)
            
    return indicators, element_ok

def validate_bids_path(path: Union[str,Path]):
    if isinstance(path, str):
        path = Path(path)
    
    if path.is_file():
        path = path.parents[0]
        
    elements = [os.fspath(path.parents[path_idx]) for path_idx in range(2)]
    indicators, element_ok = _validate_bids_name(elements)
    indicators.insert(0, " "*(len(os.fspath(path.parents[2])) + 1))
    
    if not all(element_ok):
        message = f"Non standardized BIDS name\n{path}\n {' '.join(indicators)}"
        raise BidsValidationError(message)

def validate_bids_file(file):
    elements = os.fspath(file.name).split("_")[:-1]
    indicators, element_ok = _validate_bids_name(elements)
    indicators.insert(0, " "*(len(os.fspath(file.parent))))
    
    if not all(element_ok):
        message = f"Non standardized BIDS name\n{file}\n {' '.join(indicators)}"
        raise BidsValidationError(message)

@dataclass
class BasePath:
    """Base class for handling BIDS-formatted file paths and names.

    This class provides core functionality for working with BIDS-style paths and filenames,
    including path construction, filename parsing, and attribute management.

    Attributes:
        root (Path | None): Root directory path of the BIDS dataset
        subject (str | None): Subject identifier (without 'sub-' prefix)
        session (str | None): Session identifier (without 'ses-' prefix)
        datatype (str | None): Type of data (e.g., 'eeg', 'eyetracking', 'brainstate')
        task (str | None): Task identifier (without 'task-' prefix)
        acquisition (str | None): Acquisition identifier (without 'acq-' prefix)
        run (str | None): Run number (without 'run-' prefix)
        description (str | None): Description identifier (without 'desc-' prefix)
        suffix (str | None): File suffix/type identifier
        extension (str | None): File extension (e.g., '.fif', '.edf')

    Methods:
        _make_path: Constructs the BIDS directory path
        _make_basename: Creates the BIDS-compliant filename without extension
        parse_filename: Extracts BIDS entities from a given filename

    Note:
        This is a base class that implements core BIDS path handling functionality.
        It's meant to be inherited by more specific BIDS handling classes.
    """    
    root: Optional[Path] = None
    subject: Optional[str] = None
    session: Optional[str] = None
    datatype: Optional[str] = None
    task: Optional[str] = None
    acquisition: Optional[str] = None
    run: Optional[str] = None
    description: Optional[str] = None
    suffix: Optional[str] = None
    extension: Optional[str] = None

    def __post_init__(self):
        """Initialize extension format after dataclass initialization.
        
        Ensures extension starts with a period (.) if provided.
        """
        if self.extension and "." not in self.extension:
            self.extension = "." + self.extension

    def __str__(self):
        """Return string representation of the BIDS path.
        
        Returns:
            str: Multi-line string showing all non-private attributes and their values.
        """
        string_list = []
        for attribute, value in self.__dict__.items():
            if not "_" in attribute:
                string_list.append(f"{attribute}: {value}")
        return "\n".join(string_list)

    def _make_path(self, absolute: bool=True) -> Path:
        """Construct BIDS-compliant directory path.

        Args:
            absolute (bool): If True and root is set, returns absolute path.
                            If False, returns relative path.

        Returns:
            Path: BIDS directory path (absolute or relative)
        """
        relative_path = Path(
            os.path.join(
                f"sub-{self.subject}",
                f"ses-{self.session}", 
                self.datatype if self.datatype is not None else "",
            )
        )
        if absolute and getattr(self,"root", False) and self.root is not None:
            return self.root / relative_path
        else:
            return relative_path

    def _make_basename(self):
        """Create BIDS-compliant filename without extension.
        
        Assembles filename components in correct order following BIDS specification.
        Optional components (description, run, acquisition) are inserted if present.

        Returns:
            str: BIDS-compliant filename without extension
        """
        fname_elem = [
            f"sub-{self.subject}",
            f"ses-{self.session}",
            self.suffix,
        ]
        if self.description:
            fname_elem.insert(2, f"desc-{self.description}")
        if self.run:
            fname_elem.insert(2, f"run-{self.run}")
        if self.acquisition:
            fname_elem.insert(2, f"acq-{self.acquisition}")
        if self.task:
            fname_elem.insert(2, f"task-{self.task}")
        
        fname_elem = [elem for elem in fname_elem if elem is not None]
        return "_".join(fname_elem)

        
    def _check_filename(self, file: Union[str, Path]):
        pass

        
        
    def parse_filename(self, file: Union[str, Path]) -> Dict[str, Optional[str]]:
        """Parse BIDS entities from filename.

        Extracts BIDS entities (subject, session, task, etc.) from a BIDS-compliant 
        filename.

        Args:
            file (Path): Path object or string containing BIDS filename to parse

        Returns:
            dict: Dictionary containing extracted BIDS entities
        """
        if isinstance(file, str):
            file = Path(file)

        file_parts: Dict[str, Optional[str]] = {}
        desired_keys = ["run", "desc", "acq", "task"]
        splitted_filename = file.stem.split("_")

        if len(file.parts) > 1:
            file_parts["datatype"] = file.parts[-2]
        else:
            file_parts["datatype"] = splitted_filename[-1]
            
        file_parts["subject"] = file.name.split("_")[0].split("-")[1]
        file_parts["session"] = file.name.split("_")[1].split("-")[1]

        for desired_key in desired_keys:
            if desired_key in file.stem:
                value = [
                    part.split("-")[1]
                    for part in splitted_filename
                    if desired_key in part
                ][0]
            else:
                value = None

            if desired_key == "desc":
                desired_key = "description"
            elif desired_key == "acq":
                desired_key = "acquisition"

            file_parts[desired_key] = value

        file_parts["suffix"] = splitted_filename[-1]
        file_parts["extension"] = file.suffix

        return file_parts


@dataclass
class BidsPath(BasePath):
    """Class for handling BIDS paths with additional functionality.
    
    Extends BasePath with properties for path manipulation and file handling.
    Provides methods to generate filenames and paths following BIDS conventions.

    Attributes:
        Inherits all attributes from BasePath
    """
    subject: str | None = None
    session: str | None = None
    datatype: str | None = None
    task: str | None = None
    suffix: str | None = None
    extension: str | None = None
    root: Path | None = None
    run: str | None = None
    acquisition: str | None = None
    description: str | None = None

    @classmethod
    def from_filename(cls, file: str | os.PathLike) -> 'BidsPath':
        """Create BidsPath instance from existing BIDS filename.

        Args:
            file: Path or string of BIDS-compliant filename

        Returns:
            BidsPath: New instance populated with entities from filename
        """

        file_parts = cls.parse_filename(cls(), Path(file))
        return cls(**file_parts)

    def __post_init__(self) -> None:
        """Initialize the object after __init__ has been called."""
        super().__post_init__()

    @property
    def basename(self) -> str:
        """Get BIDS-compliant filename without extension.

        Returns:
            str: Base filename following BIDS naming convention
        """
        return super()._make_basename()

    @property
    def filename(self) -> str:
        """Get complete filename with extension.

        Returns:
            str: Full filename with extension
        """
        return self.basename + (self.extension or "")

    @property
    def absolute_path(self) -> Path:
        """Get absolute path to file.

        Returns:
            Path: Absolute path if root is set, else relative path with warning
        """
        if self.root:
            return super()._make_path(absolute=True)
        else:
            warn(
                "There was no root path detected. Setting relative "
                "path as the root path"
            )
            return super()._make_path(absolute=False)

    @property
    def relative_path(self) -> Path:
        """Get path relative to root directory.

        Returns:
            Path: Relative path following BIDS folder structure
        """
        return super()._make_path(absolute=False)

    @property
    def fullpath(self) -> Path:
        """Get complete path including filename.

        Returns:
            Path: Full path with filename and extension
        """
        if getattr(self,'root', False):
            return self.absolute_path / self.filename
        else:
            return self.relative_path / self.filename

@dataclass
class BidsQuery(BidsPath):
    """Class for querying BIDS datasets using wildcards and patterns.
    
    Extends BidsPath to support flexible querying of BIDS datasets using
    wildcards and patterns. Handles conversion of query parameters to
    filesystem-compatible glob patterns.

    Attributes:
        Inherits all attributes from BidsPath
        All attributes support wildcards (*) for flexible matching
    """
    root: Optional[Path] = None
    subject: Optional[str] = None
    session: Optional[str] = None
    datatype: Optional[str] = None
    task: Optional[str] = None
    acquisition: Optional[str] = None
    run: Optional[str] = None
    description: Optional[str] = None
    suffix: Optional[str] = None
    extension: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize query parameters with wildcards.
        
        Converts None values to wildcards and adds wildcards to existing values
        where appropriate.
        """
        required_attrs = [
            "subject",
            "session",
            "datatype",
            "suffix",
            "extension",
        ]

        for attr in required_attrs:
            if getattr(self, attr) is None:
                if attr == "extension":
                    setattr(self, attr, ".*")
                else:
                    setattr(self, attr, "*")

        for attr in ["task", "run", "acquisition", "description"]:
            if getattr(self, attr) is not None:
                setattr(self,attr, getattr(self, attr) + "*")
        
        return super().__post_init__()

    @property
    def filename(self) -> str:
        """Get filename pattern for querying.

        Returns:
            str: Filename pattern with wildcards for matching
        """
        potential_cases = [
            "*_*",
            "**",
            "*.*"
        ]
        filename = super().filename
        for case in potential_cases:
            filename = filename.replace(case,"*")
        return filename

    def generate(self) -> Iterator[Path]:
        """Generate iterator of matching files.

        Returns:
            Iterator[Path]: Iterator yielding paths matching query

        Raises:
            Exception: If root path is not defined
        """
        if self.root:
            return self.root.rglob(os.fspath(self.relative_path / self.filename))
        else:
            raise Exception(
                "Root was not defined. Please instantiate the object"
                " by setting root to a desired path"
            )


@dataclass
class BidsArchitecture(BidsQuery):
    """Main class for working with BIDS dataset structure.
    
    Provides comprehensive functionality for querying, selecting, and reporting
    on BIDS datasets. Supports complex queries and selections based on BIDS
    entities.

    Attributes:
        Inherits all attributes from BidsQuery
        database (pd.DataFrame): Cached database of matching files
    """
    
    root: Path
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
        super().__post_init__()
        self._db_lookup_generator = self.generate()
    
    def __next__(self):
        return next(self._db_lookup_generator)
    
    def __iter__(self):
        return iter(self._db_lookup_generator)


    def __add__(self, other):  # noqa: ANN204, D105, ANN001
        #TODO I want to design an add dunder method that can add another 
        # architecture object to the existing one with the possibility of having
        # different roots.
        pass

    @cached_property
    def database(self) -> pd.DataFrame:
        """Scan filesystem and build DataFrame of matching files.

        Args:
            query: Glob pattern for file selection

        Returns:
            pd.DataFrame: DataFrame containing all matching files and their BIDS entities
        """
        database_keys = [
            "root",
            "subject",
            "session",
            "datatype",
            "task",
            "run",
            "acquisition",
            "description",
            "suffix",
            "extension",
            "atime",
            "mtime",
            "ctime",
            "filename",
        ]

        data_base_dict: dict[str, list] = {key: [] for key in database_keys}

        for file in self.generate():
            if 'test' in file.name.lower():
                continue
            bids_path = BidsPath.from_filename(file)
            for key, value in bids_path.__dict__.items():
                if key == "root":
                    data_base_dict['root'].append(self.root)
                else:
                    data_base_dict[key].append(value)

            file_stats = file.stat()
            data_base_dict['atime'].append(int(file_stats.st_atime))
            data_base_dict['mtime'].append(int(file_stats.st_mtime))
            data_base_dict['ctime'].append(int(file_stats.st_ctime))
            data_base_dict['filename'].append(file)


        return pd.DataFrame(data_base_dict)
    
    def print_summary(self) -> str:
        """Public string representation when calling print().

        Returns:
            str: String representation
        """
        if not getattr(self, 'database'):
            raise Exception("No database generated. Run the `get_layout` method"\
                " before.")
            
        str_list = list()
        print(f"Results for query: {self.fullpath}")
        for attribute in self.database.columns:
            if attribute == "filename":
                continue

            all_existing_values = [
                str(val) for val in self.database[attribute].unique()
            ]

            if len(all_existing_values) <= 4:
                specification_str = (
                    f"({ ', '.join([str(s) for s in all_existing_values])})"
                )
            else:
                specification_str = (
                    f"({str(all_existing_values[0])} "
                    f"... {str(all_existing_values[-1])})"
                )

            value_length = str(len(all_existing_values))

            str_list.append(
                f"{str(attribute).capitalize()}s: {value_length} "
                f"{specification_str}"
            )

        all_str = f"{'\n'.join(str_list)}\nFiles: {len(self.database)}"
        return all_str

    def _standardize_input_str(self, value: str) -> str:
        """Remove BIDS prefixes from input strings.

        Standardizes inputs by removing prefixes like 'sub-', 'ses-', etc.

        Args:
            value: Input string to standardize

        Returns:
            str: Standardized string without BIDS prefixes
        """
        prefix_list = [
            "sub-",
            "ses-",
            "task-",
            "acq-",
            "run-",
            "desc-",
        ]
        for prefix in prefix_list:
            if prefix in value.lower():
                return value.replace(prefix, "")

        return value

    def _get_range(
        self,
        dataframe_column: pd.core.series.Series,
        start: Optional[Union[int, str]] = None,
        stop: Optional[Union[int, str]] = None
    ) -> pd.Series:
        if isinstance(start, str):
            start = int(start)

        if isinstance(stop, str):
            stop = int(stop)

        dataframe_column = dataframe_column.apply(lambda s: int(s))

        if start is None or start == "*":
            start = min(dataframe_column)

        if stop is None or stop == "*":
            stop = max(dataframe_column)

        return (start <= dataframe_column) & (dataframe_column < stop)

    def _get_single_loc(
        self, dataframe_column: pd.core.series.Series, value: str
    ) -> pd.core.series.Series:
        locations_found = dataframe_column == value
        if not any(locations_found):
            Warning("No location corresponding found in the database")
            locations_found.apply(lambda s: not (s))

        return locations_found

    def _is_numerical(
        self, dataframe_column: pd.Series
    ) -> pd.Series:
        return all(dataframe_column.apply(lambda x: str(x).isdigit()))

    def _interpret_string(
        self,
        dataframe_column: pd.core.series.Series,
        string: str,
    ) -> pd.core.series.Series:
        """I want to interpret the string when there is a `-`. Check if
        The splitted results are 2 digit. If not throw an error saying that the
        input must be 2 digit separated by a `-` or 1 digit and a wild card `*`
        separated by a `-`. If everything is ok run the get range.

        Args:
            string (str): _description_
        """
        if "-" in string:
            start, stop = string.split("-")
            conditions = [
                (start.isdigit() or start == "*"),
                (stop.isdigit() or stop == "*"),
            ]

            if not all(conditions):
                raise ValueError(
                    "Input must be 2 digits separated by a `-` or "
                    "1 digit and a wild card `*` separated by a `-`"
                )

            return self._get_range(
                dataframe_column=dataframe_column,
                start=int(start) if start.isdigit() else None,
                stop=int(stop) if stop.isdigit() else None,
            )

        else:
            return self._get_single_loc(dataframe_column, string)

    def _perform_selection(
        self, 
        dataframe_column: pd.Series, 
        value: str
    ) -> pd.Series:
        if self._is_numerical(dataframe_column):
            return self._interpret_string(dataframe_column, value)
        else:
            return self._get_single_loc(dataframe_column, value)

    def select(
        self, 
        **kwargs: Dict[str, Union[str, List[str]]]
        ) -> pd.DataFrame:
        """Select files from database based on BIDS entities.

        Supports flexible selection using:
        - Single values
        - Lists of values
        - Ranges (e.g. "1-5")
        - Wildcards

        Args:
            kwargs: key-words argument for the selection:
                subject: Subject identifier(s)
                session: Session identifier(s)
                datatype: Data type identifier(s)
                task: Task identifier(s)
                run: Run number(s)
                acquisition: Acquisition identifier(s)
                description: Description identifier(s)
                suffix: Suffix identifier(s)
                extension: File extension(s)

        Returns:
            pd.DataFrame: Selected subset of database matching criteria
        """
        possible_arguments = [
            "subject",
            "session",
            "datatype",
            "task",
            "run",
            "acquisition",
            "description",
            "suffix",
            "extension",
        ]

        for argument_name in kwargs.keys():
            if argument_name not in possible_arguments:
                raise Exception(
                    "Argument must be one of the following"
                    f"{', '.join(possible_arguments)}"
                )

        condition_for_select = list()
        for name, value in kwargs.items():
            if value is not None:

                if hasattr(value, '__iter__'):
                    col = self.database[name]

                    if isinstance(value, str):
                        value = [value]

                    temp_selection = list()
                    for individual_value in value:
                        individual_value = self._standardize_input_str(individual_value)
                        temp_selection.append(
                            self._perform_selection(col, individual_value)
                        )

                    if len(temp_selection) > 1:
                        temp_selection = reduce(lambda x, y: x | y, temp_selection)

                    else:
                        temp_selection = temp_selection[0]

                    condition_for_select.append(temp_selection)

                else:
                    raise TypeError("Argument must be an iterable,"\
                        f"got {type(value)} instead")

        selection = reduce(lambda x, y: x & y, condition_for_select)
        self.database = self.database.loc[selection]
        return self
    
    def copy(self) -> 'BidsArchitecture':
        """Return a shallow copy of the BidsSelector instance."""
        return copy.copy(self)
    
    def report(self) -> 'BidsDescriptor':
        """Generate report of database contents.

        Returns:
            BidsDescriptor: Reporter object with database summary
        """
        return BidsDescriptor(self.database)


#TODO 
# - Finish to write the dunder methods of BidsDescriptor.
#
# - Add Unix style info on files (accesed date, changed date, modified date).
#
# - Add possibility to check if file exist make a report like (brainstates, eeg, 
#   eyetracking).
#
# - Add listing such as list sessions per subject, list runs per tasks, list tasks
#   per subjects etc.
#   It would be a method like: listing('session', 'subject') or listing('task', 
#   'subject').
#
# - Add possibility to report what file are breaking bids and where in the filename.
#   To do the above add a checker helper that will check if the path name and filename
#   respect the minimum bids standard. The checker will be able to flag where the
#   name breaks the bids in the error message. Design a custom error handling.
#
# - Add a print tree of the selection
#
# - Add possibility to ignore file from a .bidsignore file that will be in the 
#   root folder.
#
# - Write the glob in RUST that will generate a json file to be read in python 
#   and used in a dataframe (pandas or Polar)

@dataclass
class BidsDescriptor:
    """Class for generating reports on BIDS database contents.
    
    Creates summary reports and provides access to unique values for each
    BIDS entity in the database.

    Attributes:
        database (pd.DataFrame): Database of BIDS files
        {entity}s (tuple): Unique values for each BIDS entity
    """
    database: pd.DataFrame

    def __post_init__(self):
        """Initialize reporter by creating entity value tuples."""
        for column in self.database.columns:
            setattr(self,column+"s",tuple(self.database[column].unique()))
    
    
class BidsSelector:
    pass
