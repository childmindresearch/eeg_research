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
import re

import pandas as pd


class BidsValidationError(Exception):
    pass


def validate_bids_file(file: Path) -> bool:
    """Validate the BIDS filename and pathname.
    
    Args:
        file: Path to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        BidsValidationError: If validation fails
    """
    # Define BIDS rules
    valid_keys = {"sub", "ses", "task", "acq", "run", "recording", "desc"}
    valid_datatype_pattern = re.compile(r"^[a-z0-9]+$")
    key_value_pattern = re.compile(r"(?P<key>[a-zA-Z0-9]+)-(?P<value>[a-zA-Z0-9]+)")
    path_pattern = re.compile(r"(sub|ses)-[\w\d]+")

    errors = []

    # Parse file structure
    filename = os.fspath(file.name) if file.suffix else None
    if filename:
        bids_path_parts = file.parent.parts[-3:]

        conditions = (
            bids_path_parts[0].startswith("sub"),
            bids_path_parts[2].startswith("ses"),
            "-" in bids_path_parts[0],
            "-" in bids_path_parts[2],
        )
        if not any(conditions):
            raise BidsValidationError(
                "Path does not contain valid BIDS elements (e.g., 'sub-*')."
                "Should be in the form of"
                "'root/sub-<label>/ses-<label>/<datatype>'"
            )
        datatype = file.parent.parts[-1]
    else:
        bids_path_parts = file.parent.parts[-2:]
        datatype = file.parts[-1]

    if datatype and not valid_datatype_pattern.match(datatype):
        errors.append(
            f"Invalid datatype: '{datatype}' should be a lowercase "
            "alphanumeric string."
        )

    for part in [bids_path_parts[0], bids_path_parts[1]]:
        if not path_pattern.match(part):
            errors.append(
                f"Invalid path component: '{part}' should match the pattern "
                "'<key>-<value>' with key being 'sub' or 'ses'."
            )

    if filename:
        name_parts = file.stem.split("_")

        for i, part in enumerate(name_parts):
            if i == len(name_parts) - 1:
                continue

            match = key_value_pattern.match(part)
            if not match:
                errors.append(
                    f"Invalid format in '{part}': should be '<key>-<value>'"
                )
            else:
                key = match.group("key")
                if key not in valid_keys:
                    errors.append(
                        f"Invalid key '{key}': must be one of {sorted(valid_keys)}"
                    )
    
    path_subject = (
        bids_path_parts[0].split('-')[1] if '-' in bids_path_parts[0] else None
    )
    path_session = (
        bids_path_parts[1].split('-')[1] if '-' in bids_path_parts[2] else None
    )
    
    filename_entities = {}
    name_parts = file.stem.split('_')
    for part in name_parts:
        if '-' in part:
            key, value = part.split('-', 1)
            filename_entities[key] = value
    
    if path_subject and 'sub' in filename_entities:
        if path_subject != filename_entities['sub']:
            errors.append(
                f"Subject mismatch: path has 'sub-{path_subject}' but "
                f"filename has 'sub-{filename_entities['sub']}'"
            )
    
    if path_session and 'ses' in filename_entities:
        if path_session != filename_entities['ses']:
            errors.append(
                f"Session mismatch: path has 'ses-{path_session}' but "
                f"filename has 'ses-{filename_entities['ses']}'"
            )

    if errors:
        message = (
            f"Non-standardized BIDS name\n"
            f"{file}\n\n" + 
            "\n".join(f"{i + 1}. {error}" for i, error in enumerate(errors))
        )
        raise BidsValidationError(message)

    return True


@dataclass
class BasePath:
    """Base class for handling file paths.

    This class provides core functionality for working with paths, including
    path construction and attribute management.

    Attributes:
        root: Root directory path
        subject: Subject identifier
        session: Session identifier
        datatype: Type of data
        suffix: File suffix/type identifier
        extension: File extension
    """

    root: Optional[Path] = None
    subject: Optional[str] = None
    session: Optional[str] = None
    datatype: Optional[str] = None
    suffix: Optional[str] = None
    extension: Optional[str] = None

    def __post_init__(self) -> None:
        """Ensure extension starts with a period if provided."""
        if self.extension and not self.extension.startswith("."):
            self.extension = f".{self.extension}"

    def _make_path(self, absolute: bool = True) -> Path:
        """Construct directory path.

        Args:
            absolute: If True and root is set, returns absolute path.
                     If False, returns relative path.

        Returns:
            Path object representing the constructed path
        """
        components = []
        if self.subject:
            components.append(f"sub-{self.subject}")
        if self.session:
            components.append(f"ses-{self.session}")
        if self.datatype:
            components.append(self.datatype)

        relative_path = Path(*components)
        if absolute and self.root:
            return self.root / relative_path
        return relative_path

    def _make_basename(self) -> str:
        """Create filename without extension.

        Returns:
            Base filename constructed from available attributes
        """
        components = []
        if self.subject:
            components.append(f"sub-{self.subject}")
        if self.session:
            components.append(f"ses-{self.session}")
        if self.suffix:
            components.append(self.suffix)
        return "_".join(components)


@dataclass
class BidsPath(BasePath):
    """BIDS-compliant path handler with query capabilities.

    Extends BasePath with BIDS-specific functionality and query features.
    Handles path construction, validation, and pattern matching for BIDS datasets.

    Attributes:
        task: Task identifier
        run: Run number
        acquisition: Acquisition identifier
        description: Description identifier
    """

    task: Optional[str] = None
    run: Optional[str] = None
    acquisition: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize and normalize BIDS entities."""
        # Define valid prefixes for each attribute
        self._validate_and_normalize_entities()
        super().__post_init__()

    def _validate_and_normalize_entities(self) -> None:
        """Validate and normalize all BIDS entities."""
        prefix_mapping = {
            'subject': 'sub',
            'session': 'ses',
            'task': 'task',
            'run': 'run',
            'acquisition': 'acq',
            'description': 'desc'
        }
        
        for attr, prefix in prefix_mapping.items():
            value = getattr(self, attr)
            if value is not None and isinstance(value, str):
                # First validate
                if '-' in value:
                    given_prefix = value.split('-')[0]
                    if given_prefix != prefix:
                        raise ValueError(
                            f"Invalid prefix in {attr}='{value}'. "
                            f"Expected '{prefix}-' prefix if any, got '{given_prefix}-'"
                        )
                # Then normalize if validation passed
                setattr(self, attr, self._normalize_entity(prefix, value))

    def _normalize_entity(self, prefix: str, value: Optional[str]) -> Optional[str]:
        """Normalize BIDS entity value by removing prefix if present."""
        if value is None:
            return None
            
        value = value.strip()
        prefix_pattern = f"^{prefix}-"
        if re.match(prefix_pattern, value):
            return value[len(prefix)+1:]
        
        return value

    def _check_filename(self, file: Union[str, Path]) -> None:
        """Validate BIDS naming conventions.

        Args:
            file: Path to validate

        Raises:
            BidsValidationError: If file doesn't follow BIDS conventions
        """
        if isinstance(file, str):
            file = Path(file)
        validate_bids_file(file)

    def _make_basename(self) -> str:
        """Create BIDS-compliant filename without extension.

        Returns:
            str: BIDS-compliant filename
        """
        components = [f"sub-{self.subject}", f"ses-{self.session}"]

        # Optional entities in BIDS-specified order
        if self.task:
            components.append(f"task-{self.task}")
        if self.acquisition:
            components.append(f"acq-{self.acquisition}")
        if self.run:
            components.append(f"run-{self.run}")
        if self.description:
            components.append(f"desc-{self.description}")

        if self.suffix:
            components.append(self.suffix)

        return "_".join(filter(None, components))

    @property
    def basename(self) -> str:
        """Get BIDS-compliant filename without extension."""
        return self._make_basename()

    @property
    def filename(self) -> str:
        """Get complete filename with extension."""
        return f"{self.basename}{self.extension or ''}"

    @property
    def fullpath(self) -> Path:
        """Get complete path including filename."""
        path = self._make_path(absolute=bool(self.root))
        return path / self.filename

    def match_pattern(self, pattern: str = "*") -> bool:
        """Check if path matches given pattern.

        Args:
            pattern: Glob pattern to match against

        Returns:
            True if path matches pattern, False otherwise
        """
        return Path(self.filename).match(pattern)

    @classmethod
    def from_filename(cls, file: Union[str, Path]) -> "BidsPath":
        """Create BidsPath instance from existing filename.

        Args:
            file: BIDS-compliant filename or path

        Returns:
            New BidsPath instance with normalized entities
        """
        if isinstance(file, str):
            file = Path(file)

        # Parse BIDS entities from filename
        entities: Dict[str, Optional[str]] = {}
        
        # Extract path components
        if len(file.parts) > 2:
            entities["datatype"] = file.parts[-2]
            # Already normalized since we're extracting from BIDS structure
            entities["subject"] = file.parts[-3].split("-")[1]
            if len(file.parts) > 3:
                entities["session"] = file.parts[-4].split("-")[1]

        # Parse filename components
        name_parts = file.stem.split("_")
        for part in name_parts:
            if "-" in part:
                key, value = part.split("-", 1)
                # Values are already normalized since we're parsing from BIDS format
                if key == "sub":
                    entities["subject"] = value
                elif key == "ses":
                    entities["session"] = value
                elif key == "task":
                    entities["task"] = value
                elif key == "acq":
                    entities["acquisition"] = value
                elif key == "run":
                    entities["run"] = value
                elif key == "desc":
                    entities["description"] = value

        entities["suffix"] = name_parts[-1]
        entities["extension"] = file.suffix

        return cls(**entities)


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
                setattr(self, attr, getattr(self, attr) + "*")

        return super().__post_init__()

    @property
    def filename(self) -> str:
        """Get filename pattern for querying.

        Returns:
            str: Filename pattern with wildcards for matching
        """
        potential_cases = ["*_*", "**", "*.*"]
        filename = super().filename
        for case in potential_cases:
            filename = filename.replace(case, "*")
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


class BidsArchitecture:
    """Main class for working with BIDS dataset structure.

    Provides comprehensive functionality for querying, selecting, and reporting
    on BIDS datasets. Uses composition with BidsPath for path handling.

    Attributes:
        root (Path): Root directory of the BIDS dataset
        database (pd.DataFrame): Cached database of matching files
    """

    def __init__(
        self,
        root: Path,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        datatype: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        acquisition: Optional[str] = None,
        description: Optional[str] = None,
        suffix: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> None:
        """Initialize BidsArchitecture.

        Args:
            root: Root directory of the BIDS dataset
            subject: Subject identifier or pattern
            session: Session identifier or pattern
            datatype: Data type identifier or pattern
            task: Task identifier or pattern
            run: Run number or pattern
            acquisition: Acquisition identifier or pattern
            description: Description identifier or pattern
            suffix: Suffix identifier or pattern
            extension: File extension or pattern
        """
        self.root = root
        self._path_handler = BidsPath(
            root=root,
            subject=subject or "*",
            session=session or "*",
            datatype=datatype or "*",
            task=task,
            run=run,
            acquisition=acquisition,
            description=description,
            suffix=suffix or "*",
            extension=extension or ".*",
        )
        self._database: Optional[pd.DataFrame] = None

    @cached_property    
    def database(self) -> pd.DataFrame:
        """Get or create database of matching files.

        Returns:
            pd.DataFrame: Database containing all matching files and their BIDS entities
        """
        if self._database is None:
            self._database = self._create_database()
        return self._database
    
    def _get_unique_values(self, column: str) -> List[str]:
        """Get sorted unique non-None values for a given column.
        
        Args:
            column: Name of the database column
            
        Returns:
            List[str]: Sorted list of unique non-None values
        """
        return sorted([elem for elem in self.database[column].unique() 
                      if elem is not None])

    @property
    def subjects(self) -> List[str]:
        """Get unique subject identifiers.
        
        Returns:
            List[str]: Sorted list of subject IDs
        """
        return self._get_unique_values('subject')
    
    @property
    def sessions(self) -> List[str]:
        """Get unique session identifiers.
        
        Returns:
            List[str]: Sorted list of session IDs
        """
        return self._get_unique_values('session')
    
    @property
    def datatypes(self) -> List[str]:
        """Get unique datatypes.
        
        Returns:
            List[str]: Sorted list of datatypes
        """
        return self._get_unique_values('datatype')
    
    @property
    def tasks(self) -> List[str]:
        """Get unique task identifiers.
        
        Returns:
            List[str]: Sorted list of task IDs
        """
        return self._get_unique_values('task')
    
    @property
    def runs(self) -> List[str]:
        """Get unique run numbers.
        
        Returns:
            List[str]: Sorted list of run numbers
        """
        return self._get_unique_values('run')
    
    @property
    def acquisitions(self) -> List[str]:
        """Get unique acquisition identifiers.
        
        Returns:
            List[str]: Sorted list of acquisition IDs
        """
        return self._get_unique_values('acquisition')
    
    @property
    def descriptions(self) -> List[str]:
        """Get unique description identifiers.
        
        Returns:
            List[str]: Sorted list of description IDs
        """
        return self._get_unique_values('description')
    
    @property
    def suffixes(self) -> List[str]:
        """Get unique suffixes.
        
        Returns:
            List[str]: Sorted list of suffixes
        """
        return self._get_unique_values('suffix')
    
    @property
    def extensions(self) -> List[str]:
        """Get unique file extensions.
        
        Returns:
            List[str]: Sorted list of file extensions
        """
        return self._get_unique_values('extension')
    
    def _create_database(self) -> pd.DataFrame:
        """Scan filesystem and build DataFrame of matching files.

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

        data: Dict[str, List[Any]] = {key: [] for key in database_keys}

        pattern = self._path_handler.filename
        for file in self.root.rglob(pattern):
            if "test" in file.name.lower():
                continue

            try:
                bids_path = BidsPath.from_filename(file)
                for key, value in bids_path.__dict__.items():
                    if key == "root":
                        data["root"].append(self.root)
                    else:
                        data[key].append(value)

                file_stats = file.stat()
                data["atime"].append(int(file_stats.st_atime))
                data["mtime"].append(int(file_stats.st_mtime))
                data["ctime"].append(int(file_stats.st_ctime))
                data["filename"].append(file)
            except BidsValidationError:
                warn(f"Skipping invalid BIDS file: {file}")
                continue

        return pd.DataFrame(data)

    def _get_range(
        self,
        dataframe_column: pd.core.series.Series,
        start: int | str | None = None,
        stop: int | str | None = None,
    ) -> pd.core.series.Series:
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
        self, 
        dataframe_column: pd.core.series.Series, 
        value: str
    ) -> pd.core.series.Series:
        locations_found = dataframe_column == value
        if not any(locations_found):
            Warning("No location corresponding found in the database")
            locations_found.apply(lambda s: not (s))

        return locations_found

    def _is_numerical(
        self,
        dataframe_column: pd.core.series.Series
    ) -> pd.core.series.Series:
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
        self, dataframe_column: pd.core.series.Series, value: str
    ) -> pd.core.series.Series:
        if self._is_numerical(dataframe_column):
            return self._interpret_string(dataframe_column, value)
        else:
            return self._get_single_loc(dataframe_column, value)

    def select(self, **kwargs) -> "BidsArchitecture":
        """Select files from database based on BIDS entities.

        Args:
            **kwargs: Entity criteria to match (e.g., subject="001", task="rest")

        Returns:
            New BidsArchitecture instance with filtered database
        
        Raises:
            ValueError: If an invalid selection key is provided
        """
        valid_keys = [
            "subject", "session", "datatype", "task", "run",
            "acquisition", "description", "suffix", "extension"
        ]

        # Validate keys
        for key in kwargs:
            if key not in valid_keys:
                raise ValueError(f"Invalid selection key: {key}")

        # Create new instance
        new_instance = copy.deepcopy(self)
        
        # Build combined mask for all conditions
        mask = pd.Series(True, index=self.database.index)
        
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, list):
                    mask &= self.database[key].isin(value)
                elif isinstance(value, str):
                    value = value.strip()
                    if value:  # Skip empty strings
                        mask &= self._perform_selection(self.database[key], value)
        
        if hasattr(new_instance, 'database'):
            delattr(new_instance, 'database')
        setattr(new_instance, 
                'database', 
                self.database[mask].copy()
                )
        
        return new_instance
