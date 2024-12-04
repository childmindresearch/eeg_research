"""Module that host the BIDS architecture and selector classes.

The pybids or mne-bids packages are a little bit too strict
when it comes to deal with datatype that are not BIDS standardized. Often
in derivatives we have this kind of non-standardized datatype such as
'eyetracking' or 'brainstate' or 'respiration' that are important to separate
as individual modality. The workaround is to either do a monkey patching to the
packages by forcing it to allow such datatype or write from scratch a separate
class that would be less strict.
The monkey patching is not very pythonic and is not recommended in the dev community
Because of the side effect that can induce. Therefore I am on the road to 
write another 'bids-like' data handler (also simplier) from scratch.
This will be useful for everybody dealing with pseudo BIDS layout with the 
possibility to customize the queries.
"""


import os
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import pandas as pd


def is_numerical(dataframe: pd.DataFrame, column_name: str):
    return all(dataframe[column_name].apply(lambda string: string.isdigit()))

@dataclass
class BidsArchitecture:
    """A flexible and lightweight BIDS file selector.

    This class provides an intuitive interface to query and select files from BIDS datasets,
    including non-standardized derivatives. It handles custom modalities like eyetracking,
    brainstate, or respiration data that standard BIDS tools may not support.

    Key Features:
        - Flexible file selection using wildcards and patterns
        - Support for non-standard BIDS derivatives
        - Simple query interface for complex file patterns
        - Pandas DataFrame output for easy data manipulation

    Args:
        root (Path | str): Root directory of the BIDS dataset
        subject (str, optional): Subject identifier (e.g. "01")
        session (str, optional): Session identifier
        datatype (str, optional): Data modality (e.g. "eeg", "eyetracking")
        task (str, optional): Task identifier
        run (str, optional): Run number
        acquisition (str, optional): Acquisition identifier
        description (str, optional): Description tag
        suffix (str, optional): File suffix
        extension (str, optional): File extension without dot

    Examples:
        >>> selector = BidsArchitecture(root="data/")
        >>> selector = BidsArchitecture(
        ...     root="data/",
        ...     subject="01",
        ...     task="rest",
        ...     datatype="eeg"
        ... )

    Properties:
        files_dataframe (pd.DataFrame): Selected files with BIDS entities as columns
        query_instruction (str): Current query pattern used for file selection

    Methods:
        _construct_path(): Builds BIDS directory path pattern
        _construct_file(): Builds BIDS filename pattern
        _construct_glob_query(): Combines path and file patterns
        _get_layout(): Scans filesystem and builds DataFrame
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
        self._get_layout(self._construct_glob_query())
        
    def __str__(self) -> str:
        """Public string representation when calling print().

        Returns:
            str: String representation
        """
        str_list = list()
        print(f"Results for query: {self.query_instruction}")
        for attribute in self.database.columns:
            if attribute == "filename":
                continue

            all_existing_values = [
                str(val)
                for val in self.database[attribute].unique()
                ]

            if len(all_existing_values) <= 4:
                specification_str = (
                    f"({ ', '.join([str(s) for s in all_existing_values])})"
                )
            else:
                specification_str = (
                    f"({str(all_existing_values[0])} "\
                    f"... {str(all_existing_values[-1])})"
                )

            value_length = str(len(all_existing_values))

            str_list.append(
                f"{str(attribute).capitalize()}s: {value_length} "
                f"{specification_str}"
            )

        all_str = f"{'\n'.join(str_list)}\nFiles: {len(self.database)}"
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

    def _construct_path(self) -> Path:
        """Build the BIDS directory path pattern.
        
        Constructs a Path object representing the directory structure:
        sub-{subject}/ses-{session}/{datatype}
        
        Returns:
            Path: Directory path pattern with appropriate wildcards
        """
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
    
    def _construct_file(self) -> str:
        """Build the BIDS filename pattern.
        
        Constructs a filename pattern following BIDS conventions:
        sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-{desc}_{suffix}.{ext}
        
        Returns:
            str: Filename pattern with appropriate wildcards
        """

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
            if value:
                parts.append(f"{key}{value}")
                first_none_replaced = False
            elif not first_none_replaced:
                parts.append("*")
                first_none_replaced = True

        filename = "_".join(parts)
        
        if self.suffix:
            filename = f"{filename}_{self.suffix}"
        
        if self.extension:
            if not first_none_replaced and self.suffix is None:
                filename = filename + f"*.{self.extension}"
            else:
                filename = filename + f".{self.extension}"
        else:
            filename = filename + '*'

        filename = filename.replace("_*_","*").replace("**","*")
        return filename

    def _construct_glob_query(self) -> str:
        """Combine path and filename patterns for filesystem query.
        
        Returns:
            str: Complete glob pattern for file selection
        """
        self.query_instruction = os.fspath(
            self._construct_path() / self._construct_file()
            )
        return self.query_instruction
    
    def parse_filename(self, file: Path | str) -> dict:
        """Extract BIDS entities from filename.
        
        Args:
            file: Path or string of file to parse
            
        Returns:
            dict: BIDS entities extracted from filename
                Keys: root, subject, session, datatype, task, run,
                      acquisition, description, suffix, extension, filename
        """
        if isinstance(file, str):
            file = Path(file)

        path_parts = file.relative_to(self.root).parts

        file_parts = {}
        file_parts['root'] = self.root
        file_parts['subject'] = file.name.split('_')[0].split('-')[1]
        file_parts['session'] = file.name.split('_')[1].split('-')[1]
        try:
            file_parts['datatype'] = path_parts[2]
        except Exception:
            Warning('No datatype detected')

        desired_keys = ['task','run','desc','acq']
        splitted_filename = file.stem.split('_')
        for desired_key in desired_keys:
            
            if desired_key in file.stem:
                value = [
                    part.split('-')[1] 
                    for part in splitted_filename
                    if desired_key in part
                ][0]
            else:
                value = None

            if desired_key == 'desc':
                desired_key = 'description'
            
            elif desired_key == 'acq':
                desired_key = 'acquisition'
            
            file_parts[desired_key] = value
                
        file_parts['suffix'] = splitted_filename[-1]
        file_parts['extension'] = file.suffix
        file_parts['filename'] = file

        return file_parts

    def _get_layout(self, query: str = '*') -> pd.DataFrame:
        """Scan filesystem and build DataFrame of matching files.
        
        Args:
            query: Glob pattern for file selection
            
        Returns:
            pd.DataFrame: DataFrame containing all matching files and their BIDS entities
        """
        files_iterator = Path(self.root).rglob(query)
        database_keys = ['root','subject','session','datatype','task','run', 
                         'acquisition', 'description','suffix','extension','filename']

        data_base_dict: dict[str, list] = {key: [] for key in database_keys}

        for file in files_iterator:
            file_parts = self.parse_filename(file)
            for key, value in file_parts.items():
                data_base_dict[key].append(value)

        self.database = pd.DataFrame(data_base_dict)

        return self

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
            "run-",
            "desc-",
        ]
        for prefix in prefix_list:
            if prefix in value.lower():
                return value.replace(prefix, "")

        return value
    
    def _get_range(self,
                   dataframe_column: pd.core.series.Series,
                   start: int | str | None = None,
                   stop: int | str | None = None) -> pd.core.series.Series:
        
        if isinstance(start, str):
            start = int(start)

        if isinstance(stop, str):
            stop = int(stop)

        dataframe_column = dataframe_column.apply(lambda s: int(s))

        if start is None or start == '*':
            start = min(dataframe_column)
            
        if stop is None or stop == '*':
            stop = max(dataframe_column)
        
        return (start <= dataframe_column) & (dataframe_column < stop)
    
    def _get_single_loc(self,
                        dataframe_column: pd.core.series.Series,
                        value: str) -> pd.core.series.Series:
        
        locations_found = dataframe_column == value
        if not any(locations_found):
            raise Exception('No location corresponding found in the database')
        else:
            return dataframe_column == value

    def _is_numerical(self,
                      dataframe_column: pd.core.series.Series
                      ) -> pd.core.series.Series:
        return all(
            dataframe_column.apply(lambda string: string.isdigit())
            )
    
    def _interpret_string(self,
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
        if '-' in string:
            start, stop = string.split('-')
            conditions = [
                (start.isdigit() or start == '*'),
                (stop.isdigit() or stop == '*')
            ]
                 
            if not all(conditions):
                raise ValueError("Input must be 2 digits separated by a `-` or "\
                    "1 digit and a wild card `*` separated by a `-`")

            return self._get_range(
                dataframe_column=dataframe_column,
                start=int(start) if start.isdigit() else None,
                stop=int(stop) if stop.isdigit() else None
            )

        else:
            return self._get_single_loc(dataframe_column, string)
        
    def _perform_selection(self,
                           dataframe_column: pd.core.series.Series,
                           value: str) -> pd.core.series.Series:

        if self._is_numerical(dataframe_column):
            return self._interpret_string(dataframe_column,value)
        else:
            return self._get_single_loc(dataframe_column, value)
        
    def select(self, **kwargs) -> pd.DataFrame:
        """Select files from database based on BIDS entities.
        
        If the argument is not None check if it is a list or a string. If it is
        a string then check if there is any `-` and `*`. If it is the case then
        call the string interpreter to generate a truth table of a selection of
        a range. If not then only get the truth table directly with an == passed
        on the dataframe. Check if the string value exist in the unique values
        of the dataframe. If not throw an error.
        If it is a list then iterate over the list and for each value of the list
        perfor the check above. The truth table should be a list that is appending
        as a function of the iteration over argument to generate a final table
        that would be the selection. 
        
        Remember to throw a warning if all the table is False (if not any()).

        Args:
            subject: Subject identifier
            session: Session identifier
            datatype: Data type identifier
            task: Task identifier
            run: Run identifier
            acquisition: Acquisition identifier
            description: Description identifier
            suffix: Suffix identifier
            extension: File extension
            Returns:
            pd.DataFrame: DataFrame containing selected files and their BIDS entities
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
                raise Exception("Argument must be one of the following"\
                    f"{', '.join(possible_arguments)}")
        
        condition_for_select = list()
        for name, value in kwargs.items(): 

            if value is not None:
                conditions = [
                    isinstance(value,str),
                    isinstance(value,list)
                ]

                if any(conditions):
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
                    raise TypeError('Argument must be either string or list')
        
        selection = reduce(lambda x, y: x & y, condition_for_select)
        return self.database.loc[selection]
                    