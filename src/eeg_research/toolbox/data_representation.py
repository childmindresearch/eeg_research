import numpy as np
import pickle
import os
from pathlib import Path
from bids import BIDSLayout
"""
I want to be able to get a multidimensional numpy arrays for each modality. 
This way it will be easier to compute along desired axis such as training ML
on cross-subject way or intra-subject (using sessions). All data will be
available in one array and be easily accessible for any configurations.

Then I want to make a multimodal object that can combine several modality
together while checking if for a specific subject/session/task the modality
exist.

First the instance will need to be able to read data for given input following
the BIDS format (root path, then subject(s), session(s), task(s), run(s).
It will be possible to give a list as input to read and concatenate data. For
example if we want to create a EEGData object for sub-01, sub-02, and sub-03
a list of the 3 subject names will be pass as input. The list can be: 
- The formated subject identifier following BIDS: sub-<label>
- Only the label
- Only a list of integer
- A regular expression. 
The class will convert into BIDS format the input in any way. 
From this input a list of all existing files will be created and depending on 
the format (pickle or NWB) data will be loaded and converted into numpy array.
The array will be concatenated to give a shape of 
(subjects, sessions, tasks, runs, channels, time, frequency). This will help to 
vectorize and speed up things, but also help to perform different sort of 
analysis such as cross subject or inter subject, cross-session or cross-runs 
etc.
The term channels here has a broad meaning and can signify either eeg channels
or emg channels or other different component for a same modality (for example
brainstates, pupil size and its first and second derivatives etc.)
The array will be referenced by different indexation which will be lists (or dict) 
of string values to identify which index is which subject or session or runs.
Thanks to that it will be possible to pick a subset of the data by calling a 
method.

Some subjects have only 1 session and some 2 sessions. It will generate an imbalance
in the array. To take care of that several scenario are possible:
- If sessions = None, then all sessions will be considered and be concatenated
  in the array. In the case that the number of sessions don't match, the array
  of the subject having less session will be populated with nan for the missing session.
  But there will never be a reference to that missing session in the keys.

- If sessions = list, then ONLY subjects that have ALL the sessions in the list
  are considered and concatenated.

- If sessions = str, then ONLY subjects that have THIS session are considered.

This will be the same for task and runs.

"""

from eeg_research.system.bids_selector import BIDSselector
from dataclasses import dataclass

@dataclass
class MultimodalData:
    pass

@dataclass
class BaseModality:
        
    root: str | os.PathLike
    subject: str | None
    session: str | None
    task: str | None
    run: str | None

    def __post_init__(self) -> None:
        self.selector = BIDSselector(root = self.root,
                                subject=self.subject,
                                session=self.session,
                                task=self.task,
                                run=self.run)

    def __str__(self):
        print(self.selector)
    
    
    def read_data(self, 
                  datatype: str, 
                  suffix: str) -> 'BaseModality':
        """Read and concatenate data from the selection.
        
        For now data has to be saved into a dictionary object that is pickled.

        Returns:
            BaseModality: _description_
        """
        
        self.selector.datatype = datatype
        self.selector.suffix = suffix
        idx_list = list()
        all_data_list = list()
        data_shape_list = list()
        for sub_idx in range(self.selector.subject):
            for ses_idx in range(self.selector.session):
                for task_idx in range(self.selector.task):
                    for run_idx in range(self.selector.run):
                        idx_list.append(
                            (sub_idx, ses_idx, task_idx, run_idx)
                            )
        
        for filename in self.selector.layout:
            with open(filename, 'rb',) as file:
                data_array = pickle.load(file)

            
            data_shape_list.append(data_array['feature'])
            all_data_list.append(data_array['feature'])

        pass
        return self
        


class EEGData(BaseModality):
    def __init__(self,
                 subject: str | list[str] | None, 
                 session: str | list[str] | None, 
                 task: str | list[str] | None,
                 run: str | list[str] | None,) -> None:

        super().__init__(subject,
                         session, 
                         task, 
                         run, 
        )

        
        pass

class BrainStateData(BaseModality):
    def __init__(self,
                 subjects: str | list[str] | None, 
                 sessions: str | list[str] | None, 
                 tasks: str | list[str] | None,
                 runs: str | list[str] | None,) -> None:

        super().__init__(subjects,
                         sessions,
                         tasks,
                         runs, 
        )
        
        pass

class EyeTrackingData(BaseModality):
    def __init__(self,
                 subjects: str | list[str] | None, 
                 sessions: str | list[str] | None, 
                 tasks: str | list[str] | None, 
                 acquisitions: str | list[str] | None, 
                 runs: str | list[str] | None,) -> None:

        super().__init__(subjects, 
                         sessions, 
                         tasks, 
                         runs, 
        )

