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

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import ConnectionPatch
from mne.preprocessing import annotate_muscle_zscore


class ZscoreAnnotator:
    """A class to perform artifacts annotation with zScore.
    
    Args:
        raw (mne.io.Raw): The EEG data to annotate
    """
    def __init__(self, raw: mne.io.Raw) -> None:  # noqa: D107
        self.raw = raw
        self.artifacts_general_annotations: list = list()
    
    def detect_muscles(self, **kwargs: dict) -> 'ZscoreAnnotator':
        """Wrapper around the mne function to annotate muscle.
        
        Args:
            kwargs(dict): A dictionnary containing the arguments for
                                  the `annotate_muscle_zscore` to be parsed.
        
        Returns:
            The ZscoreAnnotator instance
        """
        muscle_annotations, _ = annotate_muscle_zscore(self.raw, **kwargs)
        self.artifacts_general_annotations.append(muscle_annotations)

        return self
        
    def detect_other_artifacts( 
                        self,
                        description: str = 'BAD_others',
                        channel_type: str | None ='eeg', 
                        z_thresh: float=3.5, 
                        min_artifact_gap: float | None =0.1, 
                        minimum_duration: float | None =0.2,
                        filtering: tuple = (None, 8.0),
                        ) -> 'ZscoreAnnotator':
        """Annotate artifacts in raw EEG data based on a z-score threshold.
        
        Parameters:
        - raw: Raw object from MNE containing EEG data.
        - channel_type: Type of channels to analyze.
        - z_thresh: Z-score threshold to use for detecting artifacts.
        - min_artifact_gap: Minimum time in seconds between separate artifacts; 
                            below this, artifacts will be grouped.
        - minimum_duration: Minimum duration for each annotation. 
                            If an annotation is shorter, it is adjusted.
        
        Returns:
        - annotations: MNE Annotations object with detected, grouped, 
                       and adjusted artifacts.
        """
        raw_copy = self.raw.copy()
        if filtering:
            raw_copy.filter(*filtering)
        if channel_type:
            picks = mne.pick_types(raw_copy.info,
                                   meg=False, 
                                   eeg=(channel_type=='eeg'), 
                                   eog=False)
        data, times = raw_copy[picks]
        z_scores = (np.abs((data - np.mean(data, axis=1, keepdims=True)) / 
                           np.std(data, axis=1, keepdims=True)))
        artifacts = (z_scores > z_thresh).any(axis=0)
        gradient = np.diff(artifacts, prepend=0)
        rising_edge_idx = np.where(gradient == 1)[0]
        falling_edge_idx = np.where(gradient == -1)[0]
        if sum(artifacts) == 0:
            return mne.Annotations()

        onsets = times[rising_edge_idx]
        ends = times[falling_edge_idx]
        durations = np.array(ends) - np.array(onsets)
        adjusted_onsets: list = list()
        adjusted_durations: list = list()
        last_end = 0

        for i, (onset, duration) in enumerate(zip(onsets, durations)):
            if minimum_duration and duration < minimum_duration:
                new_onset = max(0, onset - (minimum_duration - duration) / 2)
                new_duration = minimum_duration
            else:
                new_onset = onset
                new_duration = duration
            
            if adjusted_onsets and new_onset - last_end <= min_artifact_gap:
                adjusted_durations[-1] = new_onset + new_duration - adjusted_onsets[-1]
            else:
                adjusted_onsets.append(new_onset)
                adjusted_durations.append(new_duration)
            
            last_end = adjusted_onsets[-1] + adjusted_durations[-1]

        descriptions = [description] * len(adjusted_onsets)
        self.artifacts_general_annotations.append(
            mne.Annotations(
            onset=adjusted_onsets, 
            duration=adjusted_durations, 
            description=descriptions,
            orig_time=self.raw.info['meas_date']
            )
        )
        return self
    
    def merge_annotations(self) -> 'ZscoreAnnotator':
        """Merge MNE Annotations objects into a single Annotations object.
        
        Overlapping annotations are merged into a single annotation with the 
        description as a combination of the overlapping annotation descriptions.
        
        Returns:
        - merged_annotations: MNE Annotations object containing all merged annotations
        """
       # Initialize empty lists for onsets, durations, and descriptions
        all_onsets = []
        all_durations = []
        all_descriptions = []
        
        # Collect all annotations
        for annotations in self.artifacts_general_annotations:
            all_onsets.extend(annotations.onset)
            all_durations.extend(annotations.duration)
            all_descriptions.extend(annotations.description)
        
        # Convert to arrays for vectorized operations
        all_onsets = np.array(all_onsets) #type: ignore
        all_durations = np.array(all_durations) #type: ignore
        all_descriptions = np.array(all_descriptions) #type: ignore
        
        # Sort by onsets
        sorted_indices = np.argsort(all_onsets)
        all_onsets = all_onsets[sorted_indices]
        all_durations = all_durations[sorted_indices]
        all_descriptions = all_descriptions[sorted_indices]
        
        merged_onsets = [all_onsets[0]]
        merged_durations = [all_durations[0]] 
        merged_descriptions = [all_descriptions[0]]
        
        for i in range(1, len(all_onsets)):
            current_start = all_onsets[i]
            current_end = current_start + all_durations[i]
            last_end = merged_onsets[-1] + merged_durations[-1]
            
            if current_start <= last_end:
                merged_durations[-1] = max(last_end, current_end) - merged_onsets[-1]
                if all_descriptions[i] not in merged_descriptions[-1]:
                    
                    merged_descriptions[-1] += '_' + all_descriptions[i][4:]
            else:
                merged_onsets.append(current_start)
                merged_durations.append(all_durations[i])
                merged_descriptions.append(all_descriptions[i])
        
        self.artifact_annotations = mne.Annotations(onset=merged_onsets,
                                            duration=merged_durations,
                                            description=merged_descriptions,
                                            orig_time=self.raw.info['meas_date'])
        return self
        
    def compute_statistics(self) -> 'ZscoreAnnotator':
        """Compute the portion of the signal that is polluted."""
        mask_containing_bad = ["BAD" 
                               in description 
                               for description 
                               in self.artifact_annotations.description]
        index_bad = np.where(mask_containing_bad)
        tot_bad_seconds = self.artifact_annotations.duration[index_bad].sum()
        tot_good_seconds = self.raw.times[-1] - tot_bad_seconds

        self.statistics = dict(
            tot_good = dict(
                seconds = tot_good_seconds,
                ratio = tot_good_seconds /self.raw.times[-1]
                ),
            tot_bad = dict(
                number = sum(mask_containing_bad),
                seconds = tot_bad_seconds,
                ratio = tot_bad_seconds /self.raw.times[-1])
        )
        descriptions = np.unique(self.artifact_annotations.description)
        self.statistics['tot_bad'].update(dict(
            artifact_types = descriptions
        ))
        for description in descriptions:
            description_index = np.where(
                self.artifact_annotations.description == description
                )[0]
            tot_sec_this_description = (
                self.artifact_annotations.duration[description_index].sum()
            )
            self.statistics[str(description)] = dict(
                seconds = tot_sec_this_description,
                ratio = tot_sec_this_description /self.raw.times[-1],
                ratio_from_bad = tot_sec_this_description /tot_bad_seconds,
        )
        return self

    def print_statistics(self) -> 'ZscoreAnnotator':
        """Print in the prompt the quantity of signal polluted."""
        default_message = "STATISTICS NOT COMPUTED"
        if not getattr(self, 'statistics', False):
            print(default_message)
            return self
        
        messages_list: list[str] = list()
        messages_list.extend(f"""
ARTIFACT ANNOTATIONS STATISTICS
        EEG total duration:.................... {np.round(self.raw.times[-1],2)}s
        Number of bad segment annotated:....... {
            np.round(self.statistics['tot_bad']['number'],2)}
        Total duration of bad segments:........ {
            np.round(self.statistics['tot_bad']['seconds'],2)}s ({
            np.round(self.statistics['tot_bad']['ratio']*100,2)
            }%)
        Total duration of good signal:......... {
            np.round(self.statistics['tot_good']['seconds'],2)}s ({
            np.round(self.statistics['tot_good']['ratio']*100,2)
            }%)

        Types of artifacts annotated: {', '.join(
            self.statistics['tot_bad']['artifact_types']
            )}""")

        for artifact_type in self.statistics['tot_bad']['artifact_types']:
            this_artifact = self.statistics[artifact_type]
            
            this_artifact_duration = np.round(this_artifact['seconds'],2)
            this_artifact_perc = np.round(this_artifact['ratio']*100,2)
            
            messages_list.extend(f"""
                |__{artifact_type} duration (sec): {this_artifact_duration}s ({
                this_artifact_perc}%)""")
        
        self.statistics_message = ''.join(messages_list)
        print(self.statistics_message)
        return self

    def annotate(self, overwrite: bool = False) -> 'ZscoreAnnotator':
        """Write the annotation to the raw object."""
        if not getattr(self,'artifact_annotations', False):
            self.merge_annotations()
        if overwrite:
            to_write = self.artifact_annotations
        else:
            to_write = self.raw.annotations + self.artifact_annotations
        self.raw.set_annotations(to_write)
        return self
    
    def write_statistics(self, 
                         saving_filename: str | os.PathLike) -> 'ZscoreAnnotator':
        """Write into an external file the computed statistics.
        
        Args:
            saving_filename (str | os.PathLike): The full path and name of
                                                 the file to be written.
                                                 If no file extension is 
                                                 provided, txt will be chosen
        
        Returns:
            The ZscoreAnnotator instance
        """
        base_filename, extension = os.path.splitext(saving_filename)
        extension_dont_exists = '' in extension
        if extension_dont_exists:
            extension = '.txt'

        filename = base_filename + extension
        with open(filename,'w') as file:
            file.write(self.statistics_message)
        
        print(f'Written into {filename}')
        
        return self

    def plot_statistics(self) -> plt.figure:
        """Plot the statistics of bad segment compared to good ones.
        
        Returns:
            fig: The matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(wspace=0)

        # pie chart parameters
        overall_ratios = [self.statistics['tot_bad']['ratio'], 
                          self.statistics['tot_good']['ratio']
        ]
        labels = ['Bad segments', 'Good signal']
        explode = [0.1, 0]
        # rotate so that first wedge is split by the x-axis
        angle = -180 * overall_ratios[0]
        wedges, *_ = ax1.pie(overall_ratios, 
                            autopct='%1.1f%%', 
                            startangle=angle,
                            colors = ['tab:red', 'tab:green'],
                            labels=labels, 
                            explode=explode)

        # bar chart parameters
        artifacts_ratios = [value['ratio_from_bad'] 
                            for key,value in self.statistics.items() 
                            if "BAD" in key]
        artifacts_labels = [key for key in self.statistics.keys() if 'BAD' in key]
        bottom = 1
        width = .2
        cmap = mpl.colormaps['tab20c']
        colors = cmap.colors[4:4+len(artifacts_ratios)] #type: ignore

        # Adding from the top matches the legend.
        for j, (height, label) in enumerate(
            reversed([*zip(artifacts_ratios, artifacts_labels)])
            ):
            bottom -= height
            bc = ax2.bar(0, height, width, bottom=bottom,label=label,
                        color=colors[j])
            ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        ax2.set_title('Artifacts type')
        ax2.legend(fontsize = 8)
        ax2.axis('off')
        ax2.set_xlim(- 2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(artifacts_ratios)

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                            xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0]) # type: ignore
        con.set_linewidth(2)
        ax2.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                            xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0]) # type: ignore
        ax2.add_artist(con)
        con.set_linewidth(2)
        return fig

    # TODO
    # - Need to add the plot and then run on data
    # - Add high frequency/high amplitude detection
    # - Add electrode level detection, stats and plot
        

        
        
