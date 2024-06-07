#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-05-28
# email: samuel DOT louviot AT nki DOT rfmh DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
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
"""Signal Envelope and Time-Frequency extraction pipeline.

This pipeline is used in the project of brain state prediction from EEG data in
collaboration with John Hopkins University. It is used to extract the envelope 
of the EEG signal and the time-frequency representation of the signal.
"""
import mne
import bids
import mne_bids
from mne_bids import BIDSPath
import os
from pathlib import Path
import eeg_research.preprocessing.tools.gradient_remover as GradientRemover
import eeg_research.preprocessing.tools.utils as utils
import numpy as np
import pickle

def parse_file_entities(filename):
    file = Path(filename).name
    basename, extension = os.path.splitext(file)
    entities = dict()
    entities['extension'] = extension
    entities['suffix'] = basename.split('_')[-1]
    for entity in basename.split('_')[:-1]:
        key, value = entity.split('-')

        if key == 'sub':
            key = 'subject'
        elif key == 'ses':
            key = 'session'
        elif key == 'acq':
            key = 'acquisition'
        elif key == 'run':
            value = int(value)
        elif key == 'desc':
            key = 'description'

        entities[key] = value
    return entities

class Envelope:
    def __init__(self, raw):
        self.raw = raw
        self.frequencies = dict()
    def extract_eeg_band_envelope(self):
        self.frequencies = {
            'delta': (0.5, 4),
            'low_theta': (4, 6),
            'high_theta': (6, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 40)
        }
        envelope_list = list()
        for freqs in self.frequencies.values():
            filtered = self.raw.copy().filter(*freqs)
            envelope_list.append(
                filtered.copy().apply_hilbert(envelope = True).get_data()
                )

        self.envelopes = np.stack(envelope_list, axis = -1)

        return self

    def extract_custom_envelope(self,
                                low = 1, 
                                high = 40, 
                                step = 1):

        envelope_list = list()
        for i, low_frequency in enumerate(range(low, high, step)):
            self.frequencies[f'narrow_band_{i+1}'] = (low_frequency, low_frequency + step)
            high_frequency = low_frequency + step
            filtered = self.raw.copy().filter(low_frequency, high_frequency)
            envelope_list.append(
                filtered.copy().apply_hilbert(envelope = True).get_data()
                )
            self.envelopes = np.stack(envelope_list, axis = -1)

        return self
    
    def save(self, filename):
        print(f'saving into {filename}')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

def create_morlet_epochs(raw):
    gradient_trigger_name = GradientRemover.extract_gradient_trigger_name(raw)
    channel_types = utils.map_channel_type(raw)
    raw = utils.set_channel_types(raw, channel_types)
    raw.set_montage('easycap-M1')
    events, event_id = mne.events_from_annotations(raw)
    picked_events = mne.pick_events(events, include=[event_id[gradient_trigger_name]])
    picked_event_id = {gradient_trigger_name: event_id[gradient_trigger_name]}
    epochs_gradient = mne.Epochs(raw, 
                                picked_events, 
                                picked_event_id, 
                                tmin=0.1, 
                                tmax=2.08, 
                                baseline=None,
                                event_repeated = 'drop',
                                preload=True)
    frequencies = np.linspace(1,40,40)
    cycles = frequencies / 2
    power = epochs_gradient.compute_tfr(freqs = frequencies, 
                                        n_cycles = cycles,
                                        method='morlet',
                                        average = False,
                                        return_itc = False,
                                        n_jobs = -1)
    
    return power

def Main():
    derivatives_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES/eeg_features_extraction')
    raw_path = Path('/projects/EEG_FMRI/bids_eeg/BIDS/NEW/PREP_BV_EDF')

    for filename in raw_path.iterdir():
        file_entities = parse_file_entities(filename)
        if file_entities['task'] == 'checker' or file_entities['task'] == 'rest':
            raw = mne.io.read_raw_edf(raw_path / filename, preload=True)
            bids_path = BIDSPath(**file_entities, 
                                root=derivatives_path,
                                datatype='eeg')
            bids_path.mkdir()
            
            bids_path.update(description = 'MorletTFR')
            power = create_morlet_epochs(raw)
            saving_path = os.path.splitext(bids_path.fpath)[0] + '.hdf5'
            power.save(saving_path, overwrite=True)

            envelope = Envelope(raw)
            bids_path.update(description = 'EEGbandsEnvelopes')
            bands_envelope_filename = os.path.splitext(bids_path.fpath)[0] + '.pkl'
            envelope.extract_eeg_band_envelope().save(bands_envelope_filename)
            custom_envelope_filename = os.path.splitext(bids_path.fpath)[0] + '_custom.pkl'
            envelope.extract_custom_envelope().save(custom_envelope_filename)

if __name__ == '__main__':
    Main()