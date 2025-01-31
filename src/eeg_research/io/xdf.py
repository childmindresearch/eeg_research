from pathlib import Path
import numpy as np
import mne
import pyxdf

def _parse_channel_names(eeg_stream: dict) -> dict:
    chan_dict = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    ch_names = list()
    ch_types = list()
    for chan in chan_dict:
        ch_names.append(chan['label'][0])
        if chan['type'][0].lower() == 'marker':
            ch_types.append('stim')
        else:
            ch_types.append(chan['type'][0].lower())
    
    parsed_chan_info = {
        "ch_names": ch_names,
        "ch_types": ch_types,
    }
    
    return parsed_chan_info

class RawXDF(mne.io.BaseRaw):
    """Raw object from XDF file."""
    
    def __init__(self, 
                 ch_names: list,
                 units: str,
                 input_fname: str | Path,
                 preload: bool=False,
                 *,
                 verbose=None,
            ):
        
        eeg, _= pyxdf.load_xdf(input_fname, select_streams=5)
        sfreq = float(eeg[0]['info']['nominal_srate'][0])
        info = mne.create_info(ch_names, sfreq=sfreq)
        last_samps = int(eeg[0]['footer']['info']['sample_count'][0])-1
        super().__init__(
            info,
            preload,
            filenames=[input_fname],
            last_samps=last_samps,
            orig_format="int",
            orig_units=units,
            verbose=verbose,
        )
        self._data = eeg[0]['time_series'].T

def read_raw_xdf(filename, **kwargs):
    """Read XDF file.
    
    Args:
        filename : str | Path
            Path to XDF file.
        **kwargs : dict
            Additional keyword arguments passed to RawXDF.
    
    Returns
        raw : RawXDF
            Raw object containing XDF data.
    """
    return RawXDF(filename, **kwargs)