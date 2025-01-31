import os
from pathlib import Path

import mne
import mne_bids
import eeg_research.preprocessing.tools.utils as utils
import numpy as np
import pandas as pd

def all_conditions_met_on(chan: dict) -> bool:
    """Check if arbitrary conditions are met on the channel dictionary.

    Args:
        chan (dict): A dictionary from the xdf data object. Usually accessed by
                     `eeg_stream['info']['desc'][0]['channels'][0]['channel']`

    Returns:
        bool: True if the conditions are met, False otherwise.
    """
    try:
        return chan.get("type",[]) != [] and chan.get("label", []) != []
    except Exception:
        return False

def parse_lsl_channel_names(lsl_eeg_stream: dict) -> dict[str, list[str]]:
    """Parse the channels into MNE compatible format.
    
    Args:
        lsl_eeg_stream (dict): The XDF stream dedicated to EEG.

    Returns:
        dict[str, list[str]]: A mapping of channel names and types for MNE
    """
    chan_dict = lsl_eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_names, ch_types = [], []
    
    for chan in chan_dict:
        if not all_conditions_met_on(chan):
            continue

        ch_names.append(chan["label"][0])
        ch_type = chan["type"][0].lower()
        ch_types.append("stim" if ch_type == "marker" else ch_type)

    return {"ch_names": ch_names, "ch_types": ch_types}

def set_channel_montage(
    raw: mne.io.Raw,
    montage_name: str = "easycap-M1"
) -> tuple[mne.io.Raw, mne.channels.DigMontage]:
    """Set the channel montage and return the montage.

    Args:
        raw (mne.io.Raw): The raw EEG data in mne format.
        montage_name (str): The name of the built-in montage to use.
                            The name should be a valid name for the
                            mne.channels.make_standard_montage function.
                            In doubt, you can check the available montages by
                            calling mne.channels.get_builtin_montages().

    Returns:
        tuple[mne.io.Raw, mne.channels.Montage]: The raw EEG data and the
                                                 MNE montage object.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    channel_map = utils.map_channel_type(raw)
    utils.set_channel_types(raw, channel_map)
    raw.set_montage(montage, on_missing="warn")
    return raw, montage

def set_channel_dataframe(
    raw: mne.io.Raw,
    prep_output: dict,
) -> pd.DataFrame:
    """Create a channel quality dataframe for BIDS compatibility.
    
    Args:
        raw (mne.io.Raw): The raw EEG data
        prep_output (dict): Preprocessing pipeline output containing channel quality info
    
    Returns:
        pd.DataFrame: Channel quality information indexed by channel name
    """
    channel_names = list(raw.impedances.keys())
    
    df_dict = {"name": channel_names}
    
    for bad_label, bad_channels in prep_output["noisy_channels_original"].items():
        df_dict[bad_label] = np.isin(channel_names, bad_channels)
    
    df_dict["still_noisy"] = np.isin(
        channel_names, 
        prep_output["still_noisy_channels"]
    )
    
    df_dict["impedances"] = [
        raw.impedances[ch]["imp"] for ch in channel_names
    ]

    return pd.DataFrame(df_dict).set_index("name")

def save_channels_info(
    raw: mne.io.Raw,
    prep_output: dict,
    saving_bids_path: mne_bids.BIDSPath
) -> None:
    """Save channel information to BIDS-compatible TSV file.
    
    Args:
        raw (mne.io.Raw): The raw EEG data
        prep_output (dict): Preprocessing pipeline output
        saving_bids_path (mne_bids.BIDSPath): Path object for the output file
    """
    channel_info_fname = saving_bids_path.fpath.parent / (
        f"{saving_bids_path.basename}_channels.tsv"
    )
    
    channel_dataframe = set_channel_dataframe(raw, prep_output)
    
    try:
        if channel_info_fname.exists():
            existing_df = pd.read_csv(
                channel_info_fname, 
                sep="\t", 
                index_col=["name"]
            )
            result = existing_df.join(channel_dataframe, how="outer")
        else:
            result = channel_dataframe
            
        result.to_csv(channel_info_fname, sep="\t")
    except Exception as e:
        raise IOError(f"Failed to save channel info: {e}") from e
