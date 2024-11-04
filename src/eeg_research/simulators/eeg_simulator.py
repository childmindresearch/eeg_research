"""Simulate EEG data for testing purposes."""

import mne
import neurokit2 as nk
import numpy as np
from mne import create_info
from mne.io import RawArray

# TODO:
#   - add the simulation of:
#       - EOG
#       - gradient artifacts
#       - BCG artifacts


def simulate_light_eeg_data(
    n_channels: int = 16,
    duration: int = 2,
    sampling_frequency: int = 256,
) -> RawArray:
    """Simulate EEG data that have low impact on memory.

    When events and realistic EEG data are not needed, this function
    provides a light version of the simulate_eeg_data function.

    Args:
        n_channels (int): The number of EEG channels.
        duration (int): The duration of the EEG data in seconds.
        sampling_frequency (int): The sampling frequency of the EEG data.

    Returns:
        RawArray: The simulated EEG data.
    """
    if n_channels <= 0:
        raise ValueError("The number of channels must be greater than 0.")

    if duration <= 0:
        raise ValueError("The duration must be greater than 0.")

    eeg_data = np.random.randn(n_channels, duration * sampling_frequency)
    channel_names = [str(i) for i in range(n_channels)]
    info = create_info(channel_names, sampling_frequency, ch_types="eeg")
    raw = RawArray(eeg_data, info)

    return raw

def simulate_eeg_data(
    n_channels: int = 16,
    duration: int = 2,
    misc_channels: list = ["ecg"],
    sampling_frequency: int = 256,
    events_kwargs: dict = dict(name="R128", number=1, start=1, stop=5),
) -> RawArray:
    """Simulate EEG data.

    This function generates simulated EEG data.

    Args:
        n_channels (int, optional): The number of EEG channels.
            Defaults to 16.
        duration (int, optional): The duration of the EEG data in seconds.
            Defaults to 2.
        misc_channels (list, optional): The list of miscellaneous channels to simulate.
            Defaults to ["ecg"].
        sampling_frequency (int, optional): The sampling frequency of the EEG data.
            Defaults to 256.
        events_kwargs (dict, optional): The parameters to generate events in the EEG
            data. Defaults to dict(name="R128", number=1, start=1, stop=5).

    Returns:
        RawArray: The simulated EEG data.
    """
    if n_channels <= 0:
        raise ValueError("The number of channels must be greater than 0.")

    if duration <= 0:
        raise ValueError("The duration must be greater than 0.")

    eeg_data = np.zeros((n_channels, duration * sampling_frequency))
    for channel in range(n_channels):
        # !!!!!!
        # Careful it is sensitive to the duration. Somewhat it doesn't have
        # the same shape as the premade eeg_data array so numpy throws an error
        # saying it couldn't broadcast the array.
        # I will need to take care of that
        eeg_data[channel, :] = nk.eeg_simulate(
            duration=duration, sampling_rate=sampling_frequency, noise=0.1
        )

    channel_names = [str(i) for i in range(n_channels)]
    montage = mne.channels.make_standard_montage("biosemi16")
    ch_names = montage.ch_names
    channel_mapping = {str(i): ch_name for i, ch_name in enumerate(ch_names)}

    if misc_channels:
        misc_channels_object_list = list()
        if "ecg" in misc_channels:
            ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_frequency)

            eeg_data[ch_names.index("T8"), :] *= (ecg * 2) * 1e-6
            eeg_data[ch_names.index("T7"), :] *= -(ecg * 2) * 1e-6
            ecg = np.expand_dims(ecg, axis=0)
            raw_ecg = RawArray(
                ecg, create_info(["ecg"], sampling_frequency, ch_types="ecg")
            )
            misc_channels_object_list.append(raw_ecg)

        if "emg" in misc_channels:
            emg = nk.emg_simulate(duration=duration, sampling_rate=sampling_frequency)
            emg = np.expand_dims(emg, axis=0)
            raw_emg = RawArray(
                emg, create_info(["emg"], sampling_frequency, ch_types="emg")
            )
            misc_channels_object_list.append(raw_emg)

    info = create_info(channel_names, sampling_frequency, ch_types="eeg")
    raw = RawArray(eeg_data, info)
    raw.rename_channels(channel_mapping)
    raw.set_montage(montage, on_missing="ignore")
    if misc_channels:
        raw.add_channels(misc_channels_object_list)

    if events_kwargs:
        events_index = np.linspace(
            events_kwargs["start"] * sampling_frequency,
            events_kwargs["stop"] * sampling_frequency,
            num=events_kwargs["number"],
            endpoint=False,
        )
        print(len(events_index))
        events_name = [events_kwargs["name"]] * events_kwargs["number"]
        annotations = mne.Annotations(
            onset=events_index / sampling_frequency, duration=0, description=events_name
        )
        raw.set_annotations(annotations)

    return raw

