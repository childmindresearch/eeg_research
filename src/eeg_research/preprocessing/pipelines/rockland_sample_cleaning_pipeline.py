from pathlib import Path
import os

# Multithreading
nthreads = "5"  # 64 on synapse
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads

import numpy as np
import mne
import pyprep
import mne_bids
import channels_handling as ch
import pandas as pd
import bids_explorer.architecture as arch
import eeg_research.preprocessing.tools.utils as utils

import pyxdf


def convert_in_volts(eeg_stream: dict) -> np.ndarray:
    """Convert the EEG from the XDF object in volts.

    From the units information contains in the XDF object, the EEG signal
    is converted in volts.

    Args:
        eeg_stream (dict): The stream in the XDF object dedicated to the EEG.

    Returns:
        np.ndarray: The EEG signal converted.
    """
    units = {
        "microvolts": 1e-6,
        "millivolts": 1e-3,
        "volts": 1,
    }
    unit_matrix = list()
    chan_dict = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    signals = eeg_stream["time_series"].T
    for chan in chan_dict:
        if chan.get("unit"):
            unit_matrix.append(units.get(chan["unit"][0].lower(), 1))

    unit_matrix = np.array(unit_matrix)[:, np.newaxis]

    return np.multiply(signals, unit_matrix)


def read_raw_xdf(filename: str | Path) -> mne.io.RawArray:
    """Read the XDF file and convert it into an mne.io.RawArray.

    Args:
        filename (str | Path): The input filename to read.

    Returns:
        mne.io.RawArray: The mne raw object.
    """
    eeg, _ = pyxdf.load_xdf(filename, select_streams=[{"type": "EEG"}])
    sfreq = float(eeg[0]["info"]["nominal_srate"][0])
    info = mne.create_info(**ch.parse_channel_names(eeg[0]), sfreq=sfreq)
    return mne.io.RawArray(convert_in_volts(eeg[0]), info=info)


def run_prep(raw: mne.io.Raw) -> pyprep.PrepPipeline:
    """Run the PREP pipeline
    Args:
        raw (mne.io.Raw): The raw EEG data in mne format.

    Returns:
        pyprep.PrepPipeline: The pipeline object containing all the info.
    """
    raw, montage = ch.set_channel_montage(raw)
    raw.filter(l_freq=0, h_freq=125).resample(250)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
    }
    prep = pyprep.PrepPipeline(
        raw, montage=montage, channel_wise=True, prep_params=prep_params
    )

    return prep.fit()


def annotate_blinks(
    raw: mne.io.Raw, ch_name: list[str] = ["Fp1", "Fp2"]
) -> mne.Annotations:
    """Annotate the blinks in the EEG signal.

    Args:
        raw (mne.io.Raw): The raw EEG data in mne format.
        ch_name (list[str]): The channels to use for the EOG. Default is
                             ["Fp1", "Fp2"]. I would suggest to use the
                             channels that are the most frontal (just above
                             the eyes). In the case of an EGI system the
                             channels would be "E25" and "E8".

    Returns:
        mne.Annotations: The annotations object containing the blink events.
    """
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=ch_name)
    blink_annotations = mne.annotations_from_events(
        eog_epochs.events,
        raw.info["sfreq"],
        event_desc={eog_epochs.events[0, 2]: "blink"},
    )
    return blink_annotations


def annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
    """Annotate the muscle artefacts in the EEG signal.

    This use the mne.preprocessing.annotate_muscle_zscore function:
    https://mne.tools/stable/generated/mne.preprocessing.annotate_muscle_zscore.html
    Args:

        raw (mne.io.Raw): The raw EEG data in mne format.

    Returns:
        mne.Annotations: The annotations object containing the muscle artefacts.
    """
    muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(
        raw,
        threshold=3,
        ch_type="eeg",
        min_length_good=0.1,
        filter_freq=(95, 120),
    )

    return muscle_annotations


def read_experiment_annotations(annotation_filename: str | Path) -> mne.Annotations:
    """Read the experiment annotations from a csv file.

    The csv file should have the following columns:
    - sectionname: The name of the section.
    - starttime: The start time of the section.
    - endtime: The end time of the section.

    Args:
        annotation_filename (str | Path): The filename of the csv file
        containing the experiment annotations.

    Returns:
        mne.Annotations: The annotations object containing the experiment annotations.
    """
    df_annotation = pd.read_csv(
        annotation_filename,
        header=1,
        usecols=["sectionname", "starttime", "endtime"],
        dtype={
            "sectionname": str,
            "starttime": np.float32,
            "endtime": np.float32,
        },
    )

    df_annotation["duration"] = df_annotation["endtime"] - df_annotation["starttime"]

    annotations = mne.Annotations(
        onset=df_annotation["starttime"].values,
        duration=df_annotation["duration"].values,
        description=df_annotation["sectionname"].values,
    )

    return annotations


def combine_annotations(annotations_list: list[mne.Annotations]) -> mne.Annotations:
    """Combine the annotations into a single annotations object.

    This is important because the mne.io.Raw.set_annotations method requires a
    single annotations object. And if we feed to that method the annotations
    of the artifact we just annotated, the experiment annotations will be
    overwritten and we will lose the information about the experiment.
    So an important step is to combine all the desired annotations objects into
    a single one.

    Args:
        annotations_list (list[mne.Annotations]): The list of annotations to combine.

    Returns:
        mne.Annotations: The combined annotations object.
    """
    return sum(annotations_list)


def save_bids_tree(raw, raw_cleaned, bids_path):
    mne_bids.write_raw_bids(
        raw_cleaned.pick_types(eeg=True),
        bids_path=bids_path,
        allow_preload=True,
        format="EDF",
        overwrite=True,
    )


def save_eeg_coordinates(raw_bv, CapTrack):
    pass


def pipeline(subject, session, task, run, root=Path("data2/Projects/NKI_RS2/MoBI")):
    base = root / f"sub-{subject}/ses-{session}/"
    lsl_filename = (
        base / f"lsl/sub-{subject}_ses-{session}_task-{task}_run-{run}_lsl.xdf.gz"
    )
    bv_filename = base / f"raw/sub-{subject}_ses-{session}_run-{run}_eeg.vhdr"
    annotation_filename = (
        base / f"raw/sub-{subject}_ses-{session}_task-{task}_run-{run}_events.csv"
    )

    saving_bids_path = mne_bids.BIDSPath(
        root=root / "derivatives", subject=subject, session=session, task=task, run=run
    )
    if saving_bids_path.fpath.parent.is_dir():
        raise FileExistsError(
            f"The directory {saving_bids_path.fpath.parent} already exists"
        )

    mne.set_log_level(verbose="ERROR")
    raw = read_raw_xdf(lsl_filename)
    raw_bv = mne.io.read_raw_brainvision(bv_filename)
    prep_output = run_prep(raw)
    raw_cleaned = prep_output.raw

    blinks_annotations = annotate_blinks(raw_cleaned)
    muscle_annotations = annotate_muscle(raw_cleaned)
    experiment_annotations = read_experiment_annotations(annotation_filename)

    annotations = combine_annotations(
        [blinks_annotations, muscle_annotations, experiment_annotations]
    )

    raw_cleaned.set_annotations(annotations)
    save_bids_tree(raw, raw_cleaned, saving_bids_path)
    ch.save_channels_info(raw_bv, prep_output, saving_bids_path)


if __name__ == "__main__":
    root = Path("/data2/Projects/NKI_RS2/MoBI/")
    architecture = arch.BidsArchitecture(root=root)
    selection = architecture.select(datatype="lsl")
    for idx, file in selection:
        print(file["filename"])
        try:
            pipeline(
                root=root,
                subject=file["subject"],
                session=file["session"],
                task=file["task"],
                run=file["run"],
            )
        except Exception as e:
            print(e)
            continue
