[![DOI](https://zenodo.org/badge/657341621.svg)](https://zenodo.org/doi/10.5281/zenodo.10383685) fix this later

# EEG research

[![Build](https://github.com/childmindresearch/eeg_research/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/eeg_research/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/eeg_research/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/eeg_research)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-stable](https://img.shields.io/badge/stability-experimental-red.svg)
[![LGPL--3.0 License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://github.com/childmindresearch/eeg_research/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/eeg_research)

The purpose of this monorepository is to group all software tools and pipelines related to EEG (experiment, cleaning, analysis etc.). This repository is maintained by Child Mind Institute.

This library operates within a **BIDS** ([Brain Imaging Data Structure](https://bids.neuroimaging.io/)) environment, requiring the EEG data to be organized according to BIDS standards.

## What is BIDS?

The Brain Imaging Data Structure (BIDS) is a standard for organizing and describing neuroimaging data. BIDS datasets organize data into a series of files and folders, with each file and folder name providing important information about the data it contains. This standardization makes it easier to share and analyze data across different labs and research groups.

## Features

- EEG-fMRI cleaning pipeline:
    - **Gradient Artifact Removal**: Implements *FILL IN HERE* to minimize the impact of gradient artifacts caused by the radiofrequency pulses in the MR scanner.
    - **Ballistocardiogram (BCG) Artifact Correction**: Utilizes *FILL IN HERE* to identify and remove artifacts related to cardiac activity.
    - **Quality Control**: Offers a series of checks and visualizations to assess the effectiveness of artifact removal and the overall quality of the EEG data post-cleaning.
    - **Interactive Menu**: Provides an interactive menu for users to select specific cleaning steps and configure parameters, making the pipeline more intuitive and user-friendly to navigate.
    - **Automated Workflow**: Allows users to run the cleaning pipeline with predefined arguments for a more automated and streamlined process, ideal for batch processing multiple subjects, sessions, or tasks.

- *PLACEHOLDER PIPELINE*:
    - *FILL IN HERE*

## Requirements

To use this library, following dependencies will be installed on your system:

- [Python](https://www.python.org/downloads/release/python-3120/) (~3.12)
- [numpy](https://pypi.org/project/numpy/) (^1.26)
- [scipy](https://pypi.org/project/scipy/) (^1.14.0)
- [mne](https://pypi.org/project/mne/) (^1.7.1)
- [matplotlib](https://pypi.org/project/matplotlib/) (^3.8.4)
- [pybids](https://pypi.org/project/pybids/) (^0.16.4)
- [asrpy](https://pypi.org/project/asrpy/) (^0.0.3)
- [pyprep](https://pypi.org/project/pyprep/) (^0.4.3)
- [neurokit2](https://pypi.org/project/neurokit2/) (^0.2.7)
- [pybv](https://pypi.org/project/pybv/) (^0.7.5)
- [simple-term-menu](https://pypi.org/project/simple-term-menu/) (^1.6.4)
- [eeglabio](https://pypi.org/project/eeglabio/) (^0.0.2.post4)

## Installation

Install this package via:

```sh
pip install PLACEHOLDER_NAME
```

Or get the newest development version via:

```sh
pip install git+https://github.com/childmindresearch/eeg_research
```

## Documentation

The documentation for this library can be found [here](https://childmindresearch.github.io/eeg_research).


## Quick start

If you want to run the EEG-fMRI cleaning pipeline in interactive mode:

```sh
eeg_fmri_cleaning --root "path/to/your/datafolder" --interactive
```

If you want to run the EEG-fMRI cleaning pipeline with predefined arguments (subject, session, task, etc.) for a more automated workflow:

```sh
eeg_fmri_cleaning --root "path/to/your/datafolder" --subject "*" --session "*" --task "rest" --run "1-3" --extension ".vhdr" --datatype "eeg" --gradient --bcg --qc
```

See [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ) for a more complete example.
