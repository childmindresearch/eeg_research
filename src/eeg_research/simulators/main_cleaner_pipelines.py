#!/usr/bin/env -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-04-04
# email: samuel DOT louviot AT nki DOT rfmh DOT org
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

import argparse

import bids

from eeg_research.simulators.cleaner_pipelines import CleanerPipelines

parser = argparse.ArgumentParser(description="Run the cleaning pipelines")
parser.add_argument("--path", type=str, help="Path to the BIDS dataset")
args = parser.parse_args()


def run_cbin_cleaner(cleaner: CleanerPipelines) -> CleanerPipelines:
    """Run the cbin cleaner pipeline."""
    cleaner.read_raw()
    if cleaner._task_is("checker"):
        cleaner.run_clean_gradient_and_bcg()
    elif cleaner._task_is("checkeroff"):
        cleaner.run_clean_bcg()
    return cleaner


def run_cbin_cleaner_asr(cleaner: CleanerPipelines) -> CleanerPipelines:
    """Run the cbin cleaner pipeline with ASR."""
    cleaner = run_cbin_cleaner(cleaner)
    cleaner.run_asr()
    return cleaner


def run_cbin_cleaner_pyprep_asr(cleaner: CleanerPipelines) -> CleanerPipelines:
    """Run the cbin cleaner pipeline with PyPrep and ASR."""
    cleaner = run_cbin_cleaner(cleaner)
    cleaner.run_pyprep()
    cleaner.run_asr()
    return cleaner


def main(reading_path: str) -> None:
    """Run the cleaning pipelines."""
    layout = bids.BIDSLayout(reading_path)
    file_list = layout.get(extension=".set")

    for BIDSFile_object in file_list:
        # Totally sub-optimal need to fix it
        cleaner_cbin = CleanerPipelines(BIDSFile_object)
        cleaner_cbin_asr = CleanerPipelines(BIDSFile_object)
        cleaner_cbin_pyprepr_asr = CleanerPipelines(BIDSFile_object)
        if any(
            [BIDSFile_object.task == "checker", BIDSFile_object.task == "checkeroff"]
        ):
            try:
                run_cbin_cleaner(cleaner_cbin)
                run_cbin_cleaner_asr(cleaner_cbin_asr)
                run_cbin_cleaner_pyprep_asr(cleaner_cbin_pyprepr_asr)

            except Exception as e:
                message = f"""filename: {str(BIDSFile_object.filename)}
                error:{str(e)}

                """
                cleaner_cbin.write_report(message)


if __name__ == "__main__":
    main(args.path)
