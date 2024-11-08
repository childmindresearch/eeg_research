"""Module that get a selection of bids file from a regexp-like input."""

import argparse


def cli_bids_arg_parser() -> dict:
    """Parse command line arguments."""
    # Create the parser with RawTextHelpFormatter so that newlines are preserved
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--root",
        help="Root folder.",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--subject",
        help="Input options for subject IDs are: \n"
        "- '*' for all subjects \n"
        "- 'x' for subject x \n"
        "- 'x-y' for subjects x to y \n"
        "- 'x-*' for subjects x to the last \n"
        "- '*-y' for subjects from the first to y",
        default=None,
    )

    parser.add_argument(
        "--session",
        help="Input options for session IDs are: \n"
        "- '*' for all sessions \n"
        "- 'x' for session x \n"
        "- 'x-y' for sessions x to y \n"
        "- 'x-*' for sessions x to the last \n"
        "- '*-y' for sessions from the first to y",
        default=None,
    )

    parser.add_argument(
        "--run",
        help="Input options for run IDs are: \n"
        "- '*' for all runs \n"
        "- 'x' for run x \n"
        "- 'x-y' for runs x to y \n"
        "- 'x-*' for runs x to the last \n"
        "- '*-y' for runs from the first to y",
        default=None,
    )

    parser.add_argument(
        "--task",
        help="Input options for task IDs are: \n"
        "- '*' for all tasks \n"
        "- 'a' for task a",
        default=None,
    )

    parser.add_argument(
        "--extension",
        help="Input options for file extensions are: \n"
        "- '*' for all extensions \n"
        "- 'a' for extension a",
        default=None,
    )

    parser.add_argument(
        "--datatype",
        help="Input options for datatypes are: \n"
        "- '*' for all datatypes \n"
        "- 'a' for datatype a",
        default="eeg",
    )

    parser.add_argument(
        "--suffix",
        help="Input options for suffixes are: \n"
        "- '*' for all suffixes \n"
        "- 'a' for suffix a",
        default="eeg",
    )

    parser.add_argument(
        "--description",
        help="Description is only applicable to derivative data.",
        default=None,
    )

    parser.add_argument(
        "--interactive",
        help="Run the interactive menu",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if not any([args.interactive, args.gradient, args.bcg, args.qc]):
        parser.error(
            "Please provide at least one of the following arguments: "
            "--interactive" 
        )

    return vars(args)
