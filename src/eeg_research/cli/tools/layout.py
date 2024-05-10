"""This module provides functions for creating entities from command line arguments."""

import argparse

import bids

from eeg_research.cli.tools.parser import parse_range_arguments


def create_entities(args: argparse.Namespace, layout: bids.BIDSLayout) -> dict:
    """Create entities dictionary from arguments.

    Args:
        args: The parsed command line arguments.
        layout: The BIDSLayout object.

    Returns:
        A dictionary of entities.
    """
    # List of entity names that need to be parsed
    entity_names = [
        "subject",
        "session",
        "run",
        "task",
        "extension",
        "datatype",
        "suffix",
    ]

    # Create entities dictionary
    entities = {
        name: parse_range_arguments(layout, name, getattr(args, name))
        for name in entity_names
    }

    # Add other entities
    entities.update({"description": args.description})

    return entities
