"""
This module provides functions to process RMIA (Robust Membership Inference Attack) data files.
It includes utilities for loading, validating, and processing data, as well as saving results.
"""

import os
import glob
import timeit
import argparse
from typing import List

import numpy as np

# TODO: remove later
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from mialibrary.data_loader import load_data_rmia, save_results
from mialibrary.rmia.check_types import check_types
from mialibrary.rmia.transform_inputs import transform_inputs
from mialibrary.rmia.validate_inputs import validate_inputs
from mialibrary.rmia.compute_score import compute_score_rmia


def process_files(unprocessed_paths: List[str], offline: bool):
    """
    Process all unprocessed files.

    Parameters
    ----------
    unprocessed_paths : List[str]
        List of file paths to process.
    offline : bool
        Whether to process files in offline mode.
    """
    for score_path in sorted(unprocessed_paths):
        print(f"Processing: {score_path}", flush=True)

        in_indices_path = score_path.replace("scores", "in_indices")
        signals_path = score_path.replace("scores", "stat")
        population_signals_path = score_path.replace("scores", "stat_pop")

        # Load data
        signals, population_signals, in_indices = load_data_rmia(
            signals_path, population_signals_path, in_indices_path
        )

        # Check types
        signals_checked, population_signals_checked, in_indices_checked = check_types(
            signals, population_signals, in_indices
        )

        # Transform inputs
        signals_transformed, population_signals_transformed, in_indices, num_models = (
            transform_inputs(
                signals_checked, population_signals_checked, in_indices_checked, offline
            )
        )

        # Validate inputs
        (
            signals_validated,
            population_signals_validated,
            in_indices_validated,
        ) = validate_inputs(
            signals_transformed,
            population_signals_transformed,
            in_indices,
            signals_path,
            population_signals_path,
            in_indices_path,
        )

        # Compute scores
        start_time = timeit.default_timer()
        scores = compute_score_rmia(
            signals_validated,
            population_signals_validated,
            in_indices_validated,
            offline,
            num_models=num_models,
        )
        elapsed_time = timeit.default_timer() - start_time
        print(f"Time to compute scores: {elapsed_time:.2f} seconds")

        # Save results
        save_results(score_path, scores, in_indices)


def process_rmia_data(data_path: str, filter_pattern: str = "*", offline: bool = False):
    """
    Process RMIA data files programmatically.

    Parameters
    ----------
    data_path : str
        Path to in_indices, stat, and output score files.
    filter_pattern : str, optional
        Further filtering of stat files e.g., by dataset. Default is '*'.
    offline : bool, optional
        Whether to process files in offline mode. Default is False.
    """
    all_indices_paths = glob.glob(
        os.path.join(data_path, f"in_indices_{filter_pattern}.pkl")
    )
    all_score_paths = glob.glob(os.path.join(data_path, f"scores_{filter_pattern}.pkl"))
    unprocessed_paths = list(
        set([p.replace("in_indices", "scores") for p in all_indices_paths])
        - set(all_score_paths)
    )

    print(f"Found {len(all_indices_paths)} in_indices files")
    print(
        f"Already processed: {len(all_indices_paths) - len(unprocessed_paths)}, "
        f"to process: {len(unprocessed_paths)}"
    )

    process_files(unprocessed_paths, offline)


def main():
    """
    Main function to parse arguments and process files.
    """
    parser = argparse.ArgumentParser(description="Process RMIA data files.")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to in_indices, stat, and output score files.",
    )
    parser.add_argument(
        "--filter",
        default="*",
        help="Further filtering of stat files e.g., by dataset.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Whether to process files in offline mode.",
    )
    args = parser.parse_args()

    script_start = timeit.default_timer()

    process_rmia_data(
        data_path=args.data_path,
        filter_pattern=args.filter,
        offline=args.offline,
    )

    script_end = timeit.default_timer()
    print(f"Total script time: {script_end - script_start:.2f} seconds")


if __name__ == "__main__":
    main()
