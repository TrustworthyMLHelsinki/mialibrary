"""
Test cases for validating outputs of the process_fast_lira.py script.
Checks if the generated outputs match the expected results for 4 scenarios.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import glob

import pytest
import numpy as np

from mialib.data_loader import load_data
from mialib.lira.check_types import check_types
from mialib.lira.validate_inputs import validate_inputs
from mialib.compute_score import compute_lira_score


# Compare with correct outputs from the original script
@pytest.mark.parametrize(
    "stat_path, in_indices_path, expected_output_path, fix_variance, use_median",
    [
        # Scenario 1: Median global, fix_variance=True, use_median=True
        (
            "slow/data_median_global/stat_*.pkl",
            "slow/data_median_global/in_indices_*.pkl",
            "slow/data_median_global/scores_*.pkl",
            True,
            True,
        ),
        # Scenario 2: Median local, fix_variance=False, use_median=True
        (
            "slow/data_median_local/stat_*.pkl",
            "slow/data_median_local/in_indices_*.pkl",
            "slow/data_median_local/scores_*.pkl",
            False,
            True,
        ),
        # Scenario 3: Mean global, fix_variance=True, use_median=False
        (
            "slow/data_mean_global/stat_*.pkl",
            "slow/data_mean_global/in_indices_*.pkl",
            "slow/data_mean_global/scores_*.pkl",
            True,
            False,
        ),
        # Scenario 4: Mean local, fix_variance=False, use_median=False
        (
            "slow/data_mean_local/stat_*.pkl",
            "slow/data_mean_local/in_indices_*.pkl",
            "slow/data_mean_local/scores_*.pkl",
            False,
            False,
        ),
    ],
)
def test_validate_outputs(
    stat_path, in_indices_path, expected_output_path, fix_variance, use_median
):
    """Test if the outputs match the expected results for each scenario."""
    # Get the first matching file for each path
    stat_path = glob.glob(stat_path)[0]
    in_indices_path = glob.glob(in_indices_path)[0]
    expected_output_path = glob.glob(expected_output_path)[0]

    # Load the input data
    stat, in_indices = load_data(stat_path, in_indices_path)

    # Check types
    stat_checked, in_indices_checked = check_types(stat, in_indices)

    # Validate inputs
    stat_validated, in_indices_validated = validate_inputs(
        stat_checked, in_indices_checked, stat_path, in_indices_path
    )

    # Compute the scores
    scores = np.array(
        compute_lira_score(
            stat_validated,
            in_indices_validated,
            fix_variance=fix_variance,
            use_median=use_median,
        )
    ).ravel()

    # Load the expected output
    with open(expected_output_path, "rb") as f:
        expected_output = pickle.load(f)

    # Validate the outputs
    assert np.allclose(scores, expected_output["scores"], atol=1.0e-8), (
        f"Mismatch in outputs for {stat_path} and {in_indices_path} "
        f"with fix_variance={fix_variance}, use_median={use_median}"
    )
