"""
Test cases for validate_inputs function in process_fast_lira.py
Validates input data and checks for shape mismatches, NaN values, and other edge cases
to ensure that the input is appropriate for LiRA score computation.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from mialib.lira.validate_inputs import validate_inputs


@pytest.mark.parametrize(
    "stat, in_indices, expected_error, error_message",
    [
        # Valid input
        (
            np.random.rand(4, 2, 1),
            np.array([[True, False], [False, True], [True, False], [False, True]]),
            None,
            None,
        ),
        # Shape mismatch
        (
            np.random.rand(4, 2, 1),
            np.array([[True, False, True], [False, True, False]]),
            ValueError,
            "Shape mismatch.*",
        ),
        # Insufficient shadow models
        (
            np.random.rand(4, 3, 1),
            # example 2 with only one shadow model, example 3 without shadow model
            np.array(
                [
                    [True, False, True],
                    [False, False, True],
                    [True, False, True],
                    [False, True, True],
                ]
            ),
            ValueError,
            "LiRA attack cannot be conducted without shadow models.*",
        ),
        # Imbalanced in_indices (all True)
        (
            np.random.rand(4, 2, 1),
            np.array([[True, True], [True, True], [True, True], [True, True]]),
            ValueError,
            "Efficient LiRA needs roughly half examples in both in and out shadow model training.*",
        ),
        # Imbalanced in_indices (all False)
        (
            np.random.rand(4, 2, 1),
            np.array([[False, False], [False, False], [False, False], [False, False]]),
            ValueError,
            "Efficient LiRA needs roughly half examples in both in and out shadow model training.*",
        ),
        # NaN values in stat
        (
            np.array([[[1], [2]], [[3], [np.nan]]]),
            np.array([[True, False], [False, True]]),
            ValueError,
            "contains NaN values.*",
        ),
        # Empty inputs
        (
            np.empty((0, 2, 1)),
            np.empty((0, 2), dtype=bool),
            ValueError,
            "is empty. Cannot proceed with computation.*",
        ),
    ],
)
def test_validate_inputs(stat, in_indices, expected_error, error_message):
    """Test validate_inputs with various edge cases."""
    if expected_error:
        with pytest.raises(expected_error, match=error_message):
            validate_inputs(stat, in_indices, "stat.pkl", "in_indices.pkl")
    else:
        # Valid case: Ensure no errors and correct conversion
        stat_jax, in_indices_jax = validate_inputs(
            stat, in_indices, "stat.pkl", "in_indices.pkl"
        )
        assert stat_jax.shape == stat.shape
        assert in_indices_jax.shape == in_indices.shape
