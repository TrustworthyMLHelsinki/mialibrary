"""
Test cases for check_types in process_fast_lira.py
Ensures that the function correctly validates the type of input data
and raises appropriate errors for invalid types.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import jax.numpy as jnp

from mialib.lira.check_types import check_types


@pytest.mark.parametrize(
    "stat, in_indices, expected_stat_type, expected_in_indices_type, expected_error, error_message",
    [
        # Valid NumPy arrays
        (np.array([1, 2, 3]), np.array([4, 5, 6]), np.ndarray, np.ndarray, None, None),
        # Valid JAX arrays
        (
            jnp.array([1, 2, 3]),
            jnp.array([4, 5, 6]),
            np.ndarray,
            np.ndarray,
            None,
            None,
        ),
        # Valid list of NumPy arrays
        (
            [np.array([1, 2]), np.array([3, 4])],
            [np.array([5, 6]), np.array([7, 8])],
            np.ndarray,
            np.ndarray,
            None,
            None,
        ),
        # Invalid type for stat (string instead of list/array)
        (
            "invalid",
            np.array([4, 5, 6]),
            None,
            None,
            TypeError,
            "Expected stat to be a list of numpy arrays, a numpy array, or a jax array, .*",
        ),
        # Invalid type for in_indices (string instead of list/array)
        (
            np.array([1, 2, 3]),
            "invalid",
            None,
            None,
            TypeError,
            "Expected in_indices to be a list of numpy arrays, a numpy array, or a jax array, .*",
        ),
        # Both inputs invalid
        (
            "invalid",
            "invalid",
            None,
            None,
            TypeError,
            "Expected stat to be a list of numpy arrays, a numpy array, or a jax array, .*",
        ),
        # Empty list inputs (valid case) - to be excluded in validate_inputs
        ([], [], np.ndarray, np.ndarray, None, None),
        # Invalid nested list inputs (contains non-NumPy array element)
        (
            [np.array([1, 2]), "invalid"],
            [np.array([5, 6]), np.array([7, 8])],
            None,
            None,
            ValueError,
            "All elements in the stat list must be numpy arrays, .*",
        ),
    ],
)
def test_check_types(
    stat,
    in_indices,
    expected_stat_type,
    expected_in_indices_type,
    expected_error,
    error_message,
):
    """
    Test the check_types function with various valid and invalid inputs.
    """
    if expected_error:
        # Check that the function raises the expected error for invalid inputs
        with pytest.raises(expected_error, match=error_message):
            check_types(stat, in_indices)
    else:
        # Check that the function returns NumPy arrays for valid inputs
        stat_result, in_indices_result = check_types(stat, in_indices)
        assert isinstance(
            stat_result, expected_stat_type
        ), f"Expected stat to be {expected_stat_type}, but got {type(stat_result)}"
        assert isinstance(
            in_indices_result, expected_in_indices_type
        ), f"Expected in_indices to be {expected_in_indices_type}, but got {type(in_indices_result)}"
