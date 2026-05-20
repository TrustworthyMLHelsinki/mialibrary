"""Validation utilities for input data"""

from typing import Tuple

import numpy as np


def validate_inputs(
    stat: np.ndarray,
    stat_pop: np.ndarray,
    in_indices: np.ndarray,
    stat_path: str,
    stat_pop_path: str,
    in_indices_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate the inputs for shape, NaN values, and number of examples.
    Convert NumPy arrays to JAX arrays after validation.

    Parameters
    ----------
    stat : np.ndarray
        The statistics array.
    stat_pop : np.ndarray
        The population statistics array.
    in_indices : np.ndarray
        The in_indices array.
    stat_path : str
        Path to the stat file (used for error messages).
    stat_pop_path : str
        Path to the stat_pop file (used for error messages).
    in_indices_path : str
        Path to the in_indices file (used for error messages).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The validated and converted stat and in_indices arrays as JAX arrays.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    # Validate shapes
    if stat.ndim != 2:
        raise ValueError(f"stat must be a 2D array, but got shape {stat.shape}")
    if stat_pop.ndim != 2:
        raise ValueError(f"stat_pop must be a 2D array, but got shape {stat_pop.shape}")
    if in_indices.ndim != 2:
        raise ValueError(
            f"in_indices must be a 2D array, but got shape {in_indices.shape}"
        )
    if stat.shape != in_indices.shape:
        raise ValueError(
            f"Shape mismatch: stat has shape {stat.shape}, "
            f"but in_indices has shape {in_indices.shape}"
        )

    # Reject if empty
    if stat.size == 0 or stat_pop.size == 0 or in_indices.size == 0:
        raise ValueError(
            f"{stat_path}, {stat_pop_path} or {in_indices_path} is empty. Cannot proceed with computation."
        )

    # Reject if any NaN values are present
    if np.any(np.isnan(stat)):
        raise ValueError(
            f"{stat_path} contains NaN values. Cannot proceed with computation."
        )

    # Validate the the number of in and shadow examples
    m, n = in_indices.shape
    half_examples = m * n // 2
    tolerance = max(half_examples // 10, 2)  # 5% tolerance, at least 2
    in_num = in_indices.sum()
    if in_num < (half_examples - tolerance) or in_num > (half_examples + tolerance):
        raise ValueError(
            "Efficient LiRA needs roughly half examples in both in and out shadow model training,"
            "therefore, in_indices must contain about half True value in total."
            f"But got {in_num} True values out of {half_examples*2} total examples."
        )

    # Validate the number of shadow models
    num_models = in_indices.sum(axis=0)
    if np.any(num_models < 2):
        raise ValueError(
            f"RMIA attack cannot be conducted without shadow models. "
            f"Each column in in_indices must have at least two True values. "
            f"But got less than 2 at: {np.where(num_models < 2)}"
        )

    return stat, stat_pop, in_indices
