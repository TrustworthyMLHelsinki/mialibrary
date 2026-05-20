"""Type checking and conversion utilities"""

from typing import Tuple, Any

import numpy as np


def check_types(
    stat: Any, stat_pop: Any, in_indices: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check the types of the input arrays and convert them to numpy arrays.

    Parameters
    ----------
    stat : Any
        The statistics array, expected to be a list of numpy arrays or a numpy array.
    stat_pop : Any
        The stat_pop array, expected to be a list of numpy arrays or a numpy array.
    in_indices : Any
        The in_indices array, expected to be a list of numpy arrays or a numpy array.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The type-checked and converted stat, stat_pop, and in_indices arrays as NumPy arrays.

    Raises
    ------
    TypeError
        If any of the input arrays have an invalid type.
    ValueError
        If a list is provided but contains invalid elements.
    """

    def validate_and_convert(input_data: Any, name: str) -> np.ndarray:
        # If input is a list, ensure all elements are numpy arrays and stack them
        if isinstance(input_data, list):
            if not all(isinstance(item, np.ndarray) for item in input_data):
                raise ValueError(
                    f"All elements in the {name} list must be numpy arrays, "
                    f"but got invalid elements: {[type(item) for item in input_data]}"
                )
            input_data = np.array(input_data)

        # Final check: must be numpy array
        if not isinstance(input_data, np.ndarray):
            raise TypeError(
                f"Expected {name} to be a list of numpy arrays, a numpy array, or a jax array, "
                f"but got {type(input_data)}"
            )
        return input_data

    # Validate and convert all inputs
    stat = validate_and_convert(stat, "stat")
    stat_pop = validate_and_convert(stat_pop, "stat_pop")
    in_indices = validate_and_convert(in_indices, "in_indices")

    return stat, stat_pop, in_indices
