"""Type checking and conversion utilities"""

from typing import Tuple, Any

import numpy as np
import jax.numpy as jnp


def convert_to_jax_array(arr: np.ndarray) -> jnp.ndarray:
    """
    Convert a NumPy array to a JAX array.

    Parameters
    ----------
    arr : np.ndarray
        The NumPy array to convert.

    Returns
    -------
    jnp.ndarray
        The converted JAX array.

    Raises
    ------
    TypeError
        If arr is not a numpy array or a jax array.
    """
    if isinstance(arr, np.ndarray):
        return jnp.array(arr)
    elif isinstance(arr, jnp.ndarray):
        return arr
    else:
        raise TypeError(f"Expected a numpy or jax array, but got {type(arr)}")


def check_types(stat: Any, in_indices: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check the types of the input arrays and convert them to numpy arrays.

    Parameters
    ----------
    stat : Any
        The statistics array, expected to be a list of numpy arrays, a numpy array, or a jax array.
    in_indices : Any
        The in_indices array, expected to be a list of numpy arrays, a numpy array, or a jax array.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The type-checked and converted stat and in_indices arrays as NumPy arrays.

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

        # If input is a JAX array, convert to numpy
        elif isinstance(input_data, jnp.ndarray):
            input_data = np.array(input_data)

        # Final check: must be numpy array
        if not isinstance(input_data, np.ndarray):
            raise TypeError(
                f"Expected {name} to be a list of numpy arrays, a numpy array, or a jax array, "
                f"but got {type(input_data)}"
            )
        return input_data

    # Validate and convert both inputs
    stat = validate_and_convert(stat, "stat")
    in_indices = validate_and_convert(in_indices, "in_indices")

    return stat, in_indices
