"""Transform and validate inputs for RMIA evaluation."""

from typing import Tuple

import numpy as np

from .rmia import sigmoid


def transform_inputs(
    stat: np.ndarray,
    stat_pop: np.ndarray,
    in_indices: np.ndarray,
    offline: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Transform the inputs for RMIA evaluation by applying the sigmoid function
    to convert logits to probabilities.

    Parameters
    ----------
    stat : np.ndarray
        The statistics array of shape (num_models, num_examples, augmentations).
    stat_pop : np.ndarray
        The population statistics array of shape (num_models, num_examples, augmentations).
    in_indices : np.ndarray
        The in_indices array of shape (num_models, num_examples).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        Transformed signals, population signals, memberships, and number of models.
    """

    # Validate if stat is not probability vectors
    row_sums = stat.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-3):
        print("[WARNING] Input looks like probability vectors, not LiRA logit stats.")
        print("We expect logit statistics as input for RMIA as we do for LiRA.")

    # Ensure that the number of models is even for online RMIA
    num_models = in_indices.shape[0]

    if not offline:
        if num_models % 2 != 0:
            num_models -= 1

    # Select only the required number of models
    stat = stat[:num_models, :, :]
    stat_pop = stat_pop[:num_models, :, :]
    in_indices = in_indices[:num_models, :]

    stat = np.array(stat).reshape(num_models, -1)
    stat_pop = np.array(stat_pop).reshape(num_models, -1)
    stat_transformed = sigmoid(stat)
    stat_pop_transformed = sigmoid(stat_pop)

    return stat_transformed, stat_pop_transformed, in_indices, num_models
