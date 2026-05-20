"""
Score computation utilities for LiRA.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import device_put

from .fast_lira import (
    correct_parameters_mean_global,
    correct_parameters_mean_local,
    correct_parameters_median_local,
    correct_parameters_median_global,
    normal_logpdf,
)

jax.config.update("jax_enable_x64", True)


def compute_lira_scores_all_cases(
    all_stats: jnp.ndarray,
    all_in_indices: jnp.ndarray,
    fix_variance: bool = True,
    use_median: bool = True,
) -> jnp.ndarray:
    """
    Main function that handles all 4 cases based on fix_variance and use_median parameters.

    Parameters
    ----------
        all_stats (jnp.ndarray): Array of shape (m, n) containing statistics.
        all_in_indices (jnp.ndarray): Boolean array of shape (m, n) indicating in indices.
        fix_variance (bool): Whether to fix variance globally. Defaults to True.
        use_median (bool): Whether to use median instead of mean. Defaults to True.

    Returns
    ----------
        jnp.ndarray: Log likelihood ratio values.
    """
    if fix_variance and use_median:
        print("Median & Global")
        in_means, in_stds, out_means, out_stds = correct_parameters_median_global(
            all_stats, all_in_indices
        )
    elif fix_variance and not use_median:
        print("Mean & Global")
        in_means, in_stds, out_means, out_stds = correct_parameters_mean_global(
            all_stats, all_in_indices
        )
    elif not fix_variance and use_median:
        print("Median & Local")
        in_means, in_stds, out_means, out_stds = correct_parameters_median_local(
            all_stats, all_in_indices
        )
    else:  # not fix_variance and not use_median
        print("Mean & Local")
        in_means, in_stds, out_means, out_stds = correct_parameters_mean_local(
            all_stats, all_in_indices
        )

    # Create normal distributions and compute log likelihood ratio
    if fix_variance:
        in_logpdf = normal_logpdf(all_stats, in_means, in_stds[:, None, None])
        out_logpdf = normal_logpdf(all_stats, out_means, out_stds[:, None, None])
    else:
        in_logpdf = normal_logpdf(all_stats, in_means, in_stds)
        out_logpdf = normal_logpdf(all_stats, out_means, out_stds)

    scores = in_logpdf - out_logpdf
    return scores.mean(axis=2)  # Return mean score across augmentations


def compute_lira_score(
    stat: jnp.ndarray, in_indices: jnp.ndarray, fix_variance: bool, use_median: bool
) -> np.ndarray:
    """
    Compute LiRA scores for a given stat and in_indices using JAX.

    Parameters
    ----------
        stat (jnp.ndarray): The statistics array.
        in_indices (jnp.ndarray): The in_indices array.
        fix_variance (bool): Whether to fix variance in computation.
        use_median (bool): Whether to use median in computation.

    Returns
    -------
        np.ndarray: Computed scores.
    """
    stat = device_put(stat)
    in_indices = device_put(in_indices)
    scores = compute_lira_scores_all_cases(stat, in_indices, fix_variance, use_median)
    scores = np.array(scores)
    return scores
