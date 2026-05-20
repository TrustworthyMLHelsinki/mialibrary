"""
This script contains functions to compute LiRA scores.
Includes functions to prepare statistics, correct means and standard deviations,
and compute log likelihood ratios based on different configurations
(mean/median and local/global variance).
A JAX implementation with post-process logic to compute parameters for distributions.
"""

from typing import Tuple
import jax.numpy as jnp
from jax import jit


@jit
def prepare_stats(all_stats: jnp.ndarray, all_in_indices: jnp.ndarray) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    Prepare statistics for in and out indices,
    returning nan- and zero-masked arrays and number of models per sample.
    Needed for cases with median.

    Parameters
    ----------
        all_stats (jnp.ndarray): Array of shape (m, n, k) containing statistics.
        all_in_indices (jnp.ndarray): Boolean array of shape (m, n) indicating in indices.

    Returns
    ----------
        Tuple of 7 jnp.ndarrays:
            - not_indices: Boolean array of shape (m, n) for out indices.
            - in_stats_nan: Nan-masked array for in indices.
            - out_stats_nan: Nan-masked array for out indices.
            - in_stats_zero: Zero-masked array for in indices.
            - out_stats_zero: Zero-masked array for out indices.
            - in_num_models: Number of models per sample for in indices.
            - out_num_models: Number of models per sample for out indices.
    """
    m, _, _ = all_stats.shape  # (m, n, k)

    not_indices = ~all_in_indices.astype(bool)  # (m, n)
    in_stats_nan = jnp.where(
        all_in_indices[:, :, jnp.newaxis], all_stats, jnp.nan
    )  # (m, n, k), nan-masked
    out_stats_nan = jnp.where(
        not_indices[:, :, jnp.newaxis], all_stats, jnp.nan
    )  # (m, n, k), nan-masked

    in_stats_zero = (
        all_stats * all_in_indices[:, :, jnp.newaxis]
    )  # (m, n, k), zero-masked
    out_stats_zero = (
        all_stats * not_indices[:, :, jnp.newaxis]
    )  # (m, n, k), zero-masked

    in_num_models = jnp.sum(all_in_indices, axis=0)  # (n,)
    out_num_models = m - in_num_models  # (n,)

    return (
        not_indices,
        in_stats_nan,
        out_stats_nan,
        in_stats_zero,
        out_stats_zero,
        in_num_models,
        out_num_models,
    )


@jit
def prepare_stats_zero(
    all_stats: jnp.ndarray, all_in_indices: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Prepare statistics for in and out indices,
    returning zero-masked arrays and number of models per sample.
    Saves computation time for cases with mean.

    Parameters
    ----------
        all_stats (jnp.ndarray): Array of shape (m, n, k) containing statistics.
        all_in_indices (jnp.ndarray): Boolean array of shape (m, n) indicating in indices.

    Returns
    ----------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - not_indices: Boolean array of shape (m, n) for out indices.
            - in_stats_zero: Zero-masked array for in indices.
            - out_stats_zero: Zero-masked array for out indices.
            - in_num_models: Number of models per sample for in indices.
            - out_num_models: Number of models per sample for out indices.
    """
    m, _, _ = all_stats.shape  # (m, n, k)

    not_indices = ~all_in_indices.astype(bool)
    in_stats_zero = (
        all_stats * all_in_indices[:, :, jnp.newaxis]
    )  # (m, n, k), zero-masked
    out_stats_zero = (
        all_stats * not_indices[:, :, jnp.newaxis]
    )  # (m, n, k), zero-masked

    in_num_models = jnp.sum(all_in_indices, axis=0)  # (n,)
    out_num_models = m - in_num_models  # (n,)

    return not_indices, in_stats_zero, out_stats_zero, in_num_models, out_num_models


@jit
def correct_mean_and_std(
    stats_zero: jnp.ndarray, num_model: jnp.ndarray, indices: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (m,n)-corrected mean and std matrix for in_stats_zero and out_stats_zero.

    Parameters
    ----------
        stats_zero (jnp.ndarray): Zero-masked statistics array of shape (m, n, k).
        num_model (jnp.ndarray): Number of models per sample of shape (n,).
        indices (jnp.ndarray): Boolean array of shape (m, n) indicating indices.

    Returns
    ----------
        Tuple[jnp.ndarray, jnp.ndarray]:
            - corrected_means: Corrected mean matrix of shape (m, n, k).
            - corrected_stds: Corrected standard deviation matrix of shape (m, n, k).
    """
    num_model_extended = num_model[:, jnp.newaxis]  # (n, 1)

    full_mean = stats_zero.sum(axis=0) / num_model_extended  # (n, k), algebraic mean
    full_var = ((stats_zero - full_mean) ** 2 * indices[:, :, jnp.newaxis]).sum(
        axis=0
    ) / num_model_extended  # (n, k), algebraic variance

    # Correct mean and std without the effect of the target model
    corrected_means = ((num_model_extended * full_mean) - stats_zero) / (
        num_model_extended - 1
    )  # (m, n, k)
    corrected_vars = (
        (num_model_extended * (full_var + full_mean**2)) - stats_zero**2
    ) / (
        num_model_extended - 1
    ) - corrected_means**2  # (m, n, k)

    # Assign corrected values where indices are True, otherwise use full mean and variance
    corrected_means = jnp.where(
        indices[:, :, jnp.newaxis], corrected_means, full_mean
    )  # (m, n, k)
    corrected_vars = jnp.where(
        indices[:, :, jnp.newaxis], corrected_vars, full_var
    )  # (m, n, k)

    return corrected_means, jnp.sqrt(corrected_vars)


@jit
def correct_median(
    stats_nan: jnp.ndarray, num_model: jnp.ndarray, indices: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns (m,n)-corrected median matrix for in_stats_nan and out_stats_nan.

    Parameters
    ----------
        stats_nan (jnp.ndarray): Nan-masked statistics array of shape (m, n).
        num_model (jnp.ndarray): Number of models per sample of shape (n,).
        indices (jnp.ndarray): Boolean array of shape (m, n) indicating indices.

    Returns
    ----------
        jnp.ndarray: Corrected median matrix of shape (m, n).
    """
    m, n, k = stats_nan.shape  # (m, n, k)

    stats_nan_flattened = stats_nan.reshape(
        m, -1
    )  # (m, n*k), flatten the last dimension

    sorted_values = jnp.sort(
        stats_nan_flattened, axis=0
    )  # (m, n*k), sort values per sample

    num_model_repeated = jnp.repeat(
        num_model, k
    )  # (n*k, ), repeat num_model for each sample

    indices_repeated = jnp.repeat(
        indices, k, axis=1
    )  # (m, n*k), repeat indices for each sample

    even_indices = num_model_repeated % 2 == 0  # (n*k,)
    odd_indices = ~even_indices  # (n*k,)

    n_indices = jnp.arange(n * k)

    # Calculate relevant indices
    even_lower_idx = num_model_repeated // 2 - 1
    even_upper_idx = num_model_repeated // 2
    odd_mid_idx = (num_model_repeated - 1) // 2
    odd_lower_idx = odd_mid_idx - 1
    odd_upper_idx = odd_mid_idx + 1

    # Get potential values (lower, upper, mid) based on indices
    lower = jnp.where(
        even_indices,
        sorted_values[even_lower_idx, n_indices],  # even case
        sorted_values[odd_lower_idx, n_indices],  # odd case
    )

    upper = jnp.where(
        even_indices,
        sorted_values[even_upper_idx, n_indices],  # even case
        sorted_values[odd_upper_idx, n_indices],  # odd case
    )

    mid = jnp.where(
        odd_indices,
        sorted_values[odd_mid_idx, n_indices],  # odd case
        0.0,  # dummy value for even indices
    )

    # Calculate corrected values for odd indices
    corrected_upper = jnp.where(odd_indices, (mid + upper) / 2, upper)

    corrected_lower = jnp.where(odd_indices, (lower + mid) / 2, lower)

    # Calculate full median
    full_median = jnp.where(even_indices, (lower + upper) / 2, mid)

    # Create corrected median matrix
    ## Assign values based on where in the order the removed value is
    median_values = jnp.where(
        stats_nan_flattened < full_median,
        jnp.where(even_indices, upper, corrected_upper),
        jnp.where(even_indices, lower, corrected_lower),
    )
    ## Handle the case where the removed value is full_median
    equal_mask = stats_nan_flattened == full_median
    corrected_median_flattened = jnp.where(
        equal_mask,
        (lower + upper) / 2,
        jnp.where(indices_repeated, median_values, full_median),
    )

    corrected_median = jnp.reshape(corrected_median_flattened, (m, n, k))  # (m, n, k)

    return corrected_median


@jit
def correct_mean_global(
    stats_zero: jnp.ndarray, num_model: jnp.ndarray, indices: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Handle the mean case with global variance correction.

    Parameters
    ----------
        stats_zero (jnp.ndarray): Zero-masked statistics array of shape (m, n, k).
        num_model (jnp.ndarray): Number of models per sample of shape (n,).
        indices (jnp.ndarray): Boolean array of shape (m, n) indicating indices.

    Returns
    ----------
        Tuple[jnp.ndarray, jnp.ndarray]:
            - corrected_means: Corrected mean matrix of shape (m, n).
            - corrected_stds: Corrected standard deviation matrix of shape (m, k).
    """
    _, _, k = stats_zero.shape  # (m, n, k)

    # Prepare adjusted num_model array, removing the effect of target model
    adjusted_num_models = jnp.where(indices, num_model - 1, num_model)  # (m, n)
    num_model_per_target = jnp.sum(
        adjusted_num_models, axis=1
    )  # (m,), number of all shadow models for each target model

    corrected_means, corrected_vars = correct_mean_and_std(
        stats_zero, num_model, indices
    )  # (m, n, k)

    # Compute global variance algebraically
    squared_sum = jnp.sum(
        (adjusted_num_models[:, :, jnp.newaxis] * corrected_vars**2), axis=(1, 2)
    )  # (m, ), sum over all samples and augmentations
    global_corrected_vars = squared_sum / (
        k * num_model_per_target
    )  # (m,), divide by k times number of models

    return corrected_means, jnp.sqrt(global_corrected_vars)


@jit
def correct_median_global(
    stats_nan: jnp.ndarray,
    stats_zero: jnp.ndarray,
    num_model: jnp.ndarray,
    indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Handle the median case with global variance correction.

    Parameters
    ----------
        stats_nan (jnp.ndarray): Nan-masked statistics array of shape (m, n, k).
        stats_zero (jnp.ndarray): Zero-masked statistics array of shape (m, n, k).
        num_model (jnp.ndarray): Number of models per sample of shape (n,).
        indices (jnp.ndarray): Boolean array of shape (m, n) indicating indices.

    Returns
    ----------
        Tuple[jnp.ndarray, jnp.ndarray]:
            - corrected_median: Corrected median matrix of shape (m, n, k).
            - corrected_stds: Corrected standard deviation matrix of shape (m, k).
    """
    _, _, k = stats_nan.shape  # (m, n, k)

    # Prepare adjusted num_model array, removing the effect of target model
    adjusted_num_models = jnp.where(indices, num_model - 1, num_model)  # (m, n)
    num_model_per_target = jnp.sum(
        adjusted_num_models, axis=1
    )  # (m,), number of all shadow models for each target model

    # Compute global mean
    corrected_sum_k = (
        jnp.sum(stats_zero, axis=0) - stats_zero
    )  # (m, n, k), sum(x_i) - x_mi
    corrected_sum = jnp.sum(
        corrected_sum_k, axis=2
    )  # (m, n), summing over augmentations

    global_mean = jnp.sum(corrected_sum, axis=1) / (
        k * num_model_per_target
    )  # (m,), global mean for each target model

    # Compute median
    corrected_median = correct_median(stats_nan, num_model, indices)  # (m, n, k)

    # Compute mean of the median
    mean_median = jnp.sum(
        adjusted_num_models[:, :, jnp.newaxis] * corrected_median, axis=(1, 2)
    ) / (
        k * num_model_per_target
    )  # (m,)

    # Compute the difference between the median and the mean of the median
    delta = (
        corrected_median - mean_median[:, jnp.newaxis, jnp.newaxis]
    )  # (m, n, k), delta = median - mean(median)

    # Compute squared sum
    sum_mean_delta = (
        global_mean[:, jnp.newaxis, jnp.newaxis] + delta
    )  # (m, n, k), global mean + delta

    stats_zero_squared = stats_zero**2  # (m, n, k)

    first_term = jnp.sum(stats_zero_squared, axis=0) - stats_zero_squared
    second_term = -2 * (sum_mean_delta) * corrected_sum_k
    third_term = adjusted_num_models[:, :, jnp.newaxis] * ((sum_mean_delta) ** 2)

    squared_sum_k = jnp.sum(first_term + second_term + third_term, axis=1)  # (m, k)
    squared_sum = jnp.sum(squared_sum_k, axis=1)  # (m,)

    # Compute global variance
    global_corrected_vars = squared_sum / (k * num_model_per_target)  # (m,)

    return corrected_median, jnp.sqrt(global_corrected_vars)


@jit
def correct_parameters_mean_local(all_stats, all_in_indices):
    """Case 1: Mean with local variance (fix_variance=False, use_median=False)"""
    not_indices, in_stats_zero, out_stats_zero, in_num_models, out_num_models = (
        prepare_stats_zero(all_stats, all_in_indices)
    )

    in_means, in_stds = correct_mean_and_std(
        in_stats_zero, in_num_models, all_in_indices
    )
    out_means, out_stds = correct_mean_and_std(
        out_stats_zero, out_num_models, not_indices
    )

    return in_means, in_stds, out_means, out_stds


@jit
def correct_parameters_mean_global(all_stats, all_in_indices):
    """Case 2: Mean with global variance (fix_variance=True, use_median=False)"""
    not_indices, in_stats_zero, out_stats_zero, in_num_models, out_num_models = (
        prepare_stats_zero(all_stats, all_in_indices)
    )

    in_means, in_stds = correct_mean_global(
        in_stats_zero, in_num_models, all_in_indices
    )
    out_means, out_stds = correct_mean_global(
        out_stats_zero, out_num_models, not_indices
    )

    return in_means, in_stds, out_means, out_stds


@jit
def correct_parameters_median_local(all_stats, all_in_indices):
    """Case 3: Median with local variance (fix_variance=False, use_median=True)"""
    (
        not_indices,
        in_stats_nan,
        out_stats_nan,
        in_stats_zero,
        out_stats_zero,
        in_num_models,
        out_num_models,
    ) = prepare_stats(all_stats, all_in_indices)

    in_means = correct_median(in_stats_nan, in_num_models, all_in_indices)
    out_means = correct_median(out_stats_nan, out_num_models, not_indices)
    _, in_stds = correct_mean_and_std(in_stats_zero, in_num_models, all_in_indices)
    _, out_stds = correct_mean_and_std(out_stats_zero, out_num_models, not_indices)

    return in_means, in_stds, out_means, out_stds


@jit
def correct_parameters_median_global(all_stats, all_in_indices):
    """Case 4: Median with global variance (fix_variance=True, use_median=True)"""
    (
        not_indices,
        in_stats_nan,
        out_stats_nan,
        in_stats_zero,
        out_stats_zero,
        in_num_models,
        out_num_models,
    ) = prepare_stats(all_stats, all_in_indices)

    in_means, in_stds = correct_median_global(
        in_stats_nan, in_stats_zero, in_num_models, all_in_indices
    )
    out_means, out_stds = correct_median_global(
        out_stats_nan, out_stats_zero, out_num_models, not_indices
    )

    return in_means, in_stds, out_means, out_stds


@jit
def normal_logpdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the log probability density function of a normal distribution.

    Parameters
    ----------
        x (jnp.ndarray): Input data array.
        mu (jnp.ndarray): Mean of the normal distribution.
        sigma (jnp.ndarray): Standard deviation of the normal distribution.

    Returns
    ----------
        jnp.ndarray: Log probability density values.
    """
    sigma = jnp.where(sigma > 0, sigma, 1e-6)  # Avoid division by zero

    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2
