# Portions of this code are excerpted from:
# https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia.py
"""Functions for simple attacks."""

from typing import Tuple

import numpy as np
import scipy.stats


def get_shadow_stats(
    stat: np.array,
    in_indices: np.array,
    target_model_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare statistics for LiRA attack by separating target and shadow model statistics.

    Parameters
    ----------
        stat (np.ndarray): Array of shape (m, n, k).
        in_indices (np.ndarray): Boolean array of shape (m, n).
        target_model_idx (int): Index of the target model.

    Returns
    -------
        stat_target (np.ndarray): Statistics for the target model (n, k).
        stat_in (np.ndarray): Statistics for shadow models where in_indices is True (m, n, k), masked with np.nan.
        stat_out (np.ndarray): Statistics for shadow models where in_indices is False (m, n, k), masked with np.nan.
    """
    stat_target = stat[target_model_idx]
    shadow_stats = np.delete(stat, target_model_idx, axis=0)
    shadow_indices = np.delete(in_indices, target_model_idx, axis=0)

    stat_in = np.where(shadow_indices[:, :, np.newaxis], shadow_stats, np.nan)
    stat_out = np.where(~shadow_indices[:, :, np.newaxis], shadow_stats, np.nan)

    stat_target = stat_target.astype(np.float64)
    stat_in = stat_in.astype(np.float64)
    stat_out = stat_out.astype(np.float64)

    return stat_target, stat_in, stat_out


def get_indices_target(in_indices: np.ndarray, target_model_idx: int) -> np.ndarray:
    """
    Get the indices for the target model.

    Parameters
    ----------
        in_indices (np.ndarray): Boolean array of shape (m, n).
        target_model_idx (int): Index of the target model.

    Returns
    -------
        np.ndarray: Boolean array of shape (n,) for the target model.
    """
    return in_indices[target_model_idx]


def run_simple_lira(
    stat_target: np.ndarray,
    stat_in: np.ndarray,
    stat_out: np.ndarray,
    fix_variance: bool = True,
    use_median: bool = True,
) -> np.ndarray:
    """
    Compute LiRA scores using Gaussian distribution fitting.

    Parameters
    ----------
        stat_target (np.ndarray): Array of shape (n, k) containing target statistics.
        stat_in (np.ndarray): Array of shape (m, n, k) with nan masked in statistics.
        stat_out (np.ndarray): Array of shape (m, n, k) with nan masked out statistics.
        option (str): "both", "in", or "out".
        fix_variance (bool): Use global variance if True, else per-example.
        use_median (bool): Use median if True, else mean.

    Returns
    -------
        np.ndarray: LiRA scores for each example.
    """

    agg_func = np.nanmedian if use_median else np.nanmean
    std_func = np.nanstd

    # Aggregate mean/median
    avg_in = agg_func(stat_in, axis=0)
    avg_out = agg_func(stat_out, axis=0)

    # Compute std
    if fix_variance:
        std_in = std_func(stat_in - avg_in[np.newaxis, :, :])
        std_out = std_func(stat_out - avg_out[np.newaxis, :, :])
    else:
        std_in = std_func(stat_in, axis=0)
        std_out = std_func(stat_out, axis=0)

    eps = 1e-30
    # Compute logpdfs
    log_pr_in = scipy.stats.norm.logpdf(stat_target, avg_in, std_in + eps)
    log_pr_out = scipy.stats.norm.logpdf(stat_target, avg_out, std_out + eps)

    # Compute scores
    scores = log_pr_in - log_pr_out
    return scores.mean(axis=1)  # Return mean score across augmentations
