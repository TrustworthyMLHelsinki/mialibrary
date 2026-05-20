"""
Robust Membership Inference Attacks (RMIA) implementation.

This module provides implementations of the RMIA algorithm for membership
inference attacks as described in the paper. It supports both offline and online modes.
"""

from statistics import mode
import numpy as np


def sigmoid(x):
    """Convert the logit to softmax. Here it is sigmoid because we have binary classification."""
    return 1 / (1 + np.exp(-x))


def compute_pr_x_offline(signals, in_indices):
    """Compute pr_x for out shadow models. Line 12 in Algorithm 1 of the paper. Offline."""
    num_out_models = np.sum(~in_indices, axis=0)
    assert np.all(num_out_models > 0), "No out reference models available."
    out_signals = np.where(in_indices, 0, signals)  # select only out shadow models

    out_signals_mean = np.sum(out_signals, axis=0) / num_out_models
    corrected_out_signals_mean = (num_out_models * out_signals_mean - out_signals) / (
        num_out_models - 1
    )

    pr_x_out = np.where(in_indices, out_signals_mean, corrected_out_signals_mean)

    return pr_x_out


def compute_pr_x_online(signals, in_indices):
    """Compute pr_x for in and out shadow models. Line 10 in Algorithm 1 of the paper. Online."""
    num_in_models = np.sum(in_indices, axis=0)
    num_out_models = np.sum(~in_indices, axis=0)
    assert np.all(num_in_models > 0), "No in reference models available."
    assert np.all(num_out_models > 0), "No out reference models available."

    in_signals = np.where(in_indices, signals, 0)  # select only in shadow models
    out_signals = np.where(~in_indices, signals, 0)  # select only out shadow models

    in_signals_mean = np.sum(in_signals, axis=0) / num_in_models
    corrected_in_signals_mean = (num_in_models * in_signals_mean - in_signals) / (
        num_in_models - 1
    )
    out_signals_mean = np.sum(out_signals, axis=0) / num_out_models
    corrected_out_signals_mean = (num_out_models * out_signals_mean - out_signals) / (
        num_out_models - 1
    )

    pr_x_in = np.where(in_indices, corrected_in_signals_mean, in_signals_mean)
    pr_x_out = np.where(in_indices, out_signals_mean, corrected_out_signals_mean)

    pr_x = (pr_x_in + pr_x_out) / 2
    return pr_x


def compute_ratio_x(x, pr_x):
    """Compute ratio_x. Line 15 in Algorithm 1 of the paper."""
    assert np.all(pr_x != 0), "pr_x must not be zero"
    return x / pr_x


def compute_pr_z(population_signals):
    """Compute pr_z. Line 17 in Algorithm 1 of the paper."""
    num_models, _ = population_signals.shape
    pr_z = (np.sum(population_signals, axis=0) - population_signals) / (num_models - 1)
    return pr_z


def compute_ratio_z(population_signals, pr_z):
    """Compute ratio_z. Line 18 in Algorithm 1 of the paper."""
    assert np.all(pr_z != 0), "pr_z must not be zero"
    return population_signals / pr_z


def compute_rmia_score(ratio_x, ratio_z):
    """Compute RMIA score. Line 16-23 in Algorithm 1 of the paper."""
    m, n = ratio_x.shape
    z_sorted = np.sort(ratio_z, axis=1)  # (m, n)
    # Row-wise searchsorted (vectorized per row; small Python loop over m)
    counts_rows = [
        np.searchsorted(z_sorted[i], ratio_x[i], side="left") for i in range(m)
    ]
    counts = np.vstack(counts_rows)  # (m, n), int
    return counts / n


def compute_pr_x(signals, in_indices, offline: bool = False):
    """
    Compute pr_x based on the selected mode.

    Parameters
    ----------
    signals : np.ndarray
        The signals to process.
    in_indices : np.ndarray
        Boolean array indicating in-model indices.
    offline : bool, optional
        If True, use offline computation; otherwise, use online computation.
    Returns
    -------
    np.ndarray
        The computed pr_x values.
    """
    if not offline:
        return compute_pr_x_online(signals, in_indices)
    else:
        return compute_pr_x_offline(signals, in_indices)
