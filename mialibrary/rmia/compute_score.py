"""Score computation utilities for RMIA."""

import numpy as np

from mialibrary.rmia.rmia import (
    compute_pr_x,
    compute_ratio_x,
    compute_pr_z,
    compute_ratio_z,
    compute_rmia_score,
)


def compute_score_rmia(
    signals: np.ndarray,
    population_signals: np.ndarray,
    in_indices: np.ndarray,
    offline: bool = False,
    num_models: int = None,
) -> np.ndarray:
    """Compute RMIA scores given the statistics and in/out indices."""

    if offline:
        print("Computing RMIA scores in offline mode.")
    else:
        print("Computing RMIA scores in online mode.")
        if num_models is not None and num_models % 2 != 0:
            num_models -= 1

    signals = signals[:num_models, :]
    population_signals = population_signals[:num_models, :]
    in_indices = in_indices[:num_models, :]

    # Compute pr_x
    pr_x = compute_pr_x(signals=signals, in_indices=in_indices, offline=offline)

    # Compute ratio_x
    ratio_x = compute_ratio_x(x=signals, pr_x=pr_x)

    # Compute pr_z
    pr_z = compute_pr_z(population_signals)

    # Compute ratio_z
    ratio_z = compute_ratio_z(population_signals, pr_z)

    # Compute RMIA score
    scores = compute_rmia_score(ratio_x, ratio_z)

    return scores
