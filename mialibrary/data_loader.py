"""Data loading and saving utilities"""

import os
import pickle
from typing import Tuple, Any

import numpy as np
import jax.numpy as jnp


def load_data_lira(stat_path: str, in_indices_path: str) -> Tuple[Any, Any]:
    """
    Load stat and in_indices data from files.

    Raises
    ------
    FileNotFoundError
        If the specified files do not exist.
    Exception
        If there is an error during file reading or deserialization.
    """
    try:
        # Validate file existence
        if not os.path.exists(stat_path):
            raise FileNotFoundError(f"Stat file not found: {stat_path}")
        if not os.path.exists(in_indices_path):
            raise FileNotFoundError(f"In_indices file not found: {in_indices_path}")

        # Load data
        with open(stat_path, "rb") as f:
            stat = pickle.load(f)
        with open(in_indices_path, "rb") as f:
            in_indices = pickle.load(f)

        print("Data loaded successfully.")
        return stat, in_indices

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except pickle.UnpicklingError:
        print("Error: Failed to deserialize the data. The file may be corrupted.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def load_data_rmia(
    signals_path: str, population_signals_path: str, in_indices_path: str
) -> Tuple[Any, Any, Any]:
    """Load signals, population signals, and in_indices data from files.

    Raises
    ------
    FileNotFoundError
        If the specified files do not exist.
    Exception
        If there is an error during file reading or deserialization.
    """
    try:
        # Validate file existence
        if not os.path.exists(signals_path):
            raise FileNotFoundError(f"Signals file not found: {signals_path}")
        if not os.path.exists(population_signals_path):
            raise FileNotFoundError(
                f"Population signals file not found: {population_signals_path}"
            )
        if not os.path.exists(in_indices_path):
            raise FileNotFoundError(f"In_indices file not found: {in_indices_path}")

        # Load data
        with open(signals_path, "rb") as f:
            signals = pickle.load(f)
        with open(population_signals_path, "rb") as f:
            population_signals = pickle.load(f)
        with open(in_indices_path, "rb") as f:
            in_indices = pickle.load(f)

        print("Data loaded successfully.")
        return signals, population_signals, in_indices

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except pickle.UnpicklingError:
        print("Error: Failed to deserialize the data. The file may be corrupted.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def save_results(score_path: str, scores: jnp.ndarray, in_indices: jnp.ndarray):
    """
    Save the computed scores and labels.

    Raises
    ------
    Exception
        If there is an error during file writing or serialization.
    """
    try:
        scores_np = np.array(scores)
        y_true_np = np.where(np.array(in_indices), 1, 0)

        result = {"y_true": y_true_np, "scores": scores_np}

        with open(score_path, "wb") as f:
            pickle.dump(result, f)

        print(f"Scores computed and saved successfully to {score_path}.")

    except FileNotFoundError:
        print(f"Error: The specified path does not exist: {score_path}")
        raise
    except pickle.PicklingError:
        print("Error: Failed to serialize the data. Check the data format.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving results: {e}")
        raise
