"""
Test cases for load_data and save_results functions in data_loader.py.
Tests cover successful data loading, file not found errors, corrupted file handling, and saving results.
"""

import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from tempfile import NamedTemporaryFile
from mialib.data_loader import load_data, save_results


@pytest.fixture
def valid_data_files():
    """Fixture to create temporary files with valid data."""
    with NamedTemporaryFile(delete=False) as stat_file, NamedTemporaryFile(
        delete=False
    ) as in_indices_file:
        stat_data = {"key": "value"}
        in_indices_data = [1, 2, 3]
        # Write data and ensure the file is properly closed
        pickle.dump(stat_data, stat_file)
        pickle.dump(in_indices_data, in_indices_file)
        stat_file.close()
        in_indices_file.close()
        yield stat_file.name, in_indices_file.name, stat_data, in_indices_data
    # Cleanup
    os.remove(stat_file.name)
    os.remove(in_indices_file.name)


@pytest.fixture
def corrupted_file():
    """Fixture to create a temporary file with invalid (corrupted) data."""
    with NamedTemporaryFile(delete=False) as corrupted_file:
        corrupted_file.write(b"not a pickle")
        corrupted_file.close()
        yield corrupted_file.name
    # Cleanup
    os.remove(corrupted_file.name)


def test_load_data_success(valid_data_files):
    # Unpack fixture data
    stat_path, in_indices_path, stat_data, in_indices_data = valid_data_files

    # Test successful data loading
    stat, in_indices = load_data(stat_path, in_indices_path)
    assert stat == stat_data
    assert in_indices == in_indices_data


def test_load_data_file_not_found():
    # Test FileNotFoundError for missing files
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_stat.pkl", "non_existent_in_indices.pkl")


def test_load_data_unpickling_error(corrupted_file):
    # Test pickle.UnpicklingError
    with pytest.raises(pickle.UnpicklingError):
        load_data(corrupted_file, corrupted_file)


def test_save_results_success():
    """Test successful saving of results."""
    scores = np.array([0.1, 0.5, 0.9])
    in_indices = np.array([1, 0, 1])

    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.close()  # Close the file so save_results can write to it
        save_results(temp_file.name, scores, in_indices)

        # Verify the saved file
        with open(temp_file.name, "rb") as f:
            result = pickle.load(f)
            assert "y_true" in result
            assert "scores" in result
            assert np.array_equal(result["y_true"], np.array([1, 0, 1]))
            assert np.array_equal(result["scores"], scores)

        # Cleanup
        os.remove(temp_file.name)


def test_save_results_file_not_found():
    """Test FileNotFoundError when saving to an invalid path."""
    scores = np.array([0.1, 0.5, 0.9])
    in_indices = np.array([1, 0, 1])

    with pytest.raises(FileNotFoundError):
        save_results("/non_existent_directory/results.pkl", scores, in_indices)


def test_save_results_pickling_error(monkeypatch):
    """Test pickle.PicklingError during serialization."""
    scores = np.array([0.1, 0.5, 0.9])
    in_indices = np.array([1, 0, 1])

    # Simulate a PicklingError
    def mock_pickle_dump(*args, **kwargs):
        raise pickle.PicklingError("Mock PicklingError")

    monkeypatch.setattr(pickle, "dump", mock_pickle_dump)

    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.close()  # Close the file so save_results can write to it
        with pytest.raises(pickle.PicklingError):
            save_results(temp_file.name, scores, in_indices)

        # Cleanup
        os.remove(temp_file.name)
