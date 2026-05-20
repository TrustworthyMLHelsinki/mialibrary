"""
This script processes stats files to compute LiRA scores using JAX implementation.
It loads the necessary data, computes the scores, and saves the results.
Example usage in the command line:
python process_fast_lira.py --data_path /path/to/data
- this computes scores using mean and per-sample variance.
"""

import os
import glob
import timeit
import argparse
from typing import List


from mialibrary.data_loader import load_data_lira, save_results
from mialibrary.lira.check_types import check_types
from mialibrary.lira.validate_inputs import validate_inputs
from mialibrary.lira.compute_score import compute_lira_score


def process_files(unprocessed_paths: List[str], fix_variance: bool, use_median: bool):
    """
    Process all unprocessed files.
    """
    for score_path in sorted(unprocessed_paths):
        print(f"Processing: {score_path}", flush=True)

        in_indices_path = score_path.replace("scores", "in_indices")
        stat_path = score_path.replace("scores", "stat")

        # Load data
        stat, in_indices = load_data_lira(stat_path, in_indices_path)

        # Check types
        stat_checked, in_indices_checked = check_types(stat, in_indices)

        # Validate inputs
        stat_validated, in_indices_validated = validate_inputs(
            stat_checked, in_indices_checked, stat_path, in_indices_path
        )

        # Compute scores
        start_time = timeit.default_timer()
        scores = compute_lira_score(
            stat_validated, in_indices_validated, fix_variance, use_median
        )
        elapsed_time = timeit.default_timer() - start_time
        print(f"Time to compute scores: {elapsed_time:.2f} seconds")

        # Save results
        save_results(score_path, scores, in_indices)


def process_lira_data(
    data_path: str,
    filter_pattern: str = "*",
    fix_variance: bool = False,
    use_median: bool = False,
):
    """
    Process LiRA data files programmatically.

    Parameters
    ----------
    data_path : str
        Path to in_indices, stat, and output score files.
    filter_pattern : str
        Further filtering of stat files e.g., by dataset.
    fix_variance : bool
        Whether to fix variance in the LIRA score computation.
    use_median : bool
        Whether to use median in the LIRA score computation.
    """
    all_stat_paths = glob.glob(os.path.join(data_path, f"stat_{filter_pattern}.pkl"))
    all_score_paths = glob.glob(os.path.join(data_path, f"scores_{filter_pattern}.pkl"))
    unprocessed_paths = list(
        set([p.replace("stat", "scores") for p in all_stat_paths])
        - set(all_score_paths)
    )

    print(f"Found {len(all_stat_paths)} stats files")
    print(
        f"Already processed: {len(all_stat_paths) - len(unprocessed_paths)}, "
        f"to process: {len(unprocessed_paths)}"
    )

    process_files(unprocessed_paths, fix_variance, use_median)


def main():
    """
    Main function to parse arguments and process files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=None,
        help="Path to in_indices, stat, and output score files.",
    )
    parser.add_argument(
        "--filter",
        default="*",
        help="Further filtering of stat files e.g., by dataset.",
    )
    parser.add_argument(
        "--fix_variance",
        action="store_true",
        help="Whether to fix variance in the LIRA score computation.",
    )
    parser.add_argument(
        "--use_median",
        action="store_true",
        help="Whether to use median in the LIRA score computation.",
    )
    args = parser.parse_args()

    script_start = timeit.default_timer()

    process_lira_data(
        data_path=args.data_path,
        filter_pattern=args.filter,
        fix_variance=args.fix_variance,
        use_median=args.use_median,
    )

    script_end = timeit.default_timer()
    print(f"Total script time: {script_end - script_start:.2f} seconds")


if __name__ == "__main__":
    main()
