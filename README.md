# mialibrary
A simple library to conduct LiRA (Likelihood Ratio Attack) from Carlini et al. (2022) and RMIA (Robust Membership Inference Attack) from Zarifzadeh et al. (2024).

## Installation

You can install the library locally using `pip`:
```bash
pip install mialibrary
```
Available on PyPI: https://pypi.org/project/mialibrary/

## Core Functionality

This library takes pre-computed shadow model outputs and shadow model training indices as input and computes the likelihood ratios.
Suppose we are attacking an example (x, y), the corresponding LiRA scores are defined as the likelihood ratio:

$$
\Lambda := 
\frac{
    \mathcal{N} \big( \text{model output of } (x, y); \, \mu_{\text{in}}, \sigma_{\text{in}} \big)
}{
    \mathcal{N} \big( \text{model output of } (x, y); \, \mu_{\text{out}}, \sigma_{\text{out}} \big)
}
$$

The recommended shadow model output is logit computed from the predicted probabilities in the case of classifiers:

$$
\text{logit}(p) = \log\left(\frac{p}{1-p}\right) \\
\text{for } p = f_\theta(x)_y
$$

where $f_\theta(x)_y$ denotes the predicted probability of class $y$.

**RMIA (Robust Membership Inference Attack):**
RMIA extends membership inference by computing the probability ratio of the target point's output against a dataset of reference population points. This estimates if the target model's behavior on the specific instance is highly disproportionate compared to the baseline population distribution.

## Required Input Files

The library generally requires two input files from a model the user attacks:

- `stat_{identifier}`: Pre-computed statistics from model outputs
    - Format: list of NumPy arrays, NumPy arrays, or JAX arrays
    - Shape of stat: (m_models, n_samples, k_augmentations)

- `in_indices_{identifier}`: Training data indices
    - Format: list of NumPy arrays, NumPy arrays, or JAX arrays
    - Shape of in_indices: (m_models, n_samples)
    - Values: Boolean (True: used in training, False: not used)

**Important Note for RMIA**: To conduct the RMIA (Robust Membership Inference Attack), you will **additionally need population points**. These are pre-computed statistics from a separate reference set of samples drawn from the population, which RMIA uses to estimate the baseline distribution of model outputs. They must be formatted as:
- `stat_pop_{identifier}`: Pre-computed statistics for the population points
    - Format: list of NumPy arrays, NumPy arrays, or JAX arrays


## Output

- `scores_{identifier}`: Attack scores (e.g., LiRA log-likelihood ratios or RMIA probability ratios)
    - Format: NumPy array
    - Shape: (m_models, n_samples)
    - Used to assess membership inference success


## Basic Usage

### Computing LiRA scores
To compute the LiRA score, use the following command in your terminal:

```bash
# For standard LiRA computation using median and global variance.
python -m mialibrary.lira.process_fast_lira --data_path /path/to/data --use_median --fix_variance
```

In this example, scores will be calculated based on global variance and per-example median. With JAX acceleration, the score computation takes approximately **1-2 seconds**, regardless of the number of CPUs.

### Computing RMIA scores
To compute the RMIA score, use this command:

```bash
# RMIA score computation. Note: Ensure `stat_pop_*` files are present in the directory.
python -m mialibrary.rmia.process_rmia --data_path /path/to/data
```

More example usages can be found in `/examples` in the library.
It also includes small data which demonstrates the score computation and how the data is expected.

Note: This library focuses solely on score computation and evaluation. Model training and statistic computation should be handled separately.

