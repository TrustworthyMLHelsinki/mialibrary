# mialibrary
A simple library to conduct LiRA (Likelihood Ratio Attack) from Carlini et al. (2022) and RMIA (Robust Membership Inference Attack) from Zarifzadeh et al. (2024).

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

## Required Input Files

The library requires two input files from a model the user attacks:

- `stat_{identifier}`: Pre-computed statistics from model outputs
    - Format: list of NumPy arrays, NumPy arrays, or JAX arrays
    - Shape of stat: (m_models, n_samples, k_augmentations)

- `in_indices_{identifier}`: Training data indices
    - Format: list of NumPy arrays, NumPy arrays, or JAX arrays
    - Shape of in_indices: (m_models, n_samples)
    - Values: Boolean (True: used in training, False: not used)


## Output

- `scores_{identifier}`: LiRA scores (log-likelihood ratios)
    - Format: NumPy array
    - Shape: (m_models, n_samples)
    - Used to assess membership inference success


## Basic Usage

To compute the LiRA score, use one of the following commands in your terminal:

```bash
# For standard LiRA computation using median and global variance.
python -m mialibrary.lira.process_fast_lira --data_path /path/to/data --use_median --fix_variance
```

In this example, scores will be calculated based on global variance and per-example median. With JAX acceleration, the score computation takes approximately **1-2 seconds**, regardless of the number of CPUs.

More example usages can be found in '/examples' in the library.
It also includes small data which demonstrates the score computation and how the data is expected.

Note: This library focuses solely on score computation and evaluation. Model training and statistic computation should be handled separately.

