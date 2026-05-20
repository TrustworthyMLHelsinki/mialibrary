import pickle
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def _load_data(path):
    """
    Load data from disk for a specific configuration.
    Warning: This expects that scores and y_true have the same order as N.
    """

    with open(path, "rb") as f:
        scores_dict = pickle.load(f)

    M, N = scores_dict["y_true"].shape
    print(f"Loading {path}")
    print(f"M={M} N={N}")

    if scores_dict["y_true"].shape == (M, N) and scores_dict["scores"].shape == (M, N):
        return scores_dict["scores"], scores_dict["y_true"], M, N, path
    else:
        scores = np.empty((M, N))
        y_true = np.empty((M, N))

        for m in range(M):
            y_true[m] = scores_dict["y_true"][m * N : (m + 1) * N]
            scores[m] = scores_dict["scores"][m * N : (m + 1) * N]

        assert np.all(np.hstack(y_true) == scores_dict["y_true"])
        assert np.all(np.hstack(scores) == scores_dict["scores"])

        return scores, y_true, M, N, path


def _compute_tpr_and_fpr(y_true, scores):
    """
    Takes y_true and scores and returns tpr and fpr.
    """
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=scores)
    return fpr, tpr


# def plot_curve(x, y, xlabel, ylabel, ax, label, color, style, title=None):
#     ax.plot([0, 1], [0, 1], "k-", lw=1.0)
#     ax.plot(x, y, lw=2, label=label, color=color, linestyle=style)
#     ax.set(xlabel=xlabel, ylabel=ylabel)
#     ax.set(aspect=1, xscale="log", yscale="log")
#     if title is not None:
#         ax.title.set_text(title)


def plot_curve(
    x, y, xlabel, ylabel, ax, label, color, style, title=None, low_bound=1e-7
):
    ax.plot([0, 1], [0, 1], "k-", lw=1.0)
    ax.plot(x, y, lw=2, label=label, color=color, linestyle=style)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set(aspect=1, xscale="log", yscale="log")
    ax.set_xlim(low_bound, 1)
    ax.set_ylim(low_bound, 1)
    if title is not None:
        ax.title.set_text(title)


def plot_roc_curve(mia_path: str, lowest_fpr_to_plot=1e-7):
    """
    Plot ROC curve for a given mia "in_indices" data.
    """
    scores, y_true, M, N, path = _load_data(mia_path)

    fpr, tpr = _compute_tpr_and_fpr(y_true=y_true.ravel(), scores=scores.ravel())

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    plot_curve(
        x=fpr,
        y=tpr,
        xlabel="FPR",
        ylabel="TPR",
        ax=ax,
        color="blue",
        style="-",
        label=None,
        title=None,
        low_bound=lowest_fpr_to_plot,
    )


def _compute_tpr_at_fpr(y_true, scores):
    """
    Takes y_true and scores and returns tpr at fpr for
    fpr in [1e-3, 1e-2, 1e-1].
    """
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=scores)
    fpr_points = [1e-3, 1e-2, 1e-1]
    mia_results = np.interp(x=fpr_points, xp=fpr, fp=tpr)
    return fpr_points, mia_results


def compute_tpr_at_fpr(mia_path: str):
    """
    Create a df that contains distances in feature space and tpr at fprs.
    """
    scores, y_true, M, N, path = _load_data(mia_path)

    fpr_points, tpr_at_fpr = _compute_tpr_at_fpr(
        y_true=y_true.ravel(), scores=scores.ravel()
    )
    return fpr_points, tpr_at_fpr
