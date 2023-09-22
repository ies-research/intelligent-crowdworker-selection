import numpy as np
import pandas as pd

from itertools import combinations

from scipy.special import binom

from skactiveml.utils import is_labeled

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils import (
    check_X_y,
    check_random_state,
    check_array,
    column_or_1d,
)


def compute_annot_perf_clf(y_true, y, missing_label=-1):
    """
    Prints the performances of annotators for classification problems, i.e., micro and macro accuracies.

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        True class labels.
    y : array-like of shape (n_samples, n_annotators)
        Labels provided by the annotators.
    """
    y_true = column_or_1d(y_true)
    y = np.array(y)
    n_annotators = y.shape[1]
    acc = np.empty((2, n_annotators))
    for a in range(n_annotators):
        is_labeled_a = is_labeled(y[:, a], missing_label=missing_label)
        y_a = y[is_labeled_a, a]
        y_true_a = y_true[is_labeled_a]
        acc[0, a] = accuracy_score(y_true=y_true_a, y_pred=y_a)
        acc[1, a] = balanced_accuracy_score(y_true=y_true_a, y_pred=y_a)
    acc = pd.DataFrame(
        acc,
        index=["micro accuracy", "macro accuracy"],
        columns=np.arange(n_annotators),
    )
    return acc


def annot_sim_clf_cluster(
    X,
    y_true,
    cluster_annot_perfs,
    k_means_dict=None,
    random_state=None,
):
    """
    The knowledge of annotators is separated into clusters, where on each cluster an annotator can have different
    performances. These performances are expressed through labeling accuracies. The clusters are determined through a
    k-means algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    y_true : array-like, shape (n_samples)
        True class labels of the given samples X.
    cluster_annot_perfs : array-like of shape (n_annotators, n_clusters)
        The entry `cluster_annot_perfs[j, i]` indicates the accuracy of annotator `j` for labeling samples of
        cluster `i`.
    k_means_dict : None or dict, optional (default=None)
        Dictionary of parameters that are passed to `sklearn.cluster.MiniBatchKMeans`.
    random_state : None or int or numpy.random.RandomState, optional (default=None)
        The random state used for drawing the annotations and specifying the clusters.

    Returns
    -------
    y : np.ndarray of shape (n_samples, n_annotators)
        Class labels of simulated annotators.
    """
    # Check `X` and `y_true`.
    X, y_true = check_X_y(X, y_true, ensure_2d=True, allow_nd=False)
    n_samples = X.shape[0]

    # Check `cluster_annot_perfs`.
    cluster_annot_perfs = check_array(cluster_annot_perfs)
    if np.sum(cluster_annot_perfs < 0) or np.sum(cluster_annot_perfs > 1):
        raise ValueError("`cluster_perfs` must contain values in [0, 1]")
    n_annotators = cluster_annot_perfs.shape[0]
    n_clusters = cluster_annot_perfs.shape[1]

    # Check `k_means_dict`.
    if k_means_dict is None:
        k_means_dict = {
            "batch_size": 2 ** 13,
            "random_state": random_state,
            "max_iter": 1000,
            "n_init": 10,
        }
    if not isinstance(k_means_dict, dict):
        raise TypeError("`k_means_dict` must be a dictionary.")

    # Check and transform `random_state`.
    random_state = check_random_state(random_state)

    # Transform class labels to interval [0, n_classes-1].
    le = LabelEncoder().fit(y_true)
    y_true = le.transform(y_true)
    n_classes = len(le.classes_)

    # Compute clustering.
    y_cluster = MiniBatchKMeans(n_clusters=n_clusters, **k_means_dict).fit_predict(X)

    # Simulate annotators.
    y = np.empty((n_samples, n_annotators))
    for a_idx in range(n_annotators):
        P_predict = np.empty((n_samples, n_classes))
        for c_idx in range(n_clusters):
            is_c = y_cluster == c_idx
            p = (1 - cluster_annot_perfs[a_idx, c_idx]) / (n_classes - 1)
            P_predict[is_c] = p
            P_predict[is_c, y_true[is_c]] = cluster_annot_perfs[a_idx, c_idx]
        cumlative = P_predict.cumsum(axis=1)
        uniform = random_state.rand(len(cumlative), 1)
        y_predict = (uniform < cumlative).argmax(axis=1)
        y[:, a_idx] = le.inverse_transform(y_predict)

    return y, y_cluster


def generate_expert_cluster_combinations(n_annotators, n_clusters, n_expert_clusters, random_state, max_combs=15e7):
    """
    Helper function to randomly select expert clusters of annotators.

    Parameters
    ----------
    n_annotators : int
        Number of annotators.
    n_clusters : int
        Nuber of clusters.
    n_expert_clusters : int
        Number of expert clusters per annotator.
    random_state : int or np.random.RandomState or None, optional (default=None)
        Random state for selecting expert clusters.
    max_combs : int, optional (default=10e8)
        Maximum number of elements in the expert cluster combinations.
    """
    random_state = check_random_state(random_state)
    actual_combs = binom(n_clusters, n_expert_clusters) * n_expert_clusters
    if actual_combs <= max_combs:
        combs = []
        combs_list = np.array(list(combinations(np.arange(n_clusters), n_expert_clusters)))
        random_order = random_state.choice(np.arange(len(combs_list)), size=len(combs_list), replace=False)
        combs_list = combs_list[random_order]
        for comb in combs_list:
            combs.append(list(comb))
            if len(combs) == n_annotators:
                return np.array(combs, dtype=int)
    else:
        cluster_indices = np.arange(n_clusters)
        combs = np.vstack([random_state.choice(cluster_indices, size=n_expert_clusters) for _ in range(n_annotators)])
        return np.sort(combs, axis=1)

