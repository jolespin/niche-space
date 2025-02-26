#!/usr/bin/env python
import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import combinations
from scipy.spatial.distance import squareform


# def jaccard_distance(row1, row2):
#     """Calculate the Jaccard distance between two binary arrays."""
#     intersection = np.sum(np.logical_and(row1, row2))
#     union = np.sum(np.logical_or(row1, row2))
#     return 1 - intersection / union if union > 0 else 1


def pairwise_distances_nonredundant(X, metric: str, n_jobs=1, redundant_form: bool=False, **kws):
    # if isinstance(metric, str):
    #     try:
    #         import scipy
    #         metric = getattr(scipy.spatial.distance, metric)
    #     except AttributeError:
    #         raise ValueError(f"{metric} not supported in scipy.spatial.distance. Please provide a callable for custom metrics.")

    if isinstance(X, pd.DataFrame):
        samples = X.index
        X = X.to_numpy()
    else:
        samples = None

    n = X.shape[0]

    distances = pairwise_distances(X, metric=metric, n_jobs=n_jobs, **kws)
    if redundant_form:
        if samples is not None:
            return pd.DataFrame(distances, index=samples, columns=samples)
        else:
            return distances
    else:
        distances = squareform(distances, checks=False)

        if samples is not None:
            combinations_samples = pd.Index(map(frozenset, combinations(samples, 2)))
            return pd.Series(distances, index=combinations_samples)
        else:
            return distances
