# -*- coding: utf-8 -*-
from __future__ import annotations
import os,sys,warnings
from typing import Optional
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm


import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import issymmetric
from sklearn.base import clone
from sklearn.neighbors import (
    KNeighborsTransformer, 
    kneighbors_graph,
)
from scipy.spatial.distance import squareform

from datafold.pcfold.distance import BruteForceDist
from datafold.pcfold import PCManifoldKernel

from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances


from scipy.spatial.distance import (
    pdist,
    squareform,
)
from pyexeggutor import (
    build_logger,
    write_pickle,
    read_pickle,
    check_argument_choice,
    format_header,
)

from sklearn.metrics import (
    pairwise_distances,
    silhouette_score,
)
import ensemble_networkx as enx
import igraph as ig
import optuna
import joblib

# Metabolic Niche Space
from .utils import (
    compile_parameter_space,
    is_square_symmetric,
    stop_when_exceeding_trials,
)



def kneighbors_graph_from_transformer(X, knn_transformer=KNeighborsTransformer, mode="connectivity", include_self=True, **transformer_kwargs):
    """
    Calculate distance or connectivity with self generalized to any KNN transformer
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        knn_transformer : knn_transformer-like, [Default: KNeighborsTransformer]
            Either a fitted KNN transformer or an uninstantiated KNN transformer 
            with parameters specified **kwargs

        mode : str [Default: distance]
            Type of returned matrix: ‘connectivity’ will return the connectivity matrix with ones and zeros, 
            and ‘distance’ will return the distances between neighbors according to the given metric.

        include_self: bool [Default: True]
            Whether or not to mark each sample as the first nearest neighbor to itself. 
            If ‘auto’, then True is used for mode=’connectivity’ and False for mode=’distance’.
            

        transformer_kwargs: 
            Passed to knn_transformer if not instantiated
            
        Returns
        -------
        knn_graph : array-like, shape (n_samples, n_samples),

            scipy.sparse.csr_matrix

    """

    # mode checks
    assert mode in {"distance", "connectivity"}, "mode must be either 'distance' or 'connectivity'"

    # include_self checks
    if include_self == "auto":
        if mode == "distance":
            include_self = False
        else:
            include_self = True

    # If not instantiated, then instantiate it with **transformer_kwargs
    if not isinstance(knn_transformer, type):
        # Get params from model and add n_neighbors -= 1
        if include_self:
            assert not bool(transformer_kwargs), "Please provide uninstantiated `knn_transformer` or do not provide `transformer_kwargs`"
            warnings.warn("`include_self=True and n_neighbors=k` is equivalent to `include_self=False and n_neighbors=(k-1). Backend is creating a clone with n_neighbors=(k-1)")
            knn_transformer = clone(knn_transformer)
            n_neighbors = knn_transformer.get_params("n_neighbors")
            knn_transformer.set_params({"n_neighbors":n_neighbors - 1})
    else:
        try:
            n_neighbors = transformer_kwargs["n_neighbors"]
        except KeyError:
            raise Exception("Please provide `n_neighbors` as kwargs (https://docs.python.org/3/glossary.html#term-argument)")
        if include_self:
            transformer_kwargs["n_neighbors"] = n_neighbors - 1
        knn_transformer = knn_transformer(**transformer_kwargs)
        
    # Compute KNN graph for distances
    knn_graph = knn_transformer.fit_transform(X)
    
    # Convert to connectivity
    if mode == "connectivity":
        # Get all connectivities
        knn_graph = (knn_graph > 0).astype(float)
           
        # Set diagonal to 1.0
        if include_self:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                knn_graph.setdiag(1.0)
                
    return knn_graph

def brute_force_kneighbors_graph_from_rectangular_distance(distance_matrix, n_neighbors:int, mode="connectivity", include_self=True):
    assert mode in {"distance", "connectivity"}, "mode must be either 'distance' or 'connectivity'"

    # ==================================================================
    # Naive
    # -----
    # out = csr_matrix(distance_matrix.shape)
    # if mode == "connectivity":
    #     for i, index in enumerate(indices):
    #         out[i,index] = 1.0
    # if mode == "distance":
    #     distances = np.sort(distance_matrix, axis=1)[:, :n_neighbors]
    #     for i, index in enumerate(indices):
    #         out[i,index] = distances[i]
    # ==================================================================
    if include_self:
        n_neighbors = n_neighbors - 1
        
    # Sort indices up to n_neighbors            
    indices = np.argpartition(distance_matrix, n_neighbors, axis=1)[:, :n_neighbors]
    # Use ones for connectivity values
    if mode == "connectivity":
        data = np.ones(distance_matrix.shape[0] * n_neighbors, dtype=float)
    # Use distances values
    if mode == "distance":
        data = np.partition(distance_matrix, n_neighbors, axis=1)[:, :n_neighbors].ravel()
    # Get row indices
    row = np.repeat(np.arange(distance_matrix.shape[0]), n_neighbors)
    # Get column indicies
    col = indices.ravel()
    
    # Build COO matrix
    graph = sps.coo_matrix((data, (row, col)), shape=distance_matrix.shape)

    # Convert to CRS matrix
    return graph.tocsr()
    
def pairwise_distances_kneighbors(
    X, 
    metric: str, 
    n_neighbors=None, 
    n_jobs=1, 
    redundant_form: bool=True, 
    include_self=False,
    symmetric=True,
    **kws,
):
    """
    Calculate pairwise distances or k-nearest neighbors distances between samples.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    metric : str or callable
        Distance metric to use
    n_neighbors : int, optional
        Number of neighbors for kNN calculation
    n_jobs : int
        Number of parallel jobs
    redundant_form : bool
        Whether to return full matrix (True) or condensed form (False)
    include_self : bool
        Whether to include self as potential neighbor
    symmetric : bool
        Whether to symmetrize the kNN matrix
    **kws : dict
        Additional keywords passed to metric function
        
    Returns
    -------
    distances : array or DataFrame
        Distance matrix in requested format
    """

    if isinstance(X, pd.DataFrame):
        samples = X.index
        X = X.to_numpy()
    else:
        samples = None

    n = X.shape[0]

    if n_neighbors is None:
        # Calculate full distance matrix
        distances = pairwise_distances(X, metric=metric, n_jobs=n_jobs, **kws)
    else:
        # Calculate kNN distances
        # n_neighbors_adj = n_neighbors + (1 if include_self else 0)
        distances = kneighbors_graph(
            X, 
            n_neighbors=n_neighbors,
            mode="distance", 
            metric=metric, 
            n_jobs=n_jobs,
            include_self=include_self, 
            **kws
        ).todense()
        distances = np.asarray(distances)
        
        if symmetric:
            # Ensure symmetry by taking maximum of each pair
            distances = np.maximum(distances, distances.T)
    
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

def convert_distance_matrix_to_kneighbors_matrix(
    distance_matrix, 
    n_neighbors, 
    redundant_form=True,
    include_self=False, 
    symmetric=True,
    ):
    """
    Convert a fully-connected distance matrix to a k-nearest neighbors (kNN) distance matrix.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        The full distance matrix (n x n)
    n_neighbors : int
        Number of nearest neighbors to retain for each point
    include_self : bool
        Whether to include self as potential neighbor
    symmetric : bool
        Whether to symmetrize the kNN distance matrix
    
    Returns
    -------
    knn_matrix : np.ndarray
        The kNN distance matrix with non-neighbor distances set to 0
    """
    if isinstance(distance_matrix, pd.DataFrame):
        samples = distance_matrix.index
        distance_matrix = distance_matrix.to_numpy()
    else:
        samples = None
    n = distance_matrix.shape[0]
    knn_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    for i in range(n):
        # Get sorted indices excluding self if needed
        if not include_self:
            # Create mask excluding the diagonal element
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sorted_indices = np.argsort(distance_matrix[i][mask])
            # Map back to original indices
            orig_indices = np.arange(n)[mask][sorted_indices]
            knn_indices = orig_indices[:n_neighbors]
        else:
            sorted_indices = np.argsort(distance_matrix[i])
            knn_indices = sorted_indices[:n_neighbors]
        
        # Assign distances to the k nearest neighbors
        knn_matrix[i, knn_indices] = distance_matrix[i, knn_indices]
    
    if symmetric:
        knn_matrix = np.maximum(knn_matrix, knn_matrix.T)
    
    if redundant_form:
        if samples is not None:
            return pd.DataFrame(knn_matrix, index=samples, columns=samples)
        else:
            return knn_matrix
    else:
        knn_matrix = squareform(knn_matrix, checks=False)
        if samples is not None:
            combinations_samples = pd.Index(map(frozenset, combinations(samples, 2)))
            return pd.Series(knn_matrix, index=combinations_samples)
        else:
            return knn_matrix
    return knn_matrix

# def convert_distance_matrix_to_dynamic_kneighbors_matrix(
#     distance_matrix, 
#     n_neighbors="dynamic", 
#     elbow_threshold=0.5, 
#     redundant_form=True,
#     include_self=False, 
#     symmetric=True
#     ):
#     """
#     Convert a fully-connected distance matrix to a k-nearest neighbors (kNN) distance matrix.
#     Supports both fixed and dynamic neighbor selection.
    
#     Parameters
#     ----------
#     distance_matrix : np.ndarray or pd.DataFrame
#         The full distance matrix (n x n).
#     n_neighbors : int or "dynamic"
#         Number of nearest neighbors to retain for each point. 
#         Use "dynamic" to dynamically determine neighbors using the elbow method.
#     elbow_threshold : float
#         Threshold fraction of the maximum distance difference to identify the elbow when n_neighbors="dynamic".
#     include_self : bool
#         Whether to include self as a potential neighbor.
#     symmetric : bool
#         Whether to symmetrize the kNN distance matrix.
#     redundant_form : bool
#         Whether to return the matrix in redundant form or condensed form.
    
#     Returns
#     -------
#     knn_matrix : np.ndarray or pd.DataFrame
#         The kNN distance matrix with non-neighbor distances set to 0.
#     """
#     if isinstance(distance_matrix, pd.DataFrame):
#         samples = distance_matrix.index
#         distance_matrix = distance_matrix.to_numpy()
#     else:
#         samples = None
    
#     n = distance_matrix.shape[0]
#     knn_matrix = np.zeros_like(distance_matrix, dtype=float)
    
#     for i in range(n):
#         if n_neighbors == "dynamic":

#             node_distances = distance_matrix[i, :].copy()
#             if not include_self:
#                 # Exclude self-distance by setting it to infinity
#                 node_distances[i] = np.inf

#             # Sort distances and indices
#             sorted_indices = np.argsort(node_distances)
#             sorted_distances = node_distances[sorted_indices]
#             # Compute distance differences
#             distance_diffs = np.diff(sorted_distances)
            
#             # Identify the elbow point
#             threshold = elbow_threshold * np.max(distance_diffs)
#             elbow_point = np.argmax(distance_diffs > threshold) + 1  # +1 to include neighbors
            
#             knn_indices = sorted_indices[:elbow_point]
#         else:
#             # Fixed number of neighbors
#             if not include_self:
#                 # Create mask excluding the diagonal element
#                 mask = np.ones(n, dtype=bool)
#                 mask[i] = False
#                 sorted_indices = np.argsort(distance_matrix[i][mask])
#                 # Map back to original indices
#                 orig_indices = np.arange(n)[mask][sorted_indices]
#                 knn_indices = orig_indices[:n_neighbors]
#             else:
#                 sorted_indices = np.argsort(distance_matrix[i])
#                 knn_indices = sorted_indices[:n_neighbors]
        
#         # Assign distances to the k nearest neighbors
#         knn_matrix[i, knn_indices] = distance_matrix[i, knn_indices]
    
#     if symmetric:
#         knn_matrix = np.maximum(knn_matrix, knn_matrix.T)
    
#     if redundant_form:
#         if samples is not None:
#             return pd.DataFrame(knn_matrix, index=samples, columns=samples)
#         else:
#             return knn_matrix
#     else:
#         knn_matrix = squareform(knn_matrix, checks=False)
#         if samples is not None:
#             from itertools import combinations
#             combinations_samples = pd.Index(map(frozenset, combinations(samples, 2)))
#             return pd.Series(knn_matrix, index=combinations_samples)
#         else:
#             return knn_matrix
    
#     return knn_matrix

class KNeighborsKernel(PCManifoldKernel):
    """
    K-Nearest Neighbors Kernel
    
    Acknowledgement: 
    https://gitlab.com/datafold-dev/datafold/-/issues/166
    """
    def __init__(self, metric:str, n_neighbors:int, distance_matrix:Optional[np.ndarray]=None, copy_distance_matrix=False, verbose=0):

        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.copy_distance_matrix = copy_distance_matrix
        if distance_matrix is not None:
            if len(distance_matrix.shape) == 1:
                distance_matrix = squareform(distance_matrix)
            else:
                if copy_distance_matrix:
                    distance_matrix = distance_matrix.copy()
        self.distance_matrix = distance_matrix

        distance = BruteForceDist(metric=metric)
        super().__init__(is_symmetric=True, is_stochastic=False, distance=distance)

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, **kernel_kwargs):
        if all([
            Y is None,
            self.distance_matrix is not None,
            ]):
            n, m = X.shape
            assert self.distance_matrix.shape[0] == n, "X.shape[0] must equal distance_matrx.shape[0]"
            assert self.distance_matrix.shape[1] == n, "X.shape[0] must equal distance_matrx.shape[1]"
            distance_matrix = self.distance_matrix
            if self.verbose > 0:
                print("Precomputed distance matrix detected. Skipping pairwise distance calculations.", file=sys.stderr, flush=True)
        else:
            distance_matrix = self.distance(X, Y)
        return self.evaluate(distance_matrix)

    def evaluate(self, distance_matrix):

        # Compute KNN connectivity kernel
        distance_matrix_is_square = False
        shape = distance_matrix.shape
        if shape[0] == shape[1]:
            if issymmetric(distance_matrix):
                distance_matrix_is_square = True
        if distance_matrix_is_square:
            connectivities = kneighbors_graph(distance_matrix, n_neighbors=self.n_neighbors, metric="precomputed", include_self=True, mode="connectivity")
        else:
            connectivities = brute_force_kneighbors_graph_from_rectangular_distance(distance_matrix, n_neighbors=self.n_neighbors, include_self=True, mode="connectivity")

        return connectivities
    
class KNeighborsLeidenClustering(object):
    def __init__(
        self, 
        # General
        name:str=None,
        observation_type:str=None,
        feature_type:str=None,
        class_type:str=None,
        method:str = "one_minus",
        initial_distance_metric:str="precomputed",
        scoring_distance_metric:str="euclidean",
        n_neighbors:int="auto",
        
        # Community detection
        n_iter=10, 
        converge_iter=-1,
        minimum_membership_consistency=1.0, 
        cluster_prefix="c",
        
        # Optuna
        n_trials=10,
        n_jobs:int=1,
        n_concurrent_trials:int=1,
        initial_params:dict = None,
        objective_direction="maximize",
        checkpoint_directory=None,
        study_timeout=None,
        study_callbacks=None,
        random_state=0,
        verbose=1,
        stream=sys.stdout,
        ):
        
        # General
        if name is None:
            name = str(uuid.uuid4())
            
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.class_type = class_type        
        self.method = method
        self.initial_distance_metric = initial_distance_metric
        self.scoring_distance_metric = scoring_distance_metric
        
        # Community detection
        self.n_iter=n_iter
        self.converge_iter=converge_iter
        self.minimum_membership_consistency=minimum_membership_consistency
        self.cluster_prefix = cluster_prefix
        
        # Optuna
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.n_concurrent_trials = n_concurrent_trials
        self.initial_params = initial_params
        self.checkpoint_directory = checkpoint_directory
        self.random_state = random_state
        self.study_timeout = study_timeout
        if study_callbacks is None:
            study_callbacks = []
        self.study_callbacks = study_callbacks
        self.objective_direction = objective_direction

        # Hyperparameters
        self.is_tuned = True
        if n_neighbors == "auto":
            n_neighbors = [int, 10, 100]
            self.is_tuned = False
        if isinstance(n_neighbors, list):
            self.is_tuned = False
        self.n_neighbors = n_neighbors
        
        self._param_space = dict(
            n_neighbors = self.n_neighbors,
        )
        
        self.logger = build_logger(self.name, stream=stream)
        self.verbose = verbose
        self.is_fitted = False
        
    @staticmethod
    def one_minus(distance_matrix):
        """1 - distance transformation (e.g., Jaccard similarity)."""
        return 1 - distance_matrix

    @staticmethod
    def inverse(distance_matrix):
        """1 / (1 + distance) transformation to avoid division by zero."""
        return 1 / (1 + distance_matrix)

    @staticmethod
    def exponential(distance_matrix, sigma=1.0):
        """Exponential decay transformation (e.g., Gaussian Kernel)."""
        return np.exp(-distance_matrix / sigma)
        
    def distance_to_similarity(self, distances):
        """Applies the selected transformation method."""
        transformations = {
            "one_minus": self.one_minus,
            "inverse": self.inverse,
            "exponential": self.exponential,
        }

        if self.method not in transformations:
            raise ValueError(f"Unknown method: {self.method}. Choose from {list(transformations.keys())}")

        return transformations[self.method](distances)
    
    def tune(
        self,
        distance_matrix:pd.DataFrame,
        sampler, 
        **study_kws,
        ):

        def _objective(trial):
            try:

                # Compile parameters
                params = compile_parameter_space(
                    trial, 
                    self._param_space,
                )

                # Parameters
                n_neighbors = params["n_neighbors"]

                if n_neighbors >= distance_matrix.shape[0]:
                    raise ValueError(f"n_neighbors {n_neighbors} is larger than the number of observations {distance_matrix.shape[0]}")
                else:

                    # Convert distance matrix to non-redundant KNN
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Convert distance matrix to non-redundant KNN: n_neighbors={n_neighbors}")
                    knn = convert_distance_matrix_to_kneighbors_matrix(distance_matrix, n_neighbors=n_neighbors, redundant_form=False)
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Remove disconnected nodes: n_neighbors={n_neighbors}")                    
                    knn = knn[knn.values > 0]
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Convert to similarity: n_neighbors={n_neighbors}")                    
                    knn_similarity = self.distance_to_similarity(knn)
                    del knn

                    # Convert KNN to iGraph
                    graph = enx.convert_network(knn_similarity, ig.Graph)

                    # Identify leiden communities with multiple seeds
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Identify Leiden communities: n_neighbors={n_neighbors}")
                    progressbar_message = None
                    df_communities = enx.community_detection(graph, n_iter=self.n_iter, converge_iter=self.converge_iter, n_jobs=1, progressbar_message=progressbar_message)

                    # Identify membership co-occurrence ratios
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Identify membership co-occurrence ratios: n_neighbors={n_neighbors}")
                    node_pair_membership_cooccurrences = enx.community_membership_cooccurrence(df_communities).mean(axis=1)
                    del df_communities

                    # Identify node pairs that have co-membership 100% of the time
                    node_pairs_with_consistent_membership = set(node_pair_membership_cooccurrences[lambda x: x >= self.minimum_membership_consistency].index)
                    del node_pair_membership_cooccurrences

                    # Get list of clustered edges
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Get list of clustered edges and build clustered graph: n_neighbors={n_neighbors}")
                    clustered_edgelist = enx.get_undirected_igraph_edgelist_indices(graph, node_pairs_with_consistent_membership)
                    del node_pairs_with_consistent_membership

                    # Get clustered graph
                    graph_clustered = graph.subgraph_edges(clustered_edgelist, delete_vertices=True)
                    node_to_cluster = pd.Series(enx.get_undirected_igraph_connected_components(graph_clustered))
                    del graph

                    # Calculate silhouette scores
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Calculating silhouette scores: n_neighbors={n_neighbors}")
                    clustered_nodes = node_to_cluster.index
                    index = clustered_nodes.map(lambda x: distance_matrix.index.get_loc(x)).values
                    dist = distance_matrix.values[index,:][:,index]
                    score = silhouette_score(dist, node_to_cluster.values, metric="precomputed", sample_size=None, random_state=None) 
                    del dist

                    return score

            except Exception as e:
                self.logger.error(f"[Trial {trial.number}] Failed due to error: {e}. Marking as pruned.")
                raise optuna.TrialPruned()  # Prevents skipping trials

            finally:
                if self.checkpoint_directory:
                    joblib.dump(study, os.path.join(self.checkpoint_directory,  f"{self.name}.Optuna.{self.__class__.__name__}.pkl"))  # Save checkpoint

        # Sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)

        # Study
        study_params = {
            "direction":self.objective_direction, 
            "study_name":self.name, 
            "sampler":sampler, 
            **study_kws,
        }

        # Checkpoints
        study = None
        if self.checkpoint_directory:
            if not os.path.exists(self.checkpoint_directory):
                if self.verbose > 1: self.logger.info(f"Creating checkpoint directory: {self.checkpoint_directory}")
                os.makedirs(self.checkpoint_directory)

            serialized_checkpoint_filepath = os.path.join(self.checkpoint_directory,  f"{self.name}.Optuna.{self.__class__.__name__}.pkl")

            if os.path.exists(serialized_checkpoint_filepath):
                if self.verbose > 1: self.logger.info(f"[Loading] Checkpoint file: {serialized_checkpoint_filepath}")
                study = joblib.load(serialized_checkpoint_filepath)
            else:
                if self.verbose > 1: self.logger.info(f"[Creating] Checkpoint file: {serialized_checkpoint_filepath}")

        if study is None:
            study = optuna.create_study(**study_params)

        if self.initial_params:
            if self.verbose > 1: self.logger.info(f"Adding initial parameters to study: {self.initial_params}")
            study.enqueue_trial(self.initial_params, user_attrs={"memo": "initial_params"}, skip_if_exists=True)

        # Optimize
        callback_fn = stop_when_exceeding_trials(self.n_trials, self.logger)

        study.optimize(
            _objective, 
            n_trials=self.n_trials, 
            n_jobs=self.n_concurrent_trials,
            timeout=self.study_timeout, 
            show_progress_bar=self.verbose >= 2, 
            callbacks=self.study_callbacks + [callback_fn], 
            gc_after_trial=True,
        )

        return study

    def fit(
        self,
        X:pd.DataFrame,
        sampler=None,
        copy="auto",
        **study_kws,
        ):

        # Check inputs
        if self.initial_distance_metric == "precomputed":
            if len(X.shape) == 1:
                X = squareform(X)
            if not is_square_symmetric(X):
                raise ValueError("If X is precomputed, it must be square and symmetric")
            distance_matrix = X
        else:
            if self.initial_distance_metric == "jaccard":
                X = X.astype(bool)
            if self.verbose > 0:
                self.logger.info("[Start] Processing distance matrix")
            if self.initial_distance_metric == "euclidean":
                distance_matrix = squareform(pdist(X.values, metric=self.initial_distance_metric))
            else:
                distance_matrix = pairwise_distances(X=X.values, metric=self.initial_distance_metric, n_jobs=self.n_jobs)

        if self.verbose > 0:
            self.logger.info("[End] Processing distance matrix")

        # Store
        if copy == "auto":
            if self.initial_distance_metric == "precomputed":
                copy = False
            else:
                copy = True
        if copy:
            self.X_ = X.copy()

        # Tune
        if not self.is_tuned:
            if self.verbose > 0:
                self.logger.info("[Begin] Hyperparameter Tuning")
            self.study_ = self.tune(
                distance_matrix=distance_matrix,
                sampler=sampler, 
                **study_kws,
                )
            for k, v in self.study_.best_params.items():
                setattr(self,k,v)
            if self.verbose > 0:
                self.logger.info(f"Tuned parameters (Score={self.study_.best_value}): {self.study_.best_params}")
                self.logger.info("[End] Hyperparameter Tuning")
            self.is_tuned = True

        # Convert distance matrix to non-redundant KNN
        knn = convert_distance_matrix_to_kneighbors_matrix(distance_matrix, n_neighbors=self.n_neighbors, redundant_form=False)

        # Remove disconnected nodes and convert to similarity
        knn = knn[knn > 0]
        knn_similarity = self.distance_to_similarity(knn)
        del knn

        # Convert KNN to iGraph
        graph = enx.convert_network(knn_similarity, ig.Graph)

        # Identify leiden communities with multiple seeds
        df_communities = enx.community_detection(graph, n_iter=self.n_iter, converge_iter=self.converge_iter, n_jobs=1)

        # Identify membership co-occurrence ratios
        node_pair_membership_cooccurrences = enx.community_membership_cooccurrence(df_communities).mean(axis=1)
        del df_communities

        # Identify node pairs that have co-membership 100% of the time
        node_pairs_with_consistent_membership = set(node_pair_membership_cooccurrences[lambda x: x >= self.minimum_membership_consistency].index)
        del node_pair_membership_cooccurrences

        # Get list of clustered edges
        clustered_edgelist = enx.get_undirected_igraph_edgelist_indices(graph, node_pairs_with_consistent_membership)
        del node_pairs_with_consistent_membership

        # Get clustered graph
        self.graph_clustered_ = graph.subgraph_edges(clustered_edgelist, delete_vertices=True)
        self.labels_ = pd.Series(enx.get_undirected_igraph_connected_components(self.graph_clustered_, cluster_prefix=self.cluster_prefix))
        del graph

        # Calculate silhouette scores
        clustered_nodes = self.labels_.index
        index = clustered_nodes.map(lambda x: distance_matrix.index.get_loc(x)).values
        dist = distance_matrix.values[index,:][:,index]
        self.score_ = silhouette_score(dist, self.labels_.values, metric="precomputed", sample_size=None, random_state=None) 
        del dist
        
        self.n_observations_ = distance_matrix.shape[0]
        self.n_clusters_ = self.labels_.nunique()

        self.is_fitted = True

        return self
    
    def fit_transform(
        self,
        **kws,
        ):

        self.fit(**kws)
        return self.labels_
            
    def to_file(self, filepath):
        # stream = self.stream
        # self.stream = None
        write_pickle(self, filepath)
        # self.stream = stream
        
    @classmethod
    def from_file(cls, filepath):
        cls = read_pickle(filepath)
        return cls

    # =======
    # Built-in
    # =======
    def __repr__(self):
        pad = 4
        header = format_header(f"{self.__class__.__name__}(Name:{self.name}, ObservationType: {self.observation_type}, FeatureType: {self.feature_type})", line_character="=")

        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* initial_distance_metric: {}".format(self.initial_distance_metric),
            pad*" " + "* scoring_distance_metric: {}".format(self.scoring_distance_metric),
            pad*" " + "* cluster_prefix: {}".format(self.cluster_prefix),
            pad*" " + "* checkpoint_directory: {}".format(self.checkpoint_directory),
        ]
                                                  
        if self.is_tuned:
            fields += [
            pad*" " + "* n_neighbors: {}".format(self.n_neighbors),
            pad*" " + "* score: {}".format(self.score_),
            ]
        if self.is_fitted:
            fields += [
            pad*" " + "* n_observations: {}".format(self.n_observations_),
            pad*" " + "* n_clusters: {}".format(self.n_clusters_),
            ]  

        return "\n".join(fields)