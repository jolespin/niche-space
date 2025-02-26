#!/usr/bin/env python
import os
import sys
import glob
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
import anndata as ad
from joblib import delayed, Parallel

from scipy.spatial.distance import (
    pdist,
    squareform,
)
from pyexeggutor import (
    build_logger,
    write_pickle,
    read_pickle,
    read_list,
    check_argument_choice,
)

from sklearn.metrics import (
    pairwise_distances,
    silhouette_score,
)
# from clairvoyance.utils import ( 
#     compile_parameter_space, # Can this be in Clairvoyance
# )
# from sklearn.cluster import (
#     HDBSCAN, # Not included in sklearn <1.3
# )
import ensemble_networkx as enx
import networkx as nx
import igraph as ig

import matplotlib.pyplot as plt

# Metabolic Niche Space
from metabolic_niche_space.utils import fast_groupby
from metabolic_niche_space.neighbors import (
    pairwise_distances_kneighbors,
    convert_distance_matrix_to_kneighbors_matrix,
)

import optuna
# from metabolic_niche_space.metabolic_niche_space.neighbors import 
import time
import random

# Data
quality_label="completeness_gte90.contamination_lt5"
# quality_label="completeness_gte50.contamination_lt10"
output_directory=f"../data/cluster/mfc/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

genome_to_clusterani = pd.read_csv(f"../data/training/{quality_label}/y.tsv.gz", sep="\t", index_col=0, header=None).iloc[:,0].astype("category")
X_genomic_traits = pd.read_csv(f"../data/training/{quality_label}/X.tsv.gz", sep="\t", index_col=0).astype(bool)
X_genomic_traits_clusterani = pd.read_csv(f"../data/training/{quality_label}/X_grouped.tsv.gz", sep="\t", index_col=0).astype(bool)
eukaryotes = read_list(f"../data/cluster/ani/eukaryotic/{quality_label}/organisms.list", set)
prokaryotes = read_list(f"../data/cluster/ani/prokaryotic/{quality_label}/organisms.list", set)

print("Number of genomes: {}, Number of features: {}, Number of SLCs: {}".format(*X_genomic_traits.shape, X_genomic_traits_clusterani.shape[0]))

jaccard_distances = read_pickle(f"{output_directory}/genomic_traits_clusterani.jaccard_distance.dataframe.pkl")



study_name = "MetabolicFunctionalClass Detection"
logger = build_logger(study_name)
param_history = defaultdict(set)

def objective(trial):
    n = jaccard_distances.shape[0]
    
    low = int(np.log(n))
    high = int(np.sqrt(n))

    # Parameters to tune (This section either grabs redundant trials or gets stuck in a while loop
    #while True:
    n_neighbors = trial.suggest_int("n_neighbors", low, high)
    
    # Maximum attempts to find a unique value
    max_attempts = 100
    attempts = 0
    sleep_time = random.uniform(0, 10)
    print(f"Sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)    
    while True:
        n_neighbors = trial.suggest_int("n_neighbors", low, high)
        if n_neighbors not in param_history["n_neighbors"]:
            param_history["n_neighbors"].add(n_neighbors)
            break

        # Increment attempt counter
        attempts += 1

        # Break if max attempts exceeded
        if attempts >= max_attempts:
            raise ValueError("Exceeded maximum attempts while searching for a unique n_neighbors")
    # Convert distance matrix to non-redundant KNN
    logger.info(f"[Trial {trial.number}] Convert distance matrix to non-redundant KNN (n_neighbors = {n_neighbors})")
    knn = convert_distance_matrix_to_kneighbors_matrix(jaccard_distances, n_neighbors=n_neighbors, redundant_form=False)
    
    # Remove disconnected nodes and convert to similarity
    logger.info(f"[Trial {trial.number}] Remove disconnected nodes and convert to similarity (n_neighbors = {n_neighbors})")
    knn_similarity = 1 - knn[knn > 0]

    # Convert KNN to iGraph
    logger.info(f"[Trial {trial.number}] Convert KNN to iGraph (n_neighbors = {n_neighbors})")
    graph = enx.convert_network(knn_similarity, ig.Graph)

    # Identify leiden communities with multiple seeds
    logger.info(f"[Trial {trial.number}] Identify leiden communities with multiple seeds (n_neighbors = {n_neighbors})")
    progressbar_message = f"[Trial {trial.number}] Community detection (n_neighbors = {n_neighbors})"
    df_communities = enx.community_detection(graph, n_iter=100, converge_iter=25, n_jobs=-1, progressbar_message=progressbar_message)
    
    # Identify membership co-occurrence ratios
    logger.info(f"[Trial {trial.number}] Identify membership co-occurrence ratios (n_neighbors = {n_neighbors})")
    node_pair_membership_cooccurrences = enx.community_membership_cooccurrence(df_communities).mean(axis=1)
    
    # Identify node pairs that have co-membership 100% of the time
    node_pairs_with_consistent_membership = set(node_pair_membership_cooccurrences[lambda x: x == 1.0].index)

    # Get list of clustered edges
    clustered_edgelist = enx.get_undirected_igraph_edgelist_indices(graph, node_pairs_with_consistent_membership)
    
    # Get clustered graph
    graph_clustered = graph.subgraph_edges(clustered_edgelist, delete_vertices=True)
    
    # Clusters
    node_to_cluster = pd.Series(enx.get_undirected_igraph_connected_components(graph_clustered))
    
    
    # Calculate silhouette scores
    clustered_nodes = node_to_cluster.index
    index = clustered_nodes.map(lambda x: jaccard_distances.index.get_loc(x)).values
    logger.info(f"[Trial {trial.number}] Calculate silhouette scores using {len(index)} nodes (n_neighbors = {n_neighbors})")
    dist = jaccard_distances.values[index,:][:,index]
    score = silhouette_score(dist, node_to_cluster.values, metric="precomputed", sample_size=None, random_state=None) 

    return score




sampler = optuna.samplers.TPESampler(seed=0)

# Study
study_kws=dict()
study_params = {
    "direction":"maximize", 
    "study_name":study_name, 
    "sampler":sampler, 
    **study_kws,
}


study = optuna.create_study(**study_params)

# Optimize
# To optimize resource usage, calculate the total memory needed for each job, then calculate how long it takes for each job
# If my machine has 128GB of memory, each job takes 10GB of memory, 5 minutes to run and I want at least 6 trials then I would do:
# * 25 n_jobs, 6*25 = 150 trials, which I would expect to take 5*6 = 30 minutes
study.optimize(
    objective, 
    n_trials=32, # This should be dependent on time
    n_jobs=32, # This should maximize the amount of resources that are used
    # timeout=self.study_timeout, 
    show_progress_bar=True, 
    # callbacks=self.study_callbacks + [_stop_when_exceeding_trials], 
    gc_after_trial=True,
)
# 31257 ec2-user  20   0   74.5g  69.2g 130228 S 100.7 37.5  87:12.02 python                                                                                                                             

write_pickle(study, "mfc.optuna.study.pkl")
