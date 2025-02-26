#!/usr/bin/env python
import os
import sys
import glob
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
#import anndata as ad


from pyexeggutor import (
    build_logger,
    write_pickle,
    read_pickle,
    read_list,
    check_argument_choice,
)

from ensemble_networkx import convert_network


# Metabolic Niche Space
from metabolic_niche_space.neighbors import (
    KNeighborsLeidenClustering,
    pairwise_distances_kneighbors,
)


# Data
#quality_label="completeness_gte90.contamination_lt5"
quality_label="completeness_gte50.contamination_lt10"

output_directory=f"../data/cluster/mfc/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

# UPDATE MNS PACKAGE FIRST!

jaccard_distances = pd.read_pickle(f"../data/cluster/mfc/{quality_label}/genomic_traits_clusterani.jaccard_distance.series.pkl")
jaccard_distances = convert_network(jaccard_distances, pd.DataFrame)

n = jaccard_distances.shape[0]
n_neighbors_params = [int, int(np.log(n)), int(np.sqrt(n)/2)]
clustering = KNeighborsLeidenClustering(
    name=f"{quality_label}.MFC.FullKOfam", 
    feature_type="ko", 
    observation_type="ani-cluster", 
    class_type="MFC", 
    n_neighbors=n_neighbors_params, 
    n_trials=25, 
    checkpoint_directory="checkpoints", 
    n_jobs=-1,
)
clustering.fit(jaccard_distances)
clustering.to_file(f"../data/cluster/mfc/{quality_label}/MFC.FullKOfam.KNeighborsLeidenClustering.pkl")

    





