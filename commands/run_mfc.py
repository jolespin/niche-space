#!/usr/bin/env python
import sys
import os
import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
#import anndata as ad

from ensemble_networkx import convert_network


# Metabolic Niche Space
from nichespace.neighbors import (
    KNeighborsLeidenClustering,
    pairwise_distances_kneighbors,
)


# Data
#quality_label="completeness_gte90.contamination_lt5"
quality_label="completeness_gte50.contamination_lt10"

output_directory=f"../data/cluster/mfc/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

#jaccard_distances = pd.read_pickle(f"../data/cluster/mfc/{quality_label}/genomic_traits_clusterani.jaccard_distance.dataframe.pkl")

jaccard_distances = pd.read_pickle(f"../data/cluster/mfc/{quality_label}/genomic_traits_clusterani.jaccard_distance.series.pkl")
jaccard_distances = convert_network(jaccard_distances, pd.DataFrame)

n = jaccard_distances.shape[0]
n_neighbors_params = [int, int(np.log(n)), int(np.sqrt(n)/2)]
clusterer = KNeighborsLeidenClustering(
    name=f"MFC.FullKOfam", 
    feature_type="ko", 
    observation_type="ani-cluster", 
    class_type="MFC", 
    n_neighbors=n_neighbors_params, 
    n_trials=25, 
    n_jobs=-1,
    n_iter=100, 
    cluster_prefix="MFC-",
        
)
clusterer.fit(jaccard_distances)
clusterer.to_file(f"../data/cluster/mfc/{quality_label}/{clusterer.name}.KNeighborsLeidenClustering.pkl")    





