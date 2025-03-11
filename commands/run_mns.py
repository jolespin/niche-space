import os
import sys
import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
from pyexeggutor import (
    read_pickle,
)


# Metabolic Niche Space
from nichespace.manifold import HierarchicalNicheSpace

#quality_label="completeness_gte90.contamination_lt5"
quality_label="completeness_gte50.contamination_lt10"
output_directory=f"../data/manifold/v2025.3.3/{quality_label}/"
os.makedirs(output_directory, exist_ok=True)

genome_to_clusterani = pd.read_csv(f"../data/training/v2025.3.3/{quality_label}/y.tsv.gz", sep="\t", index_col=0, header=None).iloc[:,0].astype("category")
X = pd.read_parquet(f"../data/training/v2025.3.3/{quality_label}/X.parquet")

clusterer = read_pickle(f"../data/cluster/mfc/v2025.3.3/{quality_label}/MFC.FullKOfam.KNeighborsLeidenClustering.pkl")

clusterani_to_mfc = clusterer.labels_
genome_to_clustermfc = genome_to_clusterani.map(lambda x: clusterani_to_mfc.get(x, pd.NA))

print(X.shape)
y1 = genome_to_clusterani.loc[X.index]
y2 = genome_to_clustermfc.loc[X.index].dropna()
y1 = y1.loc[y2.index]
X = X.loc[y2.index].astype(float)
assert np.all(y1.notnull())
assert np.all(y2.notnull())
print(X.shape)
#distance_matrix = pd.read_parquet("../data/training/v2025.3.3/completeness_gte50.contamination_lt10/X1.minimum_nfeatures-100.jaccard_distance.redundant.parquet")

# Trial 20 finished with value: 0.6272154597719747 and parameters: {'n_components': 47, 'alpha': 0.01855074648175971}. Best is trial 20 with value: 0.6272154597719747.
n, m = X.shape
mns = HierarchicalNicheSpace(
    observation_type="genome",
    feature_type="ko",
    class1_type="ani-cluster",
    class2_type="mfc-cluster",
    name="NAL-GDB_MNS__v2025.3.3__SLC-MFC.medium",
    #n_neighbors=[int, int(np.log(n)), int(np.sqrt(n)/2)],
    n_neighbors=19,
    n_components=47,
    alpha=0.01855074648175971,
    #n_components=[int,40,80],
    n_trials=21,
    n_jobs=-1,
    verbose=3,
    checkpoint_directory=f"{output_directory}/checkpoints",
    cast_as_float=True,
    #parallel_kws={"require":"sharedmem"},
)
mns.fit(X, y1, y2) #, distance_matrix=distance_matrix)
mns.to_file(f"{output_directory}/{mns.name}.HierarchicalNicheSpace.pkl")

