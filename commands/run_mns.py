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
output_directory=f"../data/cluster/mfc/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

genome_to_clusterani = pd.read_csv(f"../data/training/{quality_label}/y.tsv.gz", sep="\t", index_col=0, header=None).iloc[:,0].astype("category")
X_genomic_traits = pd.read_csv(f"../data/training/{quality_label}/X.tsv.gz", sep="\t", index_col=0).astype(bool)

# genome_to_taxonomy = pd.read_csv("../data/taxonomy.tsv.gz", sep="\t", index_col=0).iloc[:,0]
# clusterani_to_taxonomy = pd.read_csv("../data/cluster/ani/cluster-ani_to_taxonomy.tsv.gz", sep="\t", index_col=0, header=None).iloc[:,0]

clusterer = read_pickle(f"../data/cluster/mfc/{quality_label}/MFC.FullKOfam.KNeighborsLeidenClustering.pkl")

clusterani_to_mfc = clusterer.labels_
genome_to_clustermfc = genome_to_clusterani.map(lambda x: clusterani_to_mfc.get(x, pd.NA))

X = X_genomic_traits
print(X.shape)
y1 = genome_to_clusterani.loc[X.index]
y2 = genome_to_clustermfc.loc[X.index].dropna()
y1 = y1.loc[y2.index]
X = X.loc[y2.index]
assert np.all(y1.notnull())
assert np.all(y2.notnull())
print(X.shape)


n, m = X.shape
mns = HierarchicalNicheSpace(
    observation_type="genome",
    feature_type="ko",
    class1_type="ani-cluster",
    class2_type="mfc-cluster",
    name="NAL-GDB_MNS_v3.SLC-MFC.medium",
    n_neighbors=[int, int(np.log(n)), int(np.sqrt(n)/2)],
    n_trials=100,
    n_jobs=-1,
    verbose=3,
    checkpoint_directory=f"../data/training/{quality_label}/checkpoints",
)
mns.fit(X, y1, y2)
mns.to_file(mns, f"../data/training/{quality_label}/{mns.name}.HierarchicalNicheSpace.pkl")

