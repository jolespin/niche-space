import os
import sys
import glob
from collections import defaultdict
from tqdm import tqdm
import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
import anndata as ad
from pyexeggutor import (
    write_pickle,
    read_pickle,
    read_list, 
)

# from clairvoyance.utils import ( 
#     compile_parameter_space, # Can this be in Clairvoyance
# )
# from sklearn.cluster import (
#     HDBSCAN, # Not included in sklearn <1.3
# )

import matplotlib.pyplot as plt

# Metabolic Niche Space
from metabolic_niche_space.manifold import GroupedNicheSpace

# Data
df_quality = pd.read_csv("../data/quality.tsv.gz",sep="\t", index_col=0)

quality_label="completeness_gte90.contamination_lt5"
# quality_label="completeness_gte50.contamination_lt10"
output_directory=f"../data/cluster/mfc/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

genome_to_clusterani = pd.read_csv(f"../data/training/{quality_label}/y.tsv.gz", sep="\t", index_col=0, header=None).iloc[:,0].astype("category")
X_genomic_traits = pd.read_csv(f"../data/training/{quality_label}/X.tsv.gz", sep="\t", index_col=0).astype(bool)
X_genomic_traits_clusterani = pd.read_csv(f"../data/training/{quality_label}/X_grouped.tsv.gz", sep="\t", index_col=0).astype(bool)
eukaryotes = read_list(f"../data/cluster/ani/eukaryotic/{quality_label}/organisms.list", set)
prokaryotes = read_list(f"../data/cluster/ani/prokaryotic/{quality_label}/organisms.list", set)

genome_to_taxonomy = pd.read_csv("../data/taxonomy.tsv.gz", sep="\t", index_col=0).iloc[:,0]
clusterani_to_taxonomy = pd.read_csv("../data/cluster/ani/cluster-ani_to_taxonomy.tsv.gz", sep="\t", index_col=0, header=None).iloc[:,0]
df_meta_mfc__genomes = pd.read_csv(f"../data/cluster/mfc/{quality_label}/identifier_mapping.mfc.genomes.with_openai.tsv.gz", sep="\t", index_col=0)
df_meta_mfc__slc = pd.read_csv(f"../data/cluster/mfc/{quality_label}/identifier_mapping.mfc.genome_clusters.with_openai.tsv.gz", sep="\t", index_col=0)

X_genomic_traits_mfc = X_genomic_traits_clusterani.groupby(df_meta_mfc__slc["id_cluster-mfc"]).sum() > 0
df_kegg = pd.read_csv("/home/ec2-user/SageMaker/s3/newatlantis-raw-veba-db-prod/VDB_v8.1/Annotate/KOfam/kegg-ortholog_metadata.tsv", sep="\t", index_col=0)
ko_to_description = df_kegg["definition"]

print("Number of genomes: {}, Number of features: {}, Number of SLCs: {}".format(*X_genomic_traits.shape, X_genomic_traits_clusterani.shape[0]))
# Number of genomes: 20377, Number of features: 2124, Number of SLCs: 6719

# CPU times: user 3.41 s, sys: 87.8 ms, total: 3.5 s
# Wall time: 3.5 s

X = X_genomic_traits
print(X.shape)
y1 = genome_to_clusterani.loc[X.index]
y2 = df_meta_mfc__genomes["id_cluster-mfc"].loc[X.index].dropna()
y1 = y1.loc[y2.index]
X = X.loc[y2.index]
assert np.all(y1.notnull())
assert np.all(y2.notnull())
print(X.shape)

# genomes_with_mfc = df_meta_mfc__genomes.index[df_meta_mfc__genomes["id_cluster-mfc"].notnull()]
# X = X_genomic_traits.loc[genomes_with_mfc]
# y = df_meta_mfc__genomes.loc[genomes_with_mfc]["id_cluster-mfc"]
# n, m = X.shape

n, m = X.shape
model_name="NAL-GDB_MNS_v2.SLC-MFC"
# mns = GroupedNicheSpace(
#     observation_type="genome",
#     feature_type="ko",
#     class1_type="ani-cluster",
#     class2_type="mfc-cluster",
#     name=model_name,
#     n_neighbors=[int, int(np.log(n)), int(np.sqrt(n)/2)],
#     n_trials=100,
#     n_jobs=-1,
#     verbose=3,
#     checkpoint_directory="checkpoints",
# )
# mns.fit(X, y1, y2)
# mns.qualitative_transform()
# mns.to_file(mns, os.path.join("checkpoints", f"{model_name}.GroupedNicheSpace.n_trials-100.pkl"))


mns = GroupedNicheSpace(
    observation_type="genome",
    feature_type="ko",
    class1_type="ani-cluster",
    class2_type="mfc-cluster",
    name=model_name,
    n_neighbors=[int, int(np.log(n)), int(np.sqrt(n)/2)],
    n_trials=250,
    n_jobs=-1,
    verbose=3,
    checkpoint_directory="checkpoints",
)
mns.fit(X, y1, y2)
mns.qualitative_transform()
mns.to_file(mns, os.path.join("checkpoints", f"{model_name}.GroupedNicheSpace.n_trials-250.pkl"))





