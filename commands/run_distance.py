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


from pyexeggutor import (
    build_logger,
    write_pickle,
    read_pickle,
    read_list,
    check_argument_choice,
)



# Metabolic Niche Space
from metabolic_niche_space.utils import fast_groupby
from metabolic_niche_space.neighbors import (
    KNeighborsLeidenClustering,
    pairwise_distances_kneighbors,
)


# Data
#quality_label="completeness_gte90.contamination_lt5"
quality_label="completeness_gte50.contamination_lt10"

output_directory=f"../data/cluster/mfc/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

genome_to_clusterani = pd.read_csv(f"../data/training/{quality_label}/genome_to_ani-cluster.tsv.gz", sep="\t", index_col=0, header=None).iloc[:,0].astype("category")
X_genomic_traits = pd.read_csv(f"../data/training/global.genomic_traits.kofam.bool-int.tsv.gz", sep="\t",index_col=0)
X_genomic_traits = X_genomic_traits.astype(pd.SparseDtype("bool", fill_value=False))
X_genomic_traits.to_pickle(f"../data/training/global.genomic_traits.kofam.bool.NUMPY-1.26.4.pkl.gz")

X_genomic_traits = X_genomic_traits.loc[genome_to_clusterani.index]
X_genomic_traits.to_pickle(f"../data/training/{quality_label}/X.pkl.gz")

X_genomic_traits_clusterani = fast_groupby(X_genomic_traits) > 0
X_genomic_traits_clusterani = X_genomic_traits_clusterani.astype(pd.SparseDtype("bool", fill_value=False))
X_genomic_traits_clusterani.to_pickle(f"../data/training/{quality_label}/X_grouped.pkl.gz")

#jaccard_distances = pairwise_distances_kneighbors(X_genomic_traits_clusterani, metric="jaccard", redundant_form=False, n_jobs=-1)
#jaccard_distances.to_pickle(f"../data/cluster/mfc/{quality_label}/genomic_traits_clusterani.jaccard_distance.series.pkl")





