#!/usr/bin/env python
import os
import sys
# import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
# import anndata as ad

from pyexeggutor import build_logger
from ensemble_networkx import write_parquet_nonredundant_pairwise_matrix



# Niche Space
from nichespace.utils import fast_groupby
from nichespace.neighbors import (
    pairwise_distances_kneighbors,
)


# Data
#quality_label="completeness_gte90.contamination_lt5"
quality_label="completeness_gte50.contamination_lt10"

output_directory=f"../data/cluster/mfc/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

logger = build_logger(stream=sys.stderr)
logger.info("Loading X_genomic_traits")
filepath = f"../data/training/v2025.3.3/{quality_label}/X.parquet"
if not os.path.exists(filepath):
    genome_to_clusterani = pd.read_csv(f"../data/training/v2025.3.3/{quality_label}/genome_to_ani-cluster.tsv.gz", sep="\t", index_col=0).iloc[:,0].astype("category")
    X_genomic_traits = pd.read_parquet(f"../data/training/v2025.3.3/{quality_label}/global.genomic_traits.kofam.bool.parquet")

    X_genomic_traits = X_genomic_traits.loc[genome_to_clusterani.index]
    X_genomic_traits.to_parquet(filepath, index=True)
else:
    X_genomic_traits = pd.read_parquet(filepath)
logger.info("X_genomic_traits: n_observations {}, n_features {}".format(*X_genomic_traits.shape))

logger.info("Grouping X_genomic_traits by cluster-ani")
filepath = f"../data/training/v2025.3.3/{quality_label}/X1.parquet"
if not os.path.exists(filepath):
    X_genomic_traits_clusterani = fast_groupby(X_genomic_traits, genome_to_clusterani) > 0
    # X_genomic_traits_clusterani = X_genomic_traits_clusterani.astype(pd.SparseDtype("bool", fill_value=False))
    X_genomic_traits_clusterani.to_parquet(filepath, index=True)
else:
    X_genomic_traits_clusterani = pd.read_parquet(filepath)
logger.info("X_genomic_traits_clusterani: n_observations {}, n_features {}".format(*X_genomic_traits_clusterani.shape))
    
logger.info("Removing X_genomic_traits from memory")
del X_genomic_traits

logger.info("Computing pairwise distance")
jaccard_distances = pairwise_distances_kneighbors(X_genomic_traits_clusterani, metric="jaccard", redundant_form=False, n_jobs=-1)

logger.info("Writing parquet")
write_parquet_nonredundant_pairwise_matrix(jaccard_distances, f"../data/cluster/mfc/v2025.3.3/{quality_label}/genomic_traits_clusterani.jaccard_distance.nonredundant.parquet")





