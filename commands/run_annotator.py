import os
import sys
import glob
from collections import defaultdict
from tqdm import tqdm
import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
from pyexeggutor import (
    write_pickle,
    read_pickle,
)

# Metabolic Niche Space
from metabolic_niche_space.manifold import (
    HierarchicalNicheSpace,
    EmbeddingAnnotator,
    DEFAULT_REGRESSOR, 
    DEFAULT_REGRESSOR_PARAM_SPACE,
)

# Load data
quality_label="completeness_gte90.contamination_lt5"
model_name="NAL-GDB_MNS_v2.SLC-MFC"
mns = read_pickle(f"../data/training/{quality_label}/{model_name}.HierarchicalNicheSpace.pkl")

# Annotate
annotator = EmbeddingAnnotator(
    name=mns.name,
    observation_type=mns.observation_type,
    feature_type=mns.feature_type,
    embedding_type="MNS",
    estimator=DEFAULT_REGRESSOR, 
    param_space=DEFAULT_REGRESSOR_PARAM_SPACE, 
    n_trials=1000, 
    n_iter=10, 
    n_concurrent_trials=1, 
    n_jobs=-1,
    verbose=3,
)

annotator.fit(
    X=mns.X_.astype(int), 
    Y=mns.diffusion_coordinates_initial_,
    X_testing=mns.X1_.astype(int),
    Y_testing=mns.diffusion_coordinates_grouped_,
)

annotator.to_file(f"../data/training/{quality_label}/{model_name}.EmbeddingAnnotator.pkl")
