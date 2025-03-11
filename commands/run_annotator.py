import os
import sys
from pyexeggutor import (
    read_pickle,
)

# Metabolic Niche Space
from nichespace.manifold import (
    EmbeddingAnnotator,
    DEFAULT_REGRESSOR, 
    DEFAULT_REGRESSOR_PARAM_SPACE,
)

# Load data
# quality_label="completeness_gte90.contamination_lt5"
quality_label="completeness_gte50.contamination_lt10"
model_name="NAL-GDB_MNS_v3.SLC-MFC.medium"
mns = read_pickle(f"../data/manifold/{quality_label}/{model_name}.HierarchicalNicheSpace.pkl")
output_directory=f"../data/annotate/{quality_label}"
os.makedirs(output_directory, exists_ok=True)

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
    checkpoint_directory=f"../data/annotate/{quality_label}/checkpoints",
)

annotator.fit(
    X=mns.X_.astype(int), 
    Y=mns.diffusion_coordinates_initial_,
    X_testing=mns.X1_.astype(int),
    Y_testing=mns.diffusion_coordinates_grouped_,
)

annotator.to_file(f"../data/annotate/{quality_label}/{model_name}.EmbeddingAnnotator.pkl")
