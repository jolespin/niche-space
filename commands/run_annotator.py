import os
import sys
from pyexeggutor import (
    read_pickle,
    build_logger,
)

# Metabolic Niche Space
from nichespace.manifold import (
    EmbeddingAnnotator,
    DEFAULT_REGRESSOR, 
    DEFAULT_REGRESSOR_PARAM_SPACE,
)

logger = build_logger(stream=sys.stderr)
# Load data
# quality_label="completeness_gte90.contamination_lt5"
logger.info("Loading MNS")
quality_label="completeness_gte50.contamination_lt10"
model_name="NAL-GDB_MNS__v2025.3.3__SLC-MFC.medium"
mns = read_pickle(f"../data/manifold/v2025.3.3/{quality_label}/{model_name}.HierarchicalNicheSpace.pkl")
output_directory=f"../data/annotate/v2025.3.3/{quality_label}"
os.makedirs(output_directory, exist_ok=True)

logger.info("Annotating embeddings")
# Annotate
annotator = EmbeddingAnnotator(
    name=mns.name,
    observation_type=mns.observation_type,
    feature_type=mns.feature_type,
    embedding_type="MNS",
    estimator=DEFAULT_REGRESSOR, 
    param_space=DEFAULT_REGRESSOR_PARAM_SPACE, 
    n_trials=500, 
    n_iter=10, 
    n_concurrent_trials=1, 
    n_jobs=-1,
    verbose=3,
    checkpoint_directory=f"../data/annotate/v2025.3.3/{quality_label}/checkpoints",
)

annotator.fit(
    X=mns.X_.astype(int), 
    Y=mns.diffusion_coordinates_initial_,
    X_testing=mns.X1_.astype(int),
    Y_testing=mns.diffusion_coordinates_grouped_,
)
logger.info("Writing file")
annotator.to_file(f"../data/annotate/v2025.3.3/{quality_label}/{model_name}.EmbeddingAnnotator.pkl")
logger.info("Complete")
