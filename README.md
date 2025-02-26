# Niche Space
The `nichespace` package is developed for computing quantitative hierarchical niche spaces and qualitative niche spaces for visualization. 
This package also includes graph theoretical clustering and embedding annotations used bayesian AutoML methods.

## Bayesian Hyperparameter Optimization
`Optuna` is used under the hood with the Tree-structured Parzen Estimator algorithm to leverage Guasissian Mixture Models.  To access the hyperparameter optimization, 
`compile_parameter_space` and `check_parameter_space` are loaded from `Clairvoyance` (whose AutoML is used by the `EmbeddingAnnotator` class) to provide user-friendly
access to `Optuna`.  

```python
n_neighbors = [int, 10, 100]
```

In the backend, will generate a [suggest_int](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int) suggestor used during optimization:

```python
n_neighbors = suggest_int("n_neighbors", 10, 100, *, step=1, log=False)
```

You can provide additional arguments as follows:

```python
learning_rate = [float, 1e-10, 1e2, {"log":True}]
```

In the backend, will generate a [suggest_float](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float) suggestor used during optimization:

```python
learning_rate = suggest_int("learning_rate", 1e-10, 1e2, log=True)
```

## Graph theoretical clustering with multi-seed Leiden community detection and KNN kernels
Graph-theoretical approaches are robust and versatile to custom distances.  With Leiden community detection, the user can
provide a single random seed that is used for stochastic processes in the backend.  The approach implemented in this package
used `EnsembleNetworkX` in the backend to compute multiple random seeds and finds the node-pairs with consistent cluster
membership.  Since the number of edges scales quadratically with the number of nodes in fully-connected networks (e.g., 
pairwise Jaccard similarity), we trim lower strength connections using `convert_distance_matrix_to_kneighbors_matrix` and 
use `Optuna` for selecting the optimal number of neighbors in the backend.  Since our boolean matrix is sparse we want to explore smaller 
numbers of neighbors than a dense dataset.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from nichespace.neighbors import (
    KNeighborsLeidenClustering,
)

# Load real boolean data and create toy dataset
n_observations = 500

X_grouped = pd.read_csv("../test/X_grouped.tsv.gz", sep="\t", index_col=0) > 0
X_toy = X_grouped.iloc[:n_observations]
X_training, X_testing = train_test_split(X_toy, test_size=0.3)

# Precompute pairwise distances
metric="jaccard"

distances = pairwise_distances_kneighbors(
    X=X_training, 
    metric=metric, 
    n_jobs=-1, 
    redundant_form=True,
)

# Determine parameter range
n = distances.shape[0]
n_neighbors_params = [int, int(np.log(n)), int(np.sqrt(n)/2)]

# Bayesian-optimized KNN Leiden Clustering
clustering = KNeighborsLeidenClustering(
    name="jaccard_similarity_clustering", 
    feature_type="ko", 
    observation_type="ani-cluster", 
    class_type="LeidenCluster", 
    n_neighbors=n_neighbors_params, 
    n_trials=5, 
    n_jobs=-1,
)
clustering.fit(distances)
# 2025-02-26 20:32:48,057 - jaccard_similarity_clustering - INFO - [End] Processing distance matrix
# 2025-02-26 20:32:48,058 - jaccard_similarity_clustering - INFO - [Begin] Hyperparameter Tuning
# [I 2025-02-26 20:32:48,059] A new study created in memory with name: jaccard_similarity_clustering
# [I 2025-02-26 20:32:48,586] Trial 0 finished with value: 0.11658078543607725 and parameters: {'n_neighbors': 7}. Best is trial 0 with value: 0.11658078543607725.
# [I 2025-02-26 20:32:49,269] Trial 1 finished with value: 0.13545759217183717 and parameters: {'n_neighbors': 8}. Best is trial 1 with value: 0.13545759217183717.
# [I 2025-02-26 20:32:49,955] Trial 2 finished with value: 0.13545759217183717 and parameters: {'n_neighbors': 8}. Best is trial 1 with value: 0.13545759217183717.
# [I 2025-02-26 20:32:50,614] Trial 3 finished with value: 0.11658078543607725 and parameters: {'n_neighbors': 7}. Best is trial 1 with value: 0.13545759217183717.
# [I 2025-02-26 20:32:51,279] Trial 4 finished with value: 0.11658078543607725 and parameters: {'n_neighbors': 7}. Best is trial 1 with value: 0.13545759217183717.
# 2025-02-26 20:32:51,407 - jaccard_similarity_clustering - WARNING - [Callback] Stopping optimization: 5 trials reached (limit=5)
# 2025-02-26 20:32:51,408 - jaccard_similarity_clustering - INFO - Tuned parameters (Score=0.13545759217183717): {'n_neighbors': 8}
# 2025-02-26 20:32:51,408 - jaccard_similarity_clustering - INFO - [End] Hyperparameter Tuning
# Community detection: 100%|██████████| 10/10 [00:00<00:00, 53.60it/s]
# =============================================================================================================
# KNeighborsLeidenClustering(Name:jaccard_similarity_clustering, ObservationType: ani-cluster, FeatureType: ko)
# =============================================================================================================
#     * initial_distance_metric: precomputed
#     * scoring_distance_metric: euclidean
#     * cluster_prefix: c
#     * checkpoint_directory: None
#     * n_neighbors: 8
#     * score: 0.13545759217183717
#     * n_observations: 350
#     * n_clusters: 11

clustering.labels_
# NAL-ESLC_277166ac6b2cd5e8acc3eee765fd8677     c1
# NAL-ESLC_2232466fa2308ebd534fb9a40bd2a62c     c1
# NAL-ESLC_269738373fbd2a4b6ba2cc2d8edfac17     c1
# NAL-ESLC_0af894e4be82ebacad9e3e44b57d2aa1     c1
# NAL-ESLC_36a7c2631f0227735806d5cab775349b     c1
#                                             ... 
# NAL-ESLC_19b2e8331e0a335bd53957c94ecdde35     c9
# NAL-ESLC_45af5e14e9bfc2dc2f34ee92488968d5    c10
# NAL-ESLC_3b8193c12b57ccc88cf6e64fe97da102    c10
# NAL-ESLC_3d9316ab5dc99a7fe4ac23e157cb75ee    c11
# NAL-ESLC_3cf6c883ec9a7f3be1cf5dd391b0d9b0    c11
# Length: 349, dtype: object

# Write to file
clustering.to_file("../test/objects/KNeighborsLeidenClustering.pkl")

# Load from file
clustering = KNeighborsLeidenClustering.from_file("../test/objects/KNeighborsLeidenClustering.pkl")
```



## Manifold Learning
### Using a custom KNN kernel to build a Diffusion Map embedding
There are several methods used for building diffusion map embeddings from continuous data.  However, fewer methods exist
for computing diffusion maps from boolean data where Euclidean distance is not appropriate.  Further, most methods that allow
for non-Euclidean distances often do not support out-of-sample transformations.  This package contains a custom `KNeighborsKernel`
that allows for KNN kernels with a wide range of distances that can be used with `datafold.dynfold.DiffusionMaps` which is aliased
as `DiffusionMapEmbedding` in this package.

##### Basic Euclidean distance
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from nichespace.neighbors import KNeighborsKernel
from nichespace.manifold import DiffusionMapEmbedding # Shortcut: from datafold.dynfold import DiffusionMaps

# Create dataset
n_samples = 1000
n_neighbors=int(np.sqrt(n_samples))
X, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2, n_clusters_per_class=1, random_state=0)
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.3)

# Build KNeighbors Kernel
kernel = KNeighborsKernel(metric="euclidean", n_neighbors=30)

# Calculate Diffusion Maps using KNeighbors
model = DiffusionMapEmbedding(
    kernel=kernel, 
    n_eigenpairs=int(np.sqrt(X_training.shape[0])), # Upper bound
) 
dmap_X = model.fit_transform(X_training)
dmap_Y = model.transform(X_testing)

# Shapes
print(dmap_X.shape, dmap_Y.shape)
# (700, 26) (300, 26)
```

##### Boolean data and Jaccard distance

```python
from sklearn.model_selection import train_test_split
from nichespace.neighbors import (
    KNeighborsKernel,
    pairwise_distances_kneighbors,
)
from nichespace.manifold import DiffusionMapEmbedding # Shortcut: from datafold.dynfold import DiffusionMaps

# Load real boolean data and create toy dataset
n_observations = 500

X_grouped = pd.read_csv("../test/X_grouped.tsv.gz", sep="\t", index_col=0) > 0
X_toy = X_grouped.iloc[:n_observations]
X_training, X_testing = train_test_split(X_toy, test_size=0.3)

# Precompute pairwise distances
metric="jaccard"

distances = pairwise_distances_kneighbors(
    X=X_training, 
    metric=metric, 
    n_jobs=-1, 
    redundant_form=True,
)

# Build KNeighbors Kernel with precomputed distances
kernel = KNeighborsKernel(
    metric=metric, 
    n_neighbors=50, 
    distance_matrix=distances.values,
)

# Calculate Diffusion Maps using KNeighbors
model = DiffusionMapEmbedding(kernel=kernel, n_eigenpairs=int(np.log(X_training.shape[0]))) # Lower bound since it's sparse
dmap_X = model.fit_transform(X_training)
dmap_Y = model.transform(X_testing)

# Shapes
print(dmap_X.shape, dmap_Y.shape)
# ((350, 5), (150, 5))
```

##### Boolean data and Jaccard distance to build a hierarchical niche space

```python
import numpy as np
import pandas as pd
from nichespace.manifold import HierarchicalNicheSpace

# Load real boolean data
n = 1000 # Just use a few samples for the test
X = pd.read_csv("../test/X.tsv.gz", sep="\t", index_col=0) > 0
Y = pd.read_csv("../test/Y.tsv.gz", sep="\t", index_col=0)
X = X.iloc[:n]
Y = Y.iloc[:n]
y1 = Y["id_cluster-ani"]
y2 = Y["id_cluster-mfc"]

n, m = X.shape
hns = HierarchicalNicheSpace(
    observation_type="genome",
    feature_type="ko",
    class1_type="ani-cluster",
    class2_type="mfc-cluster",
    name="test",
    n_neighbors=[int, int(np.log(n)), int(np.sqrt(n)/2)],
    n_trials=2,
    n_jobs=-1,
    verbose=3,
)
hns.fit(X, y1, y2)
# Grouping rows by: 100%|██████████| 2124/2124 [00:00<00:00, 5243.35 column/s]
# 2025-02-26 21:07:53,967 - test - INFO - [Start] Filtering observations and classes below feature threshold: 100
# 2025-02-26 21:07:53,974 - test - INFO - [Dropping] N = 6 y1 classes
# 2025-02-26 21:07:53,975 - test - INFO - [Dropping] N = 9 observations
# 2025-02-26 21:07:53,977 - test - INFO - [Remaining] N = 769 y1 classes
# 2025-02-26 21:07:53,978 - test - INFO - [Remaining] N = 48 y2 classes
# 2025-02-26 21:07:53,978 - test - INFO - [Remaining] N = 991 observations
# 2025-02-26 21:07:53,980 - test - INFO - [Remaining] N = 2124 features
# 2025-02-26 21:07:53,980 - test - INFO - [End] Filtering observations and classes below feature threshold
# 2025-02-26 21:07:53,986 - test - INFO - [Start] Processing distance matrix

# 2025-02-26 21:07:54,858 - test - INFO - [End] Processing distance matrix
# 2025-02-26 21:07:54,862 - test - INFO - [Begin] Hyperparameter Tuning
# [I 2025-02-26 21:07:54,864] A new study created in memory with name: test
#   0%|          | 0/2 [00:00<?, ?it/s]
# 2025-02-26 21:07:54,867 - test - INFO - [Trial 0] Fitting Diffision Map: n_neighbors=11, n_components=75, alpha=0.6027633760716439
# 2025-02-26 21:07:55,000 - test - INFO - [Trial 0] Transforming observations: n_neighbors=11, n_components=75, alpha=0.6027633760716439
# [Trial 0] Projecting initial data into diffusion space: 100%|██████████| 991/991 [00:04<00:00, 230.01it/s]
# 2025-02-26 21:07:59,395 - test - INFO - [Trial 0] Calculating silhouette score: n_neighbors=11, n_components=75, alpha=0.6027633760716439

#   0%|          | 0/2 [00:04<?, ?it/s]
# [I 2025-02-26 21:07:59,410] Trial 0 finished with value: 0.05362523711445728 and parameters: {'n_neighbors': 11, 'n_components': 75, 'alpha': 0.6027633760716439}. Best is trial 0 with value: 0.05362523711445728.
# Best trial: 0. Best value: 0.0536252:  50%|█████     | 1/2 [00:04<00:04,  4.88s/it]
# 2025-02-26 21:07:59,746 - test - INFO - [Trial 1] Fitting Diffision Map: n_neighbors=11, n_components=48, alpha=0.6458941130666561
# 2025-02-26 21:07:59,851 - test - INFO - [Trial 1] Transforming observations: n_neighbors=11, n_components=48, alpha=0.6458941130666561
# [Trial 1] Projecting initial data into diffusion space: 100%|██████████| 991/991 [00:03<00:00, 254.23it/s]
# 2025-02-26 21:08:03,821 - test - INFO - [Trial 1] Calculating silhouette score: n_neighbors=11, n_components=48, alpha=0.6458941130666561

# Best trial: 0. Best value: 0.0536252:  50%|█████     | 1/2 [00:08<00:04,  4.88s/it]
# [I 2025-02-26 21:08:03,843] Trial 1 finished with value: 0.030161153756859602 and parameters: {'n_neighbors': 11, 'n_components': 48, 'alpha': 0.6458941130666561}. Best is trial 0 with value: 0.05362523711445728.
# 2025-02-26 21:08:04,183 - test - WARNING - [Callback] Stopping optimization: 2 trials reached (limit=2)
# Best trial: 0. Best value: 0.0536252: 100%|██████████| 2/2 [00:09<00:00,  4.66s/it]
# 2025-02-26 21:08:04,186 - test - INFO - Tuned parameters (Score=0.05362523711445728): {'n_neighbors': 11, 'n_components': 75, 'alpha': 0.6027633760716439}
# 2025-02-26 21:08:04,187 - test - INFO - [End] Hyperparameter Tuning

# [Parallel Transformation] Grouped data: 100%|██████████| 769/769 [00:03<00:00, 253.23it/s]
# [Parallel Transformation] Initial data: 100%|██████████| 991/991 [00:03<00:00, 249.25it/s]
# 2025-02-26 21:08:11,519 - test - INFO - Scaling embeddings by steady-state vector
# 2025-02-26 21:08:11,521 - test - INFO - Calculating silhouette score for initial data
# =============================================================================================================================
# HierarchicalNicheSpace(Name:test, ObservationType: genome, FeatureType: ko, Class1Type: ani-cluster, Class2Type: mfc-cluster)
# =============================================================================================================================
#     * kernel_distance_metric: jaccard
#     * scoring_distance_metric: euclidean
#     * niche_prefix: n
#     * checkpoint_directory: None
#     * n_neighbors: 11
#     * n_components: 75
#     * alpha: 0.6027633760716439
#     * score: 0.030982796217236735

# Save to disk
hns.to_file("../test/objects/HierarchicalNicheSpace.pkl")

# Load from disk
hns = HierarchicalNicheSpace.from_file("../test/objects/HierarchicalNicheSpace.pkl")
```


##### Building a qualitative space from a niche space

```python
from nichespace.manifold import QualitativeSpace

X_basis, y_basis = hns.get_basis()

qualitative_hns = QualitativeSpace(
    observation_type="genome",
    feature_type="ko",
    class_type="mfc-cluster",
    name=hns.name,
    n_trials=3,
    verbose=0,
    n_neighbors=hns.n_neighbors, 
)
qualitative_hns.fit(X_basis, y_basis)
# [I 2025-02-26 21:13:17,124] A new study created in memory with name: test
# Warning: random state is set to 0.
# [I 2025-02-26 21:13:21,918] Trial 0 finished with value: 0.3292269706726074 and parameters: {'MN_ratio': 0.5488135039273248, 'FP_ratio': 4, 'lr': 0.9081174303467494}. Best is trial 0 with value: 0.3292269706726074.
# Warning: random state is set to 0.
# [I 2025-02-26 21:13:26,384] Trial 1 finished with value: 0.3461191952228546 and parameters: {'MN_ratio': 0.5448831829968969, 'FP_ratio': 3, 'lr': 0.9723822284693177}. Best is trial 1 with value: 0.3461191952228546.
# Warning: random state is set to 0.
# [I 2025-02-26 21:13:32,108] Trial 2 finished with value: 0.30728310346603394 and parameters: {'MN_ratio': 0.4375872112626925, 'FP_ratio': 5, 'lr': 1.4458575131465337}. Best is trial 1 with value: 0.3461191952228546.
# Warning: random state is set to 0.
# =============================================================================================
# QualitativeSpace(Name:test, ObservationType: genome, FeatureType: ko, ClassType: mfc-cluster)
# =============================================================================================
#     * scoring_distance_metric: euclidean
#     * checkpoint_directory: None
#     * n_neighbors: 43
#     * n_components: 3
#     * MN_ratio: 0.5448831829968969
#     * FP_ratio: 3
#     * score: 0.3461191952228546

# Save to disk
qualitative_hns.to_file("../test/objects/QualitativeSpace.pkl")

# Load from disk
qualitative_hns = QualitativeSpace.from_file("../test/objects/QualitativeSpace.pkl")
```

#### Annotating niches

```python
from nichespace.manifold import (
    EmbeddingAnnotator,
    DEFAULT_REGRESSOR, 
    DEFAULT_REGRESSOR_PARAM_SPACE,
)

m = 5 # Just annotate a few dimensions for the test
annotator = EmbeddingAnnotator(
    name=hns.name,
    observation_type=hns.observation_type,
    feature_type=hns.feature_type,
    embedding_type="HNS",
    estimator=DEFAULT_REGRESSOR,
    param_space=DEFAULT_REGRESSOR_PARAM_SPACE,
    n_trials=3,
    n_iter=2,
    n_concurrent_trials=1, # Not ready for > 1
    n_jobs=-1,
    verbose=0,
)

annotator.fit(
    X=hns.X_.astype(int), 
    Y=hns.diffusion_coordinates_initial_.iloc[:,:m],
    X_testing=hns.X1_.astype(int),
    Y_testing=hns.diffusion_coordinates_grouped_.iloc[:,:m],
)
# Running bayesian AutoML to identify relevant features:   0%|          | 0/5 [00:00<?, ?it/s][I 2025-02-26 21:38:15,381] A new study created in memory with name: n1|n_iter=1
# /home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/clairvoyance/feature_selection.py:811: UserWarning: remove_zero_weighted_features=True and removed 1615/1628 features
#   warnings.warn("remove_zero_weighted_features=True and removed {}/{} features".format((n_features_initial - n_features_after_zero_removal), n_features_initial))
# Synopsis[n1|n_iter=1] Input Features: 1628, Selected Features: 5
# Initial Training Score: -0.19783516568223325, Feature Selected Training Score: -0.06881325145571217
# Initial Testing Score: -0.23928107781121563, Feature Selected Testing Score: -0.08284976073989164
# ...
# home/ec2-user/SageMaker/environments/mns/lib/python3.9/site-packages/clairvoyance/feature_selection.py:811: UserWarning: remove_zero_weighted_features=True and removed 4/6 features
#   warnings.warn("remove_zero_weighted_features=True and removed {}/{} features".format((n_features_initial - n_features_after_zero_removal), n_features_initial))
# Running bayesian AutoML to identify relevant features: 100%|██████████| 5/5 [00:15<00:00,  3.03s/it]
# Synopsis[n5|n_iter=2] Input Features: 6, Selected Features: 2
# Initial Training Score: -0.9351419239311268, Feature Selected Training Score: -0.9132294157735279
# Initial Testing Score: -0.5985545420867351, Feature Selected Testing Score: -0.981147886984271
# ===========================================================================================
# EmbeddingAnnotator(Name:test, ObservationType: genome, FeatureType: ko, EmbeddingType: HNS)
# ===========================================================================================
#     * estimator: DecisionTreeRegressor(random_state=0)
#     * param_space: {'criterion': ['categorical', ['squared_error', 'friedman_mse']], 'min_samples_leaf': [<class 'int'>, 2, 50], 'min_samples_split': [<class 'float'>, 0.0, 0.5], 'max_features': ['categorical', ['sqrt', 'log2']], 'max_depth': ['int', 5, 50], 'min_impurity_decrease': [<class 'float'>, 1e-05, 0.01, {'log': True}], 'ccp_alpha': [<class 'float'>, 1e-05, 0.01, {'log': True}]}
#     * scorer: None
#     * n_iter: 2
#     * n_trials: 3
#     * transformation: None
#     * [X] m_features = 2124
#     * [X] n_observations = 991
#     * [Y] p_embeddings = 5
#     * [X_testing] n_observations = 769
#     * [AutoML] successful = 5
#     * [AutoML] failed = 0
    
# Save to disk
annotator.to_file("../test/objects/EmbeddingAnnotator.pkl")

# Load from disk
annotator = EmbeddingAnnotator.from_file("../test/objects/EmbeddingAnnotator.pkl")
```

### Pathway coverage and enrichment from predictive features

```python
import pandas as pd
from pyexeggutor import read_pickle # Could also use pd.read_pickle
from kegg_pathway_profiler.pathways import (
    pathway_coverage_wrapper,
)
from kegg_pathway_profiler.enrichment import (
    unweighted_pathway_enrichment_wrapper,
)

# Load KEGG module database
kegg_pathway_database = read_pickle("path/to/KEGG-Pathway-Profiler/database.pkl.gz")

# Calculate KEGG module completion ratios
data = pathway_coverage_wrapper(
    evaluation_kos=set(annotator.selected_features_["n1"]), # Annotate just the first niche
    database=kegg_pathway_database,
)
df_kegg_coverage = pd.DataFrame(data).T.sort_values("coverage", ascending=False)

# Calculate KEGG module enrichment
df_kegg_enrichment = unweighted_pathway_enrichment_wrapper(
    evaluation_kos=set(annotator.selected_features_["1"]), 
    database=kegg_pathway_database,
    background_set=set(mns.X_.columns),
).sort_values("FDR")
```