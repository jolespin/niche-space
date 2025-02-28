# Niche Space
The `nichespace` package is developed for computing quantitative hierarchical niche spaces and qualitative niche spaces for visualization. 
This package also includes graph theoretical clustering and embedding annotations used bayesian AutoML methods.

## Install
```
# PyPI
pip install niche-space

# Conda
conda install -c new-atlantis-labs niche-space
```

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
import pandas as pd
from sklearn.model_selection import train_test_split
from nichespace.neighbors import (
    pairwise_distances_kneighbors,
    KNeighborsLeidenClustering,
)

# Load real boolean data and create toy dataset
n_observations = 500

X_grouped = pd.read_csv("../test/X_grouped.tsv.gz", sep="\t", index_col=0) > 0
X_toy = X_grouped.iloc[:n_observations]

# Precompute pairwise distances
metric="jaccard"

distances = pairwise_distances_kneighbors(
    X=X_toy, 
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
# 2025-02-26 20:45:41,152 - jaccard_similarity_clustering - INFO - [End] Processing distance matrix
# 2025-02-26 20:45:41,153 - jaccard_similarity_clustering - INFO - [Begin] Hyperparameter Tuning
# [I 2025-02-26 20:45:41,155] A new study created in memory with name: jaccard_similarity_clustering
# [I 2025-02-26 20:45:43,368] Trial 0 finished with value: 0.18090088460183 and parameters: {'n_neighbors': 9}. Best is trial 0 with value: 0.18090088460183.
# [I 2025-02-26 20:45:44,103] Trial 1 finished with value: 0.17937516227113132 and parameters: {'n_neighbors': 10}. Best is trial 0 with value: 0.18090088460183.
# [I 2025-02-26 20:45:44,823] Trial 2 finished with value: 0.18090088460183 and parameters: {'n_neighbors': 9}. Best is trial 0 with value: 0.18090088460183.
# [I 2025-02-26 20:45:45,540] Trial 3 finished with value: 0.18090088460183 and parameters: {'n_neighbors': 9}. Best is trial 0 with value: 0.18090088460183.
# [I 2025-02-26 20:45:46,245] Trial 4 finished with value: 0.18437101801937045 and parameters: {'n_neighbors': 8}. Best is trial 4 with value: 0.18437101801937045.
# 2025-02-26 20:45:46,294 - jaccard_similarity_clustering - WARNING - [Callback] Stopping optimization: 5 trials reached (limit=5)
# 2025-02-26 20:45:46,294 - jaccard_similarity_clustering - INFO - Tuned parameters (Score=0.18437101801937045): {'n_neighbors': 8}
# 2025-02-26 20:45:46,294 - jaccard_similarity_clustering - INFO - [End] Hyperparameter Tuning
# Community detection: 100%|??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 10/10 [00:00<00:00, 42.96it/s]
# =============================================================================================================
# KNeighborsLeidenClustering(Name:jaccard_similarity_clustering, ObservationType: ani-cluster, FeatureType: ko)
# =============================================================================================================
#     * initial_distance_metric: precomputed
#     * scoring_distance_metric: euclidean
#     * cluster_prefix: c
#     * checkpoint_directory: None
#     * n_neighbors: 8
#     * score: 0.18437101801937045
#     * n_observations: 500
#     * n_clusters: 15

clustering.labels_
# NAL-ESLC_45c547b7c7d0fb1d2eb1ccbed3ee1ff7     c1
# NAL-ESLC_452ebe6d3c8f19f9598110bd9be48002     c1
# NAL-ESLC_484b2f94d8a6c24e94b3f1cb87c98147     c1
# NAL-ESLC_00045683b92daedb002220d52bbda9fd     c1
# NAL-ESLC_38e4d757b66995adc0018d329c3b136a     c1
#                                             ...
# NAL-ESLC_11aa3949879b1f3668ae3a190e92bd3d    c14
# NAL-ESLC_17822fab9e078f3444d17fe2e1945785    c14
# NAL-ESLC_4877918b070ebaea1dea30d43c253603    c14
# NAL-ESLC_45af5e14e9bfc2dc2f34ee92488968d5    c15
# NAL-ESLC_3b8193c12b57ccc88cf6e64fe97da102    c15
# Length: 498, dtype: object

# Write to file
clustering.to_file("../test/objects/KNeighborsLeidenClustering.pkl")

# Load from file
clustering = KNeighborsLeidenClustering.from_file("../test/objects/KNeighborsLeidenClustering.pkl")
```



## Manifold Learning

### Diffusion Maps using Euclidean distance with a custom KNN kernel on continuous data
There are several methods used for building diffusion map embeddings from continuous data.  However, fewer methods exist
for computing diffusion maps from boolean data where Euclidean distance is not appropriate.  Further, most methods that allow
for non-Euclidean distances often do not support out-of-sample transformations.  This package contains a custom `KNeighborsKernel`
that allows for KNN kernels with a wide range of distances that can be used with `datafold.dynfold.DiffusionMaps` which is aliased
as `DiffusionMapEmbedding` in this package.

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

### Diffusion Maps using Jaccard distance with a custom KNN kernel on boolean data

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

### Boolean data and Jaccard distance to build a hierarchical niche space
Instead of building a niche space using a single class for otpimization we are going to build a hierarchical niche space.  
In this example, we have genomes and KEGG orthologs with presence/absence as our feature matrix.  We've clustered the genomes 
using [skani](https://github.com/bluenote-1577/skani) based on 95% ANI and 50% alignment fraction (see [VEBA] for more information) 
to produce the `cluster-ani` class.  The feature matrix was aggregated with respect to `cluster-ani` and then clustered using 
`KNeighborsLeidenClustering` to yield `cluster-mfc` classes.  The hierarchical niche space algorithm implements the following strategy:
1. Aggregate `X` with respect to `y1` to yield `X1`
2. Compute pairwise distance of `X1`
3. Build custom KNN kernel from the pairwise distance matrix to yield `distance_matrix`
4. Bayesian hyperparameter optimization
    * Build a Diffusion Map from the custom KNN kernel
    * Transform out-of-samples `X` with the Diffision Map computed with `X1`
    * Compute silhouette scores for `X` using the `y2` labels
5. Repeat 4 until optimal hyperparameters are identified

This strategy allows for building small kernels from the `training data` (i.e., `X1`), which are faster to transform out-of-sample data, 
while also learning the hierarchical patterns in the `testing data` (i.e., `X`) with respect to `y2`.  In this example, we are building 
`Metabolic Functional Class (MFC)` categories clustering `Species-Level Clusters (SLC)` by using set and graph theoretical approaches.
The `HierarchicalNicheSpace` builds a diffusion space that represents both SLCs and genomes while learning pre-computed clustering patterns.

The steady-state vector is properly accounted for in both the `NicheSpace` and `HierarchicalNicheSpace` classes. 

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
# Grouping rows by: 100%|??????????????????????????????| 2124/2124 [00:00<00:00, 5243.35 column/s]
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
# [Trial 0] Projecting initial data into diffusion space: 100%|??????????????????????????????| 991/991 [00:04<00:00, 230.01it/s]
# 2025-02-26 21:07:59,395 - test - INFO - [Trial 0] Calculating silhouette score: n_neighbors=11, n_components=75, alpha=0.6027633760716439

#   0%|          | 0/2 [00:04<?, ?it/s]
# [I 2025-02-26 21:07:59,410] Trial 0 finished with value: 0.05362523711445728 and parameters: {'n_neighbors': 11, 'n_components': 75, 'alpha': 0.6027633760716439}. Best is trial 0 with value: 0.05362523711445728.
# Best trial: 0. Best value: 0.0536252:  50%|???????????????     | 1/2 [00:04<00:04,  4.88s/it]
# 2025-02-26 21:07:59,746 - test - INFO - [Trial 1] Fitting Diffision Map: n_neighbors=11, n_components=48, alpha=0.6458941130666561
# 2025-02-26 21:07:59,851 - test - INFO - [Trial 1] Transforming observations: n_neighbors=11, n_components=48, alpha=0.6458941130666561
# [Trial 1] Projecting initial data into diffusion space: 100%|??????????????????????????????| 991/991 [00:03<00:00, 254.23it/s]
# 2025-02-26 21:08:03,821 - test - INFO - [Trial 1] Calculating silhouette score: n_neighbors=11, n_components=48, alpha=0.6458941130666561

# Best trial: 0. Best value: 0.0536252:  50%|???????????????     | 1/2 [00:08<00:04,  4.88s/it]
# [I 2025-02-26 21:08:03,843] Trial 1 finished with value: 0.030161153756859602 and parameters: {'n_neighbors': 11, 'n_components': 48, 'alpha': 0.6458941130666561}. Best is trial 0 with value: 0.05362523711445728.
# 2025-02-26 21:08:04,183 - test - WARNING - [Callback] Stopping optimization: 2 trials reached (limit=2)
# Best trial: 0. Best value: 0.0536252: 100%|??????????????????????????????| 2/2 [00:09<00:00,  4.66s/it]
# 2025-02-26 21:08:04,186 - test - INFO - Tuned parameters (Score=0.05362523711445728): {'n_neighbors': 11, 'n_components': 75, 'alpha': 0.6027633760716439}
# 2025-02-26 21:08:04,187 - test - INFO - [End] Hyperparameter Tuning

# [Parallel Transformation] Grouped data: 100%|??????????????????????????????| 769/769 [00:03<00:00, 253.23it/s]
# [Parallel Transformation] Initial data: 100%|??????????????????????????????| 991/991 [00:03<00:00, 249.25it/s]
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


### Building a qualitative space from a niche space
We can't visualize greater than 3 dimensions and there will likely be more than 3 diffusion dimensions.  To
visualize, we embed the concatenated diffusion coordinates (both `X` and `X1` referred to as `X_basis`) with [PaCMAP](https://github.com/YingfanWang/PaCMAP)
and perform hyperparameter tuning to learn the class structure (in this example, MFC patterns).

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

### Annotating niches
Now that we have niche space embeddings, we need to assign some type of human interpretable meaning to the
embeddings in the form of annotations.  Since Diffusion Maps do not have loadings like Principal Component Analysis
we have designed an AutoML method called [`Clairvoyance`](https://github.com/jolespin/clairvoyance) which is used under the hood 
to simultaneously optimize hyperparameters and select features.  In this case, KEGG ortholog biomarkers that are predictive of the
embedding.  We use the `X` transformed diffusion coordinates as the training data and `X1` transformed diffusion coordinates as 
testing data.

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
# Running bayesian AutoML to identify relevant features: 100%|??????????????????????????????| 5/5 [00:15<00:00,  3.03s/it]
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
In this particular example, we only use the KEGG ortholog pathway subset for features so we can perform KEGG module enrichment 
and coverage analysis using our [KEGG Pathway Profiler](https://github.com/jolespin/kegg_pathway_profiler).

Since it's not a dependency, please install separately to use this workflow.
```
pip install kegg_pathway_profiler
```

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

### Accessing OpenAI API for prompt-based annotations
The `EmbeddingAnnotator` is a data-driven feature selection approach to annotations.  However, we can also leverage
LLMs with a properly configured prompt. You'll need to install the [`OpenAPI`](https://github.com/openai/openai-python) package and optionally [dotenv](https://github.com/theskumar/python-dotenv) to use this functionality. 

```
pip install openai
pip install dotenv
```
You can put your OpenAI credentials in `~/.openai`:

```
$cat path/to/.openai

api_key=------------
organization_key=---
project_key=--------
```

Here is a simple prompt about annotating a cluster enriched in Nostoc genomes and nitrogen fixation proteins:

```python
from dotenv import dotenv_values
from nichespace.llm import LLMAnnotator

# Load credentials
config = dotenv_values("~/.openai")

# Setup client
llm = LLMAnnotator(
    api_key=config["api_key"],
    organization_key=config["organization_key"],
    project_key=config["project_key"],
)

# Setup prompt
proteins = [
 ('K02588', 'nitrogenase iron protein NifH'),
 ('K02586', 'nitrogenase molybdenum-iron protein alpha chain [EC:1.18.6.1]'),
 ('K02591', 'nitrogenase molybdenum-iron protein beta chain [EC:1.18.6.1]'),
 ('K00531', 'nitrogenase delta subunit [EC:1.18.6.1]'),
 ('K22896', 'vanadium-dependent nitrogenase alpha chain [EC:1.18.6.2]'),
 ('K22897', 'vanadium-dependent nitrogenase beta chain [EC:1.18.6.2]'),
 ('K22898', 'vanadium nitrogenase delta subunit [EC:1.18.6.2]'),
 ('K22899', 'vanadium nitrogenase iron protein')]
organisms = ["d__Bacteria; p__Cyanobacteriota; c__Cyanophyceae; o__Nostocales; f__Nostocaceae; g__Nostoc"]

# Ask prompt
llm.query(prompt=f"Can you descibe the metabolic and environmental context of a group of organisms" \
                 f"enriched in [{organisms}] genomes and [{proteins}] proteins?" \
                 f"Please make the response concise and to 100 words.",
          model="o3-mini",
)
# Nostoc, a filamentous cyanobacterium from the Nostocaceae family, thrives in diverse aquatic and terrestrial ecosystems, 
# often where nitrogen is limited. Its enrichment in nitrogenase proteins—including both molybdenum-iron and vanadium-dependent 
# enzymes—indicates a robust capacity for atmospheric nitrogen fixation. This metabolic trait is crucial in converting inert N₂ 
# into biologically available forms, supporting both self-growth and symbiotic relationships with plants and other organisms. 
# By performing nitrogen fixation under microaerobic or anoxic conditions within its usually oxygen-rich habitat, Nostoc plays a 
# vital role in ecosystem nutrient cycling and soil fertility, contributing significantly to primary productivity in nutrient-poor 
# environments.

# Ask another prompt
# llm.query(prompt=prompt2)

# Write results
llm.to_json("llm_annotations.json")
```
