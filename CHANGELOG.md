#### Daily Change Log:

* [2025.2.24] - Added robust method for pickling `QualitativeSpace` objects (e.g., `annoy_index.save('index_file.ann')`)
* [2025.2.24] - Added checkpoints to `EmbeddingAnnotator` if a checkpoint directory is provided
* [2025.2.21] - Added `EmbeddingAnnotator` to `manifold` module
* [2025.2.18] - Added `PaCMAP` into `QualitativeSpace` class and removed from `HierarchicalNicheSpace` and `NicheSpace`
* [2025.2.17] - Added `KNeighborsLeidenClustering` class to `neighbors` module
* [2025.2.17 ] - Added `initial_params` to `study.enqueue_trial(params)` to `optuna` objects for starting points 
* [2025.2.12] - Added `HierarchicalNicheSpace` to `manifold` module
* [2025.2.6] - Added `NicheSpace` to `manifold` module
* [2024.8.13] - Added `enrichment`, `utils`, and `fetch` modules
* [2024.5.30] - Added `distance_matrix` option for precomputed distances in `KNeighborsKernel`.

#### Pending: 
* Port backend to use `anndata` objects for distance matrices and counts tables
* Develop a dynamic n_neighbors scheme for `convert_distance_matrix_to_kneighbors_matrix`
* Auto-handle sparse distances by checking sparsity and then making csr_matrix 
