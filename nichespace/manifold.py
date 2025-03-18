import os
import sys
import warnings
import uuid
from collections import (
    defaultdict,
    OrderedDict,
)
from typing import Optional, Union
from tqdm import tqdm
import numpy as np # Can't install NumPy 2.2.2 which is what the pkls were saved with
import pandas as pd # 'v2.2.3'
# import anndata as ad

import optuna
import joblib
from pacmap import PaCMAP
from annoy import AnnoyIndex

from scipy.spatial.distance import (
    pdist, 
    squareform,
)

from sklearn.metrics import (
    pairwise_distances,
    silhouette_score, 
)

from sklearn.tree import DecisionTreeRegressor
# --------------------------------------------------
from sklearn.utils.validation import check_is_fitted
from datafold.pcfold import TSCDataFrame
# --------------------------------------------------
from datafold.dynfold import (
    DiffusionMaps, 
    Roseland,
)

from pyexeggutor import (
    build_logger,
    write_pickle,
    read_pickle,
    format_header,
)

from clairvoyance.utils import (
    compile_parameter_space,
    check_parameter_space,
)
from clairvoyance.bayesian import (
    BayesianClairvoyanceRegression,
)

from .neighbors import KNeighborsKernel
from .utils import (
    fast_groupby,
    stop_when_exceeding_trials,
)

# ========================================================
# Defaults
# ========================================================
DEFAULT_REGRESSOR_PARAM_SPACE = {
    "criterion":["categorical", ["squared_error", "friedman_mse"]],
    "min_samples_leaf":[int, 2, 50], 
    "min_samples_split": [float, 0.0, 0.5], 
    "max_features":["categorical", ["sqrt", "log2"]],
    "max_depth":["int", 5, 50], 
    "min_impurity_decrease": [float, 1e-5, 1e-2, {"log":True}],  # Pruning for tree regularization
    "ccp_alpha": [float, 1e-5, 1e-2, {"log":True}],  
}

DEFAULT_REGRESSOR = DecisionTreeRegressor(random_state=0)

# ========================================================
# Classes 
# ========================================================

class DiffusionMapEmbedding(DiffusionMaps):
    """
    DiffusionMapEmbedding is a renamed version of the DiffusionMaps class from the datafold.dynfold package.
    It inherits all methods and properties of the original class without any modifications.
    
    Documentation: 
        https://datafold-dev.gitlab.io/datafold/api/datafold.dynfold.DiffusionMaps.html
    Citation: 
        Lehmberg et al., (2020). datafold: data-driven models for point clouds and time series on manifolds. 
        Journal of Open Source Software, 5(51), 2283, https://doi.org/10.21105/joss.02283
    """
    pass

class LandmarkDiffusionMapEmbedding(Roseland):
    """
    LandmarkDiffusionMapEmbedding is a renamed version of the Roseland class from the datafold.dynfold package.
    It inherits all methods and properties of the original class without any modifications.
    
    Documentation: 
        https://datafold-dev.gitlab.io/datafold/api/datafold.dynfold.Roseland.html
    Citation: 
        Lehmberg et al., (2020). datafold: data-driven models for point clouds and time series on manifolds. 
        Journal of Open Source Software, 5(51), 2283, https://doi.org/10.21105/joss.02283
    """
    pass


class NicheSpace(object):
    """
    # Usage:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,        # Number of samples
        n_features=100,        # Number of boolean features
        n_informative=50,      # Number of informative features
        n_redundant=25,        # Number of redundant features
        n_repeated=0,          # No repeated features
        n_classes=3,           # Number of classes
        n_clusters_per_class=1, # One cluster per class
        weights=[0.33, 0.33, 0.34],  # Balanced class distribution
        random_state=42        # For reproducibility
    )

    # Convert features to boolean (0 or 1) using a threshold at 0
    X_boolean = (X > 0).astype(int)

    # Create DataFrame
    df = pd.DataFrame(X_boolean, columns=[f'Feature_{i+1}' for i in range(100)])
    df['Class'] = y  # Add class labels

    y = df.pop("Class")
    X = df > 0

    model = NicheSpace(n_trials=3, minimum_nfeatures=1)
    model.fit(X,y)

    """
    def __init__(
        self, 
        # General
        name:str=None,
        observation_type:str=None,
        feature_type:str=None,
        class_type:str=None,
        minimum_nfeatures:int=100,

        # Diffusion Maps
        kernel_distance_metric:str="jaccard",
        # scoring_method:str="silhouette", # or IICR
        scoring_distance_metric:str="euclidean",
        n_neighbors=[int, 10, 100],
        n_components=[int, 10, 100], # n_eigenpairs in DataFold. First diffusion map vector is steady-state so 1 is automatically added to any n_components value
        alpha=[float, 0.0, 1.0],
        scale_by_steadystate:bool=True,
        niche_prefix="n",

        # Optuna
        n_trials=50,
        n_jobs:int=1,
        n_concurrent_trials:int=1,
        initial_params:dict=None,
        objective_direction="maximize",
        checkpoint_directory=None,
        study_timeout=None,
        study_callbacks=None,
        random_state=0,
        verbose=1,
        stream=sys.stdout,
        ):
        
        warnings.warn("NicheSpace has not had the performance updates that HierarchicalNicheSpace has recieved")
        
        # General
        if name is None:
            name = str(uuid.uuid4())
            
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.class_type = class_type
        self.minimum_nfeatures = minimum_nfeatures
        
        # Diffusion Maps
        self.kernel_distance_metric = kernel_distance_metric
        # self.scoring_method = scoring_method
        self.scoring_distance_metric = scoring_distance_metric
        self.scale_by_steadystate = scale_by_steadystate
        self.niche_prefix = niche_prefix
        
        # Optuna
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.n_concurrent_trials = n_concurrent_trials
        self.initial_params = initial_params
        self.checkpoint_directory = checkpoint_directory
        self.random_state = random_state
        self.study_timeout = study_timeout
        if study_callbacks is None:
            study_callbacks = []
        self.study_callbacks = study_callbacks
        self.objective_direction = objective_direction

        # Hyperparameters
        self.is_tuned = True

        if isinstance(n_neighbors, list):
            self.is_tuned = False
        self.n_neighbors = n_neighbors
        
        if isinstance(n_components, list):
            self.is_tuned = False
        self.n_components = n_components
        
        if isinstance(alpha, list):
            self.is_tuned = False
        self.alpha = alpha
        
        self.param_space = check_parameter_space(dict(
            n_neighbors = self.n_neighbors,
            n_components = self.n_components,
            alpha = self.alpha,
        ))
        
        self.logger = build_logger(self.name, stream=stream)
        self.verbose = verbose
        self.is_fitted = False
        
    def tune(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        distance_matrix:np.array,
        sampler, 
        **study_kws,
        ):

        def _objective(trial):
            try:

                # Compile parameters
                params = compile_parameter_space(
                    trial, 
                    self.param_space,
                )

                # Parameters
                n_neighbors = params["n_neighbors"]
                n_components = params["n_components"]
                alpha = params["alpha"]

                if n_neighbors >= X.shape[0]:
                    return -1 #np.nan
                else:
                    # Build kernel
                    kernel = KNeighborsKernel( 
                        metric=self.kernel_distance_metric, 
                        n_neighbors=n_neighbors, 
                        distance_matrix=distance_matrix, 
                        copy_distance_matrix=False,
                    )

                    # Calculate Diffusion Maps using KNeighbors
                    model = DiffusionMaps(kernel=kernel, n_eigenpairs=n_components+1, alpha=alpha)

                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Fitting Diffision Map: n_neighbors={n_neighbors}, n_components={n_components}, alpha={alpha}")
                    dmap = model.fit_transform(X)

                    # Score
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Calculating silhouette score: n_neighbors={n_neighbors}, n_components={n_components}, alpha={alpha}")
                    score = silhouette_score(dmap[:,1:], y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=None) # Ignore steady state vector

                    return score

            except Exception as e:
                self.logger.error(f"[Trial {trial.number}] Failed due to error: {e}. Marking as pruned.")
                raise optuna.TrialPruned()  # Prevents skipping trials

            finally:
                if self.checkpoint_directory:
                    joblib.dump(study, os.path.join(self.checkpoint_directory, f"{self.name}.Optuna.{self.__class__.__name__}.pkl"))  # Save checkpoint

        # Sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        
        # Study
        study_params = {
            "direction":self.objective_direction, 
            "study_name":self.name, 
            "sampler":sampler, 
            **study_kws,
        }
        
        # Checkpoints
        study = None
        if self.checkpoint_directory:
            if not os.path.exists(self.checkpoint_directory):
                if self.verbose > 1: self.logger.info(f"Creating checkpoint directory: {self.checkpoint_directory}")
                os.makedirs(self.checkpoint_directory)
        
            serialized_checkpoint_filepath = os.path.join(self.checkpoint_directory, f"{self.name}.Optuna.{self.__class__.__name__}.pkl")

            if os.path.exists(serialized_checkpoint_filepath):
                if self.verbose > 1: self.logger.info(f"[Loading] Checkpoint file: {serialized_checkpoint_filepath}")
                study = joblib.load(serialized_checkpoint_filepath)
            else:
                if self.verbose > 1: self.logger.info(f"[Creating] Checkpoint file: {serialized_checkpoint_filepath}")

        if study is None:
            study = optuna.create_study(**study_params)


        if self.initial_params:
            if self.verbose > 1: self.logger.info(f"Adding initial parameters to study: {self.initial_params}")
            study.enqueue_trial(self.initial_params, user_attrs={"memo": "initial_params"}, skip_if_exists=True)
            
        # Optimize
        callback_fn = stop_when_exceeding_trials(self.n_trials, self.logger)
        study.optimize(
            _objective, 
            n_trials=self.n_trials, 
            n_jobs=self.n_concurrent_trials,
            timeout=self.study_timeout, 
            show_progress_bar=self.verbose >= 2, 
            callbacks=self.study_callbacks + [stop_when_exceeding_trials], 
            gc_after_trial=True,
        )

        return study


    def fit(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        distance_matrix:np.array=None,
        sampler=None,
        copy=True,
        **study_kws,
        ):

        # Check inputs
        if not np.all(X.shape[0] == y.size):
            raise IndexError("X.shape[0] must equal y.size")
        if not np.all(X.index == y.index):
            raise IndexError("X.index must equal y.index")
        if not isinstance(y, pd.CategoricalDtype):
            y = y.astype("category")
        self.X_ = X.copy()
        self.y_ = y.copy()

            
        # Minimum number of features
        if self.minimum_nfeatures > 0:
            if self.verbose > 0:
                self.logger.info(f"[Start] Filtering observations and classes below feature threshold: {self.minimum_nfeatures}")

            number_of_features_per_observation = (X > 0).sum(axis=1)
            observations_passed_qc = number_of_features_per_observation.index[number_of_features_per_observation >= self.minimum_nfeatures]

            y = y.loc[observations_passed_qc]
            X = X.loc[observations_passed_qc]
            if self.verbose > 0:
                self.logger.info(f"[Dropping] N = {sum(number_of_features_per_observation < self.minimum_nfeatures)} observations")
                self.logger.info(f"[Remaining] N = {y.unique()} classes")
                self.logger.info(f"[Remaining] N = {X.shape[0]} observations")
                self.logger.info(f"[Remaining] N = {X.shape[1]} features")
                self.logger.info(f"[End] Filtering observations and classes below feature threshold")
            
        # Dtype
        if self.kernel_distance_metric == "jaccard":
            X = X.astype(bool)
            
        # Distance matrix
        if distance_matrix is None:
            if self.verbose > 0:
                self.logger.info("[Start] Processing distance matrix")
            if self.kernel_distance_metric == "euclidean":
                distance_matrix = squareform(pdist(X.values, metric=self.kernel_distance_metric))
            else:
                distance_matrix = pairwise_distances(X=X.values, metric=self.kernel_distance_metric, n_jobs=self.n_jobs)
            
        if len(distance_matrix.shape) == 1:
            distance_matrix = squareform(distance_matrix)
        if self.verbose > 0:
            self.logger.info("[End] Processing distance matrix")

        # Store
        self.classes_ = y.cat.categories
        if copy:
            self.X_ = X.copy()
            self.y_ = y.copy()
        
        # Tune
        if not self.is_tuned:
            if self.verbose > 0:
                self.logger.info("[Begin] Hyperparameter Tuning")
            self.study_ = self.tune(
                X=X,
                y=y,
                distance_matrix=distance_matrix,
                sampler=sampler,
                **study_kws,
                )
            for k, v in self.study_.best_params.items():
                setattr(self,k,v)
            if self.verbose > 0:
                self.logger.info(f"Tuned parameters (Score={self.study_.best_value}): {self.study_.best_params}")
                self.logger.info("[End] Hyperparameter Tuning")
            self.is_tuned = True
            
        # Build kernel
        self.kernel_ = KNeighborsKernel( 
            metric=self.kernel_distance_metric, 
            n_neighbors=self.n_neighbors, 
            distance_matrix=distance_matrix, 
            copy_distance_matrix=True,
        )

        # Calculate Diffusion Maps using KNeighbors
        self.model_ = DiffusionMaps(
            kernel=self.kernel_, 
            n_eigenpairs=self.n_components+1, 
            alpha=self.alpha,
        )
        
        # Fit
        dmap = self.model_.fit(X)

        # Complete
        dmap = self._parallel_transform(X, self.model_, progressbar_message=f"[Parallel Transformation] Initial data")
        self.diffusion_coordinates_ = pd.DataFrame(dmap, index=X.index)
        self.diffusion_coordinates_.columns = [f"{self.niche_prefix}0_steady-state"] + list(map(lambda i: f"{self.niche_prefix}{i}", range(1,dmap.shape[1])))
        self.diffusion_coordinates_.index.name = self.observation_type
        self.diffusion_coordinates_.columns.name = self.feature_type

        # Scale
        if self.scale_by_steadystate:
            if self.verbose > 0: self.logger.info("Scaling embeddings by steady-state vector")
            self.diffusion_coordinates_ = self._scale_by_first_column(self.diffusion_coordinates_)
            # Score
            if self.verbose > 0: self.logger.info("Calculating silhouette score for initial data")
            self.score_ = silhouette_score(self.diffusion_coordinates_.values, y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=self.random_state)
        else:
            # Score
            if self.verbose > 0: self.logger.info("Calculating silhouette score for initial data excluding steady-state vector")
            self.score_ = silhouette_score(self.diffusion_coordinates_.values[:,1:], y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=self.random_state)
            
        self.is_fitted = True

        return self
    
    
    def transform(
        self,
        X,
        progressbar_message=None,
        ):
        if not self.is_fitted:
            raise Exception("Please run .fit to build DiffusionMap model before continuing")
        dmap = self._parallel_transform(self, X, self.model_, progressbar_message=progressbar_message)
        if isinstance(X, pd.DataFrame):
            X_dmap = pd.DataFrame(dmap, index=X.index)
            X_dmap.columns = [f"{self.niche_prefix}0_steady-state"] + list(map(lambda i: f"{self.niche_prefix}{i}", range(1,dmap.shape[1])))
            X_dmap.index.name = self.observation_type
            X_dmap.columns.name = self.feature_type
        else:
            return dmap
        
    def get_basis(self):
        if not self.is_fitted:
            raise Exception("Please run .fit to build DiffusionMap model before continuing")
        return self.diffusion_coordinates_

    @staticmethod
    def _scale_by_first_column(X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale all columns of a DataFrame (except the first one) by the first column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame where the first column serves as the divisor.

        Returns:
        --------
        pd.DataFrame
            A new DataFrame with the first column removed and the remaining columns scaled.
        """
        values = X.values  # Convert to NumPy array for efficiency
        steady_state_vector = values[:, 0].reshape(-1, 1)  # Extract first column as divisor
        scaled_values = values[:, 1:] / steady_state_vector  # Perform element-wise division

        return pd.DataFrame(
            scaled_values, 
            index=X.index, 
            columns=X.columns[1:]  # Remove first column name from new DataFrame
        )

    @staticmethod
    def _process_row(model, row):
        """Helper function to apply model.transform to a single row"""
        return model.transform(row.reshape(1, -1))

    def _parallel_transform(self, X, model, progressbar_message=None):
        """Parallelizes the transformation using joblib"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            output = joblib.Parallel(n_jobs=self.n_jobs, prefer="threads")(
                joblib.delayed(self._process_row)(model, row.values) for id, row in tqdm(X.iterrows(), desc=progressbar_message, total=X.shape[0], position=0, leave=True)
            )
            return np.vstack(output)
        
    @classmethod
    def from_file(cls, filepath):
        cls = read_pickle(filepath)
        return cls

    def to_file(self, filepath):
        write_pickle(self, filepath)

    # =======
    # Built-in
    # =======
    def __repr__(self):
        pad = 4
        header = format_header(f"{self.__class__.__name__}(Name:{self.name}, ObservationType: {self.observation_type}, FeatureType: {self.feature_type}, ClassType: {self.class_type})", line_character="=")

        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* kernel_distance_metric: {}".format(self.kernel_distance_metric),
            pad*" " + "* scoring_distance_metric: {}".format(self.scoring_distance_metric),
            pad*" " + "* niche_prefix: {}".format(self.niche_prefix),
            pad*" " + "* checkpoint_directory: {}".format(self.checkpoint_directory),
        ]
                                                  
        if self.is_tuned:
            fields += [
            pad*" " + "* n_neighbors: {}".format(self.n_neighbors),
            pad*" " + "* n_components: {}".format(self.n_components),
            pad*" " + "* alpha: {}".format(self.alpha),
            pad*" " + "* score: {}".format(self.score_),
            ]

        return "\n".join(fields)

class HierarchicalNicheSpace(object):
    def __init__(
        self, 
        # General
        name:str=None,
        observation_type:str=None,
        feature_type:str=None,
        class1_type:str=None,
        class2_type:str=None,
        minimum_nfeatures:int=100,

        # Diffusion Maps
        kernel_distance_metric:str="jaccard",
        # scoring_method:str="silhouette", # or IICR
        scoring_distance_metric:str="euclidean",
        n_neighbors=[int, 10, 100],
        n_components=[int, 10, 100], # n_eigenpairs in DataFold. First diffusion map vector is steady-state so 1 is automatically added to any n_components value
        alpha=[float, 0.0, 1.0],
        scale_by_steadystate:bool=True,
        niche_prefix="n",
        robust_transform=True,
        parallel_backend=None,
        parallel_prefer="threads",
        parallel_kws:dict=None,

        # Optuna
        n_trials=50,
        n_jobs:int=1,
        n_concurrent_trials:int=1,
        initial_params:dict=None,
        objective_direction="maximize",
        checkpoint_directory=None,
        study_timeout=None,
        study_callbacks=None,
        random_state=0,
        verbose=1,
        stream=sys.stdout,
        cast_as_float:bool=True,

        ):
        
        # General
        if name is None:
            name = str(uuid.uuid4())
            
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.class1_type = class1_type
        self.class2_type = class2_type
        self.minimum_nfeatures = minimum_nfeatures
        
        # Diffusion Maps
        self.kernel_distance_metric = kernel_distance_metric
        # self.scoring_method = scoring_method
        self.scoring_distance_metric = scoring_distance_metric
        self.scale_by_steadystate = scale_by_steadystate
        self.niche_prefix = niche_prefix
        self.robust_transform = robust_transform
        self.parallel_kws = dict(
                backend=parallel_backend,
                prefer=parallel_prefer,
        )
        if parallel_kws:
            self.parallel_kws.update(parallel_kws)
        
        # Optuna
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.n_concurrent_trials = n_concurrent_trials
        self.initial_params = initial_params
        self.checkpoint_directory = checkpoint_directory
        self.random_state = random_state
        self.study_timeout = study_timeout
        if study_callbacks is None:
            study_callbacks = []
        self.study_callbacks = study_callbacks
        self.objective_direction = objective_direction

        # Hyperparameters
        self.is_tuned = True
        if isinstance(n_neighbors, list):
            self.is_tuned = False
        self.n_neighbors = n_neighbors
        
        if isinstance(n_components, list):
            self.is_tuned = False
        self.n_components = n_components
        
        if isinstance(alpha, list):
            self.is_tuned = False
        self.alpha = alpha
        
        self.param_space = check_parameter_space(dict(
            n_neighbors = self.n_neighbors,
            n_components = self.n_components,
            alpha = self.alpha,
        ))
        
        self.logger = build_logger(self.name, stream=stream)
        self.verbose = verbose
        self.cast_as_float = cast_as_float
        self.is_fitted = False
        
    def tune(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        X1:pd.DataFrame,
        distance_matrix:np.array,
        sampler, 
        **study_kws,
        ):

        def _objective(trial):
            try:

                # Compile parameters
                params = compile_parameter_space(
                    trial, 
                    self.param_space,
                )

                # Parameters
                n_neighbors = params["n_neighbors"]
                n_components = params["n_components"]
                alpha = params["alpha"]

                if n_neighbors >= X1.shape[0]:
                    return -1 #np.nan
                else:
                    # Build kernel
                    kernel = KNeighborsKernel( 
                        metric=self.kernel_distance_metric, 
                        n_neighbors=n_neighbors, 
                        distance_matrix=distance_matrix, 
                        copy_distance_matrix=False,
                    )

                    # Calculate Diffusion Maps using KNeighbors
                    model = DiffusionMaps(kernel=kernel, n_eigenpairs=n_components+1, alpha=alpha)

                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Fitting Diffision Map: n_neighbors={n_neighbors}, n_components={n_components}, alpha={alpha}")
                    dmap_X1 = model.fit_transform(X1.values)

                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Transforming observations: n_neighbors={n_neighbors}, n_components={n_components}, alpha={alpha}")
                    # dmap_X = model.transform(X)
                    dmap_X = self._parallel_transform(X.values, model, progressbar_message=f"[Trial {trial.number}] Projecting initial data into diffusion space")

                    # Score
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Calculating silhouette score: n_neighbors={n_neighbors}, n_components={n_components}, alpha={alpha}")
                    score = silhouette_score(dmap_X[:,1:], y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=None) # Ignore steady state vector

                    return score

            except Exception as e:
                self.logger.error(f"[Trial {trial.number}] Failed due to error: {e}. Marking as pruned.")
                raise optuna.TrialPruned()  # Prevents skipping trials

            finally:
                if self.checkpoint_directory:
                    joblib.dump(study, os.path.join(self.checkpoint_directory, f"{self.name}.Optuna.{self.__class__.__name__}.pkl"))  # Save checkpoint

                
        # Sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        
        # Study
        study_params = {
            "direction":self.objective_direction, 
            "study_name":self.name, 
            "sampler":sampler, 
            **study_kws,
        }
        
        # Checkpoints
        study = None
        if self.checkpoint_directory:
            if not os.path.exists(self.checkpoint_directory):
                if self.verbose > 1: self.logger.info(f"Creating checkpoint directory: {self.checkpoint_directory}")
                os.makedirs(self.checkpoint_directory)
            # if self.verbose > 1: self.logger.info("Creating sqlite database: {}".format(os.path.join(self.checkpoint_directory, f"{self.name}.db")))
            # study_params["storage"] = "sqlite:///" + os.path.join(self.checkpoint_directory, f"{self.name}.db")
            # study_params["load_if_exists"] = True
        
            serialized_checkpoint_filepath = os.path.join(self.checkpoint_directory, f"{self.name}.Optuna.{self.__class__.__name__}.pkl")

            if os.path.exists(serialized_checkpoint_filepath):
                if self.verbose > 1: self.logger.info(f"[Loading] Checkpoint file: {serialized_checkpoint_filepath}")
                study = joblib.load(serialized_checkpoint_filepath)
            else:
                if self.verbose > 1: self.logger.info(f"[Creating] Checkpoint file: {serialized_checkpoint_filepath}")

        if study is None:
            study = optuna.create_study(**study_params)
        
        if self.initial_params:
            if self.verbose > 1: self.logger.info(f"Adding initial parameters to study: {self.initial_params}")
            study.enqueue_trial(self.initial_params, user_attrs={"memo": "initial_params"}, skip_if_exists=True)
            
        # Optimize
        callback_fn = stop_when_exceeding_trials(self.n_trials, self.logger)

        study.optimize(
            _objective, 
            n_trials=self.n_trials, 
            n_jobs=self.n_concurrent_trials,
            timeout=self.study_timeout, 
            show_progress_bar=self.verbose >= 2, 
            callbacks=self.study_callbacks + [callback_fn], 
            gc_after_trial=True,
        )

        return study


    def fit(
        self,
        X:pd.DataFrame,
        y1:pd.Series,
        y2:pd.Series=None,
        distance_matrix:np.array=None,
        sampler=None,
        copy=True,
        **study_kws,
        ):

        # Check inputs
        ys = [y1]
        if y2 is not None:
            ys.append(y2)
        for i, y in enumerate(ys, start=1):
            y_name = f"y{i}"
            if not np.all(X.shape[0] == y.size):
                raise IndexError(f"X.shape[0] must equal {y_name}.size")
            if not np.all(X.index == y.index):
                raise IndexError(f"X.index must equal {y_name}.index")

            setattr(self, f"{y_name}_", y.copy())
        self.X_ = X.copy()
        
        # Group values
        X1 = fast_groupby(X, y1, method="sum")

        if not set(X1.index) <= set(y1.unique()):
            raise IndexError("X1.index must be ≤ y1 categories")
            
        # Minimum number of features
        if self.minimum_nfeatures > 0:
            if self.verbose > 0:
                self.logger.info(f"[Start] Filtering observations and classes below feature threshold: {self.minimum_nfeatures}")

            number_of_features_per_class = (X1 > 0).sum(axis=1)
            index_classes = number_of_features_per_class.index[number_of_features_per_class > self.minimum_nfeatures]

            mask = y1.map(lambda x: x not in index_classes)
            y1 = y1.loc[~mask]
            if y2 is not None:
                y2 = y2.loc[y1.index]
            X = X.loc[y1.index]
            if self.verbose > 0:
                self.logger.info(f"[Dropping] N = {X1.shape[0] - len(index_classes)} y1 classes")
                self.logger.info(f"[Dropping] N = {sum(mask)} observations")
                self.logger.info(f"[Remaining] N = {y1.nunique()} y1 classes")
                if y2 is not None:
                    self.logger.info(f"[Remaining] N = {y2.nunique()} y2 classes")
                self.logger.info(f"[Remaining] N = {X.shape[0]} observations")
                self.logger.info(f"[Remaining] N = {X.shape[1]} features")
                self.logger.info(f"[End] Filtering observations and classes below feature threshold")
            X1 = X1.loc[index_classes]
            
        # Dtype
        if self.kernel_distance_metric == "jaccard":
            X = X.astype(bool)
            X1 = X1.astype(bool)
            
        self.observations_ = X.index
        self.observations1_ = X1.index
        self.features_ = X.columns
            
        # Distance matrix
        serialized_checkpoint_filepath = None
        if self.checkpoint_directory:
            serialized_checkpoint_filepath = os.path.join(self.checkpoint_directory, f"{self.name}.{self.__class__.__name__}.distance_matrix.parquet")
            if os.path.exists(serialized_checkpoint_filepath):
                self.logger.info(f"Loading distance matrix from checkpoint: {serialized_checkpoint_filepath}")
                distance_matrix = pd.read_parquet(serialized_checkpoint_filepath).values
                
        if distance_matrix is None:
            if self.verbose > 0:
                self.logger.info("[Start] Processing distance matrix")
            if self.kernel_distance_metric == "euclidean":
                distance_matrix = squareform(pdist(X1, metric=self.kernel_distance_metric))
            else:
                distance_matrix = pairwise_distances(X=X1.values, metric=self.kernel_distance_metric, n_jobs=self.n_jobs)
        if len(distance_matrix.shape) == 1:
            distance_matrix = squareform(distance_matrix)
        if not distance_matrix.shape[0] == X1.shape[0]:
            raise ValueError(f"distance_matrix.shape[0] ({distance_matrix.shape[0]}) does not match X1.shape[0] ({X1.shape[0]}).  This may be a result of automatic filtering.  If so, please filter before providing input or do not provide distance matrix")
        if serialized_checkpoint_filepath:
            if not os.path.exists(serialized_checkpoint_filepath):
                self.logger.info(f"Writing distance matrix checkpoint: {serialized_checkpoint_filepath}")
                pd.DataFrame(distance_matrix, index=X1.index, columns=X1.index).to_parquet(serialized_checkpoint_filepath, index=True)
        if self.verbose > 0:
            self.logger.info("[End] Processing distance matrix")
            
        # Cast as float
        if self.cast_as_float: # Decrease overhead for parallel transform
            X = X.astype(float)
            X1 = X1.astype(float)

        # Store
        if not isinstance(y1, pd.CategoricalDtype):
            y1 = y1.astype("category")
        self.classes1_ = y1.cat.categories

        if y2 is not None:
            if not isinstance(y2, pd.CategoricalDtype):
                y2 = y2.astype("category")
            self.classes2_ = y2.cat.categories

        if copy:
            self.X_ = X.copy()
            self.y1_ = y1.copy()
            if y2 is not None:
                self.y2_ = y2.copy()
            self.X1_ = X1.copy()
        
        if y2 is not None:
            y = self.y2_.copy()
        else:
            y = self.y1_.copy()
        # Tune
        if not self.is_tuned:
            if self.verbose > 0:
                self.logger.info("[Begin] Hyperparameter Tuning")

            self.study_ = self.tune(
                X=X,
                y=y,
                X1=X1,
                distance_matrix=distance_matrix,
                sampler=sampler, 
                **study_kws,
                )
            for k, v in self.study_.best_params.items():
                setattr(self,k,v)
            if self.verbose > 0:
                self.logger.info(f"Tuned parameters (Score={self.study_.best_value}): {self.study_.best_params}")
                self.logger.info("[End] Hyperparameter Tuning")
            self.is_tuned = True
            
        # Build kernel
        self.kernel_ = KNeighborsKernel( 
            metric=self.kernel_distance_metric, 
            n_neighbors=self.n_neighbors, 
            distance_matrix=distance_matrix, 
            copy_distance_matrix=True,
        )

        # Calculate Diffusion Maps using KNeighbors
        self.model_ = DiffusionMaps(kernel=self.kernel_, n_eigenpairs=self.n_components+1, alpha=self.alpha)
        
        # Fit
        dmap = self.model_.fit(X1.values)

        # Grouped
        if self.robust_transform:
            dmap = self._parallel_transform(X1, self.model_, progressbar_message=f"[Parallel Transformation] Grouped data") # More accurate to recalculate
        self.diffusion_coordinates_grouped_ = pd.DataFrame(dmap, index=X1.index)
        self.diffusion_coordinates_grouped_.columns = [f"{self.niche_prefix}0_steady-state"] + list(map(lambda i: f"{self.niche_prefix}{i}", range(1,dmap.shape[1])))
        self.diffusion_coordinates_grouped_.index.name = self.class1_type
        self.diffusion_coordinates_grouped_.columns.name = self.feature_type

        # Complete
        dmap = self._parallel_transform(X, self.model_, progressbar_message=f"[Parallel Transformation] Initial data")
        self.diffusion_coordinates_initial_ = pd.DataFrame(dmap, index=X.index)
        self.diffusion_coordinates_initial_.columns = [f"{self.niche_prefix}0_steady-state"] + list(map(lambda i: f"{self.niche_prefix}{i}", range(1,dmap.shape[1])))
        self.diffusion_coordinates_initial_.index.name = self.observation_type
        self.diffusion_coordinates_initial_.columns.name = self.feature_type

        # Scale
        if self.scale_by_steadystate:
            if self.verbose > 0: self.logger.info("Scaling embeddings by steady-state vector")
            self.diffusion_coordinates_grouped_ = self._scale_by_first_column(self.diffusion_coordinates_grouped_)
            self.diffusion_coordinates_initial_ = self._scale_by_first_column(self.diffusion_coordinates_initial_)
            # Score
            if self.verbose > 0: self.logger.info("Calculating silhouette score for initial data")
            self.score_ = silhouette_score(self.diffusion_coordinates_initial_.values, y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=self.random_state)
        else:
            # Score
            if self.verbose > 0: self.logger.info("Calculating silhouette score for initial data excluding steady-state vector")
            self.score_ = silhouette_score(self.diffusion_coordinates_initial_.values[:,1:], y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=self.random_state)
        self.is_fitted = True

        return self
    
    def transform(
        self,
        X,
        progressbar_message=None,
        ):
        if not self.is_fitted:
            raise Exception("Please run .fit to build DiffusionMap model before continuing")
            
        if X.shape[1] != len(self.features_):
            raise ValueError("Number of X features must match number of fitted features")
            
        if isinstance(X, pd.DataFrame):
            if np.any(X.columns != self.features_):
                raise ValueError("X features must match fitted features")
                
        dmap = self._parallel_transform(X, self.model_, progressbar_message=progressbar_message)
        if isinstance(X, pd.DataFrame):

            X_dmap = pd.DataFrame(dmap, index=X.index)
            X_dmap.columns = [f"{self.niche_prefix}0_steady-state"] + list(map(lambda i: f"{self.niche_prefix}{i}", range(1,dmap.shape[1])))
            X_dmap.index.name = self.observation_type
            X_dmap.columns.name = self.feature_type
            if self.scale_by_steadystate:
                X_dmap = self._scale_by_first_column(X_dmap)
            return X_dmap
        else:
            if self.scale_by_steadystate:
                dmap = self._scale_by_first_column(dmap)
            return dmap
        
        
    def get_basis(self):
        if not self.is_fitted:
            raise Exception("Please run .fit to build DiffusionMap model before continuing")
        X_basis = pd.concat([self.diffusion_coordinates_grouped_, self.diffusion_coordinates_initial_], axis=0)
        y_basis = {
            **self.y2_.to_dict(),
            **dict(zip(self.y1_.values, self.y2_.values)),
        }
        y_basis = pd.Series(y_basis).loc[X_basis.index]
        return X_basis, y_basis
    
    @staticmethod
    def _scale_by_first_column(X):
        """
        Scale all columns of a DataFrame (except the first one) by the first column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame where the first column serves as the divisor.

        Returns:
        --------
        pd.DataFrame
            A new DataFrame with the first column removed and the remaining columns scaled.
        """
        if isinstance(X, pd.DataFrame):
            values = X.values  # Convert to NumPy array for efficiency
            steady_state_vector = values[:, 0].reshape(-1, 1)  # Extract first column as divisor
            scaled_values = values[:, 1:] / steady_state_vector  # Perform element-wise division

            return pd.DataFrame(
                scaled_values, 
                index=X.index, 
                columns=X.columns[1:]  # Remove first column name from new DataFrame
            )
        else:
            values = X.copy()  
            steady_state_vector = values[:, 0].reshape(-1, 1)  # Extract first column as divisor
            scaled_values = values[:, 1:] / steady_state_vector  # Perform element-wise division
            return scaled_values

    # @staticmethod
    # def _process_row(model, row):
    #     """Helper function to apply model.transform to a single row"""
    #     return model.transform(row.reshape(1, -1))
    
    @staticmethod
    def _process_row(model, row):
        r"""Embed out-of-sample points with Nyström extension.

        From solving the eigenproblem of the diffusion kernel :math:`K`
        (:class:`.DmapKernelFixed`)

        .. math::
            K(X,X) \Psi = \Psi \Lambda

        follows the Nyström extension for out-of-sample mappings:

        .. math::
            K(X, Y) \Psi \Lambda^{-1} = \Psi

        where :math:`K(X, Y)` is a component-wise evaluation of the kernel.

        Note, that the Nyström mapping can be used for image mappings irrespective of
        whether the computed kernel matrix :math:`K(X,X)` is symmetric.
        For details on this see :cite:t:`fernandez-2015` (especially Eq. 5).

        Parameters
        ----------
        X: TSCDataFrame, pandas.DataFrame, numpy.ndarray
            Data points of shape `(n_samples, n_features)` to be embedded.

        Returns
        -------
        TSCDataFrame, pandas.DataFrame, numpy.ndarray
            same type as `X` of shape `(n_samples, n_coords)`
        """
        X = row.reshape(1,-1)
        # check_is_fitted(model, ("X_fit_", "eigenvalues_", "eigenvectors_"))

        # X = model._validate_datafold_data(X)
        # model._validate_feature_input(X, direction="transform")

        kernel_matrix_cdist = model._dmap_kernel(model.X_fit_, X)

        # choose object to copy time information from
        if isinstance(kernel_matrix_cdist, TSCDataFrame):
            # if possible take time index from kernel_matrix (especially
            # dynamics-adapted kernels can drop samples from X)
            index_from: Optional[TSCDataFrame] = kernel_matrix_cdist
        elif isinstance(X, TSCDataFrame) and kernel_matrix_cdist.shape[0] == X.shape[0]:
            # if kernel is numpy.ndarray or scipy.sparse.csr_matrix, but X_fit_ is a time
            # series, then take indices from X_fit_ -- this only works if no samples are
            # dropped in the kernel computation.
            index_from = X
        else:
            index_from = None

        eigvec, eigvals = model._select_eigenpairs_target_coords()

        eigvec_nystroem = model._nystrom(
            kernel_matrix_cdist,
            eigvec=np.asarray(eigvec),
            eigvals=eigvals,
            index_from=index_from,
        )

        return model._perform_dmap_embedding(eigvec_nystroem)

    def _parallel_transform(self, X, model, progressbar_message=None):
        """Parallelizes the transformation using joblib"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            output = joblib.Parallel(n_jobs=self.n_jobs, **self.parallel_kws)(
                joblib.delayed(self._process_row)(model, row) for row in tqdm(X, desc=progressbar_message, total=X.shape[0], position=0, leave=True)
            )
            return np.vstack(output)

    @classmethod
    def from_file(cls, filepath):
        cls = read_pickle(filepath)
        return cls

    def to_file(self, filepath):
        write_pickle(self, filepath)

    # =======
    # Built-in
    # =======
    def __repr__(self):
        pad = 4
        header = format_header(f"{self.__class__.__name__}(Name:{self.name}, ObservationType: {self.observation_type}, FeatureType: {self.feature_type}, Class1Type: {self.class1_type}, Class2Type: {self.class2_type})", line_character="=")

        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* kernel_distance_metric: {}".format(self.kernel_distance_metric),
            pad*" " + "* scoring_distance_metric: {}".format(self.scoring_distance_metric),
            pad*" " + "* niche_prefix: {}".format(self.niche_prefix),
            pad*" " + "* checkpoint_directory: {}".format(self.checkpoint_directory),
        ]
                                                  
        if self.is_tuned:
            fields += [
            pad*" " + "* n_neighbors: {}".format(self.n_neighbors),
            pad*" " + "* n_components: {}".format(self.n_components),
            pad*" " + "* alpha: {}".format(self.alpha),
            pad*" " + "* score: {}".format(self.score_),
            ]

        return "\n".join(fields)
    
class QualitativeSpace(object):
   
    def __init__(
        self, 
        # General
        name:str=None,
        observation_type:str=None,
        feature_type:str=None,
        class_type:str=None,
        
        # PaCMAP
        n_components:int=3, 
        n_neighbors = None,
        MN_ratio = [float, 0.0, 1.0],
        FP_ratio = [int, 1, 5],
        lr = [float, 0.01, 1.5],
        pacmap_distance_metric:str='euclidean',
        n_iters=(100, 100, 250),
        initializer="pca",
        scoring_distance_metric:str="euclidean",

        # Optuna
        n_trials=25,
        n_jobs:int=1,
        n_concurrent_trials:int=1,
        initial_params:dict=None,
        objective_direction="maximize",
        checkpoint_directory=None,
        study_timeout=None,
        study_callbacks=None,
        random_state=0,
        verbose=1,
        stream=sys.stdout,
        ):
        
        # General
        if name is None:
            name = str(uuid.uuid4())
            
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.class_type = class_type
        
        self.scoring_distance_metric = scoring_distance_metric
        
        # Optuna
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.n_concurrent_trials = n_concurrent_trials
        self.initial_params = initial_params
        self.checkpoint_directory = checkpoint_directory
        self.random_state = random_state
        self.study_timeout = study_timeout
        if study_callbacks is None:
            study_callbacks = []
        self.study_callbacks = study_callbacks
        self.objective_direction = objective_direction

        # Hyperparameters
        self.is_tuned = True
        self.n_components = n_components
        self.initializer = initializer
        self.random_state = random_state
        self.pacmap_distance_metric = pacmap_distance_metric
        self.n_iters = n_iters
        

        if isinstance(n_neighbors, list):
            self.is_tuned = False
        self.n_neighbors = n_neighbors
        

        if isinstance(MN_ratio, list):
            self.is_tuned = False
        self.MN_ratio = MN_ratio
        
        if isinstance(FP_ratio, list):
            self.is_tuned = False
        self.FP_ratio = FP_ratio
        
        if isinstance(lr, list):
            self.is_tuned = False
        self.lr = lr
                
        self.param_space = check_parameter_space(dict(
            MN_ratio = self.MN_ratio,
            FP_ratio = self.FP_ratio,
            n_neighbors = self.n_neighbors,
            lr = self.lr,
        ))

        self.logger = build_logger(self.name, stream=stream)
        self.verbose = verbose
        self.is_fitted = False
        
    def tune(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        sampler, 
        **study_kws,
        ):

        def _objective(trial):
            try:
                # Compile parameters

                    
                params = compile_parameter_space(
                    trial, 
                    self.param_space,
                )

                # Parameters
                n_components = self.n_components
                initializer = self.initializer
                
                n_neighbors = params["n_neighbors"]
                MN_ratio = params["MN_ratio"]
                FP_ratio = params["FP_ratio"]
                
                continue_tuning = True
                if isinstance(n_neighbors, int):
                    if n_neighbors >= X.shape[0]:
                        continue_tuning = False
                if continue_tuning:
                    # Calculate PaCMAP
                    model = PaCMAP(
                        n_components=n_components, 
                        n_neighbors=n_neighbors, 
                        MN_ratio=MN_ratio, 
                        FP_ratio=FP_ratio,
                        random_state=self.random_state,
                        distance = self.pacmap_distance_metric,
                        num_iters = self.n_iters,
                        save_tree = False,
                    ) 

                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Fitting PaCMAP: n_neighbors={n_neighbors}, n_components={n_components}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}")
                    embedding = model.fit_transform(X, init=self.initializer)

                    # Score
                    if self.verbose > 1: self.logger.info(f"[Trial {trial.number}] Calculating silhouette score:  n_neighbors={n_neighbors}, n_components={n_components}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}")
                    score = silhouette_score(embedding, y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=None) 

                    return score
                else:
                    return -1 #np.nan


            except Exception as e:
                self.logger.error(f"[Trial {trial.number}] Failed due to error: {e}. Marking as pruned.")
                raise optuna.TrialPruned()  # Prevents skipping trials

            finally:
                if self.checkpoint_directory:
                    joblib.dump(study, os.path.join(self.checkpoint_directory, f"{self.name}.Optuna.{self.__class__.__name__}.pkl"))  # Save checkpoint

        # Sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        
        # Study
        study_params = {
            "direction":self.objective_direction, 
            "study_name":self.name, 
            "sampler":sampler, 
            **study_kws,
        }
        
        # Checkpoints
        study = None
        if self.checkpoint_directory:
            if not os.path.exists(self.checkpoint_directory):
                if self.verbose > 1: self.logger.info(f"Creating checkpoint directory: {self.checkpoint_directory}")
                os.makedirs(self.checkpoint_directory)
        
            serialized_checkpoint_filepath = os.path.join(self.checkpoint_directory,  f"{self.name}.Optuna.{self.__class__.__name__}.pkl")

            if os.path.exists(serialized_checkpoint_filepath):
                if self.verbose > 1: self.logger.info(f"[Loading] Checkpoint file: {serialized_checkpoint_filepath}")
                study = joblib.load(serialized_checkpoint_filepath)
            else:
                if self.verbose > 1: self.logger.info(f"[Creating] Checkpoint file: {serialized_checkpoint_filepath}")

        if study is None:
            study = optuna.create_study(**study_params)

        if self.initial_params:
            if self.verbose > 1: self.logger.info(f"Adding initial parameters to study: {self.initial_params}")
            study.enqueue_trial(self.initial_params, user_attrs={"memo": "initial_params"}, skip_if_exists=True)
            
        # Optimize
        callback_fn = stop_when_exceeding_trials(self.n_trials, self.logger)
        study.optimize(
            _objective, 
            n_trials=self.n_trials, 
            n_jobs=self.n_concurrent_trials,
            timeout=self.study_timeout, 
            show_progress_bar=self.verbose >= 2, 
            callbacks=self.study_callbacks + [stop_when_exceeding_trials], 
            gc_after_trial=True,
        )

        return study


    def fit(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        distance_matrix:np.array=None,
        sampler=None,
        copy=True,
        **study_kws,
        ):

        # Check inputs
        if not np.all(X.shape[0] == y.size):
            raise IndexError("X.shape[0] must equal y.size")
        if not np.all(X.index == y.index):
            raise IndexError("X.index must equal y.index")
        if not isinstance(y, pd.CategoricalDtype):
            y = y.astype("category")
        self.X_ = X.copy()
        self.y_ = y.copy()
      
        # Store
        self.classes_ = y.cat.categories
        if copy:
            self.X_ = X.copy()
            self.y_ = y.copy()
        
        # Tune
        if not self.is_tuned:
            if self.verbose > 0:
                self.logger.info("[Begin] Hyperparameter Tuning")
            self.study_ = self.tune(
                X=X,
                y=y,
                sampler=sampler, 
                **study_kws,
                )
            for k, v in self.study_.best_params.items():
                setattr(self,k,v)
            if self.verbose > 0:
                self.logger.info(f"Tuned parameters (Score={self.study_.best_value}): {self.study_.best_params}")
                self.logger.info("[End] Hyperparameter Tuning")
            self.is_tuned = True
            
       # Calculate PaCMAP
        self.model_ = PaCMAP(
            n_components=self.n_components, 
            n_neighbors=self.n_neighbors, 
            MN_ratio=self.MN_ratio, 
            FP_ratio=self.FP_ratio,
            random_state=self.random_state,
            distance = self.pacmap_distance_metric,
            num_iters = self.n_iters,
            save_tree = True,
        ) 
        if self.verbose > 1: self.logger.info(f"Fitting PaCMAP")
        self.embedding_ = self.model_.fit_transform(X, init=self.initializer)
        self.embedding_ = pd.DataFrame(
            self.embedding_,
            index=X.index,
        )
        self.embedding_.columns = self.embedding_.columns.map(lambda i: f"PaCMAP-{i+1}")
        self.embedding_.index.name = self.observation_type
        self.embedding_.columns.name = self.feature_type
        
        # Save annoy tree dimension
        self.annoy_dimension_ = self.model_.tree.f
        
        # Score
        if self.verbose > 1: self.logger.info(f"Calculating silhouette score")
        self.score_ = silhouette_score(self.embedding_, y.values, metric=self.scoring_distance_metric, sample_size=None, random_state=None) 

        self.is_fitted = True

        return self
    
    
    def transform(
        self,
        X,
        initializer="pca",
        
        ):
        if not self.is_fitted:
            raise Exception("Please run .fit to build PaCMAP model before continuing")
        if X is self.X_:
            return self.embedding_
        
        X_pacmap = self.model_.transform(X, basis=self.X_, init=initializer)
        
        if isinstance(X, pd.DataFrame): 
            X_pacmap = pd.DataFrame(
                X_pacmap,
                index=X.index,
            )
            X_pacmap.columns = X_pacmap.columns.map(lambda i: f"PaCMAP-{i+1}")
            
        return X_pacmap
    
    def plot(
        self, 
        engine:str="matplotlib",
        figsize=(8,8),
        title=None,
        **kws,
        ):
        if not hasattr(self, "embedding_"):
            raise Exception("Please run .fit to compute PaCMAP embeddings before continuing")
        if engine == "matplotlib":
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D        

            """
            Plots a 3D scatter plot using matplotlib from a DataFrame with exactly three columns.

            Parameters:
            -----------
            df : pd.DataFrame
                A DataFrame with three numerical columns representing X, Y, and Z coordinates.
            title : str, optional
                Title of the plot (default is "3D Scatter Plot").
            """
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            # Extract X, Y, Z from DataFrame
            df = self.embedding_
            x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]

            if "c" not in kws:
                kws["c"] = df.index.map(lambda x: {True:"red", False:"black"}[x in self.classes_])

            # Scatter plot
            ax.scatter(x, y, z, alpha=0.618, **kws)

            # Labels and title
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.set_zlabel(df.columns[2])
            if title:
                ax.set_title(title)

            return fig, ax
        
    @classmethod
    def from_file(cls, filepath): # , stream=sys.stdout):

        cls = read_pickle(filepath)
        if hasattr(cls, "model_"):
            ann_filepath = f"{filepath}.ann"
            if not os.path.exists(ann_filepath):
                warnings.warn(f"Could not find AnnoyIndex: {ann_filepath}")
            else:
                tree = AnnoyIndex(cls.annoy_dimension_, cls.pacmap_distance_metric)
                tree.load(ann_filepath)
                cls.model_.tree = tree
        # cls.stream = stream
        return cls


    def to_file(self, filepath):
        # # Stream
        # stream =None
        # if hasattr(self, "stream"):
        #     stream = self.stream
        #     self.stream = None
        # Annoy
        tree = None
        if hasattr(self, "model_"):
            tree = self.model_.tree
            if tree is not None:
                tree.save(f"{filepath}.ann")
                self.model_.tree = None

        write_pickle(self, filepath)
        
        # if stream is not None:
        #     self.stream = stream
        if tree is not None:
            self.model_.tree = tree
        



    # =======
    # Built-in
    # =======
    def __repr__(self):
        pad = 4
        header = format_header(f"{self.__class__.__name__}(Name:{self.name}, ObservationType: {self.observation_type}, FeatureType: {self.feature_type}, ClassType: {self.class_type})", line_character="=")

        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* scoring_distance_metric: {}".format(self.scoring_distance_metric),
            pad*" " + "* checkpoint_directory: {}".format(self.checkpoint_directory),
        ]
                                                  
        if self.is_tuned:
            fields += [
            pad*" " + "* n_neighbors: {}".format(self.n_neighbors),
            pad*" " + "* n_components: {}".format(self.n_components),
            pad*" " + "* MN_ratio: {}".format(self.MN_ratio),
            pad*" " + "* FP_ratio: {}".format(self.FP_ratio),
            pad*" " + "* score: {}".format(self.score_),
            ]

        return "\n".join(fields)
    

class EmbeddingAnnotator(object):
    def __init__(
        self, 
        # Clairvoyance
        estimator=DEFAULT_REGRESSOR,
        param_space:dict=DEFAULT_REGRESSOR_PARAM_SPACE,
        scorer=None,
        n_iter=5,
        transformation=None,
        feature_selection_performance_threshold=0.0, 
        training_testing_weights = [1.0,1.0],
        remove_zero_weighted_features=True,
        maximum_tries_to_remove_zero_weighted_features=100,
        automl_kws=None,

        # Labeling
        name:str=None,
        observation_type:str=None,
        feature_type:str=None,
        embedding_type:str=None,
        
        # Optuna
        n_trials=50,
        n_concurrent_trials:int=1,
        objective_direction="maximize",
        checkpoint_directory=None,
        study_timeout=None,
        study_callbacks=None,
        
        # Utility
        early_stopping=5,
        random_state=0,
        n_jobs=1,
        verbose=0,
        save_automl=False,
        stream=sys.stdout,
        ):

        # Clairvoyance
        self.estimator=estimator
        self.param_space=param_space
        self.scorer = scorer
        self.n_iter = n_iter
        self.transformation = transformation
        self.feature_selection_performance_threshold = feature_selection_performance_threshold
        self.training_testing_weights = training_testing_weights
        self.remove_zero_weighted_features = remove_zero_weighted_features
        self.maximum_tries_to_remove_zero_weighted_features = maximum_tries_to_remove_zero_weighted_features
        if automl_kws is None:
            automl_kws = dict()
        self.automl_kws = automl_kws
        
        # Labeling
        self.name = name
        self.observation_type = observation_type
        self.feature_type = feature_type
        self.embedding_type = embedding_type
        
        # Optuna
        self.n_trials = n_trials
        self.n_concurrent_trials = n_concurrent_trials
        self.objective_direction = objective_direction
        self.checkpoint_directory = checkpoint_directory
        self.study_timeout = study_timeout
        self.study_callbacks = study_callbacks
        # self.stream = stream
        
        # Utility
        self.logger = build_logger(self.name, stream=stream)
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.save_automl = save_automl
        self.is_fitted = False

    def _run_regression_automl(
        self,
        X,
        Y,
        id_column,
        cv,
        X_testing,
        Y_testing,
        optimize_with_training_and_testing,
        **kws,
        ):
        model = None
        if self.checkpoint_directory:
            serialized_checkpoint_filepath = os.path.join(self.checkpoint_directory, f"{self.name}.BayesianClairvoyanceRegression.{id_column}.pkl")

            if os.path.exists(serialized_checkpoint_filepath):
                self.logger.info(f"[Loading] Checkpoint file: {serialized_checkpoint_filepath}")
                model = read_pickle(serialized_checkpoint_filepath)
            else:
                self.logger.info(f"[Creating] Checkpoint file: {serialized_checkpoint_filepath}")
        
        if model is None:
            y = Y[id_column]
            y_testing = None if Y_testing is None else Y_testing[id_column]

            # Fit the AutoML model
            model = BayesianClairvoyanceRegression(
                estimator=self.estimator, 
                param_space=self.param_space,  
                study_prefix=f"{id_column}|n_iter=",
                n_iter=self.n_iter, 
                n_trials=self.n_trials, 
                feature_selection_method="addition", 
                n_jobs=self.n_jobs, 
                verbose=self.verbose,
                feature_selection_performance_threshold=self.feature_selection_performance_threshold,
                transformation=self.transformation,
                training_testing_weights=self.training_testing_weights,
                remove_zero_weighted_features=self.remove_zero_weighted_features,
                maximum_tries_to_remove_zero_weighted_features=self.maximum_tries_to_remove_zero_weighted_features,
                early_stopping=self.early_stopping,
                random_state=self.random_state,
                name=id_column,
                observation_type=self.observation_type,
                feature_type=self.feature_type,
                target_type=self.embedding_type,
                copy_X=False,
                copy_y=False,
                **self.automl_kws,
            )

            # with SuppressStderr():
            #     with warnings.catch_warnings():
            #         warnings.simplefilter("ignore", FutureWarning)
            #         warnings.filterwarnings("ignore", message="is_sparse is deprecated", category=DeprecationWarning)
            try:
                model.fit(
                    X=X, 
                    y=y, 
                    cv=cv, 
                    optimize_with_training_and_testing=optimize_with_training_and_testing, 
                    X_testing=X_testing, 
                    y_testing=y_testing,
                )
                if self.checkpoint_directory:
                    serialized_checkpoint_filepath = os.path.join(self.checkpoint_directory, f"{self.name}.BayesianClairvoyanceRegression.{id_column}.pkl")
                    model.to_file(serialized_checkpoint_filepath)

            except AssertionError as e:
                self.logger.critical(f"Model[{id_column}] AutoML failed")
            
            
        return model

        
    def fit(
        self,
        X:pd.DataFrame,
        Y:pd.DataFrame,
        cv=3,
        X_testing:pd.DataFrame=None,
        Y_testing:pd.DataFrame=None,
        optimize_with_training_and_testing="auto", 
        sort_order = ["testing", "training"],
        **kws,
        ):
        
        # Checkpoints
        if self.checkpoint_directory:
            if not os.path.exists(self.checkpoint_directory):
                if self.verbose > 1: self.logger.info(f"Creating checkpoint directory: {self.checkpoint_directory}")
                os.makedirs(self.checkpoint_directory)
        
        # Copy
        self.X_ = X.copy()
        self.Y_ = Y.copy()
        if X_testing is not None:
            if Y_testing is None:
                msg = "If X_testing is provided, user must provide Y_testing"
                self.logger.error(msg)
                raise Exception(msg)
            self.X_testing_ = X_testing.copy()
            self.Y_testing_ = Y_testing.copy()
            
        progressbar_message = f"Running bayesian AutoML to identify relevant features"
        
        # This has strangebehavior throwing warnings that do not occur when running indiviudally
        # self.models_ = joblib.Parallel(n_jobs=self.n_concurrent_trials, prefer="threads")(
        #         joblib.delayed(self._run_regression_automl)(X, Y, id_column, cv, X_testing, Y_testing, optimize_with_training_and_testing, **kws) for id_column in tqdm(Y.columns, desc=progressbar_message, total=Y.shape[1]) # , position=0, leave=True)
        #     )
        if set(sort_order) != set(["testing", "training"]):
            msg = "sort_order must contain both [testing, training]"
            raise ValueError(msg)
            
        if self.save_automl:
            self.automl_models_ = dict()
        else:
            self.automl_results_ = dict()
        self.automl_status_ok_ = dict()
        self.best_iteration_ = dict()
        self.feature_weights_ = dict()
        self.selected_features_ = dict()
        self.estimators_ = dict()
        self.scores_ = defaultdict(dict)
        self.studies_ = dict()
        

        for id_column in tqdm(Y.columns, desc=progressbar_message, total=Y.shape[1], position=0, leave=True):
            model_automl = self._run_regression_automl(
                X=X, 
                Y=Y, 
                id_column=id_column, 
                cv=cv, 
                X_testing=X_testing, 
                Y_testing=Y_testing, 
                optimize_with_training_and_testing=optimize_with_training_and_testing, 
                **kws,
            )
            if self.save_automl:
                self.automl_models_[id_column] = model_automl
            else:
                self.automl_results_[id_column] = model_automl.results_

            is_ok = model_automl.is_fitted
            self.automl_status_ok_[id_column] = is_ok
            if is_ok:
                best_iteration = model_automl.results_.sort_values([f"feature_selected_{sort_order[0]}_score", f"feature_selected_{sort_order[1]}_score"], ascending=[False, False]).iloc[0]
                self.selected_features_[id_column] = best_iteration["selected_features"]
                self.feature_weights_[id_column] = model_automl.feature_weights_[best_iteration.name]
                self.estimators_[id_column] = best_iteration["best_estimator"]
                self.best_iteration_[id_column] = best_iteration.name
                self.scores_[id_column]["training"] = best_iteration["feature_selected_training_score"]
                self.scores_[id_column]["testing"] = best_iteration["feature_selected_testing_score"]
                self.studies_[id_column] = model_automl.studies_
            del model_automl
        self.is_fitted = True
        return self

    @classmethod
    def from_file(cls, filepath):
        cls = read_pickle(filepath)
        return cls

    def to_file(self, filepath):
        write_pickle(self, filepath)

    # =======
    # Built-in
    # =======
    def __repr__(self):
        pad = 4
        header = format_header(f"{self.__class__.__name__}(Name:{self.name}, ObservationType: {self.observation_type}, FeatureType: {self.feature_type}, EmbeddingType: {self.embedding_type})", line_character="=")

        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* estimator: {}".format(self.estimator),
            pad*" " + "* param_space: {}".format(self.param_space),
            pad*" " + "* scorer: {}".format(self.scorer),
            pad*" " + "* n_iter: {}".format(self.n_iter),
            pad*" " + "* n_trials: {}".format(self.n_trials),
            pad*" " + "* transformation: {}".format(self.transformation),


        ]
                                                  
        if self.is_fitted:
            n_successful = sum(self.automl_status_ok_.values())
            n_failed = len(self.automl_status_ok_.values()) - n_successful

            fields += [
            pad*" " + "* [X] m_features = {}".format(self.X_.shape[1]),
            pad*" " + "* [X] n_observations = {}".format(self.X_.shape[0]),
            pad*" " + "* [Y] p_embeddings = {}".format(self.Y_.shape[1]),
            ]
            
            if hasattr(self, "X_testing_"):
                fields += [
                pad*" " + "* [X_testing] n_observations = {}".format(self.X_testing_.shape[0]),
                ]
                
            fields += [
                pad*" " + "* [AutoML] successful = {}".format(n_successful),
                pad*" " + "* [AutoML] failed = {}".format(n_failed),

                ]

        return "\n".join(fields)
    
__all__ = [
    "DiffusionMapEmbedding", 
    "LandmarkDiffusionMapEmbedding", 
    "NicheSpace", 
    "HierarchicalNicheSpace", 
    "QualitativeSpace",
    "EmbeddingAnnotator",
]