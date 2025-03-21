#!/usr/bin/env python
import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import optuna

def fast_groupby(X: pd.DataFrame, y: pd.Series, method: str = "sum"):
    if not np.all(X.shape[0] == y.size):
        raise IndexError("X.shape[0] must equal y.size")
    if not np.all(X.index == y.index):
        raise IndexError("X.index must equal y.index")
    if not isinstance(y, pd.CategoricalDtype):
        y = y.astype("category")
    
    # Convert y to numeric indices
    unique_classes, y_indices = np.unique(y, return_inverse=True)
    
    arrays = []
    for col in tqdm(X.columns, "Grouping rows by", unit=" column"):
        summed_values = np.bincount(y_indices, weights=X[col].values, minlength=len(unique_classes))
        
        if method == "sum":
            result_values = summed_values
        elif method == "mean":
            counts = np.bincount(y_indices, minlength=len(unique_classes))
            result_values = np.divide(summed_values, counts, out=np.zeros_like(summed_values), where=counts > 0)
        else:
            raise ValueError("Unsupported method. Use 'sum' or 'mean'.")
        
        arrays.append(result_values)
    
    X_grouped = np.vstack(arrays).T
    return pd.DataFrame(X_grouped, index=unique_classes, columns=X.columns)

def compile_parameter_space(trial, param_space): # This should be merged with `compile_parameter_space` from clairvoyance
    params = dict()
    for k, v in param_space.items():
        if isinstance(v, list):
            suggestion_type = v[0]
            if isinstance(suggestion_type, type):
                suggestion_type = suggestion_type.__name__
            suggest = getattr(trial, f"suggest_{suggestion_type}")
            suggestion = suggest(k, v[1], v[2])
        else:
            suggestion = v
        params[k] = suggestion
    return params

def is_square_symmetric(matrix, tol=1e-8, raise_exception=True):
    """Check if a matrix is square and symmetric."""
    matrix = np.array(matrix)  # Ensure it's a NumPy array
    same_shape = matrix.shape[0] == matrix.shape[1]
    all_close = np.allclose(matrix, matrix.T, atol=tol)
    all_notnull = not np.any(np.isnan(matrix))
    status_ok = all([same_shape, all_close, all_notnull])
    if raise_exception:
        if not status_ok:
            raise ValueError(f"Not symmetric\n * Square: {same_shape}\n * Upper/lower triangle close (tol={tol}): {all_close}\n * All not NaN {all_notnull}")
    return status_ok


def stop_when_exceeding_trials(n_trials, logger):
    def callback(study, trial):
        """
        Callback that stops optimization if the total number of trials exceeds `n_trials`.
        """
        finished_trial_states = [
            optuna.trial.TrialState.COMPLETE, 
            optuna.trial.TrialState.FAIL, 
            optuna.trial.TrialState.RUNNING,
        ]

        total_completed_trials = sum(1 for t in study.trials if t.state in finished_trial_states)

        if total_completed_trials >= n_trials:
            logger.warning(f"[Callback] Stopping optimization: {total_completed_trials} trials reached (limit={n_trials})")
            study.stop()
    
    return callback  # Return the function with access to `n_trials` and `logger`
