"""Candidate regressors and per-response training/evaluation pipeline."""

import numpy as np
import pandas as pd
import streamlit as st

from .model_evaluation import fit_and_measure_selected, rank_candidates
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, train_test_split,)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

def candidate_models(n_rows):
    models = {
        "Linear baseline": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial baseline": make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
        "Decision tree": DecisionTreeRegressor(min_samples_leaf=2, random_state=42),
        "Random forest": RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10)),
        "ANN (MLP)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1200, early_stopping=True, random_state=42),
        ),
        "Gaussian process": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                normalize_y=True, random_state=42, n_restarts_optimizer=0,
            ),
        ),
    }
    if n_rows >= 60:
        models["Extra trees"] = ExtraTreesRegressor( n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=180, max_depth=3, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=42, n_jobs=2,
        )
    return models


@st.cache_resource
def fit_models(X, Y, groups_tuple):
    fitted, reports = {}, {}
    all_groups = np.array(groups_tuple, dtype=object) if groups_tuple else None
    for target in Y.columns:
        target_mask = X.notna().all(axis=1) & Y[target].notna()
        Xi = X.loc[target_mask].reset_index(drop=True)
        yi = Y.loc[target_mask, target].reset_index(drop=True)
        group_values = all_groups[target_mask.to_numpy()] if all_groups is not None else None
        if len(Xi) < 30:
            continue

        if group_values is not None and pd.Series(group_values).nunique() >= 5:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(splitter.split(Xi, yi, group_values))
            X_train, X_test = Xi.iloc[train_idx], Xi.iloc[test_idx]
            y_train, y_test = yi.iloc[train_idx], yi.iloc[test_idx]
            group_train = group_values[train_idx]
            cv = GroupKFold(n_splits=min(5, pd.Series(group_train).nunique()))
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                Xi, yi, test_size=0.2, random_state=42
            )
            group_train = None
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        candidates = candidate_models(len(Xi))
        leaderboard = rank_candidates(candidates, X_train, y_train, cv, group_train)
        best_name = leaderboard[0]["Model"]
        final_model, test_pred, test_metrics = fit_and_measure_selected(
            candidates, best_name, X_train, y_train, X_test, y_test, Xi, yi
        )
        fitted[target] = final_model
        reports[target] = {
            "best_model": best_name,
            "comparison": pd.DataFrame(leaderboard),
            "test": test_metrics,
            "X_test": X_test,
            "y_test": y_test,
            "test_prediction": test_pred,
        }
    return fitted, reports
