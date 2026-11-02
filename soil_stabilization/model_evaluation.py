"""Cross-validation and held-out evaluation helpers."""

import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def rank_candidates(candidates, X_train, y_train, cv, group_train):
    """Rank model estimators by cross-validated MAE on the training split."""
    leaderboard = []
    for name, estimator in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score_args = {"groups": group_train} if group_train is not None else {}
            scores = cross_val_score(
                estimator, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error",
                n_jobs=1, **score_args,
            )
        leaderboard.append({"Model": name, "CV MAE": float(-scores.mean())})
    leaderboard.sort(key=lambda row: row["CV MAE"])
    return leaderboard

def fit_and_measure_selected(candidates, best_name, X_train, y_train, X_test, y_test, X_all, y_all):
    """Fit the selected model, calculate holdout metrics, then refit on all target rows."""
    selected = clone(candidates[best_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected.fit(X_train, y_train)
        test_prediction = selected.predict(X_test)
        final_model = clone(candidates[best_name]).fit(X_all, y_all)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, test_prediction)),
        "RMSE": float(mean_squared_error(y_test, test_prediction) ** 0.5),
        "R²": float(r2_score(y_test, test_prediction)),
        "n_test": int(len(X_test)),
        "n_training_pairs": int(len(X_all)),
        "90% absolute test-error band": float(
            pd.Series(abs(y_test.to_numpy() - test_prediction)).quantile(0.9)
        ),
    }
    return final_model, test_prediction, metrics

def metrics_frame(report):
    return pd.DataFrame([report["test"]])

def measured_prediction_frame(report):
    return pd.DataFrame({
        "Measured": report["y_test"].to_numpy(),
        "Predicted": report["test_prediction"],
    })
