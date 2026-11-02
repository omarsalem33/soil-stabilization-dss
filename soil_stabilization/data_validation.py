"""Compare independent laboratory measurements with model predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_validation(validation, available_features, available_targets, aliases, targets, models, resolve_column):
    validation_X = pd.DataFrame()
    for key in available_features:
        column = resolve_column(validation.columns, aliases[key])
        if not column:
            raise ValueError(f"Validation file is missing input: {key}")
        validation_X[key] = pd.to_numeric(validation[column], errors="raise")
    rows = []
    for target in available_targets:
        column = resolve_column(validation.columns, targets[target][1])
        if not column:
            continue
        observed = pd.to_numeric(validation[column], errors="coerce")
        keep = observed.notna()
        predicted = models[target].predict(validation_X.loc[keep])
        rows.append({
            "Response": targets[target][0], "n": int(keep.sum()),
            "MAE": mean_absolute_error(observed[keep], predicted),
            "RMSE": np.sqrt(mean_squared_error(observed[keep], predicted)),
            "R²": r2_score(observed[keep], predicted) if keep.sum() > 1 else np.nan,
            "Bias (predicted - measured)": float(np.mean(predicted - observed[keep])),
        })
    return pd.DataFrame(rows)
