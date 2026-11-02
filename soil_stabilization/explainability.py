"""Local permutation-SHAP explanation helper."""

import pandas as pd

def calculate_local_shap(model, X, row, feature_labels):
    import shap
    background = shap.sample(X, min(25, len(X)), random_state=42)
    explainer = shap.PermutationExplainer(model.predict, background)
    explanation = explainer(row, max_evals=2 * X.shape[1] + 1)
    return pd.DataFrame({
        "Input": [feature_labels[column] for column in X.columns],
        "SHAP value": explanation.values[0],
    }).sort_values("SHAP value")
