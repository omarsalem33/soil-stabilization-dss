"""Dataset audit summaries used by the Data audit page."""

def summarize_dataset(X, Y, available_features, available_targets):
    input_summary = X.describe().T
    input_summary["Missing"] = X.isna().sum()
    response_summary = Y.describe().T
    correlation = X.join(Y).corr(numeric_only=True).loc[available_features, available_targets]
    duplicates = int(X.duplicated().sum())
    return input_summary, response_summary, correlation, duplicates
