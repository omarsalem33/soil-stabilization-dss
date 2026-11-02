"""Generate a reproducible laboratory screening plan."""

import numpy as np
import pandas as pd

def generate_screening_plan(X, available_features, available_targets, target_labels, n_runs, replicates):
    from scipy.stats import qmc
    factors = [name for name in ("Lime_%", "RHA_%", "PET_%", "Fiber_Length_mm") if name in available_features]
    if len(factors) < 3:
        raise ValueError("At least three mixture factors must be present to generate a screening plan.")
    sampler = qmc.LatinHypercube(d=len(factors), seed=42)
    unit = sampler.random(int(n_runs))
    plan = pd.DataFrame(index=range(int(n_runs)), columns=available_features)
    for feature in available_features:
        plan[feature] = float(X[feature].median())
    for index, factor in enumerate(factors):
        low, high = float(X[factor].min()), float(X[factor].max())
        plan[factor] = low + unit[:, index] * (high - low)
    if "Curing_Days" in available_features:
        levels = sorted(X["Curing_Days"].unique())
        plan["Curing_Days"] = [levels[i % len(levels)] for i in range(len(plan))]
    controls = []
    for label, active in (("Untreated control", []), ("Lime-only control", ["Lime_%"]),
                          ("RHA-only control", ["RHA_%"]), ("PET-only control", ["PET_%"])):
        control = {feature: float(X[feature].median()) for feature in available_features}
        for feature in ("Lime_%", "RHA_%", "PET_%"):
            if feature in available_features:
                control[feature] = 0.0 if feature not in active else float(X[feature].median())
        control["Control_Type"] = label
        controls.append(control)
    plan["Control_Type"] = "DOE mixture"
    plan = pd.concat([pd.DataFrame(controls), plan], ignore_index=True)
    plan.insert(0, "Mixture_ID", [f"M{i:03d}" for i in range(1, len(plan) + 1)])
    plan = plan.loc[plan.index.repeat(int(replicates))].reset_index(drop=True)
    plan["Replicate"] = plan.groupby("Mixture_ID").cumcount() + 1
    plan["Run_Order"] = np.random.default_rng(42).permutation(len(plan)) + 1
    for target in available_targets:
        plan[target_labels[target][0]] = np.nan
    return plan
