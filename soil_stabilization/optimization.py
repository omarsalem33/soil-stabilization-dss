"""Multi-objective candidate-mixture search."""

import numpy as np
import pandas as pd

def search_pareto_candidates(X, row, models, prices, carbon_factors, min_ucs, max_swell, generations, target_labels):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    mix_features = [name for name in ("Lime_%", "RHA_%", "PET_%") if name in X.columns]
    if len(mix_features) != 3 or "UCS_kPa" not in models or "Swell_Percent" not in models:
        raise ValueError("Optimization requires Lime, RHA, and PET inputs plus trained UCS and swell models.")
    bounds = [X[name].agg(["min", "max"]).tolist() for name in mix_features]
    base = row.iloc[0].to_dict()
    emissions_active = all(value > 0 for value in carbon_factors)

    class MixtureProblem(Problem):
        def __init__(self):
            super().__init__(n_var=3, n_obj=4 if emissions_active else 3, n_ieq_constr=2,
                             xl=np.array([pair[0] for pair in bounds]),
                             xu=np.array([pair[1] for pair in bounds]))
        def _evaluate(self, decision, out, *args, **kwargs):
            candidates = pd.DataFrame([base] * len(decision))
            candidates.loc[:, mix_features] = decision
            ucs = models["UCS_kPa"].predict(candidates)
            swell = models["Swell_Percent"].predict(candidates)
            amounts = decision / 100 * 1000
            objectives = [-ucs, swell, amounts @ np.asarray(prices)]
            if emissions_active:
                objectives.append(amounts @ np.asarray(carbon_factors))
            out["F"] = np.column_stack(objectives)
            out["G"] = np.column_stack([min_ucs - ucs, swell - max_swell])

    result = minimize(MixtureProblem(), NSGA2(pop_size=60), ("n_gen", generations), seed=42, verbose=False)
    if result.X is None:
        return None, 0
    decision = np.atleast_2d(result.X)
    candidates = pd.DataFrame([base] * len(decision))
    candidates.loc[:, mix_features] = decision
    frame = candidates.copy()
    for target in ("UCS_kPa", "Swell_Percent"):
        frame[target_labels[target][0]] = models[target].predict(candidates)
    amounts = decision / 100 * 1000
    frame["Estimated additive cost (EGP/t dry soil)"] = amounts @ np.asarray(prices)
    if emissions_active:
        frame["Additive carbon proxy (kg CO₂e/t dry soil)"] = amounts @ np.asarray(carbon_factors)
    feasible = ((frame[target_labels["UCS_kPa"][0]] >= min_ucs) &
                (frame[target_labels["Swell_Percent"][0]] <= max_swell))
    return frame, int(feasible.sum())
