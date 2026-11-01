import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split,)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

st.set_page_config(page_title="Expansive Soil Stabilization DSS", layout="wide")
st.title("AI-Assisted Soil Stabilization Research DSS")
st.warning(
    "Research prototype only. The supplied dataset's experimental provenance has not been "
    "verified. Predictions and optimization results are not engineering approval or a "
    "substitute for laboratory tests and geotechnical design."
)

DATA_FILE = "soil_stabilization_dataset.csv"
FEATURES = [
    "Lime_%", "RHA_%", "PET_%", "Fiber_Length_mm", "Curing_Days",
    "MDD_g_cm3", "OMC_percent", "PI_percent", "Fiber_Content_%", "Water_Content_%",
]
FEATURE_LABELS = {
    "Lime_%": "Lime content (%)",
    "RHA_%": "Rice husk ash content (%)",
    "PET_%": "PET fiber content (%)",
    "Fiber_Length_mm": "Fiber length (mm)",
    "Curing_Days": "Curing time (days)",
    "MDD_g_cm3": "Maximum dry density (g/cm³)",
    "OMC_percent": "Optimum moisture content (%)",
    "PI_percent": "Plasticity index (%)",
    "Fiber_Content_%": "Fiber content (%)",
    "Water_Content_%": "Water content (%)",
}
TARGETS = {
    "UCS_kPa": ("UCS (kPa)", ["ucs_kpa", "ucs", "unconfined_compressive_strength"]),
    "Swell_Percent": ("Swell (%)", ["swell_percent", "swell", "swelling_percent"]),
    "Swell_Pressure_kPa": ("Swell pressure (kPa)", ["swell_pressure_kpa", "swelling_pressure_kpa", "swell_pressure"]),
    "CBR_percent": ("CBR (%)", ["cbr_percent", "cbr", "california_bearing_ratio"]),
    "Settlement_mm": ("Settlement (mm)", ["settlement_mm", "settlement"]),
    "Bearing_Capacity_kPa": ("Bearing capacity (kPa)", ["bearing_capacity_kpa", "bearing_capacity"]),
    "Cohesion_kPa": ("Cohesion (kPa)", ["cohesion_kpa", "cohesion"]),
    "Phi_deg": ("Friction angle (degrees)", ["phi_deg", "friction_angle_deg", "friction_angle"]),
    "Compressibility_Index": ("Compressibility index", ["compressibility_index", "compression_index", "cc"]),
}
GROUP_COLUMNS = ["Mixture_ID", "Batch_ID", "Specimen_Batch", "Source_ID", "Group_ID"]
ALIASES = {
    "Lime_%": ["lime_%", "lime_percent", "lime"],
    "RHA_%": ["rha_%", "rha_percent", "rice_husk_ash_percent", "rice_husk_ash"],
    "PET_%": ["pet_%", "pet_percent", "pet_fiber_percent"],
    "Fiber_Length_mm": ["fiber_length_mm", "fiber_length", "lf"],
    "Curing_Days": ["curing_days", "curing_time_days", "curing_time", "tc"],
    "MDD_g_cm3": ["mdd_g_cm3", "mdd", "maximum_dry_density"],
    "OMC_percent": ["omc_percent", "omc", "optimum_moisture_content"],
    "PI_percent": ["pi_percent", "pi", "plasticity_index"],
    "Fiber_Content_%": ["fiber_content_%", "fiber_content", "cf"],
    "Water_Content_%": ["water_content_%", "water_content", "w"],
}


def norm_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def resolve_column(columns, aliases):
    lookup = {norm_name(c): c for c in columns}
    for alias in aliases:
        if norm_name(alias) in lookup:
            return lookup[norm_name(alias)]
    return None


@st.cache_data
def load_source(upload_bytes):
    if upload_bytes:
        return pd.read_csv(io.BytesIO(upload_bytes)), "Uploaded CSV"
    return pd.read_csv(DATA_FILE), DATA_FILE


uploaded = st.sidebar.file_uploader("Use verified experimental CSV", type=["csv"])
try:
    raw_data, source_name = load_source(uploaded.getvalue() if uploaded else None)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

feature_columns = {
    canonical: resolve_column(raw_data.columns, aliases)
    for canonical, aliases in ALIASES.items()
}
available_features = [
    key for key in FEATURES
    if feature_columns[key] is not None
    and pd.to_numeric(raw_data[feature_columns[key]], errors="coerce").notna().sum() >= 30
]
target_columns = {
    key: resolve_column(raw_data.columns, aliases)
    for key, (_, aliases) in TARGETS.items()
}
available_targets = [key for key, col in target_columns.items() if col is not None]
group_column = next((c for c in GROUP_COLUMNS if c in raw_data.columns), None)

if len(available_features) < 3:
    st.error("The dataset must include at least three recognized input columns.")
    st.stop()
if not available_targets:
    st.error("No recognized measured response columns were found.")
    st.stop()

X_all = raw_data[[feature_columns[key] for key in available_features]].copy()
X_all.columns = available_features
Y_all = pd.DataFrame(index=raw_data.index)
for key, source_col in target_columns.items():
    if source_col is not None:
        Y_all[key] = pd.to_numeric(raw_data[source_col], errors="coerce")
for col in available_features:
    X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
valid_x = X_all.notna().all(axis=1)
X_all = X_all.loc[valid_x].reset_index(drop=True)
Y_all = Y_all.loc[valid_x].reset_index(drop=True)
groups = raw_data.loc[valid_x, group_column].reset_index(drop=True) if group_column else None
target_counts = {key: int(Y_all[key].notna().sum()) for key in TARGETS if key in Y_all}
available_targets = [key for key in TARGETS if target_counts.get(key, 0) >= 30]
Y_all = Y_all[available_targets]
if not available_targets:
    st.error("No response has at least 30 measured rows. Add experimental results before training.")
    st.stop()
if len(X_all) < 30:
    st.error("At least 30 rows with complete model inputs are required.")
    st.stop()
MODEL_LABELS = {**{k: v[0] for k, v in TARGETS.items()}}


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
        models["Extra trees"] = ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
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

        leaderboard = []
        candidates = candidate_models(len(Xi))
        for name, estimator in candidates.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_args = {"groups": group_train} if group_train is not None else {}
                scores = cross_val_score(
                    estimator, X_train, y_train, cv=cv,
                    scoring="neg_mean_absolute_error", n_jobs=1, **score_args,
                )
            leaderboard.append({"Model": name, "CV MAE": float(-scores.mean())})
        leaderboard.sort(key=lambda row: row["CV MAE"])
        best_name = leaderboard[0]["Model"]
        selected = clone(candidates[best_name])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected.fit(X_train, y_train)
            test_pred = selected.predict(X_test)
            final_model = clone(candidates[best_name]).fit(Xi, yi)
        fitted[target] = final_model
        reports[target] = {
            "best_model": best_name,
            "comparison": pd.DataFrame(leaderboard),
            "test": {
                "MAE": float(mean_absolute_error(y_test, test_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "R²": float(r2_score(y_test, test_pred)),
                "n_test": int(len(X_test)),
                "n_training_pairs": int(len(Xi)),
                "90% absolute test-error band": float(
                    np.quantile(np.abs(y_test.to_numpy() - test_pred), 0.9)
                ),
            },
            "X_test": X_test,
            "y_test": y_test,
            "test_prediction": test_pred,
        }
    return fitted, reports
group_key = tuple(groups.astype(str)) if groups is not None else tuple()
try:
    models, reports = fit_models(X_all, Y_all, group_key)
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

st.caption(
    f"Dataset: {source_name} · {len(X_all)} rows with complete inputs · "
    f"{len(available_features)} inputs · {len(available_targets)} trainable measured responses"
)
missing_targets = [TARGETS[key][0] for key in TARGETS if key not in available_targets]
if missing_targets:
    st.info(
        "These research outcomes are not in the loaded data, so the application will not "
        "invent predictions for them: " + ", ".join(missing_targets) + ". Add measured "
        "columns for them to train and evaluate those models."
    )
missing_inputs = [FEATURE_LABELS[key] for key in FEATURES if key not in available_features]
if missing_inputs:
    st.info(
        "These proposed inputs are not in the loaded data and are omitted from the models: "
        + ", ".join(missing_inputs) + ". Add measured columns before including them."
    )

tabs = st.tabs([
    "Prediction & design", "Model evaluation", "SHAP explanation",
    "Multi-objective optimization", "Experimental plan", "Independent validation",
    "Data audit",
])

with tabs[0]:
    st.subheader("Predict measured engineering responses")
    st.sidebar.header("Prediction inputs")
    values = {}
    for key in available_features:
        series = X_all[key]
        lo, hi = float(series.min()), float(series.max())
        if lo == hi:
            values[key] = lo
        elif key == "Curing_Days" and series.nunique() <= 12:
            choices = sorted(series.unique().tolist())
            values[key] = st.sidebar.select_slider(FEATURE_LABELS[key], options=choices, value=choices[-1])
        else:
            values[key] = st.sidebar.slider(
                FEATURE_LABELS[key], min_value=lo, max_value=hi,
                value=float(series.median()), step=max((hi - lo) / 100, 0.001),
            )
    row = pd.DataFrame([values], columns=available_features)
    pred = {target: float(model.predict(row)[0]) for target, model in models.items()}
    cols = st.columns(min(4, len(pred)))
    for i, (target, val) in enumerate(pred.items()):
        cols[i % len(cols)].metric(TARGETS[target][0], f"{val:.2f}")
        band = reports[target]["test"]["90% absolute test-error band"]
        cols[i % len(cols)].caption(f"Approx. holdout error band: ±{band:.2f} (90th percentile absolute error)")

    if "Cohesion_kPa" in pred and "Phi_deg" in pred:
        st.markdown("#### Exploratory shallow foundation calculation")
        st.caption(
            "Uses predicted cohesion and friction angle in a simplified general-shear equation. "
            "Shape, inclination, groundwater, eccentricity, layering, code factors, and settlement "
            "checks are omitted; this is not a design capacity."
        )
        c1, c2, c3, c4 = st.columns(4)
        width = c1.number_input("Footing width B (m)", min_value=0.1, value=1.5, step=0.1)
        depth = c2.number_input("Embedment depth Df (m)", min_value=0.0, value=1.0, step=0.1)
        gamma = c3.number_input("Soil unit weight γ (kN/m³)", min_value=1.0, value=18.0, step=0.5)
        fs = c4.number_input("Factor of safety", min_value=1.0, value=3.0, step=0.5)
        phi = np.radians(max(0.0, min(45.0, pred["Phi_deg"])))
        if phi < 1e-6:
            nq, nc, ngamma = 1.0, 5.14, 0.0
        else:
            nq = np.exp(np.pi * np.tan(phi)) * np.tan(np.pi / 4 + phi / 2) ** 2
            nc = (nq - 1) / np.tan(phi)
            ngamma = 2 * (nq + 1) * np.tan(phi)
        qult = pred["Cohesion_kPa"] * nc + gamma * depth * nq + 0.5 * gamma * width * ngamma
        c1.metric("Simplified allowable pressure (kPa)", f"{qult / fs:.1f}")

with tabs[1]:
    st.subheader("Model comparison and held-out performance")
    st.write(
        "Candidate models are ranked by cross-validated MAE on the training portion. "
        "The displayed test metrics use a separate held-out portion. If batch or mixture "
        "IDs exist, rows from the same group are kept together."
    )
    for target in available_targets:
        st.markdown(f"**{TARGETS[target][0]} · selected: {reports[target]['best_model']}**")
        st.dataframe(pd.DataFrame([reports[target]["test"]]), hide_index=True, use_container_width=True)
        st.dataframe(reports[target]["comparison"], hide_index=True, use_container_width=True)
        chart = pd.DataFrame({
            "Measured": reports[target]["y_test"].to_numpy(),
            "Predicted": reports[target]["test_prediction"],
        })
        st.scatter_chart(chart, x="Measured", y="Predicted")

with tabs[2]:
    st.subheader("Local SHAP explanation")
    st.write(
        "Shows how each measured input moved this selected prediction relative to a sample "
        "of rows from the dataset. Explanations describe model behavior, not causal effects."
    )
    explain_target = st.selectbox(
        "Response to explain", available_targets,
        format_func=lambda value: TARGETS[value][0],
    )
    if st.button("Calculate SHAP values"):
        try:
            import shap
            background = shap.sample(X_all, min(25, len(X_all)), random_state=42)
            explainer = shap.PermutationExplainer(models[explain_target].predict, background)
            explanation = explainer(row, max_evals=2 * len(available_features) + 1)
            shap_frame = pd.DataFrame({
                "Input": [FEATURE_LABELS[x] for x in available_features],
                "SHAP value": explanation.values[0],
            }).sort_values("SHAP value")
            st.bar_chart(shap_frame.set_index("Input"))
            st.dataframe(shap_frame, hide_index=True, use_container_width=True)
        except Exception as error:
            st.error(f"SHAP could not calculate an explanation: {error}")

with tabs[3]:
    st.subheader("Pareto search for candidate mixtures")
    st.write(
        "NSGA-II searches the measured mixture range for trade-offs between UCS, swelling, "
        "and additive cost. All three unit prices are required. A carbon-footprint objective "
        "is added when all three emission factors are entered."
    )
    mix_features = [f for f in ["Lime_%", "RHA_%", "PET_%"] if f in available_features]
    optimizer_ready = "UCS_kPa" in models and "Swell_Percent" in models and len(mix_features) == 3
    if not optimizer_ready:
        st.info(
            "Optimization needs measured UCS and swell response columns plus Lime, RHA, "
            "and PET inputs. The current dataset may not contain those research fields."
        )
    else:
        st.caption("Cost assumption: additive percentages are by dry-soil mass; prices are EGP/kg.")
        price_cols = st.columns(3)
        prices = [
            price_cols[i].number_input(
                f"{name} price (EGP/kg)", min_value=0.0, value=0.0, step=1.0,
                key=f"price_{key}",
            )
            for i, (name, key) in enumerate([
                ("Lime", "Lime_%"), ("RHA", "RHA_%"), ("PET", "PET_%")
            ])
        ]
        carbon_cols = st.columns(3)
        carbon = [
            carbon_cols[i].number_input(
                f"{name} factor (kg CO₂e/kg; optional)", min_value=0.0, value=0.0,
                step=0.1, key=f"carbon_{key}",
            )
            for i, (name, key) in enumerate([
                ("Lime", "Lime_%"), ("RHA", "RHA_%"), ("PET", "PET_%")
            ])
        ]
        c1, c2, c3 = st.columns(3)
        min_ucs = c1.number_input("Minimum UCS (kPa)", min_value=0.0, value=200.0, step=10.0)
        max_swell = c2.number_input("Maximum swell (%)", min_value=0.0, value=2.0, step=0.5)
        generations = c3.slider("Search generations", 20, 100, 40, 10)
        if st.button("Search Pareto candidates"):
            if not all(p > 0 for p in prices):
                st.error("Enter a positive unit price for lime, RHA, and PET before searching.")
            else:
                try:
                    from pymoo.algorithms.moo.nsga2 import NSGA2
                    from pymoo.core.problem import Problem
                    from pymoo.optimize import minimize

                    bounds = [X_all[f].agg(["min", "max"]).tolist() for f in mix_features]
                    base = row.iloc[0].to_dict()
                    emissions_active = all(value > 0 for value in carbon)
                    class MixProblem(Problem):
                        def __init__(self):
                            super().__init__(
                                n_var=3, n_obj=4 if emissions_active else 3, n_ieq_constr=2,
                                xl=np.array([pair[0] for pair in bounds]),
                                xu=np.array([pair[1] for pair in bounds]),
                            )
                        def _evaluate(self, decision, out, *args, **kwargs):
                            candidates = pd.DataFrame([base] * len(decision))
                            candidates.loc[:, mix_features] = decision
                            ucs = models["UCS_kPa"].predict(candidates)
                            swell = models["Swell_Percent"].predict(candidates)
                            amounts = decision / 100 * 1000
                            cost = amounts @ np.asarray(prices)
                            carbon_total = amounts @ np.asarray(carbon)
                            objectives = [-ucs, swell, cost]
                            if emissions_active:
                                objectives.append(carbon_total)
                            out["F"] = np.column_stack(objectives)
                            out["G"] = np.column_stack([min_ucs - ucs, swell - max_swell])

                    result = minimize(
                        MixProblem(), NSGA2(pop_size=60),
                        ("n_gen", generations), seed=42, verbose=False,
                    )
                    if result.X is None:
                        st.warning("The search did not return candidates. Check the data range and thresholds.")
                    else:
                        decision = np.atleast_2d(result.X)
                        candidates = pd.DataFrame([base] * len(decision))
                        candidates.loc[:, mix_features] = decision
                        result_frame = candidates.copy()
                        for target in ["UCS_kPa", "Swell_Percent"]:
                            result_frame[TARGETS[target][0]] = models[target].predict(candidates)
                        kg = decision / 100 * 1000
                        result_frame["Estimated additive cost (EGP/t dry soil)"] = kg @ np.asarray(prices)
                        if emissions_active:
                            result_frame["Additive carbon proxy (kg CO₂e/t dry soil)"] = kg @ np.asarray(carbon)
                        feasible = (
                            (result_frame[TARGETS["UCS_kPa"][0]] >= min_ucs)
                            & (result_frame[TARGETS["Swell_Percent"][0]] <= max_swell)
                        )
                        st.success(f"Returned {len(result_frame)} Pareto candidates; {int(feasible.sum())} meet both response thresholds.")
                        st.dataframe(result_frame.sort_values("Estimated additive cost (EGP/t dry soil)"), hide_index=True, use_container_width=True)
                        st.download_button(
                            "Download Pareto candidates CSV",
                            result_frame.to_csv(index=False).encode("utf-8"),
                            "pareto_candidates.csv", "text/csv",
                        )
                        st.warning("These are model-generated candidates. Test selected mixes independently in the laboratory before recommending them.")
                except Exception as error:
                    st.error(f"Optimization failed: {error}")

with tabs[4]:
    st.subheader("Experimental screening plan")
    st.write(
        "Creates a reproducible Latin-hypercube screening plan with untreated and single-additive "
        "controls, replicate rows, randomized run order, and empty response columns."
    )
    n_runs = st.number_input("Mixture combinations", min_value=8, max_value=120, value=24, step=4)
    replicates = st.number_input("Replicates per combination", min_value=1, max_value=5, value=3, step=1)
    if st.button("Generate screening plan"):
        try:
            from scipy.stats import qmc
            factors = [f for f in ["Lime_%", "RHA_%", "PET_%", "Fiber_Length_mm"] if f in available_features]
            if len(factors) < 3:
                st.error("At least three mixture factors must be present to generate a screening plan.")
            else:
                sampler = qmc.LatinHypercube(d=len(factors), seed=42)
                unit = sampler.random(int(n_runs))
                plan = pd.DataFrame(index=range(int(n_runs)), columns=available_features)
                for f in available_features:
                    plan[f] = float(X_all[f].median())
                for j, factor in enumerate(factors):
                    low, high = float(X_all[factor].min()), float(X_all[factor].max())
                    plan[factor] = low + unit[:, j] * (high - low)
                if "Curing_Days" in available_features:
                    levels = sorted(X_all["Curing_Days"].unique())
                    plan["Curing_Days"] = [levels[i % len(levels)] for i in range(len(plan))]
                controls = []
                for label, active in [
                    ("Untreated control", []),
                    ("Lime-only control", ["Lime_%"]),
                    ("RHA-only control", ["RHA_%"]),
                    ("PET-only control", ["PET_%"]),
                ]:
                    control = {f: float(X_all[f].median()) for f in available_features}
                    for f in ["Lime_%", "RHA_%", "PET_%"]:
                        if f in available_features:
                            control[f] = 0.0 if f not in active else float(X_all[f].median())
                    control["Control_Type"] = label
                    controls.append(control)
                plan["Control_Type"] = "DOE mixture"
                plan = pd.concat([pd.DataFrame(controls), plan], ignore_index=True)
                plan.insert(0, "Mixture_ID", [f"M{i:03d}" for i in range(1, len(plan) + 1)])
                plan = plan.loc[plan.index.repeat(int(replicates))].reset_index(drop=True)
                plan["Replicate"] = plan.groupby("Mixture_ID").cumcount() + 1
                plan["Run_Order"] = np.random.default_rng(42).permutation(len(plan)) + 1
                for target in available_targets:
                    plan[TARGETS[target][0]] = np.nan
                st.dataframe(plan, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download experimental plan CSV",
                    plan.to_csv(index=False).encode("utf-8"),
                    "experimental_screening_plan.csv", "text/csv",
                )
                st.caption("Confirm factor ranges, zero-dose controls, sample preparation, and test standards with the laboratory supervisor.")
        except Exception as error:
            st.error(f"Could not generate the plan: {error}")

with tabs[5]:
    st.subheader("Prospective independent validation")
    st.write(
        "Upload measurements from optimized mixtures that were not used to train the models. "
        "The app compares predictions with those observed results and reports prediction errors."
    )
    validation_upload = st.file_uploader("Independent validation CSV", type=["csv"], key="validation_csv")
    if validation_upload:
        try:
            validation = pd.read_csv(validation_upload)
            validation_X = pd.DataFrame()
            for key in available_features:
                col = resolve_column(validation.columns, ALIASES[key])
                if not col:
                    raise ValueError(f"Validation file is missing input: {FEATURE_LABELS[key]}")
                validation_X[key] = pd.to_numeric(validation[col], errors="raise")
            rows = []
            for target in available_targets:
                col = resolve_column(validation.columns, TARGETS[target][1])
                if not col:
                    continue
                observed = pd.to_numeric(validation[col], errors="coerce")
                keep = observed.notna()
                predicted = models[target].predict(validation_X.loc[keep])
                rows.append({
                    "Response": TARGETS[target][0],
                    "n": int(keep.sum()),
                    "MAE": mean_absolute_error(observed[keep], predicted),
                    "RMSE": np.sqrt(mean_squared_error(observed[keep], predicted)),
                    "R²": r2_score(observed[keep], predicted) if keep.sum() > 1 else np.nan,
                    "Bias (predicted - measured)": float(np.mean(predicted - observed[keep])),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info("No measured response columns matched the trained outcomes.")
        except Exception as error:
            st.error(f"Validation file could not be evaluated: {error}")

with tabs[6]:
    st.subheader("Dataset audit and research readiness")
    summary = X_all.describe().T
    summary["Missing"] = X_all.isna().sum()
    st.markdown("**Input ranges and distribution**")
    st.dataframe(summary, use_container_width=True)
    st.markdown("**Measured response distribution**")
    st.dataframe(Y_all.describe().T, use_container_width=True)
    st.markdown("**Input–response correlation (screening only)**")
    st.dataframe(pd.concat([X_all, Y_all], axis=1).corr(numeric_only=True).loc[available_features, available_targets], use_container_width=True)
    st.write(f"Duplicate input combinations: {int(X_all.duplicated().sum())}")
    if group_column:
        st.write(f"Grouped split field: {group_column} ({groups.nunique()} groups)")
    else:
        st.warning("No mixture/batch grouping column found. Replicates could leak between train and test partitions.")
    st.info(
        "Record traceable source, soil classification, additive batch properties, specimen preparation, "
        "test standards, units, replicate identity, and measurement uncertainty. Do not delete outliers "
        "without a documented reason. Add CBR, settlement, bearing capacity, swell pressure, and "
        "compressibility measurements as separate columns if those outcomes are part of the final scope."
    )
