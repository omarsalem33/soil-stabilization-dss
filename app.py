import pandas as pd
import streamlit as st

from soil_stabilization.data_io import (
    ALIASES, FEATURES, FEATURE_LABELS, TARGETS,
    load_source, prepare_training_data, resolve_column,
)
from soil_stabilization.model_training import fit_models
from soil_stabilization.model_evaluation import metrics_frame, measured_prediction_frame
from soil_stabilization.prediction import predict_responses, simplified_allowable_pressure
from soil_stabilization.explainability import calculate_local_shap
from soil_stabilization.optimization import search_pareto_candidates
from soil_stabilization.experimental_design import generate_screening_plan
from soil_stabilization.data_validation import evaluate_validation
from soil_stabilization.data_audit import summarize_dataset


st.set_page_config(page_title="Expansive Soil Stabilization DSS", layout="wide")
st.title("AI-Assisted Soil Stabilization Research DSS")
st.warning(
    "Research prototype only. The supplied dataset's experimental provenance has not been "
    "verified. Predictions and optimization results are not engineering approval or a "
    "substitute for laboratory tests and geotechnical design."
)

uploaded = st.sidebar.file_uploader("Use verified experimental CSV", type=["csv"])
try:
    raw_data, source_name = load_source(uploaded.getvalue() if uploaded else None)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

try:
    X_all, Y_all, groups, available_features, available_targets, group_column = prepare_training_data(raw_data)
except ValueError as error:
    st.error(str(error))
    st.stop()

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
    pred = predict_responses(models, row)
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
        allowable = simplified_allowable_pressure(
            pred["Cohesion_kPa"], pred["Phi_deg"], width, depth, gamma, fs
        )
        c1.metric("Simplified allowable pressure (kPa)", f"{allowable:.1f}")

with tabs[1]:
    st.subheader("Model comparison and held-out performance")
    st.write(
        "Candidate models are ranked by cross-validated MAE on the training portion. "
        "The displayed test metrics use a separate held-out portion. If batch or mixture "
        "IDs exist, rows from the same group are kept together."
    )
    for target in available_targets:
        st.markdown(f"**{TARGETS[target][0]} · selected: {reports[target]['best_model']}**")
        st.dataframe(metrics_frame(reports[target]), hide_index=True, use_container_width=True)
        st.dataframe(reports[target]["comparison"], hide_index=True, use_container_width=True)
        st.scatter_chart(measured_prediction_frame(reports[target]), x="Measured", y="Predicted")

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
            shap_frame = calculate_local_shap(models[explain_target], X_all, row, FEATURE_LABELS)
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
    mix_features = [name for name in ("Lime_%", "RHA_%", "PET_%") if name in available_features]
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
            for i, (name, key) in enumerate((("Lime", "Lime_%"), ("RHA", "RHA_%"), ("PET", "PET_%")))
        ]
        carbon_cols = st.columns(3)
        carbon = [
            carbon_cols[i].number_input(
                f"{name} factor (kg CO₂e/kg; optional)", min_value=0.0, value=0.0,
                step=0.1, key=f"carbon_{key}",
            )
            for i, (name, key) in enumerate((("Lime", "Lime_%"), ("RHA", "RHA_%"), ("PET", "PET_%")))
        ]
        c1, c2, c3 = st.columns(3)
        min_ucs = c1.number_input("Minimum UCS (kPa)", min_value=0.0, value=200.0, step=10.0)
        max_swell = c2.number_input("Maximum swell (%)", min_value=0.0, value=2.0, step=0.5)
        generations = c3.slider("Search generations", 20, 100, 40, 10)
        if st.button("Search Pareto candidates"):
            if not all(price > 0 for price in prices):
                st.error("Enter a positive unit price for lime, RHA, and PET before searching.")
            else:
                try:
                    result_frame, feasible_count = search_pareto_candidates(
                        X_all, row, models, prices, carbon, min_ucs, max_swell,
                        generations, TARGETS,
                    )
                    if result_frame is None:
                        st.warning("The search did not return candidates. Check the data range and thresholds.")
                    else:
                        st.success(
                            f"Returned {len(result_frame)} Pareto candidates; "
                            f"{feasible_count} meet both response thresholds."
                        )
                        st.dataframe(
                            result_frame.sort_values("Estimated additive cost (EGP/t dry soil)"),
                            hide_index=True, use_container_width=True,
                        )
                        st.download_button(
                            "Download Pareto candidates CSV",
                            result_frame.to_csv(index=False).encode("utf-8"),
                            "pareto_candidates.csv", "text/csv",
                        )
                        st.warning(
                            "These are model-generated candidates. Test selected mixes "
                            "independently in the laboratory before recommending them."
                        )
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
            plan = generate_screening_plan(
                X_all, available_features, available_targets, TARGETS, n_runs, replicates
            )
            st.dataframe(plan, hide_index=True, use_container_width=True)
            st.download_button(
                "Download experimental plan CSV",
                plan.to_csv(index=False).encode("utf-8"),
                "experimental_screening_plan.csv", "text/csv",
            )
            st.caption(
                "Confirm factor ranges, zero-dose controls, sample preparation, and test "
                "standards with the laboratory supervisor."
            )
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
            results = evaluate_validation(
                validation, available_features, available_targets, ALIASES,
                TARGETS, models, resolve_column,
            )
            if not results.empty:
                st.dataframe(results, hide_index=True, use_container_width=True)
            else:
                st.info("No measured response columns matched the trained outcomes.")
        except Exception as error:
            st.error(f"Validation file could not be evaluated: {error}")

with tabs[6]:
    st.subheader("Dataset audit and research readiness")
    input_summary, response_summary, correlation, duplicate_count = summarize_dataset(
        X_all, Y_all, available_features, available_targets
    )
    st.markdown("**Input ranges and distribution**")
    st.dataframe(input_summary, use_container_width=True)
    st.markdown("**Measured response distribution**")
    st.dataframe(response_summary, use_container_width=True)
    st.markdown("**Input–response correlation (screening only)**")
    st.dataframe(correlation, use_container_width=True)
    st.write(f"Duplicate input combinations: {duplicate_count}")
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
