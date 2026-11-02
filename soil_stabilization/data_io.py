"""Dataset schema, column matching, and CSV loading helpers."""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "soil_stabilization_dataset.csv"
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
    return pd.read_csv(DATA_FILE), str(DATA_FILE)


def prepare_training_data(raw_data):
    """Resolve known columns and prepare complete model inputs and measured targets."""
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
    group_column = next((col for col in GROUP_COLUMNS if col in raw_data.columns), None)

    if len(available_features) < 3:
        raise ValueError("The dataset must include at least three recognized input columns.")
    if not available_targets:
        raise ValueError("No recognized measured response columns were found.")

    X = raw_data[[feature_columns[key] for key in available_features]].copy()
    X.columns = available_features
    Y = pd.DataFrame(index=raw_data.index)
    for key, source_col in target_columns.items():
        if source_col is not None:
            Y[key] = pd.to_numeric(raw_data[source_col], errors="coerce")
    for column in available_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    valid_x = X.notna().all(axis=1)
    X = X.loc[valid_x].reset_index(drop=True)
    Y = Y.loc[valid_x].reset_index(drop=True)
    groups = raw_data.loc[valid_x, group_column].reset_index(drop=True) if group_column else None
    target_counts = {key: int(Y[key].notna().sum()) for key in TARGETS if key in Y}
    available_targets = [key for key in TARGETS if target_counts.get(key, 0) >= 30]
    Y = Y[available_targets]

    if not available_targets:
        raise ValueError("No response has at least 30 measured rows. Add experimental results before training.")
    if len(X) < 30:
        raise ValueError("At least 30 rows with complete model inputs are required.")
    return X, Y, groups, available_features, available_targets, group_column
