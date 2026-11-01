# Expansive Soil Stabilization Research DSS

## Purpose

This Streamlit application is a research prototype for an AI-assisted decision-support workflow for stabilizing expansive clay with lime, rice husk ash (RHA), and recycled PET fibers. It combines measured-data review, regression-model comparison, prediction, model explanation, mixture screening, and prospective validation.

The application does not certify a mixture or replace laboratory testing, geotechnical design, or review by the research supervisor. The provenance and laboratory basis of the project's current CSV dataset have not been verified.

## What the application does

### 1. Prediction and foundation calculation

- Reads a CSV selected in the sidebar, or the local `soil_stabilization_dataset.csv` when no upload is selected.
- Recognizes the input columns listed in the companion Excel template. Column matching is case-insensitive and also accepts a limited set of common aliases.
- Trains a separate model for each measured response that has at least 30 complete input-response pairs.
- Shows predictions only for responses represented in the uploaded data.
- When cohesion and friction angle are both available, offers a simplified shallow-foundation bearing-pressure calculation. The user can set footing width, embedment depth, soil unit weight, and factor of safety. The calculation omits several design conditions and is for exploration only.
- Shows an approximate error band based on the 90th percentile of absolute errors on the held-out test set. This is an empirical indication, not a guaranteed prediction interval.

### 2. Model comparison and evaluation

The candidate regressors include linear and polynomial baselines, a decision tree, random forest, support-vector regression, a multilayer perceptron, Gaussian process regression, and XGBoost when the package is installed. Extra Trees is included when the dataset has at least 60 usable rows.

For each response, the application:

1. Compares candidates by five-fold cross-validated mean absolute error on the training portion.
2. Evaluates the selected candidate on a separate 20% holdout set.
3. Reports MAE, RMSE, R², and the number of holdout rows.
4. Refits the selected model on all usable rows for the interactive prediction view.

If a recognized mixture, batch, source, or group identifier is supplied and has at least five distinct groups, the split keeps groups together. Otherwise, it uses a row-level split. Include `Mixture_ID` or `Batch_ID` when multiple specimens belong to the same mixture or batch.

### 3. SHAP explanation

The SHAP tab calculates a local explanation for the selected response and current input row. It displays signed SHAP contributions relative to a small background sample from the loaded data. SHAP explains how the fitted model formed a prediction; it does not establish that an input caused a soil response.

### 4. Multi-objective mixture search

The optimizer uses NSGA-II to search the observed ranges of lime, RHA, and PET percentages. Fiber length and other soil inputs stay fixed at the current prediction inputs. It requires at least 30 measured pairs for both UCS and swell, all three mixture inputs, and user-entered positive unit prices for lime, RHA, and PET.

The search returns Pareto candidates balancing:

- Higher predicted UCS
- Lower predicted swell
- Lower estimated additive cost

If the user enters emission factors for all three additives, the search also includes an additive-carbon proxy as an objective. Cost and carbon calculations assume additive percentages are fractions of dry-soil mass; unit prices are entered in EGP/kg and emission factors in kg CO₂e/kg. They do not constitute a full life-cycle assessment. Candidate mixtures must be tested independently before being recommended.

### 5. Experimental screening plan

The plan generator creates a Latin-hypercube screening set over the available lime, RHA, PET, and fiber-length ranges. It adds untreated and single-additive controls, repeats each mixture according to the selected replicate count, assigns run order, and leaves response cells blank for laboratory results. Review factor ranges, zero-dose controls, specimen preparation, and test standards with the laboratory supervisor before using the plan.

### 6. Independent validation

The validation tab accepts a separate CSV of optimized mixtures and measured laboratory results. It compares those observations with the already-fitted models and reports sample count, MAE, RMSE, R², and prediction bias. Keep validation mixtures out of the training data to preserve the independence of this assessment.

### 7. Data audit

The audit tab summarizes input ranges, response distributions, input-response correlations, duplicate input combinations, and whether a grouping field is available. Correlation is descriptive and should not be interpreted as causation.

## Upload data requirements

- The app currently accepts **CSV files only**. The Excel workbook supplied with this documentation is an editable template, not a directly uploadable file.
- In Excel, fill the `Experimental Data` sheet, then use **Save As → CSV UTF-8 (Comma delimited)**. Excel exports the active sheet only; select the intended sheet before saving.
- Use one row per tested specimen. Keep numeric measurements numeric; do not put units in data cells.
- An input column is activated when it has at least 30 numeric measurements. Leave optional input columns entirely blank if they were not recorded; partially measured input columns may reduce the usable row count because model inputs must be complete.
- Leave a response blank when that response was not measured for a specimen. The application trains each response on its own complete input-response pairs, so missing one outcome does not discard the specimen from every other outcome.
- At least 30 rows with complete input values are required, and each response needs at least 30 measured pairs to be modeled.
- Keep replicate or batch identifiers consistent across specimens. Add source, classification, method, standard, units, and replicate notes for traceability.
- The app recognizes the canonical column headers in the template and common aliases. Canonical headers are preferred.

## Canonical model inputs

| CSV column | Meaning / unit |
|---|---|
| `Lime_%` | Lime content, percent |
| `RHA_%` | Rice husk ash content, percent |
| `PET_%` | PET fiber content, percent |
| `Fiber_Length_mm` | PET fiber length, mm |
| `Curing_Days` | Curing period, days |
| `MDD_g_cm3` | Maximum dry density, g/cm³ |
| `OMC_percent` | Optimum moisture content, percent |
| `PI_percent` | Plasticity index, percent |
| `Fiber_Content_%` | Optional fiber-content input (`Cf` alias) |
| `Water_Content_%` | Optional water-content input (`w` alias) |

## Recognized response columns

| CSV column | Meaning / unit |
|---|---|
| `UCS_kPa` | Unconfined compressive strength, kPa |
| `Swell_Percent` | Swell percentage |
| `Swell_Pressure_kPa` | Swell pressure, kPa |
| `CBR_percent` | California Bearing Ratio, percent |
| `Settlement_mm` | Settlement, mm |
| `Bearing_Capacity_kPa` | Measured or independently derived bearing capacity, kPa |
| `Cohesion_kPa` | Cohesion, kPa |
| `Phi_deg` | Friction angle, degrees |
| `Compressibility_Index` | Compressibility index (dimensionless, if reported that way by the chosen method) |

Response columns are optional. A response is trained only when it has at least 30 measured rows with complete inputs. If a response is missing from the project CSV, the app will not fabricate it.

## Traceability and grouping columns

Use `Mixture_ID`, `Batch_ID`, `Specimen_Batch`, `Source_ID`, or `Group_ID` to help prevent related records from being split between training and test partitions. Other useful metadata columns include `Replicate_ID`, `Soil_Classification`, `Test_Method`, `Test_Standard`, `Test_Date`, `Data_Source`, and `Measurement_Notes`. Metadata columns are retained in the upload file but are not model inputs unless listed in the input schema.

## Current project-data coverage

The project CSV currently contains 300 rows, eight recognized input variables, and four response variables: UCS, swell percentage, cohesion, and friction angle. It does not currently provide measured CBR, settlement, swell pressure, or compressibility data, and it lacks the optional `Fiber_Content_%` and `Water_Content_%` inputs. The app can train those additional responses only after measured data are added.

The existing file has no accompanying source or laboratory-method documentation in the project folder. Confirm whether its records are measured, published, or illustrative before using model metrics, SHAP results, or optimized mixtures in a thesis or client recommendation.

## Dependencies and launch

The project `requirements.txt` lists Streamlit, NumPy, pandas, scikit-learn, XGBoost, SHAP, and pymoo. From the project directory, install the listed packages in the project environment and start the app with:

```text
streamlit run app.py
```

The DOE helper also uses SciPy, which is normally installed as a scikit-learn dependency. Confirm it is available in the project environment.

## Research tasks that remain outside the software

The application cannot perform material characterization, prepare specimens, run UCS/CBR/swell/shear/compressibility tests, determine environmental life-cycle factors, or independently approve foundation design. Those require laboratory measurements, documented material batches and test standards, validated cost/emission inputs, and engineering review. The software supports recording and analyzing such results once they are available.