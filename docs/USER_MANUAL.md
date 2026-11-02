# User Manual: Expansive Soil Stabilization DSS

## 1. Purpose

This application is a research decision-support tool for studying expansive-soil stabilization mixtures. It uses laboratory measurements to train candidate machine-learning models, estimate selected engineering responses for new mixtures, explain model behavior, explore mixture trade-offs, and compare predictions with independent laboratory validation data.

The application supports a research workflow; it does not replace laboratory testing, geotechnical design, or engineering approval. Use verified experimental measurements and preserve the source and test method for every record.

## 2. Research workflow at a glance

1. Prepare and quality-check the experimental training CSV.
2. Upload it in **Use verified experimental CSV**.
3. Review **Data audit** and correct data issues in the source file.
4. Review **Model evaluation** for each response.
5. Enter a candidate mixture in **Prediction & design**.
6. Review **SHAP explanation** to understand model behavior.
7. Optionally search candidate mixtures in **Multi-objective optimization**.
8. Plan additional laboratory work in **Experimental plan**.
9. Test selected mixtures in the laboratory, then upload those measured results in **Independent validation**.
10. Report the validation metrics and limitations with the research results.

## Project code layout

`app.py` is the Streamlit entry point and builds the user interface. The `soil_stabilization/` package holds the reusable logic: `data_io.py` prepares data, `model_training.py` creates and fits candidate models, `model_evaluation.py` ranks models and calculates holdout metrics, and the other modules handle prediction, explanations, mixture optimization, experiment planning, independent validation, and data audits. The default data CSV is in `data/`, the Excel template is in `templates/`, and this manual and the project documentation are in `docs/`.

## 3. Download, install, and run locally

The instructions below are written for the public GitHub repository `omarsalem33/soil-stabilization-dss`. Each user downloads a copy to their own computer; no developer-specific folder path is required.

### Prerequisites

- Install Git.
- Install Python 3.11 or later (Python 3.12 is a good default). On Windows, the Python launcher `py` should be available. On macOS/Linux, use `python3`.
- A terminal with Git and Python installed.

### Step 1: Copy this repository from GitHub

Install Git, then open PowerShell (Windows) or Terminal (macOS/Linux) in the parent folder where you want to save the project. Run:

```text
git clone https://github.com/omarsalem33/soil-stabilization-dss.git
cd soil-stabilization-dss
```

The repository is public, so no invitation is needed to clone it. Alternatively, on GitHub choose **Code > Download ZIP**, extract the ZIP, and open a terminal in the extracted `soil-stabilization-dss` folder. Git is recommended if you want to receive future updates with `git pull`.

The repository keeps `app.py` and `requirements.txt` in its top-level folder. Reusable processing code is under `soil_stabilization/`, the bundled default dataset is under `data/`, the Excel template is under `templates/`, and documentation is under `docs/`. The Excel template is not uploaded directly to the app: export the filled data sheet as CSV UTF-8 first. `.venv` and `.idea` are local environment/editor folders and are not required in the repository.

### Step 2: Create an isolated Python environment and install dependencies

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS/Linux terminal:**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### Step 3: Start the application

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS/Linux terminal:**

```bash
.venv/bin/python -m streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`. To stop the application, focus the terminal and press `Ctrl+C`.

### Get future updates

To download the latest committed version after the initial clone, open a terminal in the project folder and run:

```text
git pull
```


The app accepts **CSV files** for data upload. If working in Excel, complete the workbook and export each data sheet separately as **CSV UTF-8**. Check that the exported file has the intended sheet's columns and rows before uploading it.

## 4. Prepare the experimental training data

### 4.1 Record structure

- Use one row per independently recorded specimen or experimental observation.
- Use one column per variable.
- Record measured response values only in response columns. Do not put model predictions in the training response columns.
- Keep units consistent within each column and state them in the column name or an accompanying data dictionary.
- Keep a clean copy of the original laboratory data and record the source, test standard, date, and any preprocessing performed.
- Do not mix different test methods or specimen conditions without recording the distinction in columns that can be reviewed and, where suitable, used as grouping information.

### 4.2 Common input columns

The app recognizes common names and aliases for these mixture and soil variables:

| Variable | Example column name | Typical unit |
|---|---|---|
| Lime content | `Lime_%` | % of dry soil mass |
| Rice husk ash content | `RHA_%` | % of dry soil mass |
| PET fiber content | `PET_%` or `Fiber_Content_%` | % of dry soil mass |
| Fiber length | `Fiber_Length_mm` | mm |
| Curing duration | `Curing_Days` | days |
| Maximum dry density | `MDD_g_cm3` | g/cm³ |
| Optimum moisture content | `OMC_percent` | % |
| Plasticity index | `PI_percent` | % |
| Water content, if recorded | `Water_Content_%` | % |

Use the column guide/template supplied with the project if available. Optional columns can be left blank, but remove irrelevant empty columns if they cause confusion. The app needs enough complete input and target records for the response being modelled.

### 4.3 Common response columns

Depending on the measured data uploaded, the app can recognize responses such as:

- Unconfined compressive strength (UCS), e.g. `UCS_kPa`.
- Swell percentage, e.g. `Swell_Percent`.
- Swell pressure.
- California Bearing Ratio (CBR).
- Settlement.
- Cohesion, e.g. `Cohesion_kPa`.
- Friction angle, e.g. `Phi_deg`.
- Compressibility indicators.

Only responses present in the uploaded CSV and supported by sufficient complete records can be modelled. The current implementation requires at least 30 valid input/response rows for a target. This is a software minimum, not a guarantee that the dataset is scientifically adequate. More representative, well-distributed experiments are usually needed for reliable conclusions.

### 4.4 Grouping and data leakage

If the dataset contains a valid group identifier such as `Mixture_ID`, `Batch_ID`, `Specimen_Batch`, `Source_ID`, or `Group_ID`, use it to identify related specimens or batches. The app can use groups for splitting where enough groups are available. This helps keep related observations together during evaluation. Do not create arbitrary group labels just to force a grouped split.

## 5. Upload and audit the training data

1. In the left sidebar, choose **Use verified experimental CSV** and upload the experimental CSV.
2. Confirm that the expected file is selected.
3. Open **Data audit**.
4. Review detected input and response columns, missing values, invalid numeric values, and data ranges.
5. Fix issues in the original CSV and upload the corrected file again.

The app trains models from the uploaded file in the running app session. Keep your own secure copy of the data and results; do not assume uploaded files or generated outputs are permanently stored by the service.

## 6. Evaluate the models

Open **Model evaluation** after uploading the training data. For each response, review the available candidate models and the reported metrics:

- **MAE**: average absolute prediction error, in the response's unit; lower is better.
- **RMSE**: error measure that penalizes larger errors more; lower is better.
- **R²**: proportion of response variation represented by the model on the evaluation data; closer to 1 is generally better, but it can be negative when predictions perform poorly.

The app compares candidates using cross-validation on the training portion and reports performance on a held-out split. The final prediction model is fitted using the available data after evaluation. Do not select a model from R² alone; consider MAE/RMSE, sample size, data coverage, residual patterns, and scientific plausibility.

A good score on one dataset does not establish that the model will generalize to new soils, sources, laboratories, test methods, or mixture ranges.

## 7. Predict a candidate mixture

1. Open **Prediction & design**.
2. Enter the required soil and mixture inputs using the same units and definitions as the training data.
3. Review the predicted values for each response with a trained model.
4. Note any displayed error estimate and its limits. The empirical error band is based on held-out errors and is not a guaranteed prediction interval.
5. Save the inputs and predictions with a run identifier if they will be used in the research record.

Predictions are most defensible for candidate inputs within the range represented by the laboratory training data. Treat extrapolation outside that range as high uncertainty and verify it experimentally.

## 8. Explain model behavior

Open **SHAP explanation** to inspect how input features contribute to an individual prediction or model behavior. Interpret these values as explanations of the fitted model, not as proof of physical causation. Compare explanations with soil mechanics and the experimental design before drawing conclusions.

## 9. Explore mixture trade-offs

Open **Multi-objective optimization** to search for candidate mixtures balancing:

- Higher predicted UCS, subject to the minimum UCS setting.
- Lower predicted swelling, subject to the maximum swell setting.
- Lower estimated additive cost, using the entered prices and stated dry-soil-mass assumption.
- An optional carbon objective when all required emission factors are supplied.

Enter prices and emission factors only when they are supported by a traceable source and appropriate boundary assumptions. The optimization searches within the measured mixture range and depends on the trained prediction models. A Pareto candidate is a model-generated proposal, not an experimentally proven optimum. If no candidate is returned, review the data range, trained responses, and feasibility thresholds.

## 10. Plan follow-up experiments

Use **Experimental plan** to generate a proposed set of experiments within selected variable ranges. The plan can help organize candidate mixtures, controls, replication, and run order. Before laboratory execution:

- Confirm feasibility and safety with the laboratory team.
- Include appropriate untreated/control mixtures and replicates based on the research protocol.
- Randomize execution order where practical and document deviations.
- Record specimen preparation, curing, test standards, and measured results.

The generated plan is a planning aid. It does not replace a statistically justified experimental design or laboratory procedure.

## 11. Run independent validation

Validation must use laboratory observations that were not used to train or choose the final model. Keep the validation data separate until the model and analysis choices are fixed.

1. Prepare a separate validation CSV with the same input definitions and units as the training data.
2. Include the measured response columns for the outputs being validated.
3. Open **Independent validation** and upload the validation CSV in that page's uploader.
4. Review predicted-versus-measured comparisons and error metrics for each available response.
5. Report dataset size, soil/source and test conditions, validation metrics, and any limitations.

Do not upload the validation CSV as the training file if the aim is to report an independent validation result. If validation data are later added to training, they are no longer an independent test set for subsequent evaluation.

## 12. Relating app outputs to research requirements

| Research activity | App feature | Evidence to retain/report |
|---|---|---|
| Build an experimental dataset | Training CSV and Data audit | Data dictionary, source records, test standards, row counts, quality checks |
| Predict engineering responses | Prediction & design | Model, input mixture, predicted response, units, applicable data range |
| Compare ML approaches | Model evaluation | Split method, cross-validation method, MAE, RMSE, R², sample counts |
| Interpret model behavior | SHAP explanation | Explanation plots/values and careful non-causal interpretation |
| Identify mixture trade-offs | Multi-objective optimization | Objective settings, cost/carbon assumptions, Pareto candidate list |
| Propose new experiments | Experimental plan | Planned runs, controls, replicates, randomization and actual lab records |
| Test generalization on new measurements | Independent validation | Separate validation protocol and predicted-versus-measured metrics |
| Assess prediction uncertainty | Holdout error estimate and independent validation | Error distribution, validation results, and limitations; do not claim guaranteed intervals |

The software provides workflow support for these activities. The research requirement is achieved only when the user supplies valid experimental data, executes the laboratory program, performs independent validation, and documents the methods and results.

## 13. Troubleshooting

- **No response/model appears:** confirm the response column is recognized, numeric, and has at least 30 complete input/response rows.
- **A column is not detected:** use a column name from the template/guide or a recognized alias; verify spelling and numeric formatting.
- **No Pareto candidates:** check that UCS and swell models are available, entered thresholds are feasible, and the search limits overlap the measured data range.
- **Upload or parsing error:** export as CSV UTF-8, ensure a single header row, remove merged cells/notes/units rows, and check decimal separators.
- **Unexpectedly strong or weak metrics:** inspect duplicated specimens, related batches across splits, data leakage, outliers, mixed test standards, and sample coverage.
- **The app is slow or stops:** try a smaller CSV, fewer optimization generations, and a restart. Hosted free services may impose resource limits or sleep when unused.

## 14. Research and engineering limitations

- Model results depend on the quality, quantity, and coverage of experimental measurements.
- Predictions and optimization results are estimates and require laboratory confirmation.
- Cost and carbon outputs depend on user-supplied prices and emission factors and their assumptions.
- A simplified footing estimate, if shown, is illustrative and is not a design calculation.
- The app does not certify a mixture, verify source data, or replace professional engineering judgment.
- Preserve code version, dataset version, model settings, and analysis outputs for reproducibility.




