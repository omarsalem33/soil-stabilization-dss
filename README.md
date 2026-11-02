# Expansive Soil Stabilization Research DSS

A Streamlit research prototype for predicting soil-stabilization responses, comparing regression models, explaining predictions with SHAP, exploring mixture trade-offs, generating a laboratory screening plan, and checking predictions against independent laboratory measurements.

> **Research use only:** The current dataset's experimental provenance has not been verified. Predictions and optimized candidates are not engineering approval. Confirm data sources, test standards, and results with the laboratory supervisor and geotechnical engineer.

## 1. Project structure

```text
app.py                         Streamlit entry point and page layout
requirements.txt               Python dependencies
data/soil_stabilization_dataset.csv  Default dataset
soil_stabilization/            Reusable application logic
  data_io.py                   Dataset schema, column matching, CSV loading
  model_training.py            Candidate models and model fitting
  model_evaluation.py          Cross-validation ranking and held-out metrics
  prediction.py                Predictions and exploratory footing estimate
  explainability.py            SHAP explanations
  optimization.py              Pareto mixture search
  experimental_design.py       Laboratory screening plan generation
  data_validation.py           Independent validation calculations
  data_audit.py                Dataset audit summaries
templates/                     Excel data-entry template
docs/                          User and project documentation
```

Keep `app.py` and `requirements.txt` in the project root for simple local and cloud deployment. The app uses the bundled CSV only when no training file is uploaded. The Excel template is a reference for preparing data; upload CSV UTF-8 files to the app.

## 2. Install and launch on Windows

Open PowerShell and change to the project folder:

```powershell
cd "D:\Expansive Soil Stabilization Research DSS"
```

If the project virtual environment does not exist yet, create it:

```powershell
py -m venv .venv
```

Install or update the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit normally opens the app in your browser at `http://localhost:8501`. If it does not open automatically, copy that address into your browser. Keep the PowerShell window open while using the app. To stop it, press **Ctrl+C** in that window.

If PowerShell says it cannot find Python, install Python 3.11 or later (Python 3.12 is a good default) and make sure the `py` launcher is available, then repeat the setup steps.

## 3. Prepare the experimental data

The upload control accepts **CSV files**, not XLSX workbooks.

1. Open the supplied Excel template.
2. Enter one tested specimen per row in **Experimental Data**. Keep the first row of that sheet as the column headers; do not insert a title row above it.
3. Use numeric values in measurement columns. Keep units in the headers/column guide, not in the cells.
4. Leave a response blank when that test was not run. Do not enter zero to represent a missing test.
5. Keep a consistent `Mixture_ID` or `Batch_ID` for specimens that belong to the same mix or batch.
6. In Excel, select the **Experimental Data** worksheet and choose **File → Save As → CSV UTF-8 (Comma delimited)**.
7. In the app's sidebar, upload that CSV using **Use verified experimental CSV**.

Each input used by the model needs a numeric value for every row retained for that model. The app activates an input column when it contains at least 30 numeric values; blank optional input columns are ignored. At least 30 rows with complete selected inputs are required. Each response is trained separately and needs at least 30 measured input-response pairs.

The current project CSV contains eight input variables and four responses: UCS, swell percentage, cohesion, and friction angle. Other outcomes such as CBR, settlement, swell pressure, measured bearing capacity, and compressibility are modeled only after measured columns are added. The app does not create missing laboratory results.

## 4. Use each application tab

### Prediction & design

Set the mixture and soil inputs in the sidebar. The app displays predictions for outcomes that have enough measured data. If cohesion and friction angle are available, it also shows a simplified exploratory footing-pressure calculation with editable footing assumptions.

The displayed approximate error band is derived from the 90th percentile of absolute errors on the held-out test data. It is an empirical guide, not a guaranteed prediction interval.

### Model evaluation

See the candidate-model comparison for each response and the held-out MAE, RMSE, and R². Five-fold cross-validation on the training portion is used to select the model. If there are at least five groups in a recognized grouping column, rows from the same group are kept together during the train/test split.

### SHAP explanation

Choose a response and select **Calculate SHAP values**. The chart shows how each current input contributed to the selected model's prediction relative to a small background sample. SHAP describes model behavior; it does not prove causation.

### Multi-objective optimization

Optimization is available when the data include at least 30 measured UCS and swell pairs and the Lime, RHA, and PET input columns.

1. Enter positive unit prices for lime, RHA, and PET in EGP/kg.
2. Optionally enter an emission factor for all three materials to include the carbon proxy.
3. Set the minimum UCS and maximum swell limits.
4. Select **Search Pareto candidates**.
5. Review or download the candidate CSV.

The NSGA-II search varies lime, RHA, and PET within the ranges represented in the data. Fiber length and other inputs stay fixed at the current sidebar values. The cost calculation assumes additive percentages are by dry-soil mass. The carbon metric is an additive-material proxy, not a full life-cycle assessment. Test shortlisted candidates independently in the laboratory.

### Experimental plan

Choose the number of mixture combinations and replicates, then select **Generate screening plan**. The app creates a Latin-hypercube screening plan with untreated/single-additive controls, randomized run order, and blank response fields. Download the CSV and confirm factor ranges, controls, specimen preparation, and applicable test standards with the laboratory supervisor before running the tests.

### Independent validation

Use this tab only for mixtures tested independently after the training dataset/model has been established.

1. Fill the **Validation Data** worksheet with the inputs and actual laboratory responses for the independent specimens.
2. Save that worksheet as a separate CSV file.
3. Upload it using **Independent validation CSV**.
4. Review the sample count, MAE, RMSE, R², and prediction bias.

Do not include these same validation specimens in the training CSV; otherwise the validation is no longer independent. The validation sample count can be smaller than the 30-row training threshold, but more independent specimens give a more informative check.

### Data audit

Review input ranges, response distributions, input-response correlations, duplicate input combinations, and grouping-field availability. Investigate unusual values and document any exclusions before modeling.

## 5. Model and dataset notes

- Candidate models include linear and polynomial baselines, decision tree, random forest, SVR, ANN/MLP, Gaussian process, and XGBoost. Extra Trees is included for datasets with at least 60 rows.
- Models are compared separately for each outcome. Missing a response for one specimen does not remove that specimen from other response models, provided the model inputs are complete.
- If the dataset has no usable source, batch, or mixture identifier, the app falls back to row-level train/test splitting. Replicate leakage may then make evaluation look more optimistic than it is.
- The app reports what can be learned from the columns provided. It cannot run laboratory tests, determine material cost/emissions by itself, or approve a foundation design.
- The existing project's data must be checked for provenance and laboratory method before results are used in a thesis or shared as engineering recommendations.

## 6. Troubleshooting

**The app reports that the dataset is missing.**  
Keep the default dataset at `data/soil_stabilization_dataset.csv`, or upload another CSV through the sidebar.

**The app reports missing input or response columns.**  
Use the canonical column names in the Excel template and Column Guide. A response needs at least 30 measured values paired with complete model inputs to be trained.

**The browser page is not available.**  
Check that the PowerShell process is still running. If port 8501 is already in use, stop the other Streamlit process or launch with a different port, for example:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

**SHAP or optimization reports a dependency error.**  
Reinstall the requirements from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
