# Startup valuation from market data

**Predicting pre-money valuation from the information available around a funding round.** I built the model, calibration layer, evaluation pipeline, and a document-analysis interface around it.

## The problem I worked on

My first valuation workflow relied on rules, comparable companies, and LLM-written reasoning. It was useful for explaining a case, but the number could move too much when a deck omitted revenue or traction. A late-stage company with strong investor signals could be priced like an early startup simply because the text did not contain a revenue figure.

I moved the numerical estimate into a supervised model trained on historical funding rounds. The LLM still extracts facts from decks and writes the investment narrative; the valuation itself comes from code and market data.

## Results

I evaluated the valuation stack on **5,657 USD-denominated rounds** in an organization-hash calibration holdout. The target and predictions are compared in `log10(USD)`, which is more useful here than raw-dollar error across seed and growth rounds.

| Model | R² on log10 valuation | MAE on log10 valuation | Typical error factor |
| --- | ---: | ---: | ---: |
| Basic linear regression | 0.699 | 0.440 | 2.75× |
| Enriched linear regression | 0.726 | 0.424 | 2.65× |
| CatBoost + residual calibration | **0.836** | **0.323** | **2.10×** |

The full evaluation uses the same rows for each comparison. The saved calibration artifact records the split, row counts, and holdout metrics; [`results/metrics.json`](results/metrics.json) contains the comparison values. I also compared the ML output with the earlier LLM valuation method on a stratified set of 101 companies: log-scale R² was **0.772 vs 0.415**, and **45.5% vs 34.7%** of estimates landed within 2× of the historical valuation.

During base-model selection, CatBoost with categorical and text features reached **0.771 R²**; a LightGBM plus TF-IDF alternative reached **0.767**, and a median-value baseline was below zero. These are the scores stored with the trained model artifact, separate from the calibration holdout above.

## How I got there

### 1. Make the target comparable

I used **pre-money valuation in USD** as the target, filtered non-positive and implausible values, and trained in log space. This kept billion-dollar rounds from dominating the loss while preserving a direct path back to dollars at inference time.

### 2. Build features that exist at decision time

The saved model uses **138 features**: funding stage and status, geography, industry, company description, timing, investor signals, and historical market priors. I deliberately kept the current round's money raised out of this model version. Historical priors describe what earlier companies in a category or stage raised; they are distinct from the amount raised in the round being valued.

The messy part was sparse and inconsistent source data. I normalized stage names, built text and categorical fields, derived time and geography signals, and made missing values explicit. The feature construction and inference mapping live in [`app/modular_rag/ml_valuation.py`](app/modular_rag/ml_valuation.py).

### 3. Fit a model, then fix systematic residuals

I tested CatBoost against LightGBM plus TF-IDF for mixed categorical, numeric, and text inputs. After inspecting errors by stage and predicted valuation range, I added a second CatBoost model to predict the residual in log space. Its training split is based on a hash of organization name so the same company does not appear on both sides of the calibration split. The training code is in [`app/tools/build_ml_residual_calibrator.py`](app/tools/build_ml_residual_calibrator.py).

### 4. Return a range rather than one unexplained number

The point estimate is converted back to USD. The range uses the 10th and 90th percentiles of residual errors, selected by stage and prediction bucket when enough observations are available. If a segment is too small, it falls back to stage-level or global errors. This produces an estimate with a traceable calibration segment instead of an arbitrary percentage band.

### 5. Connect prediction to the analysis workflow

The Streamlit application accepts PDF decks, extracts text, retrieves relevant documents through Postgres/Supabase, and produces an investment report. Numerical valuation methods are calculated in Python; the LLM handles extraction and explanation. [`app/modular_rag/multi_method_valuation.py`](app/modular_rag/multi_method_valuation.py) combines the ML market estimate with other methods where the input supports them.

## Code map

| Path | What it does |
| --- | --- |
| [`app/modular_rag/ml_valuation.py`](app/modular_rag/ml_valuation.py) | Feature mapping, model inference, residual calibration, prediction range |
| [`app/tools/build_ml_residual_calibrator.py`](app/tools/build_ml_residual_calibrator.py) | Organization-level calibration split and residual model training |
| [`app/tools/build_ml_valuation_calibration.py`](app/tools/build_ml_valuation_calibration.py) | Segment-level residual statistics and range calibration |
| [`app/modular_rag/ml_batch_eval.py`](app/modular_rag/ml_batch_eval.py) | R², log MAE, error factors, range coverage, stage breakdown |
| [`app/modular_rag/search.py`](app/modular_rag/search.py) | Hybrid retrieval and ranking for supporting evidence |
| [`app/app_modular.py`](app/app_modular.py) | Streamlit workflow for PDF intake, analysis, valuation, and benchmarking |

## Run a prediction

Use Python 3.10+ and install the dependencies in [`app/requirements.txt`](app/requirements.txt). The repository includes the trained model and calibration artifacts, so one-record inference does not need the private training CSV or API keys.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
python scripts/predict.py examples/startup.json
```

To evaluate against a CSV that contains known USD pre-money valuations and the corresponding features:

```bash
python scripts/evaluate.py path/to/valuation_rows.csv
```

For the PDF/RAG interface, copy `app/.env.example` to `app/.env`, add your own Gemini and Supabase credentials, apply the SQL files in `app/` to your database, and run `streamlit run app/app_modular.py`.

The original licensed data exports and local databases are excluded from this repository. The public files include the inference code, trained artifacts, evaluation logic, and aggregate results.
