# Return Risk Prediction — Interview Notes (End-to-End)

## 1) Problem Statement

Build an end-to-end ML product that predicts whether a product/order is likely to be returned.

- **Business objective:** Identify high-risk orders early and take proactive action.
- **ML objective:** Binary classification
  - `1` = returned
  - `0` = not returned
- **Product objective:** Deliver prediction through a simple UI + API with user auth and prediction history.

---

## 2) Final System Architecture

### Layers
1. **Data engineering & dataset build** (`src/prepare_from_archive.py`)
2. **Model training + threshold optimization** (`src/train.py`)
3. **Optional hyperparameter tuning** (`src/tune.py`)
4. **Inference API (FastAPI)** (`backend/main.py`)
5. **UI (Streamlit)** (`frontend/app.py`)
6. **Persistence (SQLite)** (`backend/database.py`)

### Runtime flow
1. User logs in/registers from Streamlit.
2. User submits prediction form.
3. Frontend sends JSON to FastAPI `/predict`.
4. Backend loads model + encoder, does same feature engineering, predicts probability.
5. Backend applies decision threshold and returns label (`Yes`/`No`) + risk level + recommendation.
6. Backend stores prediction event in SQLite with `userid`.
7. User opens History view; frontend calls `/history/{userid}` and shows table/charts.

---

## 3) Data Creation / Preparation Strategy

## Source
- Archive CSVs in `archive/` with Amazon-like product attributes.

## Processing script
- File: `src/prepare_from_archive.py`
- Output: `data/raw/data.csv`

## Key transformations
- Parse currency/price text to numeric (`discount_price`, `actual_price`).
- Parse rating counts to numeric.
- Build core fields:
  - `product_category`
  - `product_price` (prefer actual price, fallback discount price, median fallback)
  - `discount_applied = max(actual - discount, 0)`
- Add synthetic user/order features using fixed RNG seed `42`:
  - `order_quantity` sampled from `{1,2,3,4}` with probs `[0.62, 0.24, 0.10, 0.04]`
  - `user_age ~ N(33,10)` clipped `[18,70]`
  - `user_gender` with probs `[0.48, 0.48, 0.04]`
  - `payment_method` with probs `[0.42, 0.22, 0.28, 0.08]`
  - `shipping_method` with probs `[0.64, 0.30, 0.06]`

## Label generation approach (important in interview)
This dataset uses a **synthetic return label** for demo/training validation:

- Base return probability = `0.08`
- + `0.14` if `rating < 3.5`
- + `0.05` if `discount_ratio > 0.35`
- + `0.03` if `rating_count < 25`
- + `0.04` if category is one of `{fashion, clothing, shoes}`
- Final probability clipped to `[0.03, 0.6]`
- Sampled to create `return_status ∈ {returned, not returned}`

This is acceptable for prototype/demo, but in production we should use real return outcome labels.

---

## 4) Feature Engineering Used in Training & Inference

Implemented in both:
- `src/train.py` and
- `backend/main.py`

Steps:
1. Normalize column names (lowercase, underscores)
2. Create `discount_pct = discount_applied / product_price` (safe divide)
3. Create `order_value = product_price * order_quantity`

Reasoning:
- `discount_pct` captures relative discount effect better than absolute discount alone.
- `order_value` captures interaction between quantity and unit price.

---

## 5) Model Choice and Why

## Model used
- `XGBClassifier` (XGBoost)

## Why XGBoost here
- Handles non-linear patterns well.
- Strong tabular-data performance.
- Works with mixed engineered features.
- Supports class imbalance using `scale_pos_weight`.

## Categorical handling
- `TargetEncoder` from `category-encoders` for object columns.
- Fit on training split; transform validation/test/inference consistently.

---

## 6) Exact Training Configuration (from `src/train.py`)

### Split strategy
- Total rows: **50,000**
- Train/valid/test:
  - Train: **32,000**
  - Valid: **8,000**
  - Test: **10,000**
- Two-stage split:
  - 80/20 for trainval/test
  - then 80/20 inside trainval for train/valid
- `stratify=y`, `random_state=42`

### Class balance
- Positive rate: **0.1532**
- `scale_pos_weight = negatives / positives` (computed from training split)

### XGBoost hyperparameters (training script default)
- `n_estimators=800`
- `max_depth=6`
- `min_child_weight=2`
- `learning_rate=0.03`
- `subsample=0.85`
- `colsample_bytree=0.85`
- `reg_lambda=2.0`
- `reg_alpha=0.5`
- `gamma=0.1`
- `objective='binary:logistic'`
- `eval_metric='aucpr'`
- `early_stopping_rounds=50`
- `random_state=42`
- `n_jobs=-1`

---

## 7) Threshold Optimization Strategy (Important Interview Point)

Instead of fixed threshold 0.5 only, project optimizes threshold on validation set for 3 objectives:
- Accuracy
- F1
- **Cost** (deployed objective)

### Cost function
\[
\text{Expected Cost} = \frac{FP \cdot c_{fp} + FN \cdot c_{fn}}{N}
\]
with:
- `fp_cost = 1.0`
- `fn_cost = 3.0`
- `min_recall_for_cost = 0.1`

### Search space
- Threshold sweep from `0.05` to `0.95` (181 points)

### Selected thresholds (from `models/metrics.json`)
- Best accuracy threshold: **0.72**
- Best F1 threshold: **0.46**
- Best cost threshold: **0.585**
- **Deployed threshold:** `0.585` (because `optimize_for='cost'`)

---

## 8) Final Model Performance (from `models/metrics.json`)

## At deployed threshold `0.585`
- Accuracy: **0.7547**
- Precision: **0.2501**
- Recall: **0.3009**
- F1: **0.2732**
- ROC-AUC: **0.6220**
- PR-AUC: **0.2167**
- Confusion matrix:
  - TN = 7086
  - FP = 1382
  - FN = 1071
  - TP = 461

## Comparison points
- At threshold 0.5:
  - Better recall (**0.5313**) but much lower precision/cost behavior.
- At best-accuracy threshold 0.72:
  - Very high accuracy due to class imbalance but recall collapses to 0.

**Interview framing:** We intentionally trade some recall/precision based on a business-weighted error cost function rather than pure accuracy.

---

## 9) Tuning Method (`src/tune.py`)

## Approach
- Random search via `ParameterSampler`
- Trials: `n_trials=20`
- Objective still **cost-based** with same FP/FN costs and min recall

## Best tuned configuration (report only)
From `models/tuning_report.json`:
- `subsample=0.7`
- `scale_pos_weight=4.422358221134231`
- `reg_lambda=2.0`
- `reg_alpha=0.25`
- `n_estimators=700`
- `min_child_weight=6`
- `max_depth=3`
- `learning_rate=0.03`
- `gamma=0.05`
- `colsample_bytree=0.7`
- best threshold: **0.51**
- test expected cost: **0.4551**

Note: tuning script writes report and can optionally deploy best model using `--deploy-best`.

---

## 10) API Design (FastAPI)

File: `backend/main.py`

## Endpoints
1. `GET /health`
   - returns status + model_loaded
2. `POST /register`
   - creates user with bcrypt-hashed password
3. `POST /login`
   - verifies user/password via bcrypt
4. `POST /predict`
   - input: product/order/user features (+ optional userid)
   - output: prediction label, probability, threshold, risk level, recommendation
5. `GET /history/{userid}?limit=...`
   - returns prediction history for that user

## Request schema (`backend/schemas.py`)
`OrderFeatures`:
- `userid` (optional)
- `product_category`
- `product_price`
- `order_quantity`
- `user_age`
- `user_gender`
- `payment_method`
- `shipping_method`
- `discount_applied`

## Risk buckets and recommendation logic
- High: `prob >= max(threshold + 0.15, 0.75)`
- Medium: `prob >= threshold`
- Low: otherwise

This adds business interpretation layer beyond raw probability.

---

## 11) Database Design (SQLite)

File: `backend/database.py`
DB file: `backend/returns.db`

## Tables
### `users`
- `id` (PK)
- `name`
- `userid` (unique)
- `password_hash`

### `predictions`
- `id` (PK)
- `userid`
- `product_category`
- `product_price`
- `order_quantity`
- `user_age`
- `user_gender`
- `payment_method`
- `shipping_method`
- `discount_applied`
- `prediction`
- `probability`
- `created_at`

Extra:
- migration-safe helper `_ensure_column` ensures `userid` and `created_at` exist for older DBs.

---

## 12) Frontend Design (Streamlit)

File: `frontend/app.py`

## Main views
1. **Auth view**: Login/Register tabs
2. **Prediction view**: form + prediction response + risk/recommendation
3. **Dashboard view**: data visualizations generated from `data/raw/data.csv`
4. **History view**: user-specific prediction history table + metrics + charts

## Session state keys
- `logged_in`
- `user_name`
- `user_id`
- `view`

## Dashboard plots generated
- Return status count
- Average price by category
- Return rate by category
- Price distribution
- Return rate by payment method
- Return rate by shipping method
- Return rate by age band

---

## 13) Dependencies

## Backend (`backend/requirements.txt`)
- fastapi
- uvicorn
- pandas
- scikit-learn
- joblib
- pydantic
- xgboost
- category-encoders
- bcrypt

## Frontend (`frontend/requirements.txt`)
- streamlit
- requests
- pandas
- matplotlib

---

## 14) How to Run

From project root:

1. Activate env
```powershell
.\.venv-1\Scripts\Activate.ps1
```

2. Backend
```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

3. Frontend (new terminal)
```powershell
cd frontend
python -m streamlit run app.py --server.port 8501
```

4. Open
- Frontend: `http://127.0.0.1:8501`
- Health: `http://127.0.0.1:8000/health`

---

## 15) Key Interview Talking Points (Short Version)

1. Built full-stack ML product, not just notebook model.
2. Handled class imbalance and business-cost-driven thresholding.
3. Used target encoding + XGBoost for tabular mixed features.
4. Added operational layer: auth, persistence, prediction history, dashboard.
5. Added explainable business output (risk bucket + recommendation).
6. Added tuning pipeline and reproducible data generation script.

---

## 16) Limitations & Next Improvements (good to mention proactively)

1. Current labels are synthetic for prototype; production requires real return outcomes.
2. `/history/{userid}` currently trusts userid path (no JWT auth yet).
3. Need stronger MLOps: model registry, versioned datasets, CI/CD checks.
4. Add model explainability (SHAP) per prediction.
5. Add drift monitoring and periodic retraining strategy.
6. Add stricter validation on input ranges and categorical values.

---

## 17) “Why this threshold?” answer template

> We optimized decision threshold on validation set for expected business cost, where false negatives were weighted 3x false positives. Instead of defaulting to 0.5, we selected 0.585 because it minimized cost under a minimum recall constraint. This makes the model decision policy aligned with business impact, not just generic accuracy.

---

## 18) “Why XGBoost?” answer template

> For this tabular use case with mixed numeric/categorical engineered features, XGBoost provides strong non-linear performance and robust handling of imbalance with `scale_pos_weight`. Combined with target encoding and threshold optimization, it gave a practical, production-friendly baseline.
