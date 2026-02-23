# Return Risk Prediction

End-to-end ML project that predicts if an order is likely to be returned.

## Features

- Binary return-risk prediction (`Yes` / `No`) with probability.
- FastAPI backend with auth and prediction history.
- Streamlit frontend with:
  - Login / Register
  - Prediction form and result view
  - User history and dashboard charts
- SQLite persistence for users and predictions.
- Training pipeline with threshold optimization for business cost.

## Project Structure

```
backend/              # FastAPI app, schemas, SQLite helpers
frontend/             # Streamlit app
src/                  # data prep, training, tuning scripts
data/raw/             # generated training data
models/               # model.pkl, metrics.json, threshold.txt
plots/                # dashboard images
archive/              # source CSV archive
```

## Tech Stack

- Python
- FastAPI + Uvicorn
- Streamlit
- XGBoost + category-encoders + scikit-learn
- SQLite

## Local Setup

### 1) Activate virtual environment

```powershell
cd D:\HP\Downloads\Return_Risk_Prediction-main
.\.venv-1\Scripts\Activate.ps1
```

### 2) Install dependencies (one-time)

```powershell
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt
```

### 3) Start backend (Terminal 1)

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 4) Start frontend (Terminal 2)

```powershell
cd frontend
python -m streamlit run app.py --server.port 8501
```

### 5) Open app

- Frontend: http://127.0.0.1:8501
- Backend health: http://127.0.0.1:8000/health

## API Endpoints

### `GET /health`

Returns backend status and whether model is loaded.

### `POST /register`

Register user.

Request body:

```json
{
  "name": "John Doe",
  "userid": "john01",
  "password": "secret123"
}
```

### `POST /login`

Login user.

Request body:

```json
{
  "userid": "john01",
  "password": "secret123"
}
```

### `POST /predict`

Predict return risk and save prediction to DB.

Request body:

```json
{
  "userid": "john01",
  "product_category": "Clothing",
  "product_price": 999,
  "order_quantity": 2,
  "user_age": 29,
  "user_gender": "Male",
  "payment_method": "UPI",
  "shipping_method": "Standard",
  "discount_applied": 120
}
```

Response contains:

- `prediction_label`
- `probability`
- `decision_threshold`
- `risk_level`
- `recommendation`

### `GET /history/{userid}?limit=100`

Returns recent prediction history for a user.

## Training / Rebuild

If you want to regenerate dataset and retrain model:

```powershell
python src\prepare_from_archive.py
python src\train.py
```

Outputs:

- `data/raw/data.csv`
- `models/model.pkl`
- `models/threshold.txt`
- `models/metrics.json`

## Deployment (Recommended)

- Deploy backend and frontend as separate services.
- Put a reverse proxy (Nginx) with HTTPS in front.
- Use environment variables for production URLs.
- Persist DB (`backend/returns.db`) using a mounted volume.
- Optionally move from SQLite to PostgreSQL for higher concurrency.

## Troubleshooting

- **`http://_vscodecontentref_` error in terminal:**
  You copied a markdown link by mistake. Use:
  `python -m streamlit run app.py --server.port 8501`

- **Frontend opens but prediction fails:**
  Ensure backend is running on port `8000`.

- **Model not loaded (`/health` shows false):**
  Re-run training (`python src\train.py`) so `models/model.pkl` exists.

## Interview Notes

Detailed interview-ready explanation is available in `INTERVIEW_NOTES.md`.

