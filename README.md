Product Return Risk Prediction – Minimal End‑to‑End Project
===========================================================

This file contains everything you need to implement a very basic version of the project:

Predicting Product Return Risk in Retail Using Classification

The goal is to build a simple binary classifier that predicts whether an order will be returned (1) or not (0) using information available at order time, and to expose it through:

- A training script
- A FastAPI backend for prediction
- A very simple Streamlit frontend

The code is intentionally minimal. You can copy the code blocks into the indicated files and run them directly.


1. Project Overview
-------------------

- Business problem  
  For each order, answer: Will this item or order be returned? This helps the retailer understand which products, customers, or patterns lead to high return rates and take early action.

- Machine learning problem  
  Supervised binary classification with input features available at order time (product, price, discounts, customer info, context) and target label:
  - 1 for returned or cancelled
  - 0 for not returned

- Tech stack  
  - Python
  - Scikit‑learn for the model
  - FastAPI for backend inference
  - Streamlit for a simple UI

- Datasets  
  - Export_Product_Return_Data.csv as the main training data
  - model_prediction_val.csv can be generated later for validation reporting


2. Assumptions About the Dataset
--------------------------------

Assume Export_Product_Return_Data.csv has at least these columns:

- timestamp or order_datetime
- age
- gender
- state
- category
- brand
- quantity
- price
- discount
- rating
- returned

The last column returned is assumed to be:

- "Yes" if the order was returned
- "No" if the order was not returned

If your file uses a different column name for the target, update the variable target_column in the training code accordingly.


3. Minimal Training Script (train_model.py)
-------------------------------------------

Create a file named train_model.py in the project root or in src and paste this code into it. This script:

- Loads Export_Product_Return_Data.csv
- Builds a simple preprocessing pipeline
- Trains a Logistic Regression classifier
- Prints basic metrics
- Saves the trained model pipeline to models/model.pkl

```python
import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    target_column = "returned"
    df = df.dropna(subset=[target_column])
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_and_save_model():
    csv_path = os.path.join("data", "raw", "Export_Product_Return_Data.csv")
    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    model = LogisticRegression(max_iter=500)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, os.path.join("models", "model.pkl"))
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    train_and_save_model()
```


4. Minimal FastAPI Backend (backend_app.py)
------------------------------------------

Create a file named backend_app.py in the backend folder and paste this code into it. This backend:

- Loads the trained model from models/model.pkl
- Exposes a single POST /predict endpoint
- Receives order features
- Returns predicted label and probability

The feature names in OrderFeatures must match the columns used during training, except for the returned target column.

```python
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


model = joblib.load("models/model.pkl")


class OrderFeatures(BaseModel):
    timestamp: str
    age: int
    gender: str
    state: str
    category: str
    brand: str
    quantity: int
    price: float
    discount: float
    rating: float


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: OrderFeatures):
    data = pd.DataFrame([features.dict()])
    proba = model.predict_proba(data)[0][1]
    label = int(proba >= 0.5)
    return {"prediction": label, "probability": float(proba)}
```

Run this backend with:

uvicorn backend_app:app --reload

from the backend directory or adjust the import path accordingly.


5. Minimal Streamlit Frontend (frontend_app.py)
----------------------------------------------

Create a file named frontend_app.py in the frontend folder and paste this code into it. This frontend:

- Renders a simple form for a single order
- Sends the data to the FastAPI backend
- Displays the prediction and probability

```python
import streamlit as st
import requests


st.title("Product Return Risk Prediction")


timestamp = st.text_input("Order Date and Time", value="2025-04-11 21:36:09")
age = st.number_input("Customer Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
state = st.text_input("State", value="Arunachal Pradesh")
category = st.text_input("Category", value="Formal Wear")
brand = st.text_input("Brand", value="Puma")
quantity = st.number_input("Quantity", min_value=1, value=1)
price = st.number_input("Price", min_value=0.0, value=933.0)
discount = st.number_input("Discount", min_value=0.0, value=30.0)
rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=2.5)


if st.button("Predict Return Risk"):
    payload = {
        "timestamp": timestamp,
        "age": int(age),
        "gender": gender,
        "state": state,
        "category": category,
        "brand": brand,
        "quantity": int(quantity),
        "price": float(price),
        "discount": float(discount),
        "rating": float(rating),
    }

    response = requests.post("http://localhost:8000/predict", json=payload)

    if response.status_code == 200:
        data = response.json()
        label = data["prediction"]
        probability = data["probability"]
        if label == 1:
            st.success(f"Prediction: This order is likely to be returned ({probability:.2f})")
        else:
            st.info(f"Prediction: This order is unlikely to be returned ({probability:.2f})")
    else:
        st.error("Error calling prediction API")
```

Run this frontend with:

streamlit run frontend_app.py

Make sure the FastAPI backend is running at http://localhost:8000.


6. Minimal Requirements
-----------------------

You can install required packages with:

```bash
pip install pandas scikit-learn joblib fastapi uvicorn streamlit requests
```


7. End‑to‑End Workflow Summary
------------------------------

- Step 1  
  Place Export_Product_Return_Data.csv under:

  data/raw/Export_Product_Return_Data.csv

- Step 2  
  Train the model:

  - Run train_model.py
  - This produces models/model.pkl

- Step 3  
  Start the FastAPI backend:

  - Move to the backend folder if needed
  - Run uvicorn backend_app:app --reload

- Step 4  
  Start the Streamlit frontend:

  - Move to the frontend folder if needed
  - Run streamlit run frontend_app.py

- Step 5  
  Use the Streamlit UI to send new order‑like inputs and get:

  - Binary prediction: 1 returned, 0 not returned
  - Return risk probability

