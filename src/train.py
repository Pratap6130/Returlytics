from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "data.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "training_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.txt"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "return_status" in df.columns:
        df["return_status"] = df["return_status"].astype(str).str.strip().str.lower()

    if "discount_applied" in df.columns and "product_price" in df.columns:
        denom = df["product_price"].replace(0, np.finfo(float).eps)
        df["discount_pct"] = df["discount_applied"] / denom

    if "product_price" in df.columns and "order_quantity" in df.columns:
        df["order_value"] = df["product_price"] * df["order_quantity"]

    return df


def optimize_threshold(
    y_true: pd.Series,
    probs: np.ndarray,
    objective: str,
    fp_cost: float,
    fn_cost: float,
    min_recall: float = 0.0,
) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = float("-inf")

    for t in thresholds:
        preds = (probs >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)

        if min_recall > 0 and rec < min_recall:
            continue

        if objective == "f1":
            score = f1_score(y_true, preds)
        elif objective == "accuracy":
            score = accuracy_score(y_true, preds)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            expected_cost = (fp * fp_cost + fn * fn_cost) / len(y_true)
            score = -expected_cost

        if score > best_score:
            best_score = float(score)
            best_threshold = float(t)

    return best_threshold, best_score


def threshold_sweep(
    y_true: pd.Series,
    probs: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> list[dict]:
    rows: list[dict] = []
    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        accuracy = accuracy_score(y_true, preds)
        expected_cost = (fp * fp_cost + fn * fn_cost) / len(y_true)
        rows.append(
            {
                "threshold": float(round(t, 3)),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "expected_cost": float(expected_cost),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return rows


def evaluate_at_threshold(y_true: pd.Series, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    except ValueError:
        metrics["pr_auc"] = None

    return metrics


def train(
    optimize_for: str = "cost",
    fp_cost: float = 1.0,
    fn_cost: float = 3.0,
    random_state: int = 42,
    min_recall: float = 0.1,
):
    df = pd.read_csv(DATA_PATH)
    df = feature_engineering(df)

    if "return_status" not in df.columns:
        raise ValueError("return_status column missing in data.csv")

    mapping = {
        "returned": 1,
        "not returned": 0,
    }
    df["returned"] = df["return_status"].map(mapping)
    df = df.dropna(subset=["returned"])
    df["returned"] = df["returned"].astype(int)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    X = df.drop(columns=["return_status", "returned"])
    y = df["returned"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        random_state=random_state,
        stratify=y_trainval,
    )

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoder = TargetEncoder(cols=categorical_cols) if categorical_cols else None

    if encoder is not None:
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_valid_enc = encoder.transform(X_valid)
        X_test_enc = encoder.transform(X_test)
    else:
        X_train_enc = X_train
        X_valid_enc = X_valid
        X_test_enc = X_test

    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = (negatives / positives) if positives else 1.0

    model = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        min_child_weight=2,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train_enc,
        y_train,
        eval_set=[(X_valid_enc, y_valid)],
        verbose=False,
    )

    valid_probs = model.predict_proba(X_valid_enc)[:, 1]
    test_probs = model.predict_proba(X_test_enc)[:, 1]

    threshold_map: dict[str, float] = {}
    optimization_scores: dict[str, float] = {}

    for objective in ["accuracy", "f1", "cost"]:
        t, score = optimize_threshold(
            y_valid,
            valid_probs,
            objective=objective,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            min_recall=min_recall if objective == "cost" else 0.0,
        )
        threshold_map[objective] = t
        optimization_scores[objective] = score

    deployed_threshold = threshold_map[optimize_for]

    metrics_report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "row_count": int(len(df)),
            "positive_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
        },
        "threshold_optimization": {
            "objective": optimize_for,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "min_recall_for_cost": min_recall,
            "thresholds": threshold_map,
            "optimization_scores": optimization_scores,
            "deployed_threshold": deployed_threshold,
        },
        "validation_threshold_sweep": threshold_sweep(
            y_valid,
            valid_probs,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        ),
        "test_metrics": {
            "at_0.5": evaluate_at_threshold(y_test, test_probs, 0.5),
            "at_best_accuracy": evaluate_at_threshold(
                y_test, test_probs, threshold_map["accuracy"]
            ),
            "at_best_f1": evaluate_at_threshold(y_test, test_probs, threshold_map["f1"]),
            "at_deployed_threshold": evaluate_at_threshold(
                y_test, test_probs, deployed_threshold
            ),
        },
        "baseline": {
            "majority_class_accuracy": float((y_test == 0).mean()),
        },
    }

    feature_importance = sorted(
        zip(X_train_enc.columns.tolist(), model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    metrics_report["top_feature_importance"] = [
        {"feature": name, "importance": float(score)} for name, score in feature_importance
    ]

    final_package = {
        "model": model,
        "encoder": encoder,
        "threshold": deployed_threshold,
        "thresholds": threshold_map,
        "optimize_for": optimize_for,
        "feature_columns": X.columns.tolist(),
        "trained_at_utc": metrics_report["trained_at_utc"],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_package, MODEL_PATH)
    THRESHOLD_PATH.write_text(f"{deployed_threshold:.6f}")
    METRICS_PATH.write_text(json.dumps(metrics_report, indent=2))

    deployed_metrics = metrics_report["test_metrics"]["at_deployed_threshold"]
    print(f"Optimization objective: {optimize_for}")
    print(f"Deployed threshold: {deployed_threshold:.3f}")
    print(f"Test Accuracy: {deployed_metrics['accuracy']:.3f}")
    print(f"Test Precision: {deployed_metrics['precision']:.3f}")
    print(f"Test Recall: {deployed_metrics['recall']:.3f}")
    print(f"Test F1: {deployed_metrics['f1']:.3f}")
    print(f"ROC-AUC: {deployed_metrics['roc_auc']:.3f}")
    print(
        "Confusion Matrix (tn, fp, fn, tp): "
        f"({deployed_metrics['tn']}, {deployed_metrics['fp']}, "
        f"{deployed_metrics['fn']}, {deployed_metrics['tp']})"
    )
    print("Saved: models/model.pkl, models/threshold.txt, models/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train return-risk model")
    parser.add_argument(
        "--optimize-for",
        choices=["cost", "f1", "accuracy"],
        default="cost",
        help="Objective used to choose the deployed decision threshold",
    )
    parser.add_argument("--fp-cost", type=float, default=1.0, help="Cost of false positive")
    parser.add_argument("--fn-cost", type=float, default=3.0, help="Cost of false negative")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.1,
        help="Minimum recall constraint applied for cost optimization",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        optimize_for=args.optimize_for,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        random_state=args.random_state,
        min_recall=args.min_recall,
    )
