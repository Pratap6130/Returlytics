from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split
from xgboost import XGBClassifier

from train import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
    evaluate_at_threshold,
    feature_engineering,
    optimize_threshold,
)


BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}


def safe_auc(y_true: pd.Series, probs: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return None


def safe_pr_auc(y_true: pd.Series, probs: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, probs))
    except ValueError:
        return None


def build_search_space(scale_pos_weight: float) -> dict:
    return {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [3, 4, 5, 6, 7],
        "min_child_weight": [1, 2, 4, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [1.0, 2.0, 4.0, 8.0],
        "reg_alpha": [0.0, 0.25, 0.5, 1.0],
        "gamma": [0.0, 0.05, 0.1, 0.2],
        "scale_pos_weight": [
            max(0.8, scale_pos_weight * 0.8),
            scale_pos_weight,
            scale_pos_weight * 1.2,
        ],
    }


def tune(
    n_trials: int = 20,
    fp_cost: float = 1.0,
    fn_cost: float = 3.0,
    random_state: int = 42,
    min_recall: float = 0.1,
    deploy_best: bool = False,
) -> dict:
    df = pd.read_csv(DATA_PATH)
    df = feature_engineering(df)

    if "return_status" not in df.columns:
        raise ValueError("return_status column missing in data.csv")

    mapping = {"returned": 1, "not returned": 0}
    df["returned"] = df["return_status"].map(mapping)
    df = df.dropna(subset=["returned"])
    df["returned"] = df["returned"].astype(int)

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
    base_spw = (negatives / positives) if positives else 1.0

    search_space = build_search_space(base_spw)
    sampled_params = list(
        ParameterSampler(search_space, n_iter=n_trials, random_state=random_state)
    )

    best_trial: dict | None = None
    trial_rows: list[dict] = []

    for idx, params in enumerate(sampled_params, start=1):
        model = XGBClassifier(random_state=random_state, **BASE_PARAMS, **params)
        model.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_valid_enc, y_valid)],
            verbose=False,
        )

        valid_probs = model.predict_proba(X_valid_enc)[:, 1]
        test_probs = model.predict_proba(X_test_enc)[:, 1]

        best_threshold, best_score = optimize_threshold(
            y_valid,
            valid_probs,
            objective="cost",
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            min_recall=min_recall,
        )

        test_metrics = evaluate_at_threshold(y_test, test_probs, best_threshold)
        expected_cost_test = (
            test_metrics["fp"] * fp_cost + test_metrics["fn"] * fn_cost
        ) / len(y_test)

        trial = {
            "trial": idx,
            "params": params,
            "valid_best_threshold": float(best_threshold),
            "valid_cost_score": float(best_score),
            "valid_roc_auc": safe_auc(y_valid, valid_probs),
            "valid_pr_auc": safe_pr_auc(y_valid, valid_probs),
            "test_metrics_at_threshold": test_metrics,
            "test_expected_cost": float(expected_cost_test),
        }
        trial_rows.append(trial)

        if best_trial is None or trial["valid_cost_score"] > best_trial["valid_cost_score"]:
            best_trial = {
                **trial,
                "model": model,
            }

    assert best_trial is not None

    report = {
        "tuned_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "row_count": int(len(df)),
            "positive_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
        },
        "optimization": {
            "objective": "cost",
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "min_recall": min_recall,
            "n_trials": n_trials,
        },
        "best": {
            "params": best_trial["params"],
            "threshold": best_trial["valid_best_threshold"],
            "valid_cost_score": best_trial["valid_cost_score"],
            "valid_roc_auc": best_trial["valid_roc_auc"],
            "valid_pr_auc": best_trial["valid_pr_auc"],
            "test_metrics_at_threshold": best_trial["test_metrics_at_threshold"],
            "test_expected_cost": best_trial["test_expected_cost"],
        },
        "all_trials": trial_rows,
    }

    output_path = MODEL_PATH.parent / "tuning_report.json"
    output_path.write_text(json.dumps(report, indent=2))

    if deploy_best:
        final_package = {
            "model": best_trial["model"],
            "encoder": encoder,
            "threshold": best_trial["valid_best_threshold"],
            "thresholds": {"cost": best_trial["valid_best_threshold"]},
            "optimize_for": "cost",
            "feature_columns": X.columns.tolist(),
            "trained_at_utc": report["tuned_at_utc"],
        }
        joblib.dump(final_package, MODEL_PATH)
        THRESHOLD_PATH.write_text(f"{best_trial['valid_best_threshold']:.6f}")

        compact_metrics = {
            "trained_at_utc": report["tuned_at_utc"],
            "data": report["data"],
            "threshold_optimization": report["optimization"],
            "test_metrics": {
                "at_deployed_threshold": best_trial["test_metrics_at_threshold"],
            },
            "best_params": best_trial["params"],
        }
        METRICS_PATH.write_text(json.dumps(compact_metrics, indent=2))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune return-risk model for business cost")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of sampled trials")
    parser.add_argument("--fp-cost", type=float, default=1.0, help="Cost of false positive")
    parser.add_argument("--fn-cost", type=float, default=3.0, help="Cost of false negative")
    parser.add_argument("--min-recall", type=float, default=0.1, help="Minimum recall")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deploy-best",
        action="store_true",
        help="Overwrite model artifacts with the best tuned model",
    )
    args = parser.parse_args()

    report = tune(
        n_trials=args.n_trials,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        random_state=args.random_state,
        min_recall=args.min_recall,
        deploy_best=args.deploy_best,
    )

    best = report["best"]
    test_metrics = best["test_metrics_at_threshold"]
    print(f"Tuning trials: {report['optimization']['n_trials']}")
    print(f"Best threshold (cost objective): {best['threshold']:.3f}")
    print(f"Best validation cost score: {best['valid_cost_score']:.6f}")
    print(f"Test expected cost: {best['test_expected_cost']:.6f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test Precision: {test_metrics['precision']:.3f}")
    print(f"Test Recall: {test_metrics['recall']:.3f}")
    print(f"Test F1: {test_metrics['f1']:.3f}")
    print(f"Saved tuning report: {MODEL_PATH.parent / 'tuning_report.json'}")