"""Reproducible training pipeline for GridGuard AI."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from features import FEATURE_NAMES, extract_features


def make_base_model():
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def main():
    base_dir = Path(__file__).parent
    df = pd.read_csv(base_dir / "datasetsmall.csv")
    X, y, _ = extract_features(df)

    # 1. Stratified 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc, cv_pr = [], []

    for tr_idx, val_idx in skf.split(X, y):
        clf = make_base_model()
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        cv_roc.append(roc_auc_score(y[val_idx], probs))
        cv_pr.append(average_precision_score(y[val_idx], probs))

    # 2. Holdout Split + Isotonic Probability Calibration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=make_base_model(),
        method="isotonic",
        cv=5,
    )
    calibrated_model.fit(X_train, y_train)
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]

    # 3. Cost-sensitive / F1 threshold tuning
    best_th, best_f1 = 0.50, -1.0
    for th in np.linspace(0.08, 0.50, 43):
        score = f1_score(y_test, (test_probs >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_th = float(score), float(th)

    preds = (test_probs >= best_th).astype(int)
    cm = confusion_matrix(y_test, preds)
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    trivial_acc = float(1.0 - np.mean(y_test))

    metrics = {
        "model": "Calibrated HistGradientBoostingClassifier (isotonic, 11 time-series features)",
        "dataset": "SGCC datasetsmall.csv",
        "total_consumers": int(len(y)),
        "theft_rate_pct": round(float(np.mean(y) * 100), 2),
        "decision_threshold": round(best_th, 3),
        "trivial_baseline": {
            "strategy": "Always predict Normal (0)",
            "accuracy": round(trivial_acc, 4),
            "recall": 0.0,
            "pr_auc": round(float(np.mean(y_test)), 4),
        },
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(prec, 4),
        "false_alarm_rate": round(1.0 - prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(best_f1, 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "cv_5fold": {
            "roc_auc_mean": round(float(np.mean(cv_roc)), 4),
            "roc_auc_std": round(float(np.std(cv_roc)), 4),
            "pr_auc_mean": round(float(np.mean(cv_pr)), 4),
            "pr_auc_std": round(float(np.std(cv_pr)), 4),
        },
        "features": FEATURE_NAMES,
        "disclaimer": (
            "Screening priority tool only — not proof of theft. "
            "Requires human field verification before any billing or legal action."
        ),
    }

    joblib.dump(calibrated_model, base_dir / "theft_model.pkl")
    with open(base_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Saved calibrated theft_model.pkl and model_metrics.json")
    print(f"     ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']} (baseline {metrics['trivial_baseline']['pr_auc']})")
    print(f"     At threshold {best_th:.3f} -> Prec: {prec*100:.1f}% (False Alarm: {(1-prec)*100:.1f}%) | Rec: {rec*100:.1f}% | Acc: {metrics['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()