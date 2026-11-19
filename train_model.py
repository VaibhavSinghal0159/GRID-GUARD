import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_features(df: pd.DataFrame):
    target_col = "FLAG" if "FLAG" in df.columns else df.columns[-1]
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
    drop_cols = [c for c in ["FLAG", "CONS_NO", target_col] if c in df.columns]

    readings = (
        df.drop(columns=drop_cols)
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(axis=1, limit_direction="both")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    vals = readings.values.astype(float)

    mean_safe = np.mean(vals, axis=1) + 1e-5
    std_v = np.std(vals, axis=1)
    max_v = np.max(vals, axis=1)
    p25_v = np.percentile(vals, 25, axis=1)
    mid = max(1, vals.shape[1] // 2)
    first_half = np.mean(vals[:, :mid], axis=1) + 1e-5
    second_half = np.mean(vals[:, mid:], axis=1) + 1e-5

    night_usage_ratio = np.clip(p25_v / mean_safe, 0.0, 1.0)
    spike_factor = np.clip(max_v / mean_safe, 1.0, 10.0)
    phase_imbalance = np.clip(np.abs(second_half - first_half) / mean_safe, 0.0, 1.0)
    load_mismatch = np.clip(std_v / mean_safe, 0.0, 2.0)

    feature_names = [
        "night_usage_ratio",
        "spike_factor",
        "phase_imbalance",
        "load_mismatch",
    ]
    X = np.column_stack([night_usage_ratio, spike_factor, phase_imbalance, load_mismatch])
    return X, y, feature_names


def main():
    base_dir = Path(__file__).parent
    df = pd.read_csv(base_dir / "datasetsmall.csv")
    X, y, feature_names = build_features(df)

    # 1. Stratified 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc, cv_recall, cv_precision, cv_f1 = [], [], [], []

    for tr_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        preds = (probs >= 0.50).astype(int)

        cv_auc.append(roc_auc_score(y[val_idx], probs))
        cv_recall.append(recall_score(y[val_idx], preds, zero_division=0))
        cv_precision.append(precision_score(y[val_idx], preds, zero_division=0))
        cv_f1.append(f1_score(y[val_idx], preds, zero_division=0))

    # 2. Final 80/20 Stratified Holdout Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "model": "RandomForestClassifier (n_estimators=150, max_depth=8, class_weight='balanced')",
        "dataset": "SGCC datasetsmall.csv",
        "total_consumers": int(len(y)),
        "theft_rate_pct": round(float(np.mean(y) * 100), 2),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "cv_5fold": {
            "roc_auc_mean": round(float(np.mean(cv_auc)), 4),
            "roc_auc_std": round(float(np.std(cv_auc)), 4),
            "recall_mean": round(float(np.mean(cv_recall)), 4),
            "precision_mean": round(float(np.mean(cv_precision)), 4),
            "f1_mean": round(float(np.mean(cv_f1)), 4),
        },
        "feature_importance": {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, model.feature_importances_)
        },
    }

    joblib.dump(model, base_dir / "theft_model.pkl")
    with open(base_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Saved unified theft_model.pkl and model_metrics.json")
    print(f"[OK] Holdout -> Acc: {metrics['accuracy']*100:.2f}% | Prec: {metrics['precision']*100:.2f}% | "
          f"Rec: {metrics['recall']*100:.2f}% | AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()