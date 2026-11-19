"""Shared feature extraction for GridGuard AI training and inference."""

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "night_usage_ratio",
    "spike_factor",
    "phase_imbalance",
    "load_mismatch",
    "zero_day_ratio",
    "max_zero_streak",
    "max_daily_drop",
    "iqr_to_mean",
    "trend_slope",
    "weekly_std_ratio",
    "median_to_mean",
]


def _max_zero_streak(row: np.ndarray) -> int:
    max_s = cur = 0
    for v in row:
        if v <= 1e-6:
            cur += 1
            if cur > max_s:
                max_s = cur
        else:
            cur = 0
    return max_s


def extract_features(df: pd.DataFrame):
    target_col = "FLAG" if "FLAG" in df.columns else None
    y = (
        pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
        if target_col
        else None
    )
    drop_cols = [c for c in ["FLAG", "CONS_NO"] if c in df.columns]

    readings = (
        df.drop(columns=drop_cols)
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(axis=1, limit_direction="both")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    vals = readings.values.astype(float)
    n_rows, n_days = vals.shape
    eps = 1e-5

    mean_v = np.mean(vals, axis=1)
    mean_safe = mean_v + eps
    std_v = np.std(vals, axis=1)
    max_v = np.max(vals, axis=1)
    p25_v = np.percentile(vals, 25, axis=1)
    p50_v = np.percentile(vals, 50, axis=1)
    p75_v = np.percentile(vals, 75, axis=1)

    mid = max(1, n_days // 2)
    first_half = np.mean(vals[:, :mid], axis=1)
    second_half = np.mean(vals[:, mid:], axis=1)

    # Core + time-series features
    night_usage_ratio = np.clip(p25_v / mean_safe, 0.0, 1.0)
    spike_factor = np.clip(max_v / mean_safe, 1.0, 15.0)
    phase_imbalance = np.clip(np.abs(second_half - first_half) / mean_safe, 0.0, 2.0)
    load_mismatch = np.clip(std_v / mean_safe, 0.0, 3.0)
    zero_day_ratio = np.mean(vals <= 1e-6, axis=1)
    max_zero_streak = np.array([_max_zero_streak(r) / max(1, n_days) for r in vals])

    diffs = np.diff(vals, axis=1) if n_days > 1 else np.zeros((n_rows, 1))
    max_daily_drop = np.clip(-np.min(diffs, axis=1) / mean_safe, 0.0, 10.0)
    iqr_to_mean = np.clip((p75_v - p25_v) / mean_safe, 0.0, 5.0)

    # Linear trend slope across the billing window
    t = np.arange(n_days, dtype=float)
    t_centered = t - t.mean()
    denom = float(np.sum(t_centered ** 2)) + eps
    trend_slope = np.clip((vals @ t_centered) / (denom * mean_safe), -1.0, 1.0)

    # 7-day periodic stability (weekday vs weekend proxy)
    if n_days >= 14:
        n_weeks = n_days // 7
        weekly_means = vals[:, : n_weeks * 7].reshape(n_rows, n_weeks, 7).mean(axis=2)
        weekly_std_ratio = np.clip(np.std(weekly_means, axis=1) / mean_safe, 0.0, 3.0)
    else:
        weekly_std_ratio = np.zeros(n_rows)

    median_to_mean = np.clip(p50_v / mean_safe, 0.0, 2.0)

    X = np.column_stack([
        night_usage_ratio,
        spike_factor,
        phase_imbalance,
        load_mismatch,
        zero_day_ratio,
        max_zero_streak,
        max_daily_drop,
        iqr_to_mean,
        trend_slope,
        weekly_std_ratio,
        median_to_mean,
    ])

    raw_stats = {
        "mean_raw": mean_v,
        "std_raw": std_v,
        "max_raw": max_v,
        "first_half_mean": first_half,
        "second_half_mean": second_half,
        "night_usage_ratio": night_usage_ratio,
        "spike_factor": spike_factor,
        "phase_imbalance": phase_imbalance,
        "load_mismatch": load_mismatch,
        "zero_day_ratio": zero_day_ratio,
        "max_daily_drop": max_daily_drop,
    }
    return X, y, raw_stats