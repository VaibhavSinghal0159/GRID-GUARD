"""
Electricity Theft Detector v4
==============================
Scores grid nodes using the trained RandomForest model (theft_model.pkl)
on real smart-meter readings from datasetsmall.csv.
Output: theft_detection_results_v4.json (same schema as v3/v4)

Usage:
    python electricity_theft_detector_v4.py
"""

import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Fixed house positions (Chandigarh sector grid)
HOUSE_POSITIONS = [
    (30.7352, 76.7700), (30.7251, 76.7741), (30.7353, 76.7806),
    (30.7376, 76.7834), (30.7317, 76.7785), (30.7289, 76.7868),
    (30.7385, 76.7726), (30.7318, 76.7750), (30.7276, 76.7847),
    (30.7253, 76.7770), (30.7305, 76.7763), (30.7286, 76.7703),
    (30.7325, 76.7719), (30.7415, 76.7710), (30.7292, 76.7820),
    (30.7410, 76.7766), (30.7271, 76.7708), (30.7362, 76.7849),
    (30.7430, 76.7865), (30.7406, 76.7770), (30.7324, 76.7860),
    (30.7266, 76.7765), (30.7367, 76.7834), (30.7370, 76.7708),
    (30.7360, 76.7801), (30.7282, 76.7786), (30.7286, 76.7788),
    (30.7287, 76.7879), (30.7298, 76.7848), (30.7244, 76.7858),
    (30.7394, 76.7774), (30.7246, 76.7877), (30.7346, 76.7837),
    (30.7274, 76.7794), (30.7356, 76.7779), (30.7410, 76.7823),
    (30.7382, 76.7802), (30.7382, 76.7780), (30.7349, 76.7766),
    (30.7432, 76.7722), (30.7332, 76.7845), (30.7405, 76.7725),
    (30.7265, 76.7825), (30.7352, 76.7771), (30.7352, 76.7788),
    (30.7256, 76.7871), (30.7283, 76.7805), (30.7255, 76.7787),
    (30.7383, 76.7848), (30.7301, 76.7753),
]

TRANSFORMER_POS = (30.7333, 76.7794)

ZONE_NAMES = [
    "Isolated High Risk", "High Risk Cluster A", "High Risk Cluster B",
    "Normal Zone", "Moderate Risk Zone"
]


def _load_model_and_data(base_dir: Path, n_nodes: int):
    model_path = base_dir / "theft_model.pkl"
    csv_path = base_dir / "datasetsmall.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=n_nodes, random_state=42).reset_index(drop=True)

    target_col = "FLAG" if "FLAG" in sample_df.columns else sample_df.columns[-1]
    drop_cols = [c for c in ["FLAG", "CONS_NO", target_col] if c in sample_df.columns]

    readings = (
        sample_df.drop(columns=drop_cols)
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(axis=1, limit_direction="both")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    vals = readings.values
    mean_raw = np.mean(vals, axis=1)
    std_raw = np.std(vals, axis=1)
    max_raw = np.max(vals, axis=1)
    p25_raw = np.percentile(vals, 25, axis=1)

    mean_safe = mean_raw + 1e-5
    mid = max(1, vals.shape[1] // 2)
    first_half_mean = np.mean(vals[:, :mid], axis=1) + 1e-5
    second_half_mean = np.mean(vals[:, mid:], axis=1) + 1e-5

    night_usage_ratio = np.clip(p25_raw / mean_safe, 0.0, 1.0)
    spike_factor = np.clip(max_raw / mean_safe, 1.0, 10.0)
    phase_imbalance = np.clip(np.abs(second_half_mean - first_half_mean) / mean_safe, 0.0, 1.0)
    load_mismatch = np.clip(std_raw / mean_safe, 0.0, 2.0)

    X = np.column_stack([night_usage_ratio, spike_factor, phase_imbalance, load_mismatch])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
        probs = np.where(preds == 1, 0.75, 0.20)

    stats = {
        "mean_raw": mean_raw,
        "std_raw": std_raw,
        "max_raw": max_raw,
        "night_usage_ratio": night_usage_ratio,
        "spike_factor": spike_factor,
        "phase_imbalance": phase_imbalance,
        "load_mismatch": load_mismatch,
        "first_half_mean": first_half_mean,
        "second_half_mean": second_half_mean,
    }
    return probs, stats


def _derive_reasons(night_r: float, spike_f: float, phase_imb: float, load_mm: float):
    candidates = [
        (load_mm / 2.0, "load mismatch"),
        (spike_f / 10.0, "sudden spike"),
        (phase_imb, "phase imbalance"),
        (night_r, "high night usage"),
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    primary = candidates[0][1]
    secondary = [label for _, label in candidates[1:3]]
    return primary, secondary


def generate_random_data() -> dict:
    """
    Score fixed grid nodes using the trained RandomForest model (theft_model.pkl)
    on real SGCC smart-meter readings from datasetsmall.csv.
    Function name kept as generate_random_data for backward compatibility with api.py.
    """
    base_dir = Path(__file__).parent
    num_houses = len(HOUSE_POSITIONS)
    probs, stats = _load_model_and_data(base_dir, num_houses)

    total_mean_sum = max(float(np.sum(stats["mean_raw"])), 1e-5)
    houses_raw = []

    for i, (lat, lng) in enumerate(HOUSE_POSITIONS):
        house_id = i + 1
        prob = float(probs[i])
        risk_score = int(np.clip(round(prob * 100), 0, 100))

        # Deterministic small offset from index so markers do not overlap
        angle = i * 2.39996  # golden angle in radians
        radius = 0.00025 * ((i % 5) + 1) / 5.0
        jlat = round(lat + radius * math.cos(angle), 6)
        jlng = round(lng + radius * math.sin(angle), 6)

        night_r = float(stats["night_usage_ratio"][i])
        spike_f = float(stats["spike_factor"][i])
        phase_imb = float(stats["phase_imbalance"][i])
        load_mm = float(stats["load_mismatch"][i])

        avg_con = round(float(np.clip(stats["mean_raw"][i], 0.5, 120.0)), 4)
        std_con = round(float(np.clip(stats["std_raw"][i], 0.1, 80.0)), 4)
        max_con = round(max(avg_con + 0.5, float(np.clip(stats["max_raw"][i], 1.0, 250.0))), 4)
        con_diff = round(float(stats["second_half_mean"][i] - stats["first_half_mean"][i]), 4)
        load_share = round(float(stats["mean_raw"][i] / total_mean_sum), 4)
        anomaly_score = round(-(0.30 + 0.55 * prob), 6)

        primary_reason, secondary = _derive_reasons(night_r, spike_f, phase_imb, load_mm)

        if risk_score >= 65:
            risk_level = "high"
            confidence = "very high (\u226580%)" if risk_score >= 80 else "high (65\u201379%)"
            zone = ZONE_NAMES[i % 3]
        elif risk_score >= 35:
            risk_level = "medium"
            confidence = "medium (35\u201364%)"
            zone = ZONE_NAMES[3 + (i % 2)]
        else:
            risk_level = "low"
            confidence = "low (<35%)"
            zone = "Normal Zone"
            secondary = []

        houses_raw.append({
            "house_id": house_id,
            "lat": jlat,
            "lng": jlng,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": {"primary": primary_reason, "secondary": secondary},
            "confidence": confidence,
            "zone": zone,
            "anomaly_score": anomaly_score,
            "average_consumption": avg_con,
            "max_consumption": max_con,
            "std_consumption": std_con,
            "night_usage_ratio": round(night_r, 4),
            "consumption_difference": con_diff,
            "transformer_load_share": load_share,
        })

    houses_raw.sort(key=lambda h: h["risk_score"], reverse=True)
    for rank, h in enumerate(houses_raw, start=1):
        h["priority_rank"] = rank

    top5 = houses_raw[:5]
    total_high_risk = sum(1 for h in houses_raw if h["risk_level"] == "high")

    total_consumption = round(sum(h["average_consumption"] for h in houses_raw), 4)
    expected_load = round(total_consumption * 0.90, 4)
    transformer_loss = round(max(0.0, total_consumption - expected_load), 4)
    loss_pct = round(transformer_loss / max(total_consumption, 1.0), 4)
    est_loss_rupees = round(transformer_loss * 8.0, 2)

    if loss_pct > 0.12:
        zone_status = "Critical"
    elif loss_pct > 0.06:
        zone_status = "Warning"
    else:
        zone_status = "Normal"

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "transformer": {
            "type": "transformer",
            "lat": TRANSFORMER_POS[0],
            "lng": TRANSFORMER_POS[1],
            "loss": transformer_loss,
            "loss_percentage": loss_pct,
            "status": zone_status,
            "estimated_loss_in_rupees": est_loss_rupees,
        },
        "transformer_metrics": {
            "total_house_consumption": total_consumption,
            "expected_transformer_load": expected_load,
            "transformer_loss": transformer_loss,
            "loss_ratio": loss_pct,
            "zone_status": zone_status,
            "estimated_loss_in_rupees": est_loss_rupees,
            "last_updated": now,
        },
        "insights": {
            "top_5_houses": top5,
            "total_high_risk": total_high_risk,
            "zone_status": zone_status,
            "estimated_loss": est_loss_rupees,
        },
        "houses": houses_raw,
        "last_updated": now,
    }


def save_to_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {path}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "theft_detection_results_v4.json"
    data = generate_random_data()
    save_to_json(data, output_path)

    high = data["insights"]["total_high_risk"]
    loss = data["insights"]["estimated_loss"]
    status = data["transformer"]["status"]
    print(f"[OK] Detection complete: {high} high-risk houses | Zone: {status} | Loss: Rs.{loss}")
    print(f"[OK] Top suspect: House #{data['insights']['top_5_houses'][0]['house_id']} "
          f"(score {data['insights']['top_5_houses'][0]['risk_score']}/100)")