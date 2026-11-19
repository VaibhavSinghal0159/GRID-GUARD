"""
Electricity Theft Screening Engine v4
=====================================
Runs calibrated ML screening (theft_model.pkl) on 50 sampled SGCC smart-meter
profiles mapped to fixed Chandigarh coordinates for dashboard visualization.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import extract_features

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
    "Isolated High Priority", "High Priority Cluster A", "High Priority Cluster B",
    "Normal Zone", "Moderate Priority Zone",
]


def _derive_reasons(stats: dict, i: int):
    candidates = [
        (float(stats["load_mismatch"][i]) / 3.0, "load mismatch"),
        (float(stats["spike_factor"][i]) / 15.0, "sudden spike"),
        (float(stats["phase_imbalance"][i]) / 2.0, "phase imbalance"),
        (float(stats["zero_day_ratio"][i]), "zero-reading streak"),
        (float(stats["max_daily_drop"][i]) / 10.0, "abrupt consumption drop"),
        (float(stats["night_usage_ratio"][i]), "elevated baseline floor"),
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], [label for _, label in candidates[1:3]]


def run_theft_screening() -> dict:
    """Score 50 real SGCC smart-meter profiles using calibrated theft_model.pkl."""
    base_dir = Path(__file__).parent
    model_path = base_dir / "theft_model.pkl"
    csv_path = base_dir / "datasetsmall.csv"
    metrics_path = base_dir / "model_metrics.json"

    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=len(HOUSE_POSITIONS), random_state=42).reset_index(drop=True)
    X, _, stats = extract_features(sample_df)

    probs = model.predict_proba(X)[:, 1]
    threshold = 0.20
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            threshold = float(json.load(f).get("decision_threshold", 0.20))

    # Scale calibrated probabilities relative to the tuned decision threshold into a 0-100 screening priority index
    total_mean_sum = max(float(np.sum(stats["mean_raw"])), 1e-5)
    houses_raw = []

    for i, (lat, lng) in enumerate(HOUSE_POSITIONS):
        prob = float(probs[i])
        # Map calibrated probability to 0-100 screening priority (threshold -> 65)
        if prob >= threshold:
            risk_score = int(np.clip(round(65 + 35 * (prob - threshold) / max(1e-5, 1.0 - threshold)), 65, 100))
        else:
            risk_score = int(np.clip(round(64 * (prob / max(1e-5, threshold))), 0, 64))

        angle = i * 2.39996
        radius = 0.00025 * ((i % 5) + 1) / 5.0
        jlat = round(lat + radius * math.cos(angle), 6)
        jlng = round(lng + radius * math.sin(angle), 6)

        avg_con = round(float(np.clip(stats["mean_raw"][i], 0.5, 120.0)), 4)
        std_con = round(float(np.clip(stats["std_raw"][i], 0.1, 80.0)), 4)
        max_con = round(max(avg_con + 0.5, float(np.clip(stats["max_raw"][i], 1.0, 250.0))), 4)
        con_diff = round(float(stats["second_half_mean"][i] - stats["first_half_mean"][i]), 4)
        load_share = round(float(stats["mean_raw"][i] / total_mean_sum), 4)

        primary_reason, secondary = _derive_reasons(stats, i)

        if risk_score >= 65:
            risk_level = "high"
            confidence = f"screening priority ({prob * 100:.1f}% est. prob — verify on site)"
            zone = ZONE_NAMES[i % 3]
        elif risk_score >= 35:
            risk_level = "medium"
            confidence = f"moderate priority ({prob * 100:.1f}% est. prob)"
            zone = ZONE_NAMES[3 + (i % 2)]
        else:
            risk_level = "low"
            confidence = f"low priority ({prob * 100:.1f}% est. prob)"
            zone = "Normal Zone"
            secondary = []

        houses_raw.append({
            "house_id": i + 1,
            "lat": jlat,
            "lng": jlng,
            "risk_score": risk_score,
            "calibrated_probability": round(prob, 4),
            "risk_level": risk_level,
            "reason": {"primary": primary_reason, "secondary": secondary},
            "confidence": confidence,
            "zone": zone,
            "anomaly_score": round(-prob, 6),
            "average_consumption": avg_con,
            "max_consumption": max_con,
            "std_consumption": std_con,
            "night_usage_ratio": round(float(stats["night_usage_ratio"][i]), 4),
            "consumption_difference": con_diff,
            "transformer_load_share": load_share,
        })

    houses_raw.sort(key=lambda h: h["risk_score"], reverse=True)
    for rank, h in enumerate(houses_raw, start=1):
        h["priority_rank"] = rank

    top5 = houses_raw[:5]
    total_high_risk = sum(1 for h in houses_raw if h["risk_level"] == "high")

    # Illustrative exposure estimate based on flagged houses' consumption (explicitly labeled synthetic)
    total_consumption = round(sum(h["average_consumption"] for h in houses_raw), 4)
    flagged_consumption = sum(h["average_consumption"] for h in houses_raw if h["risk_level"] == "high")
    illustrative_loss = round(flagged_consumption * 0.25, 4)  # illustrative 25% under-billing assumption on flagged nodes
    expected_load = round(max(0.0, total_consumption - illustrative_loss), 4)
    loss_pct = round(illustrative_loss / max(total_consumption, 1.0), 4)
    est_loss_rupees = round(illustrative_loss * 8.0, 2)

    zone_status = "Critical" if loss_pct > 0.12 else ("Warning" if loss_pct > 0.05 else "Normal")
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "disclaimer": "Screening priority only — not proof of theft. Transformer loss and Chandigarh coordinates are illustrative.",
        "illustrative_synthetic_loss": True,
        "transformer": {
            "type": "transformer",
            "lat": TRANSFORMER_POS[0],
            "lng": TRANSFORMER_POS[1],
            "loss": illustrative_loss,
            "loss_percentage": loss_pct,
            "status": zone_status,
            "estimated_loss_in_rupees": est_loss_rupees,
            "is_illustrative": True,
        },
        "transformer_metrics": {
            "total_house_consumption": total_consumption,
            "expected_transformer_load": expected_load,
            "transformer_loss": illustrative_loss,
            "loss_ratio": loss_pct,
            "zone_status": zone_status,
            "estimated_loss_in_rupees": est_loss_rupees,
            "is_illustrative": True,
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


# Backward-compatible alias
generate_random_data = run_theft_screening


def save_to_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    output_path = Path(__file__).parent / "theft_detection_results_v4.json"
    data = run_theft_screening()
    save_to_json(data, output_path)
    print(f"[OK] High-priority screening flags: {data['insights']['total_high_risk']} | Zone: {data['transformer']['status']}")