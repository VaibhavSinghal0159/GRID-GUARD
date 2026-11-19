# GridGuard AI — Electricity Theft Detection System

> **AI-powered electricity theft detection & inspection-triage dashboard for power distribution utilities.**  
> Built for Hack Helix — detecting grid anomalies with supervised ML, prioritizing field inspections, and simulating revenue loss.

---

## 📌 Project Overview

GridGuard AI is a full-stack intelligence platform that uses a supervised **Random Forest Classifier** trained on real-world **State Grid Corporation of China (SGCC)** smart-meter data (`datasetsmall.csv`) to detect electricity theft patterns across a distribution grid. It simulates a DISCOM (distribution company) operations center with:

- A **live geospatial map** showing transformer + household risk overlays on the Chandigarh sector grid
- **Deterministic ML inference** via a trained `RandomForestClassifier` (`theft_model.pkl`) scoring 50 real SGCC consumer profiles
- **Theft injection simulation** — stress-test grid scenarios (0–50%) with instant client-side preview and full backend recalculation
- **AI-powered explanations** for each flagged house (via Google Gemini, with rule-based fallback)
- **Live ML metrics endpoint** (`/api/model-metrics`), **CSV report export**, and inspection priority ranking

The project overlays 50 sampled SGCC smart-meter profiles onto fixed **Chandigarh sector grid** (Sector 17 area) coordinates with one distribution transformer.

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS + shadcn/ui components |
| Maps | Leaflet + react-leaflet |
| Charts | Recharts |
| Backend | Python FastAPI (uvicorn) |
| ML Model | Random Forest Classifier (`scikit-learn`) — supervised classification |
| Dataset | SGCC Smart-Meter Benchmark (`datasetsmall.csv`, 25,863 consumers) |
| AI Insights | Google Gemini 2.5 Flash (with rule-based fallback) |
| State | TanStack React Query |

---

## 🤖 ML Model — Random Forest Classifier

### How It Works

The offline training pipeline (`train_model.py` / `GridGuard_ML.ipynb`) and live inference engine (`electricity_theft_detector_v4.py`) use a **Random Forest Classifier** (`n_estimators=150`, `max_depth=8`, `class_weight="balanced"`):

1. **Data Cleaning:** Missing daily kWh readings in `datasetsmall.csv` are repaired via row-wise linear interpolation (`interpolate(axis=1)`), remaining nulls are filled with `0.0`, and negative readings are clipped to `0.0`.
2. **Feature Engineering:** Each consumer's daily kWh series is compressed into 4 behavioral