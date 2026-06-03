"""
DROPWISE — Gradio Web Interface
================================
Interactive dissolved oxygen (DO) prediction platform for aquaculture operators
and field monitors. Enter real-time IoT sensor values to receive a DO classification
(Low / Medium / High) with plain-language interpretation.

Usage:
    python dropwise_app.py

Requires rf_model_timeaware.pkl and scaler.pkl to be present (run dropwise_main.py first).
"""

import gradio as gr
import pandas as pd
import joblib
import numpy as np

# ── Load trained model and scaler ────────────────────────────────────────────
try:
    model = joblib.load("rf_model_timeaware.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(
        "Model files not found. Please run dropwise_main.py first to train and save the model."
    )

FEATURES = [
    "average_fish_weight_g",
    "survival_rate_percent",
    "disease_occurrence_cases",
    "temperature_degc",
    "precipitation_inches",
    "ph",
    "turbidity_ntu",
]

# ── Interpretation labels (matching paper pseudocode, Table 2) ────────────────
INTERPRETATIONS = {
    "Low": (
        "⚠️ LOW Dissolved Oxygen — Poor water quality, potentially harmful for fish.\n\n"
        "DO levels below 6.5 mg/L are stressful for tilapia and other aquatic species, "
        "slowing metabolism, reducing immunity, and increasing mortality risk. "
        "Immediate corrective action is recommended (e.g., aeration, water exchange)."
    ),
    "Medium": (
        "🔵 MEDIUM Dissolved Oxygen — Acceptable, but not ideal for aquaculture.\n\n"
        "DO levels between 6.5–7.5 mg/L are survivable but suboptimal. "
        "Regular monitoring and preventive measures are advised to avoid further deterioration."
    ),
    "High": (
        "✅ HIGH Dissolved Oxygen — Excellent water quality for fish health.\n\n"
        "DO levels above 7.5 mg/L support optimal fish growth, reproduction, and survival. "
        "Current conditions are well-balanced. Continue regular monitoring."
    ),
}

CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}


# ── Prediction function ───────────────────────────────────────────────────────
def predict_do(
    avg_fish_weight,
    survival_rate,
    disease_occurrence,
    temperature,
    precipitation,
    ph,
    turbidity,
):
    input_data = pd.DataFrame(
        [[avg_fish_weight, survival_rate, disease_occurrence,
          temperature, precipitation, ph, turbidity]],
        columns=FEATURES,
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    label = CLASS_LABELS.get(prediction, "Unknown")
    interpretation = INTERPRETATIONS.get(label, "No interpretation available.")

    return f"Predicted DO Category: {label}\n\n{interpretation}"


# ── Gradio Interface ──────────────────────────────────────────────────────────
inputs = [
    gr.Number(label="Average Fish Weight (g)", value=200.0, minimum=0),
    gr.Number(label="Survival Rate (%)", value=85.0, minimum=0, maximum=100),
    gr.Number(label="Disease Occurrence (cases)", value=0, minimum=0),
    gr.Number(label="Temperature (°C)", value=26.0),
    gr.Number(label="Precipitation (inches)", value=0.1, minimum=0),
    gr.Number(label="pH", value=7.2, minimum=0, maximum=14),
    gr.Number(label="Turbidity (NTU)", value=5.0, minimum=0),
]

iface = gr.Interface(
    fn=predict_do,
    inputs=inputs,
    outputs=gr.Textbox(label="DROPWISE Prediction", lines=8),
    title="🐟 DROPWISE — Dissolved Oxygen Predictor",
    description=(
        "Enter real-time IoT sensor readings from your aquaculture pond to classify "
        "dissolved oxygen (DO) levels as Low, Medium, or High, with actionable guidance.\n\n"
        "DROPWISE uses a Random Forest classifier trained on time-aware cross-validation "
        "across physicochemical and biological indicators."
    ),
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    iface.launch()
