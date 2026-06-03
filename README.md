# DROPWISE 🐟

**Dissolved Oxygen Real-time Prediction with Intelligent Sensor-based Evaluation**

A machine learning platform for classifying dissolved oxygen (DO) levels in freshwater aquaculture ecosystems using IoT sensor data, deployed via an interactive Gradio web interface.

---

## Overview

DROPWISE classifies DO concentrations into three categories — **Low**, **Medium**, and **High** — based on physicochemical and biological parameters measured by IoT sensors. Four supervised learning algorithms are compared using time-aware cross-validation (15-fold TimeSeriesSplit) to ensure realistic performance estimates.

| Model | Accuracy | F1-Score |
|---|---|---|
| Random Forest | 82.78% | 0.7912 |
| LightGBM | 82.71% | 0.7895 |
| CatBoost | 82.71% | 0.7895 |
| Logistic Regression | 81.06% | 0.7680 |

---

## Features Used

- Average fish weight (g)
- Survival rate (%)
- Disease occurrence (cases)
- Temperature (°C)
- Precipitation (inches)
- pH
- Turbidity (NTU)

---

## Dataset

The dataset is publicly available on Kaggle:  
[IoT Monitoring of Water Quality and Tilapia](https://www.kaggle.com/datasets/jocelyndumlao/iot-monitoring-of-water-quality-and-tilapia)  
License: CC0 Public Domain

Download the CSV and place it in the project root as `Data_Water.csv`.

---

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/DROPWISE.git
cd DROPWISE
pip install -r requirements.txt
```

---

## Usage

**Step 1 — Train the model:**
```bash
python dropwise_main.py
```
This will preprocess the data, run 15-fold time-aware cross-validation across all four models, save the best model (`rf_model_timeaware.pkl`), and generate all evaluation figures.

**Step 2 — Launch the Gradio interface:**
```bash
python dropwise_app.py
```
Open the URL shown in your terminal to access the interactive prediction interface.

---

## Project Structure

```
DROPWISE/
├── dropwise_main.py       # Full ML pipeline (preprocessing, training, evaluation)
├── dropwise_app.py        # Gradio web interface
├── requirements.txt       # Python dependencies
├── README.md
└── Data_Water.csv         # Dataset (download from Kaggle, not included)
```

---

## Citation

If you use this code in your research, please cite:

> Chauhan, R., Sarfi, Z., & Singh, D. (2025). DROPWISE: A Machine Learning-Driven IoT Framework for Real-Time Dissolved Oxygen Prediction in Freshwater Ecosystems. *Cogent Engineering*.

---

## License

MIT License
