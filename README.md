# 💳 Credit Card Fraud Detection System

A machine learning–powered web application that detects fraudulent credit card transactions in real time. Built with **Streamlit** for an interactive UI and a **Random Forest** classifier trained with **SMOTE** oversampling on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo Screenshots](#demo-screenshots)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Results & Performance](#results--performance)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Credit card fraud is a critical issue affecting millions of transactions worldwide. This project provides an end-to-end solution — from model training to a production-ready web interface — that allows users to:

1. **Upload** a batch of credit card transactions (CSV).
2. **Detect** potentially fraudulent transactions using a pre-trained ML model.
3. **Visualize** risk distributions, model performance metrics, and flagged transactions.
4. **Download** the annotated results for further investigation.

The model uses an **optimized probability threshold** (tuned on the Precision-Recall curve) rather than the default 0.5, which is crucial for the highly imbalanced fraud detection domain.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Real-Time Fraud Scoring** | Every transaction receives a fraud probability score and risk level (Low / Medium / High). |
| 📊 **Interactive Visualizations** | Probability distributions, box plots by prediction class, and confusion matrix heatmaps. |
| 📈 **Model Performance Dashboard** | Classification report, confusion matrix, and Precision-Recall AUC curve (when ground truth labels are available). |
| 🗂️ **Batch Upload** | Upload any CSV file matching the expected schema (`Time`, `Amount`, `V1`–`V28`). |
| 🎯 **Demo Mode** | Explore the app instantly using a bundled sample of 1,000 transactions. |
| 📥 **Export Results** | Download full prediction results as a CSV with fraud probability, predicted label, and risk level columns. |
| ⚡ **Cached Model Loading** | Uses Streamlit's `@st.cache_resource` for fast, memory-efficient artifact loading. |
| 🎛️ **Optimal Threshold** | Applies a pre-computed optimal decision threshold instead of the naive 0.5 cutoff. |

---

## Demo Screenshots

> After running the app, the interface includes:
> - A **sidebar** with model info and usage instructions.
> - **Metric cards** showing total transactions, actual fraud count, and average transaction amount.
> - **Tabbed visualizations** for distribution plots, top risk transactions, and flagged transaction details.
> - A **model performance section** with confusion matrix and PR curve (when labels are provided).

---

## Project Architecture

```
User uploads CSV
       │
       ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Streamlit   │────▶│  Feature Scaling  │────▶│  Random Forest  │
│  Frontend    │     │  (StandardScaler) │     │  Classifier     │
└──────────────┘     └───────────────────┘     └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌────────────────┐
                                               │  Probability   │
                                               │  Thresholding  │
                                               └────────┬───────┘
                                                        │
                                         ┌──────────────┼──────────────┐
                                         ▼              ▼              ▼
                                   ┌──────────┐  ┌───────────┐  ┌───────────┐
                                   │ Metrics  │  │  Visuals  │  │  CSV      │
                                   │ Summary  │  │  & Charts │  │  Download │
                                   └──────────┘  └───────────┘  └───────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io/) 1.56.0 |
| **ML Model** | scikit-learn 1.8.0 — Random Forest Classifier |
| **Data Processing** | Pandas 3.0.2, NumPy 2.4.4 |
| **Visualization** | Matplotlib 3.10.9, Seaborn 0.13.2 |
| **Imbalance Handling** | SMOTE (Synthetic Minority Oversampling Technique) — applied during training |
| **Language** | Python 3.10+ |

---

## Dataset

This project uses the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle.

| Property | Value |
|---|---|
| **Total Transactions** | 284,807 |
| **Fraud Cases** | 492 (0.172%) |
| **Features** | `Time`, `Amount`, `V1`–`V28` (PCA-transformed) |
| **Target** | `Class` (0 = Normal, 1 = Fraud) |

> **Note:** The features `V1`–`V28` are principal components obtained via PCA transformation. The original features are not available due to confidentiality.

---

## Model Details

### Training Pipeline

1. **Exploratory Data Analysis** — Class distribution, correlation analysis, and feature distributions.
2. **Preprocessing** — `StandardScaler` applied to `Time` and `Amount` columns (V1–V28 are already scaled via PCA).
3. **Handling Class Imbalance** — SMOTE is used to synthetically oversample the minority (fraud) class during training.
4. **Model Selection** — Multiple models were compared; **Random Forest** was selected based on superior PR-AUC performance.
5. **Threshold Optimization** — Instead of using the default 0.5, the optimal threshold is selected from the Precision-Recall curve to maximize the F1 score on the validation set.

### Serialized Artifacts

| File | Description |
|---|---|
| `fraud_model.pkl` | Trained Random Forest classifier (~3.5 MB) |
| `scaler.pkl` | Fitted `StandardScaler` for `Time` and `Amount` |
| `optimal_threshold.pkl` | Pre-computed optimal probability threshold |
| `feature_cols.pkl` | Ordered list of feature column names expected by the model |

### Evaluation Plots (Generated During Training)

| File | Description |
|---|---|
| `eda_plots.png` | Exploratory data analysis visualizations |
| `feature_importance.png` | Top feature importances from the Random Forest model |
| `model_comparison.png` | Comparison of candidate models (e.g., Logistic Regression, Random Forest, etc.) |
| `cost_optimization.png` | Threshold tuning and cost analysis |

---

## Getting Started

### Prerequisites

- **Python 3.10** or higher
- **pip** (Python package manager)
- **Git** (optional, for cloning)

### Installation

1. **Clone the repository** (or download the ZIP):

   ```bash
   git clone https://github.com/<your-username>/fraud_app.git
   cd fraud_app
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at **http://localhost:8501**.

---

## Usage Guide

### Option 1 — Demo Mode
1. Launch the app.
2. Check the **"Use demo data (1000 transactions)"** checkbox.
3. The app will load the bundled `test_sample.csv` and display results immediately.

### Option 2 — Upload Your Own Data
1. Prepare a CSV file with the following columns:
   - `Time` — seconds elapsed from the first transaction in the dataset.
   - `Amount` — transaction amount in USD.
   - `V1` through `V28` — PCA-transformed features.
   - `Class` *(optional)* — ground truth label (0 = Normal, 1 = Fraud). If included, the app will show model performance metrics.
2. Click **"Browse files"** or drag-and-drop your CSV.
3. The app will automatically validate columns, run predictions, and display:
   - **Fraud probability** for each transaction.
   - **Risk level** categorization (🟢 Low / 🟡 Medium / 🔴 High).
   - **Summary metrics** and **visualizations**.
4. Click **"📥 Download Full Results as CSV"** to export the annotated predictions.

---

## Project Structure

```
fraud_app/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── fraud_model.pkl          # Trained Random Forest model
├── scaler.pkl               # Fitted StandardScaler
├── optimal_threshold.pkl    # Optimal decision threshold
├── feature_cols.pkl         # Feature column names
│
├── test_sample.csv          # Demo dataset (1,000 transactions)
│
├── eda_plots.png            # EDA visualizations (training output)
├── feature_importance.png   # Feature importance chart (training output)
├── model_comparison.png     # Model comparison chart (training output)
├── cost_optimization.png    # Cost/threshold optimization chart (training output)
│
└── venv/                    # Python virtual environment (not tracked in git)
```

---

## Results & Performance

The model was evaluated on a held-out test set. Key metrics:

| Metric | Value |
|---|---|
| **Precision (Fraud class)** | High — minimizes false alarms |
| **Recall (Fraud class)** | High — catches most actual fraud |
| **PR-AUC** | Displayed in the app when ground truth labels are provided |
| **Optimal Threshold** | Automatically loaded from `optimal_threshold.pkl` |

> **Tip:** Upload data with the `Class` column included to see the full performance dashboard, including the confusion matrix and Precision-Recall curve, directly in the app.

---

## Future Improvements

- [ ] Add support for real-time single-transaction prediction via a REST API (e.g., FastAPI).
- [ ] Integrate deep learning models (Autoencoder-based anomaly detection).
- [ ] Add SHAP / LIME explanations for individual predictions.
- [ ] Deploy to cloud (Streamlit Cloud, AWS, or GCP).
- [ ] Add user authentication and role-based access control.
- [ ] Support additional file formats (Excel, JSON).
- [ ] Implement alerting/notification system for high-risk transactions.

---

## License

This project is developed as part of an academic coursework (6th Semester — Machine Learning). Feel free to use and modify for educational purposes.

---

## Acknowledgements

- **Dataset**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group at ULB.
- **Framework**: [Streamlit](https://streamlit.io/) for the rapid web app development.
- **Libraries**: scikit-learn, pandas, NumPy, Matplotlib, Seaborn.
- **Technique**: SMOTE from [imbalanced-learn](https://imbalanced-learn.org/) for handling class imbalance.

---

<p align="center">
  Made with ❤️ using Python & Streamlit
</p>
