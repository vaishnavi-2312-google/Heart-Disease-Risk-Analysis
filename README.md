# ❤️ Heart Disease Risk Analysis and Prediction System

**IBM Internship Project**  
**Student:** Vaishnavi  
**Domain:** Artificial Intelligence & Data Science  

---

## 📌 Project Overview

This project presents a complete **Heart Disease Risk Analysis and Prediction System** built on the UCI Cleveland Heart Disease dataset. It integrates a full analytics pipeline — descriptive, diagnostic, predictive, and prescriptive — and delivers an interactive **Streamlit** dashboard with machine learning–powered risk scoring.

---

## 🎯 Problem Statement

Cardiovascular diseases are among the leading causes of mortality worldwide. Early, data-driven identification of at-risk individuals can support timely clinical attention and healthcare resource allocation. This project applies machine learning to patient clinical data to build an analytical risk-scoring model and comprehensive visualisation dashboard.

---

## 🏆 Objectives

1. Perform thorough exploratory data analysis on the UCI Cleveland Heart Disease dataset.
2. Apply descriptive analytics to summarise patient population characteristics.
3. Conduct diagnostic analytics to identify feature associations with heart disease.
4. Build and evaluate a Random Forest classification model for risk prediction.
5. Provide prescriptive insights and data-driven analytical recommendations.
6. Deliver an interactive, filterable Streamlit dashboard for data exploration.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Name | UCI Heart Disease (Cleveland) |
| File | `processed.cleveland.data` |
| Records | 303 patient records |
| Features | 13 clinical attributes |
| Target | `num` (binarised: 0 = No Disease, 1 = Disease) |
| Missing Values | `?` in `ca` and `thal` columns (6 rows) |
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) |

### Columns

| Column | Description |
|--------|-------------|
| age | Age in years |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (1–4) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True) |
| restecg | Resting ECG results (0, 1, 2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = Yes) |
| oldpeak | ST depression induced by exercise vs rest |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels coloured by fluoroscopy (0–3) |
| thal | Thalassemia (3 = Normal, 6 = Fixed defect, 7 = Reversable defect) |
| num | **Target** — diagnosis of heart disease (0 = No disease; >0 = Disease) |

---

## 🛠 Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Primary language |
| Pandas | ≥2.0.0 | Data loading, cleaning, manipulation |
| NumPy | ≥1.26.0 | Numerical computation |
| Scikit-learn | ≥1.4.0 | Machine learning and evaluation |
| Plotly | ≥5.18.0 | Interactive visualisations |
| Streamlit | ≥1.32.0 | Interactive web dashboard |
| Python stdlib (urllib, io, os) | — | Dataset download / file handling |

---

## 📁 Project Structure

```
heart-disease-analyzer/
│
├── app.py                       ← Complete application (ALL logic here)
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
└── Vaishnavi_ProjectReport.docx ← Capstone project report
```

> ⚠️ **No `data/`, `models/`, or `src/` folders are required.**  
> The dataset is loaded automatically from the `heart+disease/` folder (if present locally) or downloaded from the UCI ML Repository.

---

## 🔄 Data Preprocessing

All preprocessing is implemented inside `app.py`:

1. **Duplicate removal** — no duplicates found in the Cleveland dataset
2. **Missing value handling** — `?` values in `ca` imputed with median; `thal` imputed with mode
3. **Target binarisation** — `num` column: 0 → 0 (No Disease), 1/2/3/4 → 1 (Disease)
4. **Type casting** — categorical/ordinal columns cast to `int`
5. **One-hot encoding** — applied to `cp`, `restecg`, `slope`, `thal`
6. **Feature matrix construction** — built dynamically; target column excluded

---

## 🔬 Exploratory Data Analysis

EDA section includes:
- Age distribution by target
- Sex distribution
- Chest pain type vs target
- Cholesterol and blood pressure distributions
- Maximum heart rate distribution
- Exercise-induced angina vs target
- ST depression (oldpeak) distribution
- Thalassemia vs target
- Resting ECG distribution
- Pearson correlation heatmap

---

## 📐 Analytics Methodology

| Phase | Question | Approach |
|-------|----------|----------|
| Descriptive | What happened? | Summary statistics, distributions, KPIs |
| Diagnostic | What patterns are observed? | Feature vs target correlations and group rates |
| Predictive | What does the model predict? | Random Forest classification |
| Prescriptive | What actions can be derived? | Risk segment analysis, recommendations |

---

## 🤖 Machine Learning Methodology

- **Algorithm:** `RandomForestClassifier`
- **n_estimators:** 200
- **max_depth:** 12
- **class_weight:** `"balanced"`
- **random_state:** 42
- **Split:** 80% train / 20% test (stratified)
- **Encoding:** One-hot encoding for multi-class categoricals
- **Storage:** Model trained in memory; reused via `@st.cache_resource`

> ⚠️ **The Random Forest model is trained in memory and reused through Streamlit caching while the application is running. No `.pkl` or `.joblib` files are created or required.**

---

## 📊 Model Evaluation Metrics

The following metrics are calculated dynamically from actual model execution:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision–Recall Curve
- Feature Importance

*See the Predictive Analytics section of the dashboard for actual values.*

---

## 📈 Dashboard Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | 🏠 Home | Project overview, objectives, tech stack |
| 2 | 📂 Dataset Overview | Schema, data quality, preprocessing log |
| 3 | 📈 KPI Dashboard | Key metrics with sidebar filters |
| 4 | 🔎 EDA | Comprehensive visual exploration |
| 5 | 📋 Descriptive Analytics | Summary statistics, population analysis |
| 6 | 🔬 Diagnostic Analytics | Feature associations with target |
| 7 | 🤖 Predictive Analytics | ML model, evaluation, interactive prediction |
| 8 | 💡 Prescriptive Insights | Risk segments, analytical insights |
| 9 | 📌 Business Recommendations | 7 data-driven recommendations |

---

## 💡 Key Findings

1. **Asymptomatic chest pain (Type 4)** is associated with the highest disease prevalence in this dataset.
2. **Number of major vessels (ca)** is the strongest predictor in the Random Forest model.
3. **Lower maximum heart rate (thalach)** is negatively correlated with heart disease.
4. **Exercise-induced angina** is strongly associated with higher disease rates.
5. **Reversable thalassemia defect** shows higher disease association than fixed defect.
6. **Males** show higher observed disease prevalence than females in this dataset.

*All findings are based on correlational analysis of this dataset and are NOT causal claims.*

---

## 🔧 Installation & Running Instructions

### Prerequisites

- Python 3.10 or later
- `pip` package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Dataset Setup

The application automatically searches for:
1. `heart+disease/processed.cleveland.data` (local, relative to `app.py`)
2. UCI ML Repository (automatic download fallback via `urllib.request`)

No manual dataset placement is required if internet access is available.

---

## 🔮 Future Improvements

1. Incorporate the full multi-centre dataset (Hungarian, Swiss, Long Beach) for improved model generalisation.
2. Experiment with XGBoost, LightGBM, and ensemble stacking for potentially higher performance.
3. Add SHAP (SHapley Additive exPlanations) for model interpretability.
4. Implement cross-validation for more robust metric estimation.
5. Add automated data quality reports using tools like `ydata-profiling`.
6. Deploy on Streamlit Community Cloud for public accessibility.
7. Explore time-series augmentation if temporal patient data becomes available.

---

## 📚 References

1. Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64, 304–310.
2. UCI Machine Learning Repository — Heart Disease Dataset: https://archive.ics.uci.edu/ml/datasets/heart+disease
3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
4. Scikit-learn Documentation: https://scikit-learn.org/
5. Streamlit Documentation: https://docs.streamlit.io/
6. Plotly Python Graphing Library: https://plotly.com/python/

---

## ⚠️ Medical Disclaimer

> This project is for **educational and analytical purposes only**. Machine-learning predictions are **not medical diagnoses** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## 👤 Author

**Vaishnavi**  
IBM Internship Capstone Project  
Domain: Artificial Intelligence & Data Science
