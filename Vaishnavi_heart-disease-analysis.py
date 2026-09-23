"""
Heart Disease Risk Analysis and Prediction System
IBM Internship Capstone Project
Student: Vaishnavi
Domain: Artificial Intelligence & Data Science

Dataset: UCI Heart Disease (Cleveland) — processed.cleveland.data
Source:  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
"""

import os
import io
import urllib.request
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "heart+disease", "processed.cleveland.data")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

# Human-readable labels used for display
COLUMN_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl",
    "restecg":  "Resting ECG Results",
    "thalach":  "Maximum Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Exercise vs Rest)",
    "slope":    "Slope of Peak Exercise ST Segment",
    "ca":       "Number of Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia",
    "num":      "Heart Disease Diagnosis (target)",
}

CP_MAP   = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}
SEX_MAP  = {0: "Female", 1: "Male"}
FBS_MAP  = {0: "≤120 mg/dl", 1: ">120 mg/dl"}
ECG_MAP  = {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}
EXANG_MAP= {0: "No", 1: "Yes"}
SLOPE_MAP= {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_MAP = {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}

DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** This project is for educational and analytical purposes only. "
    "Machine-learning predictions are not medical diagnoses and should not be used as a "
    "substitute for professional medical advice, diagnosis, or treatment."
)

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load raw dataset from local path; fall back to UCI download."""
    if os.path.isfile(LOCAL_PATH):
        raw = pd.read_csv(LOCAL_PATH, header=None, names=COLUMNS, na_values="?")
    else:
        try:
            with urllib.request.urlopen(DATASET_URL, timeout=15) as resp:
                content = resp.read().decode("utf-8")
            raw = pd.read_csv(
                io.StringIO(content), header=None, names=COLUMNS, na_values="?"
            )
        except Exception as exc:
            st.error(
                f"❌ Could not load dataset.\n\n"
                f"Place `processed.cleveland.data` in the same folder as `app.py` "
                f"or ensure internet access for automatic download.\n\nError: {exc}"
            )
            st.stop()
    return raw


@st.cache_data(show_spinner=False)
def preprocess_data(raw: pd.DataFrame):
    """
    Full preprocessing pipeline.
    Returns (df_clean, df_encoded, feature_cols, target_col, preprocessing_log).
    """
    log = []
    df = raw.copy()

    # 1. Record initial shape
    log.append(f"Initial dataset shape: {df.shape[0]} rows × {df.shape[1]} columns.")

    # 2. Duplicate removal
    n_dup = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    log.append(f"Duplicate rows removed: {n_dup}.")

    # 3. Missing-value handling
    missing_before = df.isnull().sum()
    mv_info = missing_before[missing_before > 0].to_dict()
    log.append(f"Missing values (marked '?' → NaN) before imputation: {mv_info}.")

    # Impute 'ca' (numerical) with median
    df["ca"] = df["ca"].fillna(df["ca"].median())
    # Impute 'thal' (categorical-like, stored as float) with mode
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])
    log.append("Missing 'ca' imputed with median; missing 'thal' imputed with mode.")

    # 4. Binarise target: 0 = no disease, 1 = disease (original values 1-4 → 1)
    df["target"] = (df["num"] > 0).astype(int)
    log.append(
        "Target 'num' binarised: 0 → 0 (No Heart Disease), 1/2/3/4 → 1 (Heart Disease)."
    )

    # 5. Cast columns to proper types
    int_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
    for c in int_cols:
        df[c] = df[c].astype(int)
    log.append("Categorical/ordinal columns cast to int.")

    # 6. Build encoded dataframe for ML
    df_enc = df.drop(columns=["num"]).copy()
    # One-hot encode multi-class categoricals: cp, restecg, slope, thal
    ohe_cols = ["cp", "restecg", "slope", "thal"]
    df_enc = pd.get_dummies(df_enc, columns=ohe_cols, drop_first=False)
    log.append(f"One-hot encoding applied to: {ohe_cols}.")

    feature_cols = [c for c in df_enc.columns if c != "target"]
    log.append(f"Final feature count: {len(feature_cols)}.")

    return df, df_enc, feature_cols, "target", log


# ─── Model Training ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_model(_df_enc: pd.DataFrame, feature_cols: list, target_col: str):
    """Train a RandomForestClassifier and return model + split data."""
    X = _df_enc[feature_cols]
    y = _df_enc[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred       = clf.predict(X_test)
    y_prob       = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "cm":        confusion_matrix(y_test, y_pred),
        "cr":        classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]),
        "fpr":       roc_curve(y_test, y_prob)[0],
        "tpr":       roc_curve(y_test, y_prob)[1],
        "pr_prec":   precision_recall_curve(y_test, y_prob)[0],
        "pr_rec":    precision_recall_curve(y_test, y_prob)[1],
        "feature_importances": pd.Series(
            clf.feature_importances_, index=feature_cols
        ).sort_values(ascending=False),
    }

    return clf, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics


# ─── Sidebar Filters ──────────────────────────────────────────────────────────

def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=120,
    )
    st.sidebar.title("🔍 Dashboard Filters")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider(
        "Age Range", age_min, age_max, (age_min, age_max)
    )

    sex_opts = ["All"] + [SEX_MAP[v] for v in sorted(df["sex"].unique())]
    sex_sel  = st.sidebar.selectbox("Sex", sex_opts)

    target_opts = ["All", "No Heart Disease (0)", "Heart Disease (1)"]
    target_sel  = st.sidebar.selectbox("Heart Disease Status", target_opts)

    cp_opts = ["All"] + [CP_MAP.get(v, str(v)) for v in sorted(df["cp"].unique())]
    cp_sel  = st.sidebar.selectbox("Chest Pain Type", cp_opts)

    # Apply filters
    dff = df.copy()
    dff = dff[(dff["age"] >= age_range[0]) & (dff["age"] <= age_range[1])]
    if sex_sel != "All":
        sex_code = [k for k, v in SEX_MAP.items() if v == sex_sel][0]
        dff = dff[dff["sex"] == sex_code]
    if target_sel != "All":
        t_code = 0 if "0" in target_sel else 1
        dff = dff[dff["target"] == t_code]
    if cp_sel != "All":
        cp_code = [k for k, v in CP_MAP.items() if v == cp_sel][0]
        dff = dff[dff["cp"] == cp_code]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Filtered records: **{len(dff)}** / {len(df)}")
    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)
    return dff


# ─── Section Renderers ────────────────────────────────────────────────────────

def render_home():
    st.title("❤️ Heart Disease Risk Analysis and Prediction System")
    st.markdown("### IBM Internship Capstone Project")
    st.markdown("**Student:** Vaishnavi &nbsp;|&nbsp; **Domain:** Artificial Intelligence & Data Science")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📌 Project Overview")
        st.markdown(
            """
            This capstone project presents a comprehensive **Heart Disease Risk Analysis and Prediction System**
            built on the UCI Cleveland Heart Disease dataset. The system integrates a full analytics pipeline
            covering descriptive, diagnostic, predictive, and prescriptive analytics, culminating in an
            interactive Streamlit dashboard.
            """
        )
        st.subheader("🎯 Problem Statement")
        st.markdown(
            """
            Cardiovascular diseases are among the leading causes of mortality worldwide. Early identification
            of at-risk individuals through data-driven analysis can support timely clinical attention.
            This project uses machine learning to model patterns observed in clinical patient data and provides
            an analytical risk-scoring interface.
            """
        )
        st.subheader("🏆 Objectives")
        objectives = [
            "Perform thorough exploratory data analysis on the Cleveland Heart Disease dataset.",
            "Apply descriptive analytics to summarise patient population characteristics.",
            "Conduct diagnostic analytics to identify feature associations with heart disease.",
            "Build and evaluate a Random Forest classification model for risk prediction.",
            "Provide prescriptive insights and data-driven analytical recommendations.",
            "Deliver an interactive, filterable Streamlit dashboard for data exploration.",
        ]
        for obj in objectives:
            st.markdown(f"✅ {obj}")
    with col2:
        st.subheader("📊 Dataset")
        st.markdown(
            """
            - **Name:** UCI Heart Disease (Cleveland)
            - **Records:** 303 patient records
            - **Features:** 13 clinical attributes
            - **Target:** `num` — binarised (0/1)
            - **Source:** UCI Machine Learning Repository
            """
        )
        st.subheader("🔬 Analytics Pipeline")
        steps = {
            "📋 Descriptive": "What happened? — Summary statistics and distributions",
            "🔍 Diagnostic": "What patterns/relationships are observed?",
            "🤖 Predictive": "What does the ML model predict?",
            "💡 Prescriptive": "What analytical actions can be derived?",
        }
        for step, desc in steps.items():
            st.info(f"**{step}**\n{desc}")
    st.subheader("🛠 Technology Stack")
    tech = {
        "Python 3.10+": "Primary programming language",
        "Pandas": "Data loading, cleaning, manipulation",
        "NumPy": "Numerical computation",
        "Scikit-learn": "Machine learning and evaluation",
        "Plotly": "Interactive visualisations",
        "Streamlit": "Interactive web dashboard",
        "Python Standard Library (urllib, io, os)": "Dataset download / file handling",
    }
    tech_df = pd.DataFrame(list(tech.items()), columns=["Technology", "Purpose"])
    st.dataframe(tech_df, use_container_width=True, hide_index=True)
    st.warning(DISCLAIMER)


def render_dataset_overview(raw: pd.DataFrame, df: pd.DataFrame, log: list):
    st.title("📂 Dataset Overview")
    st.markdown("---")

    # Quality metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", raw.shape[0])
    col2.metric("Total Columns", raw.shape[1])
    col3.metric("Missing Values (raw)", int(raw.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(raw.duplicated().sum()))

    st.subheader("🎯 Target Variable")
    col1, col2, col3 = st.columns(3)
    col1.metric("Target Column", "num  →  target (binarised)")
    col2.metric("No Heart Disease (0)", int((df["target"] == 0).sum()))
    col3.metric("Heart Disease (1)", int((df["target"] == 1).sum()))

    st.subheader("📋 Column Descriptions")
    desc_data = []
    for col in COLUMNS:
        dtype = raw[col].dtype
        n_miss = int(raw[col].isnull().sum())
        n_uniq = int(raw[col].nunique(dropna=False))
        desc_data.append({"Column": col, "Label": COLUMN_LABELS[col],
                          "Dtype": str(dtype), "Missing": n_miss, "Unique Values": n_uniq})
    st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)

    st.subheader("📊 Data Types Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numerical Features**")
        num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        st.dataframe(raw[num_cols].describe().round(2), use_container_width=True)
    with col2:
        st.markdown("**Categorical / Ordinal Features**")
        cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "num"]
        st.dataframe(raw[cat_cols].describe().round(2), use_container_width=True)

    st.subheader("🔍 Sample Rows (first 10)")
    st.dataframe(raw.head(10), use_container_width=True)

    st.subheader("📝 Preprocessing Log")
    for entry in log:
        st.markdown(f"- {entry}")


def render_kpi_dashboard(dff: pd.DataFrame):
    st.title("📈 KPI Dashboard")
    st.markdown("*(Values reflect sidebar filters)*")
    st.markdown("---")

    total   = len(dff)
    disease = int((dff["target"] == 1).sum())
    no_dis  = int((dff["target"] == 0).sum())
    pct_dis = round(disease / total * 100, 1) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", total)
    col2.metric("With Heart Disease", disease, f"{pct_dis}%")
    col3.metric("Without Heart Disease", no_dis, f"{100-pct_dis}%")
    col4.metric("Disease Prevalence", f"{pct_dis}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Age", f"{dff['age'].mean():.1f} yrs")
    col2.metric("Avg Cholesterol", f"{dff['chol'].mean():.1f} mg/dl")
    col3.metric("Avg Resting BP", f"{dff['trestbps'].mean():.1f} mm Hg")
    col4.metric("Avg Max Heart Rate", f"{dff['thalach'].mean():.1f} bpm")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        target_dist = dff["target"].value_counts().reset_index()
        target_dist.columns = ["Target", "Count"]
        target_dist["Label"] = target_dist["Target"].map({0: "No Disease", 1: "Disease"})
        fig = px.pie(
            target_dist, names="Label", values="Count",
            title="Target Distribution",
            color_discrete_sequence=["#3b82d4", "#e05252"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sex_dist = dff["sex"].map(SEX_MAP).value_counts().reset_index()
        sex_dist.columns = ["Sex", "Count"]
        fig = px.bar(
            sex_dist, x="Sex", y="Count", title="Sex Distribution",
            color="Sex", color_discrete_sequence=["#7c5cd8", "#3b82d4"],
        )
        st.plotly_chart(fig, use_container_width=True)


def render_eda(dff: pd.DataFrame):
    st.title("🔎 Exploratory Data Analysis")
    st.markdown("*(Charts reflect sidebar filters)*")
    st.markdown("---")

    # Age distribution
    st.subheader("Age Distribution")
    fig = px.histogram(
        dff, x="age", nbins=20, color="target",
        color_discrete_map={0: "#3b82d4", 1: "#e05252"},
        labels={"age": "Age", "target": "Heart Disease"},
        title="Age Distribution by Heart Disease Status",
        barmode="overlay",
    )
    fig.update_layout(legend=dict(title="0=No Disease, 1=Disease"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chest Pain Type vs Target")
        cp_tgt = (
            dff.groupby(["cp", "target"])
            .size()
            .reset_index(name="Count")
        )
        cp_tgt["CP Label"]  = cp_tgt["cp"].map(CP_MAP)
        cp_tgt["Tgt Label"] = cp_tgt["target"].map({0: "No Disease", 1: "Disease"})
        fig = px.bar(
            cp_tgt, x="CP Label", y="Count", color="Tgt Label",
            barmode="group", title="Chest Pain Type vs Heart Disease",
            color_discrete_sequence=["#3b82d4", "#e05252"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Sex vs Target")
        sex_tgt = (
            dff.groupby(["sex", "target"])
            .size()
            .reset_index(name="Count")
        )
        sex_tgt["Sex Label"] = sex_tgt["sex"].map(SEX_MAP)
        sex_tgt["Tgt Label"] = sex_tgt["target"].map({0: "No Disease", 1: "Disease"})
        fig = px.bar(
            sex_tgt, x="Sex Label", y="Count", color="Tgt Label",
            barmode="group", title="Sex vs Heart Disease",
            color_discrete_sequence=["#3b82d4", "#e05252"],
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cholesterol Distribution")
        fig = px.box(
            dff, x="target", y="chol",
            color="target",
            color_discrete_map={0: "#3b82d4", 1: "#e05252"},
            labels={"target": "Heart Disease", "chol": "Cholesterol (mg/dl)"},
            title="Cholesterol by Heart Disease Status",
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No Disease", "Disease"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Max Heart Rate Distribution")
        fig = px.box(
            dff, x="target", y="thalach",
            color="target",
            color_discrete_map={0: "#3b82d4", 1: "#e05252"},
            labels={"target": "Heart Disease", "thalach": "Max Heart Rate (bpm)"},
            title="Maximum Heart Rate by Heart Disease Status",
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No Disease", "Disease"])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Exercise-Induced Angina vs Target")
        exang_tgt = (
            dff.groupby(["exang", "target"])
            .size()
            .reset_index(name="Count")
        )
        exang_tgt["Angina"]    = exang_tgt["exang"].map(EXANG_MAP)
        exang_tgt["Tgt Label"] = exang_tgt["target"].map({0: "No Disease", 1: "Disease"})
        fig = px.bar(
            exang_tgt, x="Angina", y="Count", color="Tgt Label",
            barmode="group", title="Exercise-Induced Angina vs Heart Disease",
            color_discrete_sequence=["#3b82d4", "#e05252"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("ST Depression (oldpeak) vs Target")
        fig = px.violin(
            dff, x="target", y="oldpeak",
            color="target",
            color_discrete_map={0: "#3b82d4", 1: "#e05252"},
            labels={"target": "Heart Disease", "oldpeak": "ST Depression"},
            title="ST Depression by Heart Disease Status",
            box=True,
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No Disease", "Disease"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap (Numerical Features)")
    num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak", "target"]
    corr = dff[num_feats].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resting Blood Pressure Distribution")
    fig = px.histogram(
        dff, x="trestbps", nbins=25, color="target",
        color_discrete_map={0: "#3b82d4", 1: "#e05252"},
        labels={"trestbps": "Resting BP (mm Hg)", "target": "Heart Disease"},
        title="Resting Blood Pressure Distribution",
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Thalassemia vs Target")
    thal_tgt = (
        dff.groupby(["thal", "target"])
        .size()
        .reset_index(name="Count")
    )
    thal_tgt["Thal Label"] = thal_tgt["thal"].map(THAL_MAP)
    thal_tgt["Tgt Label"]  = thal_tgt["target"].map({0: "No Disease", 1: "Disease"})
    fig = px.bar(
        thal_tgt, x="Thal Label", y="Count", color="Tgt Label",
        barmode="group", title="Thalassemia Type vs Heart Disease",
        color_discrete_sequence=["#3b82d4", "#e05252"],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_descriptive(dff: pd.DataFrame):
    st.title("📋 Descriptive Analytics")
    st.markdown("**Question answered:** *What happened in the dataset?*")
    st.markdown("---")

    total   = len(dff)
    disease = int((dff["target"] == 1).sum())
    no_dis  = int((dff["target"] == 0).sum())

    st.subheader("Population Summary")
    st.markdown(
        f"The filtered dataset contains **{total}** patient records. "
        f"Of these, **{disease}** ({round(disease/total*100,1)}%) show heart disease presence "
        f"and **{no_dis}** ({round(no_dis/total*100,1)}%) show no heart disease."
    )

    st.subheader("Numerical Feature Summary Statistics")
    num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    desc = dff[num_feats].describe().round(2)
    desc.index.name = "Statistic"
    st.dataframe(desc, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Group Distribution")
        dff_copy = dff.copy()
        bins = [0, 40, 50, 60, 70, 100]
        labels = ["<40", "40-49", "50-59", "60-69", "70+"]
        dff_copy["Age Group"] = pd.cut(dff_copy["age"], bins=bins, labels=labels, right=False)
        age_grp = (
            dff_copy.groupby(["Age Group", "target"])
            .size()
            .reset_index(name="Count")
        )
        age_grp["Tgt Label"] = age_grp["target"].map({0: "No Disease", 1: "Disease"})
        fig = px.bar(
            age_grp, x="Age Group", y="Count", color="Tgt Label",
            barmode="stack", title="Age Group Distribution by Target",
            color_discrete_sequence=["#3b82d4", "#e05252"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Fasting Blood Sugar Distribution")
        fbs_dist = dff["fbs"].map(FBS_MAP).value_counts().reset_index()
        fbs_dist.columns = ["FBS", "Count"]
        fig = px.pie(
            fbs_dist, names="FBS", values="Count",
            title="Fasting Blood Sugar (> 120 mg/dl)",
            color_discrete_sequence=["#3b82d4", "#7c5cd8"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resting ECG Results")
    ecg_dist = dff["restecg"].map(ECG_MAP).value_counts().reset_index()
    ecg_dist.columns = ["ECG Result", "Count"]
    fig = px.bar(
        ecg_dist, x="ECG Result", y="Count",
        title="Resting ECG Result Distribution",
        color="ECG Result",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Descriptive Findings")
    insights = [
        f"The average patient age is **{dff['age'].mean():.1f}** years (range: {int(dff['age'].min())}–{int(dff['age'].max())}).",
        f"Average serum cholesterol is **{dff['chol'].mean():.1f} mg/dl**.",
        f"Average resting blood pressure is **{dff['trestbps'].mean():.1f} mm Hg**.",
        f"Average maximum heart rate achieved is **{dff['thalach'].mean():.1f} bpm**.",
        f"Males constitute **{round(dff['sex'].mean()*100,1)}%** of the dataset.",
        f"**{round((dff['fbs']==1).mean()*100,1)}%** of patients have fasting blood sugar > 120 mg/dl.",
    ]
    for ins in insights:
        st.markdown(f"📌 {ins}")


def render_diagnostic(dff: pd.DataFrame):
    st.title("🔬 Diagnostic Analytics")
    st.markdown("**Question answered:** *Which factors are associated with heart disease in this dataset?*")
    st.warning(
        "⚠️ Note: Associations observed below are correlational, not causal. "
        "They reflect patterns in this dataset and should not be interpreted as clinical diagnoses."
    )
    st.markdown("---")

    # Disease rates by sex
    st.subheader("Disease Prevalence by Sex")
    sex_rate = (
        dff.groupby("sex")["target"]
        .mean()
        .reset_index()
    )
    sex_rate["Sex"]          = sex_rate["sex"].map(SEX_MAP)
    sex_rate["Disease Rate"] = (sex_rate["target"] * 100).round(1)
    fig = px.bar(
        sex_rate, x="Sex", y="Disease Rate",
        title="Heart Disease Rate (%) by Sex",
        color="Sex", color_discrete_sequence=["#7c5cd8", "#3b82d4"],
        text="Disease Rate",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Observed disease rate: Males show a higher prevalence in this dataset.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age vs Disease")
        fig = px.box(
            dff, x="target", y="age",
            color="target",
            color_discrete_map={0: "#3b82d4", 1: "#e05252"},
            labels={"target": "Heart Disease", "age": "Age"},
            title="Age Distribution by Heart Disease Status",
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No Disease", "Disease"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Patients with heart disease tend to be slightly older in this dataset.")
    with col2:
        st.subheader("Max Heart Rate vs Disease")
        fig = px.box(
            dff, x="target", y="thalach",
            color="target",
            color_discrete_map={0: "#3b82d4", 1: "#e05252"},
            labels={"target": "Heart Disease", "thalach": "Max Heart Rate (bpm)"},
            title="Max Heart Rate by Heart Disease Status",
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No Disease", "Disease"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lower maximum heart rate is associated with higher disease prevalence here.")

    st.subheader("Chest Pain Type Disease Rates")
    cp_rate = (
        dff.groupby("cp")["target"]
        .agg(["mean", "count"])
        .reset_index()
    )
    cp_rate["CP Label"]      = cp_rate["cp"].map(CP_MAP)
    cp_rate["Disease Rate %"] = (cp_rate["mean"] * 100).round(1)
    cp_rate["Patient Count"] = cp_rate["count"]
    fig = px.bar(
        cp_rate, x="CP Label", y="Disease Rate %",
        color="Disease Rate %",
        color_continuous_scale="Reds",
        text="Disease Rate %",
        title="Disease Rate (%) by Chest Pain Type",
        hover_data=["Patient Count"],
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Asymptomatic chest pain (Type 4) is associated with the highest observed disease rate, "
        "which is a counter-intuitive but well-documented pattern in this dataset."
    )

    st.subheader("ST Depression (oldpeak) by Disease")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            dff, x="oldpeak", color="target",
            color_discrete_map={0: "#3b82d4", 1: "#e05252"},
            barmode="overlay", nbins=20,
            title="ST Depression Distribution",
            labels={"oldpeak": "ST Depression", "target": "Heart Disease"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        slope_rate = (
            dff.groupby("slope")["target"]
            .mean()
            .reset_index()
        )
        slope_rate["Slope Label"]   = slope_rate["slope"].map(SLOPE_MAP)
        slope_rate["Disease Rate %"] = (slope_rate["target"] * 100).round(1)
        fig = px.bar(
            slope_rate, x="Slope Label", y="Disease Rate %",
            title="Disease Rate by Slope of ST Segment",
            color="Disease Rate %",
            color_continuous_scale="Reds",
            text="Disease Rate %",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Number of Major Vessels (ca) vs Disease")
    ca_rate = (
        dff.groupby("ca")["target"]
        .mean()
        .reset_index()
    )
    ca_rate["Disease Rate %"] = (ca_rate["target"] * 100).round(1)
    fig = px.bar(
        ca_rate, x="ca", y="Disease Rate %",
        title="Disease Rate (%) by Number of Major Vessels",
        labels={"ca": "Major Vessels (0–3)", "Disease Rate %": "Disease Rate (%)"},
        color="Disease Rate %",
        color_continuous_scale="Reds",
        text="Disease Rate %",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Higher number of vessels coloured by fluoroscopy is associated with higher disease rate in this dataset."
    )

    st.subheader("Feature Correlation with Target")
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    corr_vals = dff[num_cols + ["target"]].corr()["target"].drop("target").sort_values()
    fig = px.bar(
        x=corr_vals.values,
        y=corr_vals.index,
        orientation="h",
        title="Pearson Correlation of Numerical Features with Target",
        labels={"x": "Correlation Coefficient", "y": "Feature"},
        color=corr_vals.values,
        color_continuous_scale="RdBu_r",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Negative correlation with thalach (max heart rate) and positive correlation with "
        "oldpeak (ST depression) and ca (vessels) are observed. These are correlational, not causal."
    )


def render_predictive(df_enc, feature_cols, metrics, clf, X_test, y_test, y_pred, y_prob, df_clean):
    st.title("🤖 Predictive Analytics")
    st.markdown("**Question answered:** *What does the machine learning model predict?*")
    st.warning(DISCLAIMER)
    st.markdown("---")

    n_total = len(df_enc)
    n_train = int(n_total * 0.8)
    n_test  = n_total - n_train

    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            | Parameter | Value |
            |-----------|-------|
            | Algorithm | Random Forest Classifier |
            | n_estimators | 200 |
            | max_depth | 12 |
            | class_weight | balanced |
            | random_state | 42 |
            | Test Split | 20% (stratified) |
            | Dataset Size | {n_total} |
            | Training Size | {n_train} |
            | Test Size | {n_test} |
            | Number of Features | {len(feature_cols)} |
            """
        )
    with col2:
        st.subheader("Performance Metrics")
        m = metrics
        metric_rows = {
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Precision": f"{m['precision']*100:.2f}%",
            "Recall": f"{m['recall']*100:.2f}%",
            "F1 Score": f"{m['f1']*100:.2f}%",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
        }
        for name, val in metric_rows.items():
            st.metric(name, val)

    st.subheader("Confusion Matrix")
    cm = metrics["cm"]
    cm_labels = ["No Disease (0)", "Disease (1)"]
    fig = px.imshow(
        cm, text_auto=True, x=cm_labels, y=cm_labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual"},
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics["fpr"], y=metrics["tpr"],
            mode="lines",
            name=f"ROC (AUC = {metrics['roc_auc']:.4f})",
            line=dict(color="#3b82d4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="gray", dash="dash"),
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Precision–Recall Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics["pr_rec"], y=metrics["pr_prec"],
            mode="lines",
            name="Precision-Recall",
            line=dict(color="#e05252", width=2),
        ))
        fig.update_layout(
            title="Precision–Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    st.code(metrics["cr"])

    st.subheader("Top 15 Feature Importances")
    fi = metrics["feature_importances"].head(15).reset_index()
    fi.columns = ["Feature", "Importance"]
    fig = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        title="Top 15 Feature Importances (Random Forest)",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Interactive Prediction Interface ───────────────────────────────────────
    st.markdown("---")
    st.subheader("🩺 Interactive Patient Risk Prediction")
    st.info(
        "Enter patient clinical attributes below to generate a model-based analytical risk score. "
        "This is NOT a medical diagnosis."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_age      = st.number_input("Age (years)", 20, 90, 55)
            p_sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: SEX_MAP[x])
            p_cp       = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: CP_MAP[x])
            p_trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
            p_chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        with col2:
            p_fbs      = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: FBS_MAP[x])
            p_restecg  = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: ECG_MAP[x])
            p_thalach  = st.number_input("Max Heart Rate (bpm)", 60, 220, 150)
            p_exang    = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: EXANG_MAP[x])
            p_oldpeak  = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        with col3:
            p_slope    = st.selectbox("ST Slope", options=[1, 2, 3], format_func=lambda x: SLOPE_MAP[x])
            p_ca       = st.selectbox("Major Vessels (0–3)", options=[0, 1, 2, 3])
            p_thal     = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: THAL_MAP[x])

        submitted = st.form_submit_button("🔍 Generate Risk Prediction", use_container_width=True)

    if submitted:
        # Build input row matching training feature columns
        input_dict = {
            "age": p_age, "sex": p_sex, "trestbps": p_trestbps, "chol": p_chol,
            "fbs": p_fbs, "thalach": p_thalach, "exang": p_exang, "oldpeak": p_oldpeak,
            "ca": p_ca,
        }
        # One-hot fields
        for v in [1, 2, 3, 4]:
            input_dict[f"cp_{v}"] = int(p_cp == v)
        for v in [0, 1, 2]:
            input_dict[f"restecg_{v}"] = int(p_restecg == v)
        for v in [1, 2, 3]:
            input_dict[f"slope_{v}"] = int(p_slope == v)
        for v in [3, 6, 7]:
            input_dict[f"thal_{v}"] = int(p_thal == v)

        input_df = pd.DataFrame([input_dict])
        # Align columns to training feature set
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        pred      = clf.predict(input_df)[0]
        pred_prob = clf.predict_proba(input_df)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(
                f"### ⚠️ Predicted Class: **Higher Predicted Risk**\n\n"
                f"**Model Risk Score: {pred_prob*100:.1f}%**\n\n"
                f"The model predicts a higher analytical risk score based on the entered attributes."
            )
        else:
            st.success(
                f"### ✅ Predicted Class: **Lower Predicted Risk**\n\n"
                f"**Model Risk Score: {pred_prob*100:.1f}%**\n\n"
                f"The model predicts a lower analytical risk score based on the entered attributes."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", "Higher Risk" if pred == 1 else "Lower Risk")
            st.metric("Risk Score", f"{pred_prob*100:.1f}%")
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Risk Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e05252" if pred == 1 else "#3b82d4"},
                    "steps": [
                        {"range": [0, 40], "color": "#d1fae5"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                },
            ))
            st.plotly_chart(fig, use_container_width=True)
        st.warning(DISCLAIMER)


def render_prescriptive(dff: pd.DataFrame, metrics: dict):
    st.title("💡 Prescriptive / Decision Insights")
    st.markdown("**Question answered:** *What data-driven actions or monitoring insights can be derived from the analysis?*")
    st.warning(DISCLAIMER)
    st.markdown("---")

    st.subheader("📊 High-Risk Observed Patient Segments")
    col1, col2 = st.columns(2)

    with col1:
        # Asymptomatic CP rate
        cp4_rate = round(dff[dff["cp"] == 4]["target"].mean() * 100, 1) if len(dff[dff["cp"] == 4]) > 0 else 0
        st.metric("Disease Rate — Asymptomatic CP (Type 4)", f"{cp4_rate}%")

        # Male disease rate
        male_rate = round(dff[dff["sex"] == 1]["target"].mean() * 100, 1) if len(dff[dff["sex"] == 1]) > 0 else 0
        st.metric("Disease Rate — Male Patients", f"{male_rate}%")

        # Exercise angina rate
        exang_rate = round(dff[dff["exang"] == 1]["target"].mean() * 100, 1) if len(dff[dff["exang"] == 1]) > 0 else 0
        st.metric("Disease Rate — Exercise-Induced Angina", f"{exang_rate}%")

    with col2:
        # Age > 60
        old_rate = round(dff[dff["age"] >= 60]["target"].mean() * 100, 1) if len(dff[dff["age"] >= 60]) > 0 else 0
        st.metric("Disease Rate — Age ≥ 60", f"{old_rate}%")

        # High oldpeak
        hi_op = round(dff[dff["oldpeak"] >= 2]["target"].mean() * 100, 1) if len(dff[dff["oldpeak"] >= 2]) > 0 else 0
        st.metric("Disease Rate — oldpeak ≥ 2", f"{hi_op}%")

        # ca > 0
        ca_pos = round(dff[dff["ca"] > 0]["target"].mean() * 100, 1) if len(dff[dff["ca"] > 0]) > 0 else 0
        st.metric("Disease Rate — Major Vessels (ca > 0)", f"{ca_pos}%")

    st.subheader("🎯 Model-Based Risk Stratification")
    st.markdown(
        f"""
        The Random Forest model achieved an ROC-AUC of **{metrics['roc_auc']:.4f}**, 
        indicating good discriminative ability between the two classes.

        **Analytical use cases** (educational only):
        - Use the model's predicted probability as a **risk prioritisation score** for further data review.
        - Patients with high risk scores could be flagged analytically for closer examination of their clinical features.
        - The model is NOT intended for clinical decision-making.
        """
    )

    st.subheader("📌 Key Insights from the Analysis")
    insights = [
        {
            "Insight": "Asymptomatic chest pain (Type 4) is associated with the highest disease rate in this dataset.",
            "Evidence": f"~{dff[dff['cp']==4]['target'].mean()*100:.1f}% of asymptomatic patients show heart disease.",
            "Action": "Analytical screening models may benefit from weighting asymptomatic presentation heavily.",
        },
        {
            "Insight": "Higher number of fluoroscopy-coloured major vessels (ca) is strongly associated with disease.",
            "Evidence": "Monotonic increase in disease rate with ca count is observed in this dataset.",
            "Action": "ca is a top feature in the Random Forest model and should be included in any analytical model.",
        },
        {
            "Insight": "Lower maximum heart rate (thalach) is associated with higher disease prevalence.",
            "Evidence": "Negative correlation observed between thalach and target.",
            "Action": "Thalach may be a useful inclusion in risk stratification algorithms.",
        },
        {
            "Insight": "Reversible thalassemia defect is associated with higher disease rates than fixed defect.",
            "Evidence": "Thalassemia type distribution shows different disease rates per category.",
            "Action": "Thal type should be included in any analytical feature set for heart disease modelling.",
        },
        {
            "Insight": "Exercise-induced angina is associated with significantly higher disease rates.",
            "Evidence": f"~{dff[dff['exang']==1]['target'].mean()*100:.1f}% of patients with exercise angina show disease.",
            "Action": "exang is a strong discriminating feature in this dataset.",
        },
    ]
    for ins in insights:
        with st.expander(f"📍 {ins['Insight']}"):
            st.markdown(f"**Evidence:** {ins['Evidence']}")
            st.markdown(f"**Analytical Action:** {ins['Action']}")


def render_recommendations(dff: pd.DataFrame, metrics: dict):
    st.title("📌 Business & Project Recommendations")
    st.markdown("**Based on actual findings from the analysis.**")
    st.warning(DISCLAIMER)
    st.markdown("---")

    recs = [
        {
            "title": "1. Prioritise analytical screening for asymptomatic patients",
            "finding": f"Asymptomatic chest pain (Type 4) is associated with ~{dff[dff['cp']==4]['target'].mean()*100:.1f}% disease prevalence in this dataset.",
            "evidence": "EDA and diagnostic analytics show the highest disease rate among asymptomatic patients.",
            "action": "Analytical workflows could flag asymptomatic patients for data-driven risk scoring before further clinical review.",
            "reason": "Counter-intuitive symptom patterns may be analytically identified earlier.",
        },
        {
            "title": "2. Incorporate vessel count (ca) as a high-priority feature",
            "finding": "Number of major vessels coloured by fluoroscopy is the strongest predictor in the Random Forest model.",
            "evidence": "Feature importance analysis and disease-rate-by-ca analysis confirm this finding.",
            "action": "Ensure ca is always included in any analytical heart disease model built on this dataset.",
            "reason": "Excluding this feature would significantly reduce model discriminative power.",
        },
        {
            "title": "3. Use the model's risk score as a prioritisation aid, not a diagnosis",
            "finding": f"The model achieves ROC-AUC of {metrics['roc_auc']:.4f} and F1 of {metrics['f1']:.4f}.",
            "evidence": "Model evaluation metrics demonstrate good but imperfect discrimination.",
            "action": "Treat model output as an analytical scoring tool for data exploration, not clinical advice.",
            "reason": "Machine learning models cannot replace clinical judgement or examinations.",
        },
        {
            "title": "4. Expand the dataset for improved model robustness",
            "finding": "The Cleveland dataset contains only 303 records, which limits model generalisation.",
            "evidence": "Small dataset size is a key limitation noted in this project.",
            "action": "Incorporate Hungarian, Swiss, and Long Beach datasets (combined: ~920 records) for broader training.",
            "reason": "Larger, more diverse data improves model robustness and reduces variance.",
        },
        {
            "title": "5. Apply the dashboard for population-level analytics",
            "finding": "The interactive dashboard enables filterable exploration of patient subgroups.",
            "evidence": "KPI dashboard and EDA sections support dynamic data investigation.",
            "action": "Use the dashboard for population health analytics and exploratory data tasks.",
            "reason": "Visual analytics dashboards accelerate pattern discovery in clinical datasets.",
        },
        {
            "title": "6. Monitor model performance over time with new data",
            "finding": "Static models trained on historical data may degrade as patient demographics shift.",
            "evidence": "Model trained on 1988 Cleveland data; applicability to modern populations needs validation.",
            "action": "Re-train and re-evaluate the model periodically when new data becomes available.",
            "reason": "Concept drift is a real concern for clinical ML models.",
        },
        {
            "title": "7. Address class imbalance with advanced techniques if extending the project",
            "finding": "Dataset has mild class imbalance (54% no disease / 46% disease).",
            "evidence": "Class distribution analysis shows near-balanced but not perfectly equal classes.",
            "action": "If extending this model, techniques such as SMOTE or further class weighting can be explored.",
            "reason": "Balanced evaluation metrics (F1, ROC-AUC) are more informative than raw accuracy alone.",
        },
    ]

    for rec in recs:
        with st.expander(f"✅ {rec['title']}"):
            st.markdown(f"**Finding:** {rec['finding']}")
            st.markdown(f"**Evidence:** {rec['evidence']}")
            st.markdown(f"**Suggested Action:** {rec['action']}")
            st.markdown(f"**Reason:** {rec['reason']}")

    st.markdown("---")
    st.info(
        "**Model Note:** The Random Forest model is trained in memory and reused through "
        "Streamlit caching while the application is running. No external model files (`.pkl`, `.joblib`) "
        "are created or required."
    )


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Heart Disease Risk Analysis",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load and process data
    with st.spinner("Loading dataset..."):
        raw = load_raw_data()
    with st.spinner("Preprocessing data..."):
        df_clean, df_enc, feature_cols, target_col, preproc_log = preprocess_data(raw)
    with st.spinner("Training model..."):
        clf, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics = train_model(
            df_enc, feature_cols, target_col
        )

    # Sidebar filters (applied on df_clean for visualisations)
    dff = apply_sidebar_filters(df_clean)

    # Navigation
    st.sidebar.markdown("---")
    sections = [
        "🏠 Home",
        "📂 Dataset Overview",
        "📈 KPI Dashboard",
        "🔎 Exploratory Data Analysis",
        "📋 Descriptive Analytics",
        "🔬 Diagnostic Analytics",
        "🤖 Predictive Analytics",
        "💡 Prescriptive / Decision Insights",
        "📌 Business Recommendations",
    ]
    section = st.sidebar.radio("Navigate to Section", sections)
    st.sidebar.markdown("---")
    st.sidebar.caption("**IBM Internship Capstone** | Vaishnavi | AI & Data Science")

    if section == "🏠 Home":
        render_home()
    elif section == "📂 Dataset Overview":
        render_dataset_overview(raw, df_clean, preproc_log)
    elif section == "📈 KPI Dashboard":
        render_kpi_dashboard(dff)
    elif section == "🔎 Exploratory Data Analysis":
        render_eda(dff)
    elif section == "📋 Descriptive Analytics":
        render_descriptive(dff)
    elif section == "🔬 Diagnostic Analytics":
        render_diagnostic(dff)
    elif section == "🤖 Predictive Analytics":
        render_predictive(df_enc, feature_cols, metrics, clf, X_test, y_test, y_pred, y_prob, df_clean)
    elif section == "💡 Prescriptive / Decision Insights":
        render_prescriptive(dff, metrics)
    elif section == "📌 Business Recommendations":
        render_recommendations(dff, metrics)


if __name__ == "__main__":
    main()
