import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ChurnAI Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# THEME SWITCH
# ---------------------------------------------------
theme = st.sidebar.selectbox(
    "🎨 Theme",
    ["Ocean Blue", "Royal Purple", "Emerald Green", "Midnight Dark"]
)

themes = {
    "Ocean Blue": {
        "bg": "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)",
        "card": "rgba(255,255,255,0.08)",
        "accent": "#00C6FF"
    },
    "Royal Purple": {
        "bg": "linear-gradient(135deg, #42275a 0%, #734b6d 100%)",
        "card": "rgba(255,255,255,0.08)",
        "accent": "#cc2b5e"
    },
    "Emerald Green": {
        "bg": "linear-gradient(135deg, #134E5E 0%, #71B280 100%)",
        "card": "rgba(255,255,255,0.08)",
        "accent": "#56ab2f"
    },
    "Midnight Dark": {
        "bg": "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
        "card": "rgba(255,255,255,0.05)",
        "accent": "#f953c6"
    }
}

t = themes[theme]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {t['bg']};
}}

.glass-card {{
    background: {t['card']};
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}}

.kpi-card {{
    background: {t['card']};
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(8px);
}}

.brand-header {{
    font-size: 36px;
    font-weight: 700;
    color: {t['accent']};
    letter-spacing: 1px;
}}

.sub-header {{
    font-size: 14px;
    color: rgba(255,255,255,0.6);
    margin-top: -10px;
}}

.stButton>button {{
    background: linear-gradient(90deg, {t['accent']}, #0072FF);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 15px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}}

.stButton>button:hover {{
    opacity: 0.85;
    transform: scale(1.02);
}}

div[data-testid="stMetric"] {{
    background: {t['card']};
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: rgba(255,255,255,0.05);
    padding: 8px;
    border-radius: 12px;
}}

.stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 8px 16px;
}}

.stTabs [aria-selected="true"] {{
    background: {t['accent']};
    color: white;
}}

.footer {{
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-size: 13px;
    margin-top: 40px;
    padding: 20px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# BRANDING HEADER
# ---------------------------------------------------
col_logo, col_title = st.columns([1, 10])

with col_logo:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2721/2721279.png",
        width=65
    )

with col_title:
    st.markdown("<div class='brand-header'>ChurnAI Enterprise Pro</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>AI-Powered Customer Retention Intelligence Platform</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# LOAD DATA + MODEL
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("churn_dataset.xlsx")

df = load_data()

df_display = df.copy()

df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

model = joblib.load("churn_model.pkl")

X = df[["Age", "Tenure", "Sex"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.markdown("### 🔎 Smart Filters")

age_range = st.sidebar.slider(
    "Age Range",
    int(df.Age.min()),
    int(df.Age.max()),
    (20, 60)
)

tenure_range = st.sidebar.slider(
    "Tenure Range",
    int(df.Tenure.min()),
    int(df.Tenure.max()),
    (0, 10)
)

filtered_df = df[
    (df.Age >= age_range[0]) & (df.Age <= age_range[1]) &
    (df.Tenure >= tenure_range[0]) & (df.Tenure <= tenure_range[1])
]

# ---------------------------------------------------
# ANIMATED KPI
# ---------------------------------------------------
st.markdown("### 📊 Business Intelligence Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

def animated_metric(col, label, value, prefix=""):
    with col:
        placeholder = col.empty()
        for i in range(0, value + 1, max(1, value // 30)):
            placeholder.metric(label, f"{prefix}{i}")
            time.sleep(0.01)
        placeholder.metric(label, f"{prefix}{value}")

animated_metric(kpi1, "👥 Total Customers", len(filtered_df))
animated_metric(kpi2, "📉 Churn Rate", int(filtered_df["Churn"].mean() * 100), "%")
animated_metric(kpi3, "🎂 Avg Age", int(filtered_df["Age"].mean()))
animated_metric(kpi4, "⏳ Avg Tenure", int(filtered_df["Tenure"].mean()))

st.divider()

# ---------------------------------------------------
# MAIN TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Explorer",
    "📊 Visual Analytics",
    "🔗 Correlations",
    "🤖 Model Evaluation",
    "🔮 Live Prediction"
])

# ===================================================
# TAB 1 — DATA EXPLORER
# ===================================================
with tab1:

    st.markdown("### 🔍 Smart Data Search")

    search = st.text_input("🔎 Search anything in dataset...", "")

    if search:
        result = filtered_df[
            filtered_df.apply(
                lambda row: row.astype(str).str.contains(search, case=False).any(),
                axis=1
            )
        ]
        st.success(f"Found {len(result)} matching records")
        st.dataframe(result, use_container_width=True)
    else:
        st.dataframe(filtered_df, use_container_width=True)

    colA, colB = st.columns(2)
    colA.write(f"Total Records: **{len(filtered_df)}**")
    colB.write(f"Total Columns: **{len(filtered_df.columns)}**")

    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

# ===================================================
# TAB 2 — VISUAL ANALYTICS
# ===================================================
with tab2:

    st.markdown("### 📊 Visual Analytics Dashboard")

    show_charts = st.button("🚀 Generate All Charts")

    if show_charts:

        colC, colD = st.columns(2)

        with colC:
            st.subheader("Churn Distribution")
            fig1 = px.histogram(
                filtered_df,
                x="Churn",
                color="Churn",
                color_discrete_sequence=["#00C6FF", "#ff4b2b"],
                template="plotly_dark"
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with colD:
            st.subheader("Churn Pie Chart")
            churn_counts = filtered_df["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Churn", "Count"]
            fig_pie = px.pie(
                churn_counts,
                names="Churn",
                values="Count",
                color_discrete_sequence=["#00C6FF", "#ff4b2b"],
                hole=0.4
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        colE, colF = st.columns(2)

        with colE:
            st.subheader("Age Distribution")
            fig_age = px.histogram(
                filtered_df,
                x="Age",
                color_discrete_sequence=["#56ab2f"],
                template="plotly_dark"
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)

        with colF:
            st.subheader("Age vs Tenure")
            fig_scatter = px.scatter(
                filtered_df,
                x="Age",
                y="Tenure",
                color="Churn",
                color_discrete_sequence=["#00C6FF", "#ff4b2b"],
                template="plotly_dark",
                opacity=0.7
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ===================================================
# TAB 3 — CORRELATIONS
# ===================================================
with tab3:

    st.markdown("### 🔗 Correlation Analysis")

    show_corr = st.button("🔗 Analyze Correlations")

    if show_corr:

        colG, colH = st.columns(2)

        with colG:
            st.subheader("Correlation Heatmap")
            corr = filtered_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu",
                template="plotly_dark"
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

        with colH:
            st.subheader("Pair Plot")
            fig_pair = px.scatter_matrix(
                filtered_df,
                dimensions=["Age", "Tenure"],
                color="Churn",
                color_discrete_sequence=["#00C6FF", "#ff4b2b"],
                template="plotly_dark"
            )
            fig_pair.update_layout(height=500)
            st.plotly_chart(fig_pair, use_container_width=True)

# ===================================================
# TAB 4 — MODEL EVALUATION
# ===================================================
with tab4:

    st.markdown("### 🤖 Model Performance Analysis")

    show_eval = st.button("📈 Evaluate Model")

    if show_eval:

        colI, colJ = st.columns(2)

        with colI:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                template="plotly_dark"
            )
            fig_cm.update_layout(height=450)
            st.plotly_chart(fig_cm, use_container_width=True)

        with colJ:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"AUC = {roc_auc:.2f}",
                line=dict(color="#00C6FF", width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color="gray")
            ))
            fig_roc.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# ===================================================
# TAB 5 — LIVE PREDICTION
# ===================================================
with tab5:

    st.markdown("### 🔮 AI Prediction Engine")

    colK, colL = st.columns(2)

    with colK:
        age = st.slider("Age", 18, 100, 30)
        tenure = st.slider("Tenure (Years)", 0, 30, 5)

    with colL:
        gender = st.selectbox("Gender", ["Male", "Female"])

    sex = 1 if gender == "Male" else 0

    if st.button("🚀 Run AI Prediction"):

        input_data = np.array([[age, tenure, sex]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        churn_prob = probability[0][1] * 100

        if prediction[0] == 1:
            st.error(f"❌ High Risk Customer ({churn_prob:.2f}%)")
        else:
            st.success(f"✅ Low Risk Customer ({churn_prob:.2f}%)")

        st.progress(int(churn_prob))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<div class='footer'>
© 2026 ChurnAI Enterprise Pro — Powered by AI & Machine Learning
</div>
""", unsafe_allow_html=True)