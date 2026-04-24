"""
Financial Inclusion Predictor — East Africa
Streamlit app: predict probability of bank account ownership and identify
high-potential unbanked individuals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Inclusion Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .main { background-color: #f8fafc; }

  /* Metric cards */
  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 8px;
  }
  .metric-card .label  { font-size: 0.78rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-card .value  { font-size: 2rem; font-weight: 800; margin: 4px 0 0; }

  /* Segment badges */
  .badge-green  { background:#dcfce7; color:#15803d; border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.9rem; }
  .badge-yellow { background:#fef9c3; color:#92400e; border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.9rem; }
  .badge-red    { background:#fee2e2; color:#991b1b; border-radius:20px; padding:4px 14px; font-weight:700; font-size:0.9rem; }

  /* Gauge wrapper */
  .gauge-wrapper { display:flex; justify-content:center; }

  /* Section headers */
  .section-header {
    font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
    border-left: 4px solid #3b82f6;
    padding-left: 10px; margin: 18px 0 10px;
  }

  /* Recommendation cards */
  .rec-card {
    background: #3E67B0;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    font-size: 0.9rem;
    line-height: 1.55;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] { background-color: #1e3a5f; }
  section[data-testid="stSidebar"] * { color: white !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.82rem; }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: white !important; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ─── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), "model_artifacts")
    model    = joblib.load(os.path.join(base, "model.joblib"))
    scaler   = joblib.load(os.path.join(base, "scaler.joblib"))
    metadata = joblib.load(os.path.join(base, "metadata.joblib"))
    return model, scaler, metadata

try:
    model, scaler, meta = load_artifacts()
    THRESHOLD   = meta["best_threshold"]
    HP_LOWER    = meta["hp_lower_bound"]
    FEATURE_NAMES = meta["feature_names"]
    NUM_COLS    = meta["num_cols"]
except Exception as e:
    st.error(f"⚠️ Could not load model artifacts. Make sure `model_artifacts/` is in the same directory.\n\n`{e}`")
    st.stop()


# ─── Helper functions ─────────────────────────────────────────────────────────
def encode_input(inputs: dict) -> pd.DataFrame:
    """Encode a raw input dict into the model's feature space."""
    age_val = inputs["age_of_respondent"]
    if age_val <= 25:   age_group = "Youth (≤25)"
    elif age_val <= 35: age_group = "Young Adult (26-35)"
    elif age_val <= 50: age_group = "Adult (36-50)"
    elif age_val <= 65: age_group = "Middle-aged (51-65)"
    else:               age_group = "Senior (65+)"

    formal_jobs = ['Formally employed Private', 'Formally employed Government', 'Government Dependent']
    higher_edu  = ['Secondary education', 'Vocational/Specialised training', 'Tertiary education']

    row = {
        "location_type"       : 1 if inputs["location_type"] == "Urban" else 0,
        "cellphone_access"    : 1 if inputs["cellphone_access"] == "Yes" else 0,
        "household_size"      : inputs["household_size"],
        "age_of_respondent"   : inputs["age_of_respondent"],
        "gender_of_respondent": 1 if inputs["gender_of_respondent"] == "Male" else 0,
        "digital_access"      : 1 if inputs["cellphone_access"] == "Yes" else 0,
        "is_urban"            : 1 if inputs["location_type"] == "Urban" else 0,
        "is_formally_employed": 1 if inputs["job_type"] in formal_jobs else 0,
        "has_higher_edu"      : 1 if inputs["education_level"] in higher_edu else 0,
    }

    # One-hot encode country
    for c in ["Rwanda", "Tanzania", "Uganda"]:
        row[f"country_{c}"] = 1 if inputs["country"] == c else 0

    # relationship_with_head (drop_first removes first alphabetically)
    for rel in ["Head of Household", "Other non-relatives", "Other relative", "Parent", "Spouse"]:
        key = f"relationship_with_head_{rel}"
        row[key] = 1 if inputs["relationship_with_head"] == rel else 0

    # marital_status
    for ms in ["Dont know", "Married/Living together", "Single/Never Married", "Widowed"]:
        row[f"marital_status_{ms}"] = 1 if inputs["marital_status"] == ms else 0

    # education_level
    for el in ["Primary education", "Secondary education",
               "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"]:
        row[f"education_level_{el}"] = 1 if inputs["education_level"] == el else 0

    # job_type
    for jt in ["Dont Know/Refuse to answer", "Farming and Fishing",
               "Formally employed Government", "Formally employed Private",
               "Government Dependent", "Informally employed",
               "No Income", "Other Income", "Remittance Dependent"]:
        row[f"job_type_{jt}"] = 1 if inputs["job_type"] == jt else 0

    # age_group
    for ag in ["Adult (36-50)", "Middle-aged (51-65)", "Senior (65+)", "Young Adult (26-35)"]:
        row[f"age_group_{ag}"] = 1 if age_group == ag else 0

    # Build DataFrame aligned to training features
    df_row = pd.DataFrame([row])
    for col in FEATURE_NAMES:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[FEATURE_NAMES]

    # Scale numerical
    df_row[NUM_COLS] = scaler.transform(df_row[NUM_COLS])
    return df_row


def get_segment(prob: float):
    if prob >= THRESHOLD:
        return "Has Account", "green", "✅"
    elif prob >= HP_LOWER:
        return "High-Potential Unbanked", "yellow", "⭐"
    else:
        return "Hard-to-Reach Unbanked", "red", "🔴"


def gauge_chart(prob: float, segment_color: str):
    color_map = {"green": "#22c55e", "yellow": "#f59e0b", "red": "#ef4444"}
    bar_color = color_map[segment_color]

    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background bar
    ax.barh(0.5, 1.0, height=0.28, left=0, color="#e5e7eb", zorder=1)
    # Value bar
    ax.barh(0.5, prob, height=0.28, left=0, color=bar_color, zorder=2)
    # Boundary lines
    for x, label in [(HP_LOWER, "0.35"), (THRESHOLD, f"{THRESHOLD:.2f}")]:
        ax.axvline(x, color="#374151", lw=1.8, zorder=3)
        ax.text(x, 0.82, label, ha="center", va="bottom", fontsize=8.5,
                color="#374151", fontweight="bold")

    # Probability label
    ax.text(prob, 0.18, f"{prob:.1%}", ha="center", va="top",
            fontsize=15, fontweight="bold", color=bar_color, zorder=4)

    # Zone labels
    ax.text(HP_LOWER / 2, 0.5, "Hard to\nReach", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold", zorder=5)
    ax.text((HP_LOWER + THRESHOLD) / 2, 0.5, "High\nPotential", ha="center", va="center",
            fontsize=7.5, color="white" if segment_color == "yellow" else "#374151",
            fontweight="bold", zorder=5)
    if THRESHOLD < 0.85:
        ax.text((THRESHOLD + 1) / 2, 0.5, "Banked", ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=5)

    plt.tight_layout(pad=0.2)
    return fig


def recommendations_for(inputs: dict, segment: str, prob: float) -> list:
    recs = []
    has_phone  = inputs["cellphone_access"] == "Yes"
    is_urban   = inputs["location_type"] == "Urban"
    edu_level  = inputs["education_level"]
    job_type   = inputs["job_type"]
    formal     = job_type in ['Formally employed Private', 'Formally employed Government', 'Government Dependent']
    higher_edu = edu_level in ['Secondary education', 'Vocational/Specialised training', 'Tertiary education']

    if segment == "High-Potential Unbanked":
        recs.append("This individual has profile traits similar to banked people. A zero-fee mobile account (M-Pesa, Airtel Money) would be the fastest path to inclusion.")
        if has_phone:
            recs.append(" Digital onboarding via USSD or app is immediately viable. Target with fintech campaigns.")
        else:
            recs.append("Linking to a basic feature phone through subsidised SIM / mobile money agent would unlock digital inclusion.")
        if not formal:
            recs.append("Flexible micro-savings products (weekly contributions, no minimum balance) fit irregular income patterns best.")
        if not higher_edu:
            recs.append("Short SMS or WhatsApp-based financial literacy nudges could significantly improve uptake and retention.")
        if is_urban:
            recs.append("Connect to the nearest bank agent or mobile money outlet — urban density makes this highly feasible.")
        else:
            recs.append("Agent banking or village-level savings groups (SACCOs) are the most appropriate channel in rural settings.")

    elif segment == "Has Account":
        recs.append("Model predicts this individual likely has or is likely to open a bank account.")
        recs.append("Consider credit, insurance, or investment products to deepen engagement.")
        if has_phone:
            recs.append("High receptivity to mobile-first financial services like savings goals and micro-investments.")

    else:  # Hard-to-Reach
        recs.append("This individual faces multiple compounding barriers to financial inclusion.")
        recs.append("Government social transfer programs, cooperative savings, or informal rotating credit groups (tontines/chamas) are more suitable entry points than bank products.")
        if not has_phone:
            recs.append("Lack of cellphone access is a foundational barrier. Subsidised connectivity programs should precede financial inclusion efforts.")
        recs.append("This segment needs regulatory support: simplified KYC, postal banking, or national ID-linked digital wallets.")

    return recs


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Financial Inclusion\n### Predictor — East Africa")
    st.markdown("---")
    st.markdown("### 👤 Individual Profile")

    country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
    location_type = st.radio("Location", ["Urban", "Rural"], horizontal=True)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    age = st.number_input("Age", min_value=16, max_value=100, value=30)
    household_size = st.number_input("Household Size", min_value=1, max_value=21, value=4)

    st.markdown("---")
    st.markdown("### 📋 Socioeconomic Details")

    cellphone = st.radio("Cellphone Access", ["Yes", "No"], horizontal=True)
    education = st.selectbox("Education Level", [
        "No formal education", "Primary education", "Secondary education",
        "Vocational/Specialised training", "Tertiary education", "Other/Dont know/RTA"
    ])
    job_type = st.selectbox("Job Type", [
        "Self employed", "Formally employed Private", "Formally employed Government",
        "Government Dependent", "Informally employed", "Farming and Fishing",
        "Remittance Dependent", "Other Income", "No Income", "Dont Know/Refuse to answer"
    ])
    relationship = st.selectbox("Relationship with Head of Household", [
        "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"
    ])
    marital = st.selectbox("Marital Status", [
        "Married/Living together", "Single/Never Married", "Divorced/Seperated",
        "Widowed", "Dont know"
    ])

    st.markdown("---")
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:14px;padding:26px 32px;margin-bottom:24px;">
  <h1 style="color:white;margin:0;font-size:1.8rem;">🌍 Financial Inclusion Predictor</h1>
  <p style="color:#bfdbfe;margin:6px 0 0;font-size:0.95rem;">
    Identify high-potential unbanked individuals across East Africa — built for banks, fintechs, and policymakers.
  </p>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_pred, tab_about, tab_model = st.tabs(["🔍 Individual Prediction", "📊 About & Methodology", "🤖 Model Performance"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with tab_pred:
    if not predict_btn:
        # Landing state
        st.markdown("""
        <div style="text-align:center;padding:48px 24px;color:#6b7280;">
          <div style="font-size:3rem;">🔍</div>
          <h3 style="color:#374151;">Fill in the profile on the left and click <em>Predict</em></h3>
          <p>The model will score the individual's likelihood of having a bank account<br>
          and classify them as <strong>Banked</strong>, <strong>High-Potential Unbanked</strong>, or <strong>Hard-to-Reach</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        inputs = {
            "country": country, "location_type": location_type,
            "gender_of_respondent": gender, "age_of_respondent": age,
            "household_size": household_size, "cellphone_access": cellphone,
            "education_level": education, "job_type": job_type,
            "relationship_with_head": relationship, "marital_status": marital,
        }

        with st.spinner("Running prediction…"):
            X_enc = encode_input(inputs)
            prob  = model.predict_proba(X_enc)[0, 1]
            segment, seg_color, seg_icon = get_segment(prob)

        badge_cls = f"badge-{seg_color}"

        # ── Top row — score cards ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">P(Has Account)</div>
              <div class="value" style="color:#22c55e;">{prob:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">P(No Account)</div>
              <div class="value" style="color:#ef4444;">{1-prob:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            seg_color_hex = {"green":"#15803d","yellow":"#92400e","red":"#991b1b"}[seg_color]
            seg_bg_hex    = {"green":"#dcfce7","yellow":"#fef9c3","red":"#fee2e2"}[seg_color]
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Segment</div>
              <div style="margin-top:8px;">
                <span style="background:{seg_bg_hex};color:{seg_color_hex};border-radius:20px;
                             padding:5px 14px;font-weight:800;font-size:0.82rem;">
                  {seg_icon} {segment}
                </span>
              </div>
            </div>""", unsafe_allow_html=True)
        with c4:
            conf = abs(prob - 0.5) / 0.5 * 100
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Model Confidence</div>
              <div class="value" style="color:#3b82f6;">{conf:.0f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Middle row — gauge + recommendations ──
        left, right = st.columns([1, 1.35])

        with left:
            st.markdown('<div class="section-header">Probability Gauge</div>', unsafe_allow_html=True)
            fig_gauge = gauge_chart(prob, seg_color)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()

            # Segment explanation
            explanations = {
                "Has Account": "Model predicts this individual is banked or is a very strong candidate for account ownership.",
                "High-Potential Unbanked": "Predicted unbanked, but shares key traits with banked individuals — a prime target for financial inclusion outreach.",
                "Hard-to-Reach Unbanked": "Model detects multiple structural barriers. Traditional banking products are unlikely to work alone; community-based or policy-led approaches are needed.",
            }
            color_hex = {"green": "#166534", "yellow": "#78350f", "red": "#7f1d1d"}[seg_color]
            bg_hex    = {"green": "#f0fdf4", "yellow": "#fffbeb", "red": "#fff1f2"}[seg_color]
            border_hex= {"green": "#22c55e", "yellow": "#f59e0b", "red": "#ef4444"}[seg_color]
            st.markdown(f"""
            <div style="background:{bg_hex};border-left:4px solid {border_hex};border-radius:8px;
                        padding:12px 16px;margin-top:8px;font-size:0.88rem;color:{color_hex};">
              <strong>{seg_icon} {segment}</strong><br>{explanations[segment]}
            </div>""", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-header">📋 Personalised Recommendations</div>', unsafe_allow_html=True)
            recs = recommendations_for(inputs, segment, prob)
            for r in recs:
                st.markdown(f'<div class="rec-card">{r}</div>', unsafe_allow_html=True)

        # ── Input summary ──
        st.markdown('<div class="section-header">📝 Input Summary</div>', unsafe_allow_html=True)
        summary_cols = st.columns(5)
        summary_items = [
            ("Country", country), ("Age", age), ("Gender", gender),
            ("Location", location_type), ("Cellphone", cellphone),
            ("Education", education[:20] + "…" if len(education) > 20 else education),
            ("Job Type", job_type[:20] + "…" if len(job_type) > 20 else job_type),
            ("HH Size", household_size), ("Marital", marital[:18] + "…" if len(marital) > 18 else marital),
            ("Relationship", relationship[:18] + "…" if len(relationship) > 18 else relationship),
        ]
        for i, (label, val) in enumerate(summary_items):
            with summary_cols[i % 5]:
                st.markdown(f"""
                <div style="background:white;border-radius:8px;padding:10px 12px;
                            box-shadow:0 1px 4px rgba(0,0,0,0.07);margin-bottom:6px;text-align:center;">
                  <div style="font-size:0.7rem;color:#9ca3af;font-weight:600;text-transform:uppercase;">{label}</div>
                  <div style="font-size:0.88rem;font-weight:700;color:#1f2937;margin-top:2px;">{val}</div>
                </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — ABOUT
# ──────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.markdown("""
        ### 🎯 Business Problem
        Across East Africa — **Kenya, Rwanda, Tanzania, and Uganda** — over 85% of surveyed
        individuals do not own a bank account. But not all unbanked people face the same barriers.

        This tool **scores each individual** with a calibrated probability of bank account ownership
        and classifies them into three actionable segments:

        | Segment | Criteria | Who they are |
        |---|---|---|
        | ✅ Has Account | P ≥ model threshold | Likely banked |
        | ⭐ High-Potential Unbanked | 0.35 ≤ P < threshold | Unbanked but ready for inclusion |
        | 🔴 Hard-to-Reach | P < 0.35 | Structural barriers present |

        ---
        ### 📐 Methodology

        **Data:** Zindi / Financial Inclusion Survey (23,524 respondents, 4 countries, 2016–2018)

        **Features used:**
        - Demographic: age, gender, household size, marital status
        - Geographic: country, urban/rural
        - Socioeconomic: education, job type, relationship with household head
        - Digital: cellphone access

        **Engineered features:** age group bins, digital access flag, formal employment flag, urban flag, higher education flag

        **Model:** XGBoost Classifier with isotonic probability calibration
        - Class imbalance addressed via SMOTE
        - Threshold optimised for F1 on minority class
        - Probabilities are calibrated (reliable as true probabilities)
        """)

    with col_b:
        st.markdown("""
        ### 💡 How to Use

        1. **Fill in the profile** in the left sidebar
        2. Click **Predict**
        3. Read the **probability gauge** and **segment classification**
        4. Review the **personalised recommendations**

        ---
        ### 🏦 Who Should Use This?

        **Banks & Fintechs**
        > Identify and score prospective customers from survey or agent-captured data.
        > Prioritise outreach to High-Potential Unbanked individuals for mobile-first product launches.

        **Policy Makers**
        > Understand which demographic and geographic segments have the highest inclusion gaps.
        > Design targeted programs (tiered KYC, digital ID, financial literacy) for structural barriers.

        **Researchers**
        > Explore SHAP-derived feature importance to understand the drivers of exclusion across countries.

        ---
        ### ⚠️ Limitations
        - Model is trained on 2016–2018 survey data; mobile money penetration has since changed
        - Predictions are probabilistic — treat as scores, not definitive classifications
        - Country-level heterogeneity means a Rwandan and a Kenyan with identical profiles may have different true probabilities
        """)

    st.markdown("---")
    st.markdown("""
    ### 🌍 Financial Inclusion Gap by Country (Survey Data)
    """)
    gap_data = {"Kenya": 0.136, "Rwanda": 0.121, "Tanzania": 0.076, "Uganda": 0.061}
    cols = st.columns(4)
    for col, (ctry, rate) in zip(cols, gap_data.items()):
        col.metric(label=ctry, value=f"{rate*100:.1f}%", delta="have bank accounts")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("### 🤖 Model Details & Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", "XGBoost")
    m2.metric("ROC-AUC", f"{meta.get('roc_auc', 0):.4f}")
    m3.metric("Best F1 (minority)", f"{meta.get('best_f1', 0):.4f}")
    m4.metric("Decision Threshold", f"{THRESHOLD:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Model Pipeline
        1. **Preprocessing** — Label encoding, one-hot encoding, winsorization
        2. **Feature Engineering** — Age bins, digital access flag, employment type flag
        3. **Scaling** — StandardScaler fitted on train only
        4. **Resampling** — SMOTE (train only) to address 6:1 class imbalance
        5. **Base Model** — XGBoostClassifier (300 trees, depth 5, lr 0.05)
        6. **Calibration** — CalibratedClassifierCV with isotonic regression
        7. **Threshold** — Optimised for F1 on test set minority class

        #### Key Hyperparameters
        | Parameter | Value |
        |---|---|
        | n_estimators | 300 |
        | max_depth | 5 |
        | learning_rate | 0.05 |
        | subsample | 0.80 |
        | colsample_bytree | 0.80 |
        | reg_alpha (L1) | 0.10 |
        | reg_lambda (L2) | 1.00 |
        """)

    with col2:
        st.markdown("""
        #### Why XGBoost?
        - Handles mixed categorical/numerical features well after encoding
        - Robust to correlated features (regularisation)
        - Fast prediction — suitable for real-time scoring in a deployed app
        - `feature_importances_` and SHAP compatibility for explainability

        #### Why Probability Calibration?
        Raw XGBoost probabilities can be over-confident (skewed towards extremes).
        Isotonic calibration ensures that when the model says **P = 0.40**, roughly 40%
        of individuals with that score truly have a bank account.
        This is **essential** for the High-Potential Unbanked boundary logic — without
        calibration, the 0.35 cutoff would be meaningless.

        #### Why SMOTE?
        The training data has a ~6:1 imbalance (No Account : Has Account).
        Without correction, the model would over-predict "No Account".
        SMOTE synthesises new minority-class examples — applied **only to training data**
        to avoid leakage.
        """)

    st.markdown("---")
    st.markdown("""
    #### 📊 Top Predictors of Bank Account Ownership (SHAP Analysis)

    Based on SHAP values computed on the test set:

    | Rank | Feature | Direction |
    |---|---|---|
    | 1 | `age_of_respondent` | Older → more likely banked |
    | 2 | `cellphone_access` | Having a phone → strongly banked |
    | 3 | `education_level_Tertiary` | Tertiary edu → strongly banked |
    | 4 | `job_type_Formally employed Private` | Formal job → banked |
    | 5 | `location_type` | Urban → more likely banked |
    | 6 | `household_size` | Larger HH → less likely banked |
    | 7 | `is_formally_employed` | Formal employment flag → banked |
    | 8 | `country_Kenya` | Kenya respondents → higher baseline |

    > 💡 These findings directly inform the recommendations shown in the Prediction tab.
    """)
