"""
Heart Disease Risk Prediction — Streamlit App
Run with:  streamlit run streamlit_app.py
"""
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from feature_engineering import FeatureEngineer  # noqa: F401  (needed to unpickle the model)

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card { background-color: #F8F9F9; border-radius: 8px; padding: 1rem; text-align:center; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("../models/heart_disease_pipeline.pkl")
    metadata = joblib.load("../models/model_metadata.pkl")
    return pipeline, metadata


try:
    pipeline, metadata = load_pipeline()
except FileNotFoundError:
    st.error(
        "Model file not found. Run the training notebook "
        "(`notebooks/heart_disease_analysis.ipynb`) first to generate "
        "`models/heart_disease_pipeline.pkl`."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    f"Powered by a tuned **{metadata['model_name']}** model "
    f"(test ROC-AUC: {metadata['metrics']['ROC-AUC']:.2f}, "
    f"accuracy: {metadata['metrics']['Accuracy']:.2f}) trained on the Cleveland Heart Disease dataset."
)

# st.info(
#     "⚕️ **Disclaimer:** This tool is for educational / portfolio purposes only. "
#     "It is not a medical device and must not be used for real diagnosis. "
#     "Always consult a qualified physician.",
#     icon="ℹ️",
# )

# ----------------------------------------------------------------------------
# Sidebar — patient inputs
# ----------------------------------------------------------------------------
st.sidebar.header("Patient Details")

with st.sidebar.form("patient_form"):
    st.subheader("Demographics")
    age = st.slider("Age", 18, 100, 54)
    sex = st.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)

    st.subheader("Symptoms & Vitals")
    cp = st.selectbox(
        "Chest Pain Type", options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina", 1: "Atypical Angina",
            2: "Non-anginal Pain", 3: "Asymptomatic"
        }[x],
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 246)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", [1, 0],
                    format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise-Induced Angina?", [1, 0],
                      format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

    st.subheader("Test Results")
    restecg = st.selectbox(
        "Resting ECG Result", options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
    )
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", options=[0, 1, 2],
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
    )
    ca = st.selectbox("Number of Major Vessels Colored (Fluoroscopy)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia", options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x],
    )

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
}
input_df = pd.DataFrame([input_dict])


def render_result(risk_pct: float, pred: int, confidence: float):
    """Render an animated circular gauge + result card as a single HTML/CSS/JS block."""
    is_high = pred == 1
    accent = "#E74C3C" if is_high else "#27AE60"
    accent_soft = "#FDEDEC" if is_high else "#EAFAF1"
    label = "High Risk" if is_high else "Low Risk"
    sub_label = "Heart disease likely present" if is_high else "Heart disease unlikely"
    icon = "⚠️" if is_high else "✅"
    pulse = "pulseRing 1.8s ease-out infinite" if is_high else "none"

    radius = 84
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - risk_pct / 100)

    html = f"""
    <div class="result-wrap">
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        * {{ font-family: 'Inter', sans-serif; box-sizing: border-box; }}

        .result-wrap {{
          display: flex;
          justify-content: center;
          align-items: center;
          padding: 10px 0 6px 0;
        }}
        .result-card {{
          width: 100%;
          max-width: 560px;
          background: linear-gradient(180deg, #ffffff 0%, {accent_soft} 130%);
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 20px;
          padding: 28px 32px 24px 32px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.08);
          text-align: center;
          opacity: 0;
          transform: translateY(18px);
          animation: cardIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}

        @keyframes cardIn {{
          to {{ opacity: 1; transform: translateY(0); }}
        }}

        .badge {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: {accent};
          color: white;
          font-weight: 800;
          font-size: 13px;
          letter-spacing: 0.4px;
          padding: 6px 14px;
          border-radius: 999px;
          margin-bottom: 14px;
          opacity: 0;
          animation: fadeUp 0.5s ease-out 0.35s forwards;
        }}

        @keyframes fadeUp {{
          from {{ opacity: 0; transform: translateY(8px); }}
          to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .gauge-box {{ position: relative; width: 220px; height: 220px; margin: 6px auto 10px auto; }}

        .gauge-box svg {{ transform: rotate(-90deg); }}

        .gauge-bg {{
          fill: none;
          stroke: #EEF1F3;
          stroke-width: 14;
        }}

        .gauge-fg {{
          fill: none;
          stroke: {accent};
          stroke-width: 14;
          stroke-linecap: round;
          stroke-dasharray: {circumference:.2f};
          stroke-dashoffset: {circumference:.2f};
          animation: drawGauge 1.3s cubic-bezier(0.16, 1, 0.3, 1) 0.15s forwards,
                     {pulse};
          transform-origin: 50% 50%;
        }}

        @keyframes drawGauge {{
          to {{ stroke-dashoffset: {offset:.2f}; }}
        }}

        @keyframes pulseRing {{
          0%   {{ filter: drop-shadow(0 0 0px {accent}); }}
          50%  {{ filter: drop-shadow(0 0 10px {accent}); }}
          100% {{ filter: drop-shadow(0 0 0px {accent}); }}
        }}

        .gauge-center {{
          position: absolute;
          top: 0; left: 0; right: 0; bottom: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
        }}

        .gauge-icon {{
          font-size: 30px;
          margin-bottom: 2px;
          opacity: 0;
          animation: popIn 0.4s ease-out 1.1s forwards;
        }}

        @keyframes popIn {{
          from {{ opacity: 0; transform: scale(0.5); }}
          to   {{ opacity: 1; transform: scale(1); }}
        }}

        .gauge-number {{
          font-size: 40px;
          font-weight: 800;
          color: {accent};
          line-height: 1;
        }}

        .gauge-caption {{
          font-size: 12px;
          color: #8a8f98;
          margin-top: 4px;
          font-weight: 600;
          letter-spacing: 0.3px;
        }}

        .result-title {{
          font-size: 21px;
          font-weight: 800;
          color: #1c1f26;
          margin: 4px 0 2px 0;
        }}

        .result-sub {{
          font-size: 14px;
          color: #6b7280;
          margin-bottom: 18px;
        }}

        .stat-row {{
          display: flex;
          justify-content: center;
          gap: 14px;
          opacity: 0;
          animation: fadeUp 0.5s ease-out 0.6s forwards;
        }}

        .stat {{
          background: white;
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 12px;
          padding: 10px 18px;
          min-width: 108px;
        }}

        .stat-value {{
          font-size: 18px;
          font-weight: 800;
          color: #1c1f26;
        }}

        .stat-label {{
          font-size: 11px;
          color: #9aa0a8;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.4px;
          margin-top: 2px;
        }}
      </style>

      <div class="result-card">
        <div class="badge">{icon} {label.upper()}</div>

        <div class="gauge-box">
          <svg width="220" height="220" viewBox="0 0 220 220">
            <circle class="gauge-bg" cx="110" cy="110" r="{radius}"></circle>
            <circle class="gauge-fg" cx="110" cy="110" r="{radius}"></circle>
          </svg>
          <div class="gauge-center">
            <div class="gauge-icon">{icon}</div>
            <div class="gauge-number" id="riskNum">0%</div>
            <div class="gauge-caption">RISK SCORE</div>
          </div>
        </div>

        <div class="result-title">{sub_label}</div>
        <div class="result-sub">Estimated probability of heart disease: <b>{risk_pct:.1f}%</b></div>

        <div class="stat-row">
          <div class="stat">
            <div class="stat-value">{"Disease" if is_high else "No Disease"}</div>
            <div class="stat-label">Predicted Class</div>
          </div>
          <div class="stat">
            <div class="stat-value">{confidence:.1f}%</div>
            <div class="stat-label">Confidence</div>
          </div>
        </div>
      </div>

      <script>
        (function() {{
          const target = {risk_pct:.1f};
          const el = document.getElementById('riskNum');
          const duration = 1300;
          const start = performance.now() + 150;
          function tick(now) {{
            const t = Math.max(0, Math.min(1, (now - start) / duration));
            const eased = 1 - Math.pow(1 - t, 3);
            const val = (target * eased).toFixed(1);
            el.textContent = val + '%';
            if (t < 1) {{
              requestAnimationFrame(tick);
            }} else {{
              el.textContent = target.toFixed(1) + '%';
            }}
          }}
          requestAnimationFrame(tick);
        }})();
      </script>
    </div>
    """
    components.html(html, height=470)


left, mid, right = st.columns([1, 2, 1])

if submitted:
    proba = pipeline.predict_proba(input_df)[0]
    pred = pipeline.predict(input_df)[0]
    risk_pct = proba[1] * 100
    confidence = max(proba) * 100

    with mid:
        render_result(risk_pct, pred, confidence)
else:
    with mid:
        st.subheader("👈 Enter patient details and click Predict")
        st.write(
            "Fill in the form in the sidebar with the patient's clinical values, "
            "then click **Predict Risk** to see the model's risk estimate."
        )

# ----------------------------------------------------------------------------
# Model info tab
# ----------------------------------------------------------------------------
st.divider()
with st.expander("📊 About the model"):
    st.markdown(f"""
    - **Algorithm:** {metadata['model_name']} (selected after comparing 8 algorithms
      with 5-fold cross-validation, then hyperparameter-tuned with `GridSearchCV`)
    - **Training data:** Cleveland Heart Disease dataset (297 patients, UCI ML Repository)
    - **Test set performance:**
        - Accuracy: {metadata['metrics']['Accuracy']:.3f}
        - Precision: {metadata['metrics']['Precision']:.3f}
        - Recall: {metadata['metrics']['Recall']:.3f}
        - F1-score: {metadata['metrics']['F1']:.3f}
        - ROC-AUC: {metadata['metrics']['ROC-AUC']:.3f}
    - **Engineered features:** age group, cholesterol-per-age, heart-rate reserve,
      and a composite risk-factor count, in addition to the 13 raw clinical inputs.
    - See `notebooks/heart_disease_analysis.ipynb` for the full EDA, model comparison,
      tuning process, and interpretability analysis.
    """)
