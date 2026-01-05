import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #eef2f7 !important;
    color: #0f172a !important;
    font-family: 'Segoe UI', sans-serif;
}

.card {
    background: #ffffff;
    padding: 28px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
}

.header {
    text-align: center;
    padding: 20px 0 30px 0;
}

.header h1 {
    font-size: 34px;
    font-weight: 800;
    color: #1e40af;
}

.header p {
    font-size: 15px;
    color: #475569;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 18px;
    color: #1f2937;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    font-size: 16px;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1e3a8a);
}

.result-card {
    text-align: center;
    padding: 35px;
}

.status {
    font-size: 32px;
    font-weight: 900;
    margin: 20px 0;
}

.low {
    color: #15803d;
}

.high {
    color: #b91c1c;
}

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
}

.badge-low {
    background: #dcfce7;
    color: #166534;
}

.badge-high {
    background: #fee2e2;
    color: #7f1d1d;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ model.pkl not found. Train the model first.")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>🫀 Heart Disease Prediction System</h1>
    <p>Professional AI-powered cardiovascular risk assessment</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
left, right = st.columns([3, 2])

# ================= INPUT PANEL =================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Patient Information</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 10, 100, 55)
    gender_txt = c2.selectbox("Gender", ["Female", "Male"])
    height = c3.number_input("Height (cm)", 50, 250, 165)

    c4, c5, c6 = st.columns(3)
    weight = c4.number_input("Weight (kg)", 10, 200, 75)
    ap_hi = c5.number_input("Systolic BP", 50, 250, 140)
    ap_lo = c6.number_input("Diastolic BP", 30, 150, 90)

    c7, c8, c9 = st.columns(3)
    chol_txt = c7.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    gluc_txt = c8.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    active_txt = c9.selectbox("Physical Activity", ["No", "Yes"])

    c10, c11 = st.columns(2)
    smoke_txt = c10.selectbox("Smoking", ["No", "Yes"])
    alco_txt = c11.selectbox("Alcohol Consumption", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("🔍 Predict Heart Disease", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RESULT PANEL =================
with right:
    st.markdown("<div class='card result-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Health Risk Assessment</div>", unsafe_allow_html=True)

    if predict:
        # Encoding
        gender = 1 if gender_txt == "Female" else 2
        cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[chol_txt]
        gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc_txt]
        smoke = 1 if smoke_txt == "Yes" else 0
        alco = 1 if alco_txt == "Yes" else 0
        active = 1 if active_txt == "Yes" else 0

        features = np.array([[age, gender, height, weight,
                               ap_hi, ap_lo, cholesterol,
                               gluc, smoke, alco, active]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        st.divider()

        if prediction == 1:
            st.markdown("<div class='status high'>HIGH RISK</div>", unsafe_allow_html=True)
            st.markdown("<span class='badge badge-high'>Heart Disease Detected</span>", unsafe_allow_html=True)
            st.write(f"**Risk Probability:** {probability:.2f}%")
        else:
            st.markdown("<div class='status low'>LOW RISK</div>", unsafe_allow_html=True)
            st.markdown("<span class='badge badge-low'>Healthy Heart</span>", unsafe_allow_html=True)
            st.write(f"**Health Confidence:** {100 - probability:.2f}%")

    else:
        st.info("Please fill patient details and click **Predict Heart Disease**")

    st.markdown("</div>", unsafe_allow_html=True)
