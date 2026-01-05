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
    background-color: #f9fafb !important;
    color: #111827 !important;
}

.main-card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    color: #2563eb;
    font-weight: 800;
}

.header p {
    color: #6b7280;
    font-size: 16px;
}

.section-title {
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 15px;
}

.result-box {
    text-align: center;
    padding: 30px;
    border-radius: 16px;
    background: #f1f5f9;
}

.big-text {
    font-size: 26px;
    font-weight: 800;
}

.low { color: #16a34a; }
.high { color: #dc2626; }

button {
    font-size: 16px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ model.pkl not found. Run training first.")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>🫀 Heart Disease Prediction System</h1>
    <p>AI-powered health risk analysis</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
left, right = st.columns([3, 2])

# ================= INPUT PANEL =================
with left:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Patient Details</div>", unsafe_allow_html=True)

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
    alco_txt = c11.selectbox("Alcohol", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("🔍 Predict Heart Disease", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RESULT PANEL =================
with right:
    st.markdown("<div class='main-card result-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

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
        prob = model.predict_proba(features)[0][1] * 100

        st.divider()

        if prediction == 1:
            st.markdown("<div class='big-text high'>❗ YES</div>", unsafe_allow_html=True)
            st.write("**High Risk of Heart Disease**")
            st.write(f"**Confidence:** {prob:.2f}%")
        else:
            st.markdown("<div class='big-text low'>✅ NO</div>", unsafe_allow_html=True)
            st.write("**Low Risk / Healthy Heart**")
            st.write(f"**Confidence:** {100 - prob:.2f}%")

    else:
        st.info("Fill details and click **Predict Heart Disease**")

    st.markdown("</div>", unsafe_allow_html=True)
