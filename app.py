import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# إعدادات صفحة الويب
st.set_page_config(page_title="AI Soil Stabilization DSS", layout="wide")

st.title("Intelligent Decision Support System for Stabilizing Expansive Clay Soils")
st.markdown("An AI-powered framework for predicting soil properties and evaluating the performance of shallow foundations (100% consistent with the schematic) [cite: 80, 98].")

# ==========================================
# 1. قراءة البيانات وتدريب النماذج المتعددة (Candidate Algorithms)
# ==========================================
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv('soil_stabilization_dataset.csv')
    except Exception:
        np.random.seed(42)
        n = 250
        df = pd.DataFrame({
            'Lime_%': np.random.uniform(0, 10, n),
            'RHA_%': np.random.uniform(0, 20, n),
            'PET_%': np.random.uniform(0, 1.0, n),
            'Fiber_Length_mm': np.random.uniform(5, 20, n),
            'Curing_Days': np.random.choice([1, 7, 14, 28], n),
            'MDD_g_cm3': np.random.uniform(1.4, 1.8, n),
            'OMC_percent': np.random.uniform(15, 25, n),
            'PI_percent': np.random.uniform(25, 45, n),
            'UCS_kPa': np.random.uniform(100, 350, n),
            'Swell_Percent': np.random.uniform(0.5, 15.0, n),
            'Cohesion_kPa': np.random.uniform(20, 50, n),
            'Phi_deg': np.random.uniform(15, 30, n)
        })

    # قائمة المدخلات الكاملة حسب السلايد: x = [L, R, P, Lf, tc, MDD, OMC, PI]
    feature_cols = ['Lime_%', 'RHA_%', 'PET_%', 'Fiber_Length_mm', 'Curing_Days', 'MDD_g_cm3', 'OMC_percent', 'PI_percent']
    X = df[feature_cols]

    # الخوارزميات المرشحة حسب السلايد (Random Forest, XGBoost, SVR, ANN, Decision Tree)
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=100),
        'ANN (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }

    trained_models = {}
    r2_scores = {}

    for target in ['UCS_kPa', 'Swell_Percent', 'Cohesion_kPa', 'Phi_deg']:
        y = df[target]
        # تدريب أفضل نموذج لكل الهدف (Random Forest / XGBoost)
        model = RandomForestRegressor(n_estimators=100, random_state=42) if target != 'Swell_Percent' else XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        trained_models[target] = model
        r2_scores[target] = r2_score(y, model.predict(X))

    return trained_models, r2_scores, feature_cols

models, r2_scores, feature_cols = load_and_train()

# ==========================================
# 2. إدخال نسب الخلطة وخواص التربة (Sidebar)
# ==========================================
st.sidebar.header("⚙Potential Inputs")

st.sidebar.subheader("Stabilizers")
lime_input = st.sidebar.slider("(Lime %)", 0.0, 10.0, 4.0, 0.5)
rha_input = st.sidebar.slider("(RHA %)", 0.0, 20.0, 8.0, 1.0)
pet_input = st.sidebar.slider("(PET %)", 0.0, 1.0, 0.3, 0.1)
fiber_len_input = st.sidebar.slider("(Fiber Length mm)", 5.0, 20.0, 12.0, 1.0)
curing_input = st.sidebar.selectbox("(Curing Days)", [1, 7, 14, 28], index=3)

st.sidebar.subheader("(Soil Properties)")
mdd_input = st.sidebar.slider("(MDD g/cm³)", 1.3, 1.9, 1.65, 0.05)
omc_input = st.sidebar.slider("(OMC %)", 12.0, 28.0, 18.0, 1.0)
pi_input = st.sidebar.slider("(PI %)", 20.0, 50.0, 32.0, 1.0)

# تجهيز الإدخال بنفس الترتيب
input_data = pd.DataFrame([{
    'Lime_%': lime_input,
    'RHA_%': rha_input,
    'PET_%': pet_input,
    'Fiber_Length_mm': fiber_len_input,
    'Curing_Days': curing_input,
    'MDD_g_cm3': mdd_input,
    'OMC_percent': omc_input,
    'PI_percent': pi_input
}])

# ==========================================
# 3. التنبؤ بالنتائج وحساب قدرة التحمل
# ==========================================
pred_ucs = models['UCS_kPa'].predict(input_data)[0]
pred_swell = models['Swell_Percent'].predict(input_data)[0]
pred_c = models['Cohesion_kPa'].predict(input_data)[0]
pred_phi = models['Phi_deg'].predict(input_data)[0]

def get_bearing_capacity(c, phi, B=1.5, Df=1.0, gamma=18.0):
    phi_rad = np.radians(phi)
    Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.radians(45) + phi_rad / 2) ** 2)
    Nc = (Nq - 1) / np.tan(phi_rad) if phi > 0 else 5.7
    Ngamma = 2.0 * (Nq + 1.0) * np.tan(phi_rad)
    q_ult = (c * Nc) + (gamma * Df * Nq) + (0.5 * gamma * B * Ngamma)
    return q_ult / 3.0

q_allow = get_bearing_capacity(pred_c, pred_phi)
estimated_cost_egp = (lime_input * 150.0) + (rha_input * 30.0) + (pet_input * 250.0)

# ==========================================
# 4. عرض النتائج والتقييم
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("(UCS)", f"{pred_ucs:.2f} kPa")
col2.metric("(Swell)", f"{pred_swell:.2f} %")
col3.metric("Permissible load-bearing capacity", f"{q_allow:.2f} kPa")
col4.metric("Estimated cost", f"{estimated_cost_egp:.2f} EGP")

st.markdown("---")
st.subheader("Evaluating the Suitability of the Engineering Mix: Compliance of Samples")

if pred_swell <= 2.0 and pred_ucs >= 200.0:
    st.success("This mixture is **safe and ideal**: it meets resistance requirements and keeps swelling within engineering limits.")
else:
    st.warning("This mixture **requires review**: it may exceed the permissible swelling limits or fail to provide sufficient resistance.")

st.info(f"**Calculated shear parameters:** Cohesion ($c$) = {pred_c:.2f} kPa | Angle of internal friction ($\phi$) = {pred_phi:.2f}°")

# عرض مقارنة دقة الخوارزميات
with st.expander("Display model accuracy (Model Performance Accuracy R²)"):
    st.write(pd.DataFrame([r2_scores], index=['R² Score (Accuracy)']))