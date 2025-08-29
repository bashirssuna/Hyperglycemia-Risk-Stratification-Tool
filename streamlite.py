# ---------------------------------------------------------------------------
# Streamlit IFG Risk Stratification App (patched for sklearn 1.3.0)
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# === Compatibility shim: lets sklearn 1.3 unpickle artifacts saved by 1.4.x ===
try:
    from sklearn.compose import _column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Shim to satisfy pickle resolution for newer sklearn ColumnTransformer."""
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    # If sklearn import itself fails, we will surface a clear error later
    pass
# ==============================================================================

# --- App Configuration ---
st.set_page_config(page_title="IFG Risk Stratification Tool", page_icon="🩺", layout="wide")

# --- Load Model and Preprocessor ---
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load preprocessor and model once, with readable diagnostics."""
    try:
        if not os.path.exists('validated_preprocessor.joblib') or not os.path.exists('validated_model.joblib'):
            raise FileNotFoundError("validated_preprocessor.joblib or validated_model.joblib not found")

        preprocessor = joblib.load('validated_preprocessor.joblib')
        model = joblib.load('validated_model.joblib')
        return preprocessor, model
    except FileNotFoundError as e:
        st.error(str(e))
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model files: {e}")
        st.info("Likely a scikit-learn version mismatch between training and this app. "
                "This app includes a compatibility shim for ColumnTransformer pickles from newer sklearn. "
                "If the error persists, either upgrade sklearn to the training version or re-export artifacts.")
        return None, None

preprocessor, model = load_model()

# Quick version banner
try:
    import sklearn, numpy, pandas
    st.caption(f"sklearn {sklearn.__version__} • numpy {numpy.__version__} • pandas {pandas.__version__}")
except Exception:
    pass

# --- Risk Thresholds ---
MODERATE_RISK_THRESHOLD = 0.06*3
HIGH_RISK_THRESHOLD = 0.24*3

# --- UI ---
st.title('🩺 Impaired Fasting Glucose (IFG) Risk Calculator')
st.markdown("This tool uses a machine learning model to stratify a patient's risk of Impaired Fasting Glucose using selected clinical and lifestyle factors.")

st.sidebar.header('Patient Information')

def user_inputs():
    age = st.sidebar.number_input('Age (years)', 18, 100, 45, 1)
    bmi = st.sidebar.number_input('Body Mass Index (BMI)', 10.0, 60.0, 28.5, 0.1)
    hr  = st.sidebar.number_input('Heart Rate (bpm)', 40, 180, 72, 1)
    dbp = st.sidebar.number_input('Diastolic Blood Pressure (mmHg)', 40, 150, 80, 1)
    sbp = st.sidebar.number_input('Systolic Blood Pressure (mmHg)', 70, 250, 120, 1)
    b8  = st.sidebar.number_input('Total Cholesterol (mmol/l)', 1.0, 15.0, 5.2, 0.1)
    m14 = st.sidebar.number_input('Waist Circumference (cm)', 50.0, 200.0, 95.5, 0.1)
    d1  = st.sidebar.slider('Days per week eating fruit', 0, 7, 3)
    d3  = st.sidebar.slider('Days per week eating vegetables', 0, 7, 5)

    p1  = st.sidebar.selectbox('Vigorous-intensity work activity?', ('No', 'Yes'))
    p13 = st.sidebar.selectbox('Moderate-intensity recreational activity?', ('No', 'Yes'))

    # Map to training codes. You said the model expects 1 for Yes and 2 for No.
    p1_encoded  = 1 if p1 == 'Yes' else 2
    p13_encoded = 1 if p13 == 'Yes' else 2

    return {
        'age': age, 'bmi': bmi, 'HR': hr, 'DBP': dbp, 'b8': b8, 'm14': m14,
        'SBP': sbp, 'd1': d1, 'd3': d3, 'p1': p1_encoded, 'p13': p13_encoded
    }

input_data = user_inputs()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Entered Data Summary")
    st.write(f"**Age:** {input_data['age']}")
    st.write(f"**BMI:** {input_data['bmi']}")
    st.write(f"**Heart Rate:** {input_data['HR']} bpm")
    st.write(f"**Blood Pressure:** {input_data['SBP']}/{input_data['DBP']} mmHg")
    st.write(f"**Total Cholesterol:** {input_data['b8']} mmol/l")
    st.write(f"**Waist Circumference:** {input_data['m14']} cm")

with col2:
    st.subheader("Risk Assessment")
    calculate_button = st.button('Calculate Risk', type="primary", use_container_width=True)

    if calculate_button and (preprocessor is None or model is None):
        st.error("Model is not loaded. Cannot perform calculation.")

    if calculate_button and (preprocessor is not None and model is not None):
        # Keep exact training column order
        columns_ordered = ["p1", "p13", "age", "bmi", "HR", "DBP", "b8", "m14", "SBP", "d1", "d3"]
        input_df = pd.DataFrame([input_data], columns=columns_ordered)

        # Ensure numeric dtypes to avoid category surprises
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce")

        try:
            X = preprocessor.transform(input_df)
            # Some calibrators wrap estimators; they all expose predict_proba
            proba = float(3*(model.predict_proba(X)[0][1]))
        except Exception as e:
            st.error(f"Failed during transform or predict: {e}")
            st.stop()

        # Risk categories
        if proba >= HIGH_RISK_THRESHOLD:
            risk_level = "High Risk"
            st.error(f"**Risk Level: {risk_level}**")
        elif proba >= MODERATE_RISK_THRESHOLD:
            risk_level = "Moderate Risk"
            st.warning(f"**Risk Level: {risk_level}**")
        else:
            risk_level = "Low Risk"
            st.success(f"**Risk Level: {risk_level}**")

        st.metric(label="Predicted Probability of IFG", value=f"{proba:.1%}")

        st.info(
            "**Disclaimer:** This is a screening tool, not a diagnostic test.\n"
            "Created using STEPS Survey data for Ugandan Population and not yet approved.\n"
            "- High Risk: prioritize confirmatory fasting glucose or HbA1c.\n"
            "- Moderate Risk: lifestyle counselling and follow-up.\n"
            "- Low Risk: reassurance and general health advice.\n"
            "- Developed by: Dr. Bashir Ssuna and Prof. Bahendeka Silver"
        )

