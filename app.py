import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==============================
# Load Saved Objects
# ==============================
model = joblib.load("models/career_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_encoders = joblib.load("models/feature_encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.title("🎓 Student Career Prediction System")

# ==============================
# User Inputs
# ==============================

age = st.number_input("Age", min_value=16, max_value=60, value=21)
gpa = st.number_input("GPA", min_value=0.0, max_value=10.0, value=7.0)

python_skill = st.selectbox(
    "Python Skill Level",
    list(feature_encoders["Python"].classes_)
)

sql_skill = st.selectbox(
    "SQL Skill Level",
    list(feature_encoders["SQL"].classes_)
)

java_skill = st.selectbox(
    "Java Skill Level",
    list(feature_encoders["Java"].classes_)
)

projects = st.selectbox(
    "Project",
    list(feature_encoders["Projects"].classes_)
)

major = st.selectbox(
    "Major",
    list(feature_encoders["Major"].classes_)
)

gender = st.selectbox(
    "Gender",
    list(feature_encoders["Gender"].classes_)
)


# ==============================
# Prediction
# ==============================

if st.button("Predict Career"):

    try:
        # Encode categorical inputs
        input_data = {
            "Gender": feature_encoders["Gender"].transform([gender])[0],
            "Age": age,
            "GPA": gpa,
            "Major": feature_encoders["Major"].transform([major])[0],
            "Projects": feature_encoders["Projects"].transform([projects])[0],
            "Python": feature_encoders["Python"].transform([python_skill])[0],
            "SQL": feature_encoders["SQL"].transform([sql_skill])[0],
            "Java": feature_encoders["Java"].transform([java_skill])[0]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Reorder columns exactly like training
        input_df = input_df[model_columns]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        # Decode output
        career = target_encoder.inverse_transform(prediction)[0]

        st.success(f"🎯 Predicted Career: **{career}**")

    except Exception as e:
        st.error(f"Error: {e}")