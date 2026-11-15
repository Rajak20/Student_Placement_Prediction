import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model (2).pkl"
SCALER_PATH = BASE_DIR / "scaler (5).pkl"


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)
    return model, scaler


st.title("🎓 Student Placement Prediction")
st.write(
    "Enter the student's academic and lifestyle details to estimate "
    "the placement prediction."
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError as error:
    st.error(
        f"Required model file not found: {error.filename}. "
        "Keep both .pkl files in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

with st.form("placement_form"):
    st.subheader("Student details")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.number_input(
            "Study hours per day", min_value=0.0, max_value=24.0,
            value=5.0, step=0.5
        )
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0,
            value=75.0, step=1.0
        )
        sleep_hours = st.number_input(
            "Sleep hours per day", min_value=0.0, max_value=24.0,
            value=7.0, step=0.5
        )
        internet_usage = st.number_input(
            "Internet usage (hours per day)", min_value=0.0,
            max_value=24.0, value=4.0, step=0.5
        )

    with col2:
        assignments_completed = st.number_input(
            "Assignments completed", min_value=0, max_value=1000,
            value=8, step=1
        )
        previous_score = st.number_input(
            "Previous score (%)", min_value=0.0, max_value=100.0,
            value=65.0, step=1.0
        )
        exam_score = st.number_input(
            "Exam score (%)", min_value=0.0, max_value=100.0,
            value=80.0, step=1.0
        )

    submitted = st.form_submit_button(
        "Predict Placement", type="primary", use_container_width=True
    )

if submitted:
    feature_names = [
        "study_hours",
        "attendance",
        "sleep_hours",
        "internet_usage",
        "assignments_completed",
        "previous_score",
        "exam_score",
    ]

    input_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            sleep_hours,
            internet_usage,
            assignments_completed,
            previous_score,
            exam_score,
        ]],
        columns=feature_names,
    )

    try:
        # Apply the same StandardScaler transformation used during training.
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.divider()
        st.subheader("Prediction result")

        # Assumes the training target was encoded as 0 = Not Placed, 1 = Placed.
        # Change these labels if your dataset used the opposite encoding.
        label_map = {0: "Not Placed", 1: "Placed"}
        result = label_map.get(prediction, f"Class {prediction}")

        if prediction == 1:
            st.success(f"### {result}")
        elif prediction == 0:
            st.warning(f"### {result}")
        else:
            st.info(f"### {result}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            classes = model.classes_
            probability_data = {
                label_map.get(cls, f"Class {cls}"): float(prob) * 100
                for cls, prob in zip(classes, probabilities)
            }
            st.write("Prediction probabilities")
            st.bar_chart(
                pd.DataFrame(
                    {"Probability (%)": probability_data}
                )
            )

        st.caption(
            "The result is a model estimate, not a guarantee of actual placement. "
            "The displayed class labels assume 0 = Not Placed and 1 = Placed."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")

with st.expander("About this model"):
    st.write(
        "- Algorithm: Gaussian Naive Bayes (GaussianNB)\n"
        "- Preprocessing: StandardScaler\n"
        "- Input features: study_hours, attendance, sleep_hours, "
        "internet_usage, assignments_completed, previous_score, exam_score"
    )
