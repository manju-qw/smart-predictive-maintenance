
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Smart Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Load the trained model
saved_data = joblib.load("models/random_forest_model.joblib")
model = saved_data["model"]
encoded_columns = saved_data["encoded_columns"]

st.title("⚙️ Smart Predictive Maintenance System")
st.write("Machine failure prediction using Machine Learning")

st.divider()

st.subheader("Enter Machine Details")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type", ["L", "M", "H"]
    )
    air_temp = st.number_input(
        "Air Temperature (K)", min_value=250.0,
        max_value=350.0, value=298.1
    )
    process_temp = st.number_input(
        "Process Temperature (K)", min_value=250.0,
        max_value=400.0, value=308.6
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)", min_value=0,
        max_value=10000, value=1500
    )
    torque = st.number_input(
        "Torque (Nm)", min_value=0.0,
        max_value=200.0, value=40.0
    )
    tool_wear = st.number_input(
        "Tool Wear (minutes)", min_value=0,
        max_value=300, value=100
    )

if st.button("Predict Machine Failure", type="primary"):
    machine_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    machine_data = pd.get_dummies(
        machine_data, columns=["Type"], dtype=int
    )
    machine_data = machine_data.reindex(
        columns=encoded_columns, fill_value=0
    )

    prediction = model.predict(machine_data)[0]
    probability = model.predict_proba(machine_data)[0][1] * 100

    st.subheader("Prediction Result")
    st.metric("Estimated Failure Probability", f"{probability:.2f}%")

    if prediction == 1:
        st.error("Failure risk detected. Inspect the machine.")
    else:
        st.success("The model did not predict a failure.")

    if probability >= 60:
        st.warning("Demonstration Risk Level: HIGH")
    elif probability >= 30:
        st.info("Demonstration Risk Level: MEDIUM")
    else:
        st.success("Demonstration Risk Level: LOW")

    st.caption(
        "This is a model-based estimate, not a guarantee of machine safety. "
        "Risk thresholds are illustrative and require validation."
    )
