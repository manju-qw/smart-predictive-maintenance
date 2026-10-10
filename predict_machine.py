
import joblib
import pandas as pd

# Load the trained model
saved_data = joblib.load("models/random_forest_model.joblib")

model = saved_data["model"]
features = saved_data["features"]
encoded_columns = saved_data["encoded_columns"]


def predict_machine(machine_type, air_temp, process_temp,
                    rotational_speed, torque, tool_wear):

    # Prepare machine input
    machine_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Convert machine type into numeric columns
    machine_data = pd.get_dummies(
        machine_data, columns=["Type"], dtype=int
    )

    # Match the columns used during training
    machine_data = machine_data.reindex(
        columns=encoded_columns, fill_value=0
    )

    # Predict machine failure
    prediction = model.predict(machine_data)[0]
    probability = model.predict_proba(machine_data)[0][1] * 100

    print("\n--- MACHINE PREDICTION RESULT ---")

    if prediction == 1:
        print("Prediction: MACHINE FAILURE RISK DETECTED")
    else:
        print("Prediction: NO FAILURE PREDICTED")

    print(f"Estimated failure probability: {probability:.2f}%")

    if probability >= 60:
        print("Risk Level: HIGH")
    elif probability >= 30:
        print("Risk Level: MEDIUM")
    else:
        print("Risk Level: LOW")


# Example machine readings
predict_machine(
    machine_type="M",
    air_temp=298.1,
    process_temp=308.6,
    rotational_speed=1500,
    torque=40.0,
    tool_wear=100
)
