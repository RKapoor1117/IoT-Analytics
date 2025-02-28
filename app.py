import streamlit as st
import pickle
import numpy as np

# Load the trained model using pickle
with open("rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("IoT Predictive Maintenance - Machine Failure Detection")
st.write("Enter sensor data to predict machine failure.")

# Machine Type Input (Dropdown)
machine_type = st.selectbox("Machine Type", options=[0, 1, 2], index=0)

# Continuous Input Fields
air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
process_temp = st.number_input("Process Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
rpm = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=100, step=1)

# Binary Features (Dropdowns for 0 or 1)
twf = st.selectbox("Tool Wear Failure (TWF)", options=[0, 1], index=0)
hdf = st.selectbox("Heat Dissipation Failure (HDF)", options=[0, 1], index=0)
pwf = st.selectbox("Power Failure (PWF)", options=[0, 1], index=0)
osf = st.selectbox("Overstrain Failure (OSF)", options=[0, 1], index=0)
rnf = st.selectbox("Random Failure (RNF)", options=[0, 1], index=0)

# Prediction function
def predict_failure():
    input_data = np.array([[machine_type, air_temp, process_temp, rpm, torque, tool_wear, twf, hdf, pwf, osf, rnf]])
    print(f"Features given for prediction: {input_data.shape[1]} features")
    
    prediction = model.predict(input_data)
    return "Failure" if prediction[0] == 1 else "No Failure"

# Prediction button
if st.button("Predict Machine Failure"):
    result = predict_failure()
    st.write(f"Prediction: **{result}**")
