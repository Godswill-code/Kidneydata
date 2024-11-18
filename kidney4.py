import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open(r'best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open(r'scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

# List of features (Ensure these are individual strings, not a nested list)
input_features = [
    'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
    'hemo', 'pcv', 'wc', 'rc', 'rbc_encoded', 'pc_encoded', 'pcc_encoded',
    'ba_encoded', 'htn_encoded', 'dm_encoded', 'cad_encoded', 'appet_encoded',
    'pe_encoded', 'ane_encoded'
]

# Define Streamlit app
def main():
    st.title("CKD Prediction")

    # Initialize input data dictionary
    input_data = {}

    # Collect user inputs
    for feature in input_features:
        input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data
    input_data_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.write("Prediction:", "CKD Detected" if prediction[0] == 1 else "No CKD Detected")

if __name__ == "__main__":
    main()
