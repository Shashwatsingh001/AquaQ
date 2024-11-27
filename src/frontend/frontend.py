import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the saved model
model_filename = "C:/Users/KIIT/Desktop/AqualQ/src/SavedModels/random_model.sav"
rf_model = joblib.load(model_filename)

# Water quality mapping
water_quality_mapping = {'Good': 2, 'Needs Treatment': 1, 'Poor': 0}
inverse_water_quality_mapping = {0: 'Poor', 1: 'Needs Treatment', 2: 'Good'}

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'actual_labels' not in st.session_state:
    st.session_state['actual_labels'] = None
if 'X_features' not in st.session_state:
    st.session_state['X_features'] = None
if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False

# Streamlit frontend layout
st.title("Water Quality Prediction System")
st.sidebar.header("Input Parameters")

# File upload option
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

# User input form for manual data entry
st.sidebar.subheader("Or Enter Values Manually")
chlorophyll = st.sidebar.number_input("Chlorophyll", min_value=-2.0, max_value=2.0, step=0.1)
dissolved_oxygen = st.sidebar.number_input("Dissolved Oxygen", min_value=0.0, max_value=10.0, step=0.1)
dissolved_oxygen_matter = st.sidebar.number_input("Dissolved Oxygen Matter", min_value=200.0, max_value=4000.0, step=1.0)
suspended_matter = st.sidebar.number_input("Suspended Matter", min_value=200.0, max_value=1500.0, step=1.0)
salinity = st.sidebar.number_input("Salinity", min_value=-5.0, max_value=5.0, step=0.1)
temperature = st.sidebar.number_input("Temperature", min_value=0.0, max_value=50.0, step=0.1)
turbidity = st.sidebar.number_input("Turbidity", min_value=-1.0, max_value=1.0, step=0.1)
ph = st.sidebar.number_input("pH", min_value=1.0, max_value=14.0, step=0.1)

# Collect manual input data
manual_input_data = {
    'Chlorophyll': chlorophyll,
    'Dissolved Oxygen': dissolved_oxygen,
    'Dissolved Oxygen Matter': dissolved_oxygen_matter,
    'Suspended Matter': suspended_matter,
    'Salinty': salinity,
    'Temperature': temperature,
    'Turbidity': turbidity,
    'pH': ph
}
manual_df = pd.DataFrame([manual_input_data])

# Handle file upload and predictions
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['file_uploaded'] = True
    st.write("Uploaded Data Preview:", df.head())

    # Prepare features
    X = df.drop(columns=['Water Quality', 'Drinking Water', 'Date'], errors='ignore')

    # Store actual labels if present
    if 'Water Quality' in df.columns:
        st.session_state['actual_labels'] = df['Water Quality'].map(water_quality_mapping).values

    # Store features in session state
    st.session_state['X_features'] = X

# Predict Water Quality and Drinking Water for uploaded file
if st.sidebar.button("Predict (File Data)") and st.session_state['file_uploaded']:
    # Predict Water Quality
    y_pred_quality = rf_model.predict(st.session_state['X_features'])
    st.session_state['predictions'] = y_pred_quality

    # Derive Drinking Water predictions
    y_pred_drinking = ['Yes' if q == 2 else 'No' for q in y_pred_quality]

    # Display Predictions
    predicted_quality = [inverse_water_quality_mapping[q] for q in y_pred_quality]
    predictions = pd.DataFrame({
        'Predicted Water Quality': predicted_quality,
        'Predicted Drinking Water': y_pred_drinking
    })
    st.write("Predictions:", predictions)

    # Accuracy if actual labels are available
    if st.session_state['actual_labels'] is not None:
        accuracy = accuracy_score(st.session_state['actual_labels'], y_pred_quality)
        st.write(f"Accuracy: {accuracy:.2f}")

# Manual Input Prediction
if st.sidebar.button("Predict (Manual Data)"):
    # Predict Water Quality
    y_pred_quality_manual = rf_model.predict(manual_df)[0]
    y_pred_drinking_manual = 'Yes' if y_pred_quality_manual == 2 else 'No'

    st.write("Manual Input Prediction:")
    st.write(f"Predicted Water Quality: {inverse_water_quality_mapping[y_pred_quality_manual]}")
    st.write(f"Drinking Water Recommendation: {y_pred_drinking_manual}")

# Confusion Matrix
if st.sidebar.button("View Confusion Matrix") and st.session_state['predictions'] is not None:
    conf_matrix = confusion_matrix(st.session_state['actual_labels'], st.session_state['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Poor', 'Needs Treatment', 'Good'],
                yticklabels=['Poor', 'Needs Treatment', 'Good'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

# Classification Report
if st.sidebar.button("View Classification Report") and st.session_state['predictions'] is not None:
    report = classification_report(st.session_state['actual_labels'], st.session_state['predictions'], target_names=['Poor', 'Needs Treatment', 'Good'])
    st.text_area("Classification Report", report)

st.sidebar.subheader("About the Model")
st.sidebar.write("""
    This Random Forest model predicts the water quality (Good, Needs Treatment, Poor) based on input features such as Chlorophyll, Turbidity, pH, etc.
    Based on the prediction, it also recommends if the water is safe for drinking.
""")