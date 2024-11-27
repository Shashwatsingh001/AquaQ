import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the trained model
model_path = "C:/Users/KIIT/Desktop/AqualQ/src/SavedModels/water_quality_model.h5"
nn_model = load_model(model_path)

# Load the scaler
scaler_path = "C:/Users/KIIT/Desktop/AqualQ/src/SavedModels/scaler.pkl"
scaler = joblib.load(scaler_path)

# Water quality mapping
water_quality_mapping = {0: "Poor", 1: "Needs Treatment", 2: "Good"}
inverse_water_quality_mapping = {v: k for k, v in water_quality_mapping.items()}

# Function to derive drinking water predictions
def derive_drinking_water(quality_pred):
    return ['Yes' if q == 2 else 'No' for q in quality_pred]

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'actual_labels' not in st.session_state:
    st.session_state['actual_labels'] = None
if 'X_features' not in st.session_state:
    st.session_state['X_features'] = None

# Streamlit Layout
st.title("Neural Network-based Water Quality Prediction System")
st.sidebar.header("Input Parameters")

# File upload option
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

# Manual Input for predictions
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

# File-based prediction
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())
    
    # Handle column name mismatches in uploaded data
    if 'Salinity' in df.columns:
        df.rename(columns={'Salinity': 'Salinty'}, inplace=True)

    # Feature Selection
    if 'Water Quality' in df.columns:
        st.session_state['actual_labels'] = df['Water Quality'].map(inverse_water_quality_mapping).values
    X = df.drop(columns=['Water Quality', 'Drinking Water', 'Date'], errors='ignore')
    st.session_state['X_features'] = scaler.transform(X)

    # Prediction button for file-based data
    if st.sidebar.button("Predict (File Data)"):
        y_pred_quality_probs = nn_model.predict(st.session_state['X_features'])
        st.session_state['predictions'] = np.argmax(y_pred_quality_probs, axis=1)

        # Display predictions
        predictions = pd.DataFrame({
            'Predicted Water Quality': [water_quality_mapping[q] for q in st.session_state['predictions']],
            'Drinking Water': derive_drinking_water(st.session_state['predictions'])
        })
        st.write("Predictions:", predictions)

# Manual prediction
if st.sidebar.button("Predict (Manual Data)"):
    X_manual_scaled = scaler.transform(manual_df)
    y_pred_manual_prob = nn_model.predict(X_manual_scaled)
    y_pred_manual_quality = np.argmax(y_pred_manual_prob, axis=1)[0]
    y_pred_manual_drinking = 'Yes' if y_pred_manual_quality == 2 else 'No'

    st.write(f"Predicted Water Quality: {water_quality_mapping[y_pred_manual_quality]}")
    st.write(f"Drinking Water Recommendation: {y_pred_manual_drinking}")

# Confusion Matrix and Classification Report
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

if st.sidebar.button("View Classification Report") and st.session_state['predictions'] is not None:
    report = classification_report(st.session_state['actual_labels'], st.session_state['predictions'], target_names=['Poor', 'Needs Treatment', 'Good'])
    st.text_area("Classification Report", report)

# Model Information
st.sidebar.subheader("About the Model")
st.sidebar.write("""
    This deep learning model uses a neural network with 4 layers to predict water quality based on environmental factors. 
    It also recommends whether the water is safe for drinking.
""")
