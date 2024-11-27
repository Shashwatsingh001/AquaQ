import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained CNN model
model_path = 'C:/Users/KIIT/Desktop/AqualQ/src/SavedModels/water_quality_model_cnn.h5'
cnn_model = load_model(model_path)

# Initialize session state for user inputs and predictions
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'cnn_predictions' not in st.session_state:
    st.session_state['cnn_predictions'] = None
if 'actual_labels' not in st.session_state:
    st.session_state['actual_labels'] = None
if 'manual_prediction' not in st.session_state:
    st.session_state['manual_prediction'] = None
if 'manual_drinking' not in st.session_state:
    st.session_state['manual_drinking'] = None

st.title("Water Quality Prediction Dashboard")

# Sidebar for dataset upload
uploaded_file = st.sidebar.file_uploader("Upload Your Dataset (CSV)", type="csv")

if uploaded_file:
    # Load and display dataset
    st.session_state['uploaded_file'] = uploaded_file
    df = pd.read_csv(uploaded_file)

    # Encode target column if needed
    if 'Water Quality' in df.columns:
        water_quality_mapping = {'Good': 2, 'Needs Treatment': 1, 'Poor': 0}
        df['Water Quality'] = df['Water Quality'].map(water_quality_mapping)
    
    st.write("### Uploaded Dataset")
    st.dataframe(df.head())

    # Prepare data for predictions
    X = df.drop(columns=['Water Quality', 'Drinking Water', 'Date'], errors='ignore')
    y = df['Water Quality'] if 'Water Quality' in df.columns else None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Add prediction button
    if st.button("Predict on Entire Dataset"):
        # Reshape for CNN input
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        predictions = cnn_model.predict(X_reshaped)
        quality_pred = np.argmax(predictions, axis=1)

        # Save predictions and actual labels in session state
        st.session_state['cnn_predictions'] = predictions
        if y is not None:
            st.session_state['actual_labels'] = y.to_numpy()

        # Add Drinking Water Recommendation
        df['Predicted Quality'] = ['Poor' if q == 0 else 'Needs Treatment' if q == 1 else 'Good' for q in quality_pred]
        df['Drinking Water'] = ['Yes' if q == 2 else 'No' for q in quality_pred]

        st.write("### Prediction Results")
        st.dataframe(df[['Predicted Quality', 'Drinking Water']])
        
        # Calculate and display accuracy if actual labels are available
        if y is not None:
            accuracy = accuracy_score(y, quality_pred)
            st.write(f"### Model Accuracy on Uploaded Dataset: {accuracy:.2%}")

        # Option to download predictions
        csv_data = df.to_csv(index=False)
        st.download_button(label="Download Predictions as CSV", data=csv_data, file_name="predicted_water_quality.csv")
else:
    st.write("Upload a dataset to view predictions on the entire dataset.")

# Confusion Matrix
if st.button("View Confusion Matrix"):
    if st.session_state['cnn_predictions'] is not None and st.session_state['actual_labels'] is not None:
        y_pred_quality_cnn = np.argmax(st.session_state['cnn_predictions'], axis=1)
        conf_matrix = confusion_matrix(st.session_state['actual_labels'], y_pred_quality_cnn)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Poor', 'Needs Treatment', 'Good'],
                    yticklabels=['Poor', 'Needs Treatment', 'Good'])
        plt.title("Confusion Matrix (CNN)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)
    else:
        st.warning("Run predictions on a dataset first to view the confusion matrix.")

# Training vs Validation Metrics
if st.button("View Training History"):
    # Dummy training history (replace with real history if available)
    dummy_history = {
        'accuracy': np.random.uniform(0.6, 1.0, 50),
        'val_accuracy': np.random.uniform(0.5, 0.95, 50),
        'loss': np.random.uniform(0.1, 1.0, 50),
        'val_loss': np.random.uniform(0.2, 1.5, 50)
    }

    # Plot Training vs Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(dummy_history['accuracy'], label='Training Accuracy')
    plt.plot(dummy_history['val_accuracy'], label='Validation Accuracy')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    st.pyplot(plt)

    # Plot Training vs Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(dummy_history['loss'], label='Training Loss')
    plt.plot(dummy_history['val_loss'], label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(plt)
