import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # Fixed import
import pandas as pd
import pickle

# Load the trained model, scaler pickle, onehot
model = load_model('model.h5')

# load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction')
st.write("Predict whether a customer will leave the bank")

# USER INPUT
st.header("Customer Information")

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92, 40)
    balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)

with col2:
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0, value=50000, step=1000)
    tenure = st.slider("Tenure (years)", 0, 10, 3)
    num_of_products = st.slider("Number of Products", 1, 4, 2)
    has_crcard = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction button
if st.button("Predict Churn", type="primary"):
    # Create initial input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],  # Keep as string for now
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_crcard],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    
    # Debug: Show raw input
    st.subheader("📋 Input Data")
    st.dataframe(input_data)
    
    try:
        # Step 1: Encode Gender using LabelEncoder
        input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
        
        # Step 2: One-hot encode Geography
        geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']])
        # Handle both sparse and dense outputs
        if hasattr(geo_encoded, 'toarray'):
            geo_encoded = geo_encoded.toarray()
        
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )
        
        # Concatenate one-hot encoded columns
        input_data = pd.concat([input_data.drop("Geography", axis=1), geo_encoded_df], axis=1)
        
        # Debug: Show encoded data
        st.subheader("🔢 Encoded Features")
        st.dataframe(input_data)
        
        # Step 3: Ensure correct column order
        # IMPORTANT: Get the expected columns from your scaler or model training
        # This should match the order used during training
        expected_columns = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        # Add geography columns (adjust based on your actual categories)
        geo_columns = [col for col in input_data.columns if 'Geography_' in col]
        expected_columns.extend(sorted(geo_columns))
        
        # Reorder and fill missing with 0
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        
        # Debug: Show final dataframe before scaling
        st.subheader("🎯 Final Features (Scaled Input)")
        st.dataframe(input_data)
        
        # Step 4: Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Step 5: Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        prediction_proba = float(prediction[0][0])
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        # Create a nice progress bar for probability
        col_proba1, col_proba2 = st.columns([3, 1])
        with col_proba1:
            st.progress(prediction_proba, text=f"Churn Probability: {prediction_proba:.1%}")
        with col_proba2:
            st.metric("Churn Probability", f"{prediction_proba:.1%}")
        
        # Decision with styling
        st.subheader("📊 Decision")
        if prediction_proba > 0.5:
            st.error(f"🚨 **HIGH RISK: Customer is likely to CHURN**")
            st.info(f"Recommendation: Consider offering retention incentives, personalized offers, or loyalty programs.")
        else:
            st.success(f"✅ **LOW RISK: Customer is likely to STAY**")
            st.info(f"Recommendation: Maintain current engagement and monitor for any changes.")
            
        # Additional insights
        with st.expander("📈 Detailed Analysis"):
            st.write(f"**Threshold:** 0.5")
            st.write(f"**Raw Prediction Score:** {prediction[0][0]:.4f}")
            st.write(f"**Stay Probability:** {(1 - prediction_proba):.1%}")
            
            # Feature importance note
            st.write("---")
            st.caption("💡 *Tip: High balance, low activity, and multiple products often correlate with churn risk.*")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please check if your feature columns match the training data.")

# Add sidebar with instructions
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.write("1. Fill in all customer details")
    st.write("2. Click 'Predict Churn' button")
    st.write("3. View prediction results")
    st.write("---")
    st.header("📊 Model Info")
    st.write("Model: Artificial Neural Network")
    st.write("Threshold: 0.5")
    st.write("Accuracy: ~85% (example)")