import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model and handle scaler
try:
    import os
    from sklearn.preprocessing import StandardScaler
    
    # Define model paths
    model_path = MODELS_DIR / 'random_forest_model.pkl'
    scaler_path = MODELS_DIR / 'scaler.pkl'
    feature_order_path = MODELS_DIR / 'feature_order.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.sidebar.error("Error: Model file not found!")
        st.stop()
    
    # Load model
    model = joblib.load(model_path)
    
    # Try to load feature order from saved file or model
    if os.path.exists(feature_order_path):
        feature_order = joblib.load(feature_order_path)
        st.sidebar.success("Loaded feature order from file")
    elif hasattr(model, 'feature_names_in_'):
        feature_order = list(model.feature_names_in_)
        # Save the feature order for future use
        os.makedirs('models', exist_ok=True)
        joblib.dump(feature_order, feature_order_path)
    else:
        # Fallback to default order if no feature names available
        feature_order = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        st.sidebar.warning("Using default feature order")
    
    # Handle scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        st.sidebar.success("Model and scaler loaded successfully!")
    else:
        st.sidebar.warning("Scaler not found. Creating a new scaler...")
        # Create and save a new scaler with the correct feature order
        df = pd.read_csv(DATA_DIR / 'creditcard.csv')
        
        # Ensure we have all required columns
        missing_cols = set(feature_order) - set(df.columns)
        if missing_cols:
            st.sidebar.error(f"Missing columns in dataset: {missing_cols}")
            st.stop()
            
        X = df[feature_order]  # Use the exact feature order
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save the scaler and feature order
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_order, feature_order_path)
        st.sidebar.success("New scaler and feature order saved!")
    
    # Store feature order in session state
    st.session_state.feature_order = feature_order
    st.sidebar.write("Feature order:", feature_order)
    
except Exception as e:
    st.sidebar.error(f"Error: {str(e)}")
    import traceback
    st.sidebar.text(traceback.format_exc())
    st.stop()

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.write("""
This app detects potential credit card fraud using machine learning. 
Enter the transaction details below to check if it's potentially fraudulent.
""")

# Create input fields with better time handling
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", 
                           min_value=0.0, 
                           value=100.0, 
                           step=0.01, 
                           format="%.2f")
    
with col2:
    # Add a date and time picker
    transaction_time = st.time_input("Transaction Time")
    # Convert time to seconds since midnight
    time_seconds = (transaction_time.hour * 3600 + 
                   transaction_time.minute * 60 + 
                   transaction_time.second)
    
    # Add a date picker
    transaction_date = st.date_input("Transaction Date")
    
    # Calculate time in seconds since first transaction
    # For demo purposes, we'll use the current date as reference
    # In a real app, you'd want to use the actual first transaction date
    first_transaction_date = pd.Timestamp('2023-01-01')  # Example reference date
    days_since_first = (pd.Timestamp(transaction_date) - first_transaction_date).days
    time = (days_since_first * 24 * 3600) + time_seconds

# Generate features based on amount and time
def generate_features(amount, time):
    try:
        # Load a sample of real fraud transactions
        df = pd.read_csv('creditcard.csv')
        fraud_samples = df[df['Class'] == 1].sample(1).iloc[0]
        
        features = {}
        
        # Add V1-V28 features based on real fraud patterns
        for i in range(1, 29):
            v_name = f'V{i}'
            base_value = float(fraud_samples[v_name])
            
            # Adjust features based on amount and time
            amount_factor = np.log1p(amount) / 10
            time_factor = np.log1p(time) / 10000  # Scale time effect
            
            # Apply transformations
            features[v_name] = base_value * (1 + 0.1 * amount_factor + 0.05 * time_factor)
            features[v_name] += np.random.normal(0, 0.1)  # Add small noise
        
        # Add normalized amount and time
        features['Amount'] = float(amount)
        features['Time'] = float(time)
            
        return features
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None

# Prediction function
def predict_fraud(features):
    try:
        # Get the feature order from session state
        feature_order = st.session_state.get('feature_order')
        if feature_order is None:
            st.error("Feature order not found. Please restart the app.")
            return None, None
        
        # Create a dictionary with all features set to 0.0
        row = {col: 0.0 for col in feature_order}
        
        # Update with the features we have
        for key, value in features.items():
            if key in row:
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    row[key] = 0.0  # Default to 0 if conversion fails
        
        # Create DataFrame with exact feature order
        df = pd.DataFrame([row])
        
        # Ensure columns are in the correct order
        missing_cols = set(feature_order) - set(df.columns)
        if missing_cols:
            st.error(f"Missing columns in prediction: {missing_cols}")
            return None, None
            
        df = df[feature_order]
        
        # Debug: Show the features being used
        with st.sidebar.expander("Debug: Features being used"):
            st.json({k: round(v, 4) for k, v in row.items() if v != 0})
        
        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        proba = model.predict_proba(scaled_features)[0][1]  # Probability of being fraud
        
        return prediction, proba
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None

# Predict button
if st.button("Check for Fraud"):
    with st.spinner('Analyzing transaction...'):
        # Generate features with both amount and time
        features = generate_features(amount, time)
        
        if features is not None:
            # Display transaction details
            st.subheader("📝 Transaction Details")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Amount", f"${amount:,.2f}")
                st.metric("Time", f"{transaction_date.strftime('%Y-%m-%d')} {transaction_time.strftime('%H:%M:%S')}")
            
            # Make prediction
            prediction, proba = predict_fraud(features)
            
            # Display result
            if prediction is not None:
                st.subheader("🔍 Analysis Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Transaction Amount", f"${amount:,.2f}")
                
                with col2:
                    if prediction == 1:
                        st.error(f"🚨 High Risk of Fraud ({proba*100:.1f}%)")
                        st.warning("This transaction appears to be potentially fraudulent. Please verify with additional checks.")
                    else:
                        st.success(f"✅ Low Risk of Fraud ({proba*100:.1f}%)")
                        st.info("This transaction appears to be legitimate based on our analysis.")
                
                # Show feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Top Contributing Factors")
                    feature_importance = pd.DataFrame({
                        'Feature': [f'V{i}' for i in range(1, 29)],
                        'Importance': model.feature_importances_
                    })
                    st.bar_chart(feature_importance.set_index('Feature').sort_values('Importance', ascending=False).head(5))

# Add some space at the bottom
st.markdown("---")
st.info("ℹ️ Note: This is a demonstration application. Always verify suspicious transactions through official channels.")
