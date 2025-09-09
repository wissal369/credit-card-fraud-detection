from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import random
import logging
from datetime import datetime

model = joblib.load("PROJECTS/models/random_forest_model.pkl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Function to generate synthetic V1-V28 features based on amount and time
def generate_features(amount, time_value):
    # Load a sample of known fraud patterns
    try:
        # Try to load the dataset to get real fraud patterns
        df = pd.read_csv('creditcard.csv')
        fraud_samples = df[df['Class'] == 1].sample(1).iloc[0]
        
        # Create features based on real fraud patterns, adjusted by the input amount
        features = {}
        for i in range(1, 29):
            # Use the real fraud pattern but scale it by the input amount
            v_name = f'V{i}'
            base_value = fraud_samples[v_name]
            
            # Adjust based on the amount (frauds often have specific amount patterns)
            amount_factor = np.log1p(amount) / 10  # Scale the amount effect
            features[v_name] = base_value * (1 + 0.1 * amount_factor)
            
            # Add small random noise to make it less deterministic
            features[v_name] += np.random.normal(0, 0.1)
            
        return features
        
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        # Fallback to simple generation if there's an error
        features = {}
        for i in range(1, 29):
            base_value = np.log1p(amount) * (1 + np.sin(time_value / (60*60*24) * 2 * np.pi))
            noise = np.random.normal(0, 0.5)
            features[f'V{i}'] = base_value * (0.8 + 0.4 * i/28) + noise
        return features

# Load the model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    # Generate a default time (current timestamp)
    default_time = int(datetime.now().timestamp())
    return render_template('index.html', default_time=default_time)

# Load fraud examples at startup - using the same data as in the notebook
fraud_examples = []
try:
    # Load the data with the same parameters as in the notebook
    df = pd.read_csv('creditcard.csv')
    
    # Get the first 3 fraud cases (same as shown in the notebook)
    fraud_df = df[df['Class'] == 1].head(3)
    
    # Convert to list of dictionaries, ensuring we keep all features
    fraud_examples = fraud_df.to_dict('records')
    
    logger.info(f"Loaded {len(fraud_examples)} fraud examples from the dataset")
    logger.info(f"Example 1 Amount: {fraud_examples[0]['Amount']}")
    logger.info(f"Example 2 Amount: {fraud_examples[1]['Amount']}")
    logger.info(f"Example 3 Amount: {fraud_examples[2]['Amount']}")
    
except Exception as e:
    logger.error(f"Error loading fraud examples: {str(e)}")
    logger.exception("Full error details:")

@app.route('/examples')
def show_examples():
    return render_template('examples.html')

@app.route('/predict_fraud_example', methods=['POST'])
def predict_fraud_example():
    try:
        example_id = int(request.form.get('example_id', 1)) - 1  # Convert to 0-based index
        if example_id < 0 or example_id >= len(fraud_examples):
            example_id = 0  # Default to first example if invalid
            
        example = fraud_examples[example_id]
        
        # Prepare the input data
        input_data = {k: v for k, v in example.items() if k not in ['Class']}
        input_df = pd.DataFrame([input_data])
        
        # Get the feature columns in the right order
        feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_df = input_df[feature_columns]
        
        # Scale the features
        scaled_data = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probability': float(np.max(probability[0])),
            'class': 'Fraud' if prediction[0] == 1 else 'Not Fraud',
            'amount': example['Amount'],
            'time': datetime.fromtimestamp(example['Time']).strftime('%Y-%m-%d %H:%M:%S'),
            'is_example': True,
            'example_id': example_id + 1
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        error_msg = f"Error processing fraud example: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return render_template('error.html', error=error_msg)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Starting prediction process")
        
        # Get basic transaction data
        try:
            amount = float(request.form.get('amount', 0))
            logger.info(f"Amount: {amount}")
        except ValueError:
            raise ValueError("Invalid amount provided")
        
        # Parse the datetime input
        try:
            datetime_str = request.form.get('transaction_datetime')
            if datetime_str:
                logger.info(f"Received datetime string: {datetime_str}")
                # Convert from format: YYYY-MM-DDThh:mm
                transaction_dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M')
                # Convert to timestamp (seconds since epoch)
                time_value = (transaction_dt - datetime(1970, 1, 1)).total_seconds()
                logger.info(f"Converted timestamp: {time_value}")
            else:
                time_value = datetime.now().timestamp()
                logger.info(f"Using current timestamp: {time_value}")
        except Exception as e:
            raise ValueError(f"Invalid datetime format. Please use YYYY-MM-DDThh:mm format. Error: {str(e)}")
        
        # Generate synthetic V1-V28 features
        try:
            synthetic_features = generate_features(amount, time_value)
            logger.info("Generated synthetic features")
        except Exception as e:
            raise ValueError(f"Error generating features: {str(e)}")
        
        # Create input data with all required features
        input_data = {
            'Time': time_value,
            'Amount': amount,
            **{f'V{i}': synthetic_features[f'V{i}'] for i in range(1, 29)}
        }
        
        logger.info("Created input data dictionary")
        
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Get the feature names in the order expected by the model
            feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            input_df = input_df[feature_columns]
            
            logger.info("Created input DataFrame with correct feature order")
            
            # Scale the features
            scaled_data = scaler.transform(input_df)
            logger.info("Successfully scaled input features")
            
            # Make prediction
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)
            
            logger.info(f"Prediction: {prediction[0]}, Probability: {np.max(probability[0])}")
            
            # Prepare response
            result = {
                'prediction': int(prediction[0]),
                'probability': float(np.max(probability[0])),
                'class': 'Fraud' if prediction[0] == 1 else 'Not Fraud',
                'amount': amount,
                'time': datetime.fromtimestamp(time_value).strftime('%Y-%m-%d %H:%M:%S') if time_value > 0 else 'Current Time',
                'features': {k: round(v, 4) for k, v in input_data.items() if k in ['Time', 'Amount']}
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Prediction error: {error_msg}", exc_info=True)
        if request.is_json:
            return jsonify({'error': error_msg, 'details': str(e)}), 400
        else:
            return render_template('error.html', error=error_msg, details=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
