from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            # Convert string values to float
            data = {k: float(v) for k, v in data.items()}
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Scale the features
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probability': float(np.max(probability[0])),
            'class': 'Fraud' if prediction[0] == 1 else 'Not Fraud'
        }
        
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', result=result)
    
    except Exception as e:
        error_msg = str(e)
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        else:
            return render_template('error.html', error=error_msg)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
