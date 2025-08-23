from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)

MODEL_PATH = 'dem_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")

artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']
print("User-trained model and scaler loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("\n--- PREDICT REQUEST RECEIVED ---")
        print(f"1. Incoming JSON data: {data}")
        print(f"2. Expected feature order: {feature_names}")

        feature_values = []
        for name in feature_names:
            value = data.get(name)
            if value is None:
                raise ValueError(f"Missing required feature in request: {name}")
            feature_values.append(value)

        print(f"3. Ordered feature values: {feature_values}")
        features_array = np.array(feature_values).reshape(1, -1)
        if pd.isna(features_array).any():
            raise ValueError("Data contains NaN values before scaling.")

        print(f"4. Array before scaling: {features_array}")
        features_scaled = scaler.transform(features_array)
        print(f"5. Array after scaling: {features_scaled}")

        decomposed_preds = model.predict_decomposed(features_scaled)
        print("6. Prediction successful.")
        
        response = {
            'baseline_prediction': decomposed_preds['baseline_prediction'].tolist(),
            'explanation_adjustment': decomposed_preds['explanation_adjustment'].tolist(),
            'final_prediction': decomposed_preds['final_prediction'].tolist()
        }
        return jsonify(response)
        
    except Exception as e:
        print("\n--- AN ERROR OCCURRED IN /predict ---")
        traceback.print_exc()
        print("-------------------------------------\n")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)