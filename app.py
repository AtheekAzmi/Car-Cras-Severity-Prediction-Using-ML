from flask import Flask, render_template, request, jsonify
from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

# Suppress all warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and preprocessing objects
print("="*70)
print("LOADING MODELS AND PREPROCESSORS")
print("="*70)
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')
print("✓ CatBoost model loaded")

scaler = joblib.load('scaler.pkl')
print("✓ Scaler loaded")

label_encoder = joblib.load('label_encoder.pkl')
print("✓ Label encoder loaded")

feature_encoders = joblib.load('feature_encoders.pkl')
print("✓ Feature encoders loaded")
print("="*70)

# Feature columns in correct order
FEATURE_COLUMNS = [
    'Crash Speed (km/h)', 'Impact Angle (degrees)', 'Airbag Deployed', 
    'Seatbelt Used', 'Weather Conditions', 'Road Conditions', 'Crash Type', 
    'Vehicle Type', 'Vehicle Age (years)', 'Brake Condition', 'Tire Condition', 
    'Driver Age', 'Driver Experience (years)', 'Alcohol Level (BAC%)', 
    'Distraction Level', 'Time of Day', 'Traffic Density', 'Visibility Distance (m)'
]

prediction_counter = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_counter
    prediction_counter += 1
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print(f"PREDICTION REQUEST #{prediction_counter}")
    print(f"Timestamp: {timestamp}")
    print("="*70)
    
    try:
        data = request.json
        
        print("\n[STEP 1] Received Input Data:")
        print("-" * 70)
        for col in FEATURE_COLUMNS:
            print(f"  {col:35s} : {data[col]}")
        
        # Create input list in correct order
        input_list = []
        for col in FEATURE_COLUMNS:
            input_list.append(data[col])
        
        # Create DataFrame
        input_df = pd.DataFrame([input_list], columns=FEATURE_COLUMNS)
        print("\n[STEP 2] Created DataFrame:")
        print("-" * 70)
        print(input_df.to_string(index=False))
        
        # Encode categorical features
        categorical_features = ['Airbag Deployed', 'Seatbelt Used', 'Weather Conditions', 
                               'Road Conditions', 'Crash Type', 'Vehicle Type', 
                               'Brake Condition', 'Tire Condition', 'Distraction Level', 
                               'Time of Day', 'Traffic Density']
        
        print("\n[STEP 3] Encoding Categorical Features:")
        print("-" * 70)
        for col in categorical_features:
            original_value = str(input_df.at[0, col])
            encoded_value = feature_encoders[col].transform([original_value])[0]
            input_df.at[0, col] = encoded_value
            print(f"  {col:25s} : '{original_value}' → {encoded_value}")
        
        # Convert all columns to numeric
        input_df = input_df.astype(float)
        print("\n[STEP 4] Converted to Numeric DataFrame:")
        print("-" * 70)
        print(input_df.to_string(index=False))
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        print("\n[STEP 5] Scaled Features (first 5 values):")
        print("-" * 70)
        print(f"  {input_scaled[0][:5]}")
        
        # Predict
        print("\n[STEP 6] Making Prediction...")
        print("-" * 70)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        print(f"  Raw Prediction (encoded): {prediction}")
        
        # Decode prediction
        severity = label_encoder.inverse_transform([int(prediction)])[0]
        
        print(f"  Decoded Prediction: {severity}")
        
        # Prepare probability distribution
        prob_dist = {}
        print("\n[STEP 7] Probability Distribution:")
        print("-" * 70)
        for i, class_name in enumerate(label_encoder.classes_):
            prob_dist[class_name] = float(probabilities[i])
            print(f"  {class_name:20s} : {probabilities[i]:.4f} ({probabilities[i]*100:.2f}%)")
        
        confidence = float(max(probabilities))
        
        print("\n[STEP 8] Final Result:")
        print("-" * 70)
        print(f"  Predicted Severity: {severity}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("="*70)
        print(f"✓ PREDICTION #{prediction_counter} COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'prediction': severity,
            'probabilities': prob_dist,
            'confidence': confidence
        })
        
    except Exception as e:
        import traceback
        print("\n" + "!"*70)
        print(f"ERROR IN PREDICTION #{prediction_counter}")
        print("!"*70)
        print(traceback.format_exc())
        print("!"*70 + "\n")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_sample_data', methods=['GET'])
def get_sample_data():
    """Generate random sample data for autofill"""
    import random
    
    # First, get the actual values from the encoders to ensure compatibility
    sample_data = {
        'Crash Speed (km/h)': random.randint(40, 200),
        'Impact Angle (degrees)': random.randint(10, 180),
        'Airbag Deployed': random.choice(list(feature_encoders['Airbag Deployed'].classes_)),
        'Seatbelt Used': random.choice(list(feature_encoders['Seatbelt Used'].classes_)),
        'Weather Conditions': random.choice(list(feature_encoders['Weather Conditions'].classes_)),
        'Road Conditions': random.choice(list(feature_encoders['Road Conditions'].classes_)),
        'Crash Type': random.choice(list(feature_encoders['Crash Type'].classes_)),
        'Vehicle Type': random.choice(list(feature_encoders['Vehicle Type'].classes_)),
        'Vehicle Age (years)': random.randint(0, 20),
        'Brake Condition': random.choice(list(feature_encoders['Brake Condition'].classes_)),
        'Tire Condition': random.choice(list(feature_encoders['Tire Condition'].classes_)),
        'Driver Age': random.randint(18, 80),
        'Driver Experience (years)': random.randint(0, 50),
        'Alcohol Level (BAC%)': round(random.uniform(0, 0.3), 3),
        'Distraction Level': random.choice(list(feature_encoders['Distraction Level'].classes_)),
        'Time of Day': random.choice(list(feature_encoders['Time of Day'].classes_)),
        'Traffic Density': random.choice(list(feature_encoders['Traffic Density'].classes_)),
        'Visibility Distance (m)': random.randint(20, 500)
    }
    
    print("\n" + "="*70)
    print("AUTOFILL SAMPLE DATA GENERATED")
    print("="*70)
    for key, value in sample_data.items():
        print(f"  {key:35s} : {value}")
    print("="*70 + "\n")
    
    return jsonify(sample_data)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FLASK APP STARTING")
    print("="*70)
    print("Server running at: http://127.0.0.1:5000")
    print("Press CTRL+C to quit")
    print("="*70 + "\n")
    app.run(debug=True, port=5000)