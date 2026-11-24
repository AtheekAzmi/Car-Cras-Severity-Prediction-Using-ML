# Car Crash Severity Prediction Using ML

A Flask web app that predicts the **severity of a car crash** (Minor, Severe or Fatal) from crash, vehicle, driver and environmental details. Predictions come from a trained **CatBoost** classifier, and the app shows the predicted class together with a probability for each severity level.

## Features

- Web form with 18 input features covering the crash, the vehicle, the driver and the conditions
- **Autofill** button that fills the form with random valid sample data, for quick testing
- Predicted severity shown as a colour-coded badge, with a confidence score
- Probability bars for every severity class
- Step-by-step logging of each request in the server console (raw input, encoding, scaling, prediction)

## Tech Stack

| Layer | Tools |
| --- | --- |
| Backend | Python, Flask |
| Model | CatBoost (served); an ANN (TensorFlow/Keras) model is also included |
| Preprocessing | scikit-learn (label encoders, scaler), pandas, NumPy, joblib |
| Frontend | HTML, CSS, vanilla JavaScript (`templates/index.html`) |

## Project Structure

```
├── app.py                 # Flask app: loads the model and serves the UI and API
├── templates/
│   └── index.html         # Prediction form and results UI
├── catboost_model.cbm     # Trained CatBoost classifier (used for predictions)
├── ann_model.h5           # Trained neural network model (not used by app.py)
├── scaler.pkl             # Feature scaler
├── feature_encoders.pkl   # Label encoders for the categorical features
├── label_encoder.pkl      # Encoder for the target severity classes
└── requirement.txt        # Python dependencies
```

## Input Features

| Feature | Type / Values |
| --- | --- |
| Crash Speed (km/h) | Number |
| Impact Angle (degrees) | Number |
| Airbag Deployed | Yes, No |
| Seatbelt Used | Yes, No |
| Weather Conditions | Clear, Rain, Fog, Snow |
| Road Conditions | Dry, Wet, Icy |
| Crash Type | Head-on, Rear-end, Side impact, Rollover |
| Vehicle Type | Sedan, SUV, Truck, Motorcycle |
| Vehicle Age (years) | Number |
| Brake Condition | Good, Worn out |
| Tire Condition | Good, Worn out |
| Driver Age | Number |
| Driver Experience (years) | Number |
| Alcohol Level (BAC%) | Number (e.g. 0.05) |
| Distraction Level | None, Phone, Drowsiness, Other |
| Time of Day | Morning, Afternoon, Night |
| Traffic Density | Low, Medium, High |
| Visibility Distance (m) | Number |

**Target:** crash severity, one of Minor, Severe or Fatal.

## How It Works

1. The form sends the 18 feature values as JSON to `/predict`.
2. Categorical features are converted with the saved label encoders.
3. All features are scaled with the saved scaler.
4. The CatBoost model predicts the class and the class probabilities.
5. The predicted class is decoded back to its label (Minor / Severe / Fatal) and returned with the confidence and probability distribution.

## Getting Started

### Prerequisites

- Python 3.9 – 3.11 (TensorFlow 2.15 does not support newer versions)
- pip

### Installation

```bash
git clone https://github.com/AtheekAzmi/Car-Cras-Severity-Prediction-Using-ML.git
cd Car-Cras-Severity-Prediction-Using-ML

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirement.txt
```

### Run the App

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Click **Autofill** to load sample values, or enter your own, then click **Predict**.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Prediction form UI |
| POST | `/predict` | Returns the predicted severity, confidence and class probabilities |
| GET | `/get_sample_data` | Returns random valid sample input for autofill |

### Example request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Crash Speed (km/h)": 120,
    "Impact Angle (degrees)": 45,
    "Airbag Deployed": "Yes",
    "Seatbelt Used": "No",
    "Weather Conditions": "Rain",
    "Road Conditions": "Wet",
    "Crash Type": "Head-on",
    "Vehicle Type": "Sedan",
    "Vehicle Age (years)": 8,
    "Brake Condition": "Worn out",
    "Tire Condition": "Good",
    "Driver Age": 30,
    "Driver Experience (years)": 10,
    "Alcohol Level (BAC%)": 0.08,
    "Distraction Level": "Phone",
    "Time of Day": "Night",
    "Traffic Density": "High",
    "Visibility Distance (m)": 100
  }'
```

### Example response

```json
{
  "success": true,
  "prediction": "Severe",
  "confidence": 0.72,
  "probabilities": { "Fatal": 0.18, "Minor": 0.10, "Severe": 0.72 }
}
```

*(Values above are illustrative.)*

## Disclaimer

This project was built for academic and learning purposes. Its predictions should not be used for real-world safety, insurance or legal decisions.

## Author

**Atheek Azmi** – [GitHub](https://github.com/AtheekAzmi)
