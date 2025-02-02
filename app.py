from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
from flask_cors import CORS 
CORS(app)
# Load model and encoders (only loading, no training)
try:
    model = joblib.load("virtue_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    virtue_columns = [col for col in pd.read_excel("virtue_recommendation_dataset.xlsx").columns if "Recommendation" not in col]
    recommendation_columns = [col for col in pd.read_excel("virtue_recommendation_dataset.xlsx").columns if "Recommendation" in col]

except FileNotFoundError:
    print("Error: Model or encoders not found. Run train_model.py first.")
    exit()  # Exit the app if the model files are missing


@app.route('/predict', methods=['POST'])
def predict():
    # ... (your prediction logic as before) ...
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        input_df = pd.DataFrame([data], columns=virtue_columns)

        predictions = model.predict(input_df)

        decoded_predictions = {}
        for i, col in enumerate(recommendation_columns):
            le = label_encoders[col]
            decoded_predictions[col] = le.inverse_transform(predictions[:, i]).tolist()

        return jsonify(decoded_predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=5000)  # Set debug=False in production