from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model pipeline
model_pipeline_path = "D:\Ml Exploration\ML-Explorations\DriveWise\model\model_pipeline.pkl"
with open(model_pipeline_path, "rb") as f:
    model_pipeline = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Ensure the expected fields are in the request
        required_fields = [
            'Car Make', 'Car Model', 'Mileage (in km)', 'Engine Size (L)', 'Fuel Type',
            'Year of Manufacture', 'Rental Duration (days)', 'Demand', 'Car Condition', 'Location'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Convert to DataFrame for the model
        input_df = pd.DataFrame([data])

        # Predict using the pipeline
        prediction = model_pipeline.predict(input_df)[0]

        # Return prediction as JSON response
        return jsonify({'Predicted Rental Price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
