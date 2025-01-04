from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        month = int(request.form['month'])
        location = request.form['location']
        weather = request.form['weather']
        hour = int(request.form['hour'])
        
        # Create DataFrame for the new data
        new_data = pd.DataFrame([[month, location, weather, hour]], columns=['Month', 'Location', 'Weather', 'Hour'])

        # One-Hot Encoding for 'Location' and 'Weather'
        new_data_encoded = pd.get_dummies(new_data, columns=['Location', 'Weather'], drop_first=True)

        # Reindex the encoded DataFrame to match the columns of the training data
        new_data_encoded = new_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        # Scale the encoded data
        new_data_scaled = scaler.transform(new_data_encoded)

        # Make the prediction
        prediction = model.predict(new_data_scaled)
        prediction_result = prediction[0]
        
        return render_template('index.html', prediction_text=f'Predicted Traffic Volume: {prediction_result}')
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
