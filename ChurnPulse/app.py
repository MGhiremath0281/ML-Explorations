from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model, scaler, and LabelEncoder from .pkl files
with open('logistic.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('logistic_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('encoder.pkl', 'rb') as encoder_file:  # Load the saved encoders
    encoders = pickle.load(encoder_file)  # This should be a dictionary of encoders for each column

# Expected list of features (19 total)
expected_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                     'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                     'MonthlyCharges', 'TotalCharges']

# Function to preprocess input data
def preprocess_data(new_data):
    # List of categorical columns
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                           'PaymentMethod']

    # Ensure all expected columns are in the input data (fill missing ones with defaults)
    for col in expected_features:
        if col not in new_data.columns:
            if col in categorical_columns:
                new_data[col] = -1  # Default value for categorical features
            else:
                new_data[col] = 0  # Default value for numerical features

    # Apply LabelEncoding for categorical columns
    for col in categorical_columns:
        if col in encoders:  # Ensure that the encoder exists for the column
            encoder = encoders[col]
            try:
                new_data[col] = encoder.transform(new_data[col])  # Encode the column
            except ValueError as e:
                # If there are unseen labels, replace them with -1 (or any default value)
                print(f"Warning: Unseen labels in {col}. Replacing with default value (-1).")
                new_data[col] = -1

    # List of numerical columns to scale
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Apply scaling to numerical columns
    new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])

    return new_data

# Route for displaying the HTML form
@app.route('/', methods=['GET'])
def form():
    return render_template('form.html')

# Route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        data = request.form.to_dict()

        # Convert the form data into a pandas DataFrame
        new_data = pd.DataFrame([data])

        # Ensure the input data has all necessary columns (even if some are missing, fill with defaults)
        new_data_processed = preprocess_data(new_data)

        # Ensure there are 19 features (check before prediction)
        if new_data_processed.shape[1] != 19:
            return jsonify({'error': f'Expected 19 features, but got {new_data_processed.shape[1]} features.'}), 400

        # Make a prediction using the model
        prediction = model.predict(new_data_processed)

        # Return the prediction result
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        return render_template('form.html', result=result)
    
    except Exception as e:
        # If any error occurs, return it in the response
        return jsonify({'error': str(e)}), 400

# Run the Flask app (default port is 5000, can be changed if needed)
if __name__ == '__main__':
    app.run(debug=True, port=5000)
