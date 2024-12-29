from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model, label encoder, and scaler
model = pickle.load(open('model.pkl', 'rb'))  # Load the machine learning model
le = pickle.load(open('encoder.pkl', 'rb'))        # Load the label encoder
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler (e.g., StandardScaler)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Variable to store prediction result

    if request.method == 'POST':
        # Get form data
        features = [float(request.form[field]) for field in request.form]

        # Convert features to a numpy array
        input_data = np.array(features).reshape(1, -1)

        # Scale the input data using the scaler
        scaled_data = scaler.transform(input_data)

        # Predict the prognosis using the model
        prediction = model.predict(scaled_data)

        # Decode the prediction using the label encoder
        decoded_prediction = le.inverse_transform(prediction)
        prediction = decoded_prediction[0]  # Store the decoded prediction

    return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
