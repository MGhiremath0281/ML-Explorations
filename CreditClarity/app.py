from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle  # Use pickle to load the model

app = Flask(__name__)

# Load your trained model (make sure the model is available at this path)
with open('grid_search_model.pkl', 'rb') as model_file:
    grid_search_model = pickle.load(model_file)  # Load model using pickle

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = {
            'Dependents': request.form['Dependents'],
            'Married': int(request.form['Married']),
            'Self_Employed': int(request.form['Self_Employed']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': int(request.form['Credit_History']),
            'Education': int(request.form['Education']),
            'Gender_Male': int(request.form['Gender_Male']),
            'Property_Area_Semiurban': int(request.form['Property_Area_Semiurban']),
            'Property_Area_Urban': int(request.form['Property_Area_Urban']),
            'Total_Income': float(request.form['Total_Income'])
        }

        # Convert data to DataFrame
        new_data = pd.DataFrame([data])

        # Encoding and scaling (same as in your example)
        label_encoder = LabelEncoder()
        new_data['Dependents'] = label_encoder.fit_transform(new_data['Dependents'])

        scaler = StandardScaler()
        new_data[['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Total_Income']] = scaler.fit_transform(
            new_data[['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Total_Income']]
        )

        # Predict using the trained model
        prediction = grid_search_model.predict(new_data)  # Use the model loaded with pickle

        # Return prediction result
        result = "approved" if prediction[0] == 1 else "rejected"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
