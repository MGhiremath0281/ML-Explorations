from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        dependents = request.form['dependents']
        married = int(request.form['married'])
        self_employed = int(request.form['self_employed'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        credit_history = float(request.form['credit_history'])
        education = int(request.form['education'])
        gender_male = int(request.form['gender_male'])
        property_area_semiurban = int(request.form['property_area_semiurban'])
        property_area_urban = int(request.form['property_area_urban'])
        total_income = float(request.form['total_income'])

        # Prepare the input for prediction
        input_data = np.array([[dependents, married, self_employed, loan_amount, loan_term, 
                                credit_history, education, gender_male, 
                                property_area_semiurban, property_area_urban, total_income]])

        # Scale the numerical features
        input_data[:, [3, 4, 5, 10]] = scaler.transform(input_data[:, [3, 4, 5, 10]])

        # Make a prediction
        prediction = svm_model.predict(input_data)

        # Convert prediction to human-readable output
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
