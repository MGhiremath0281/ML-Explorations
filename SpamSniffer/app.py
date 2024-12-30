import pickle
import numpy as np
import re
from datetime import datetime
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load saved models and vectorizer
with open('SVM_kernal.pkl', 'rb') as file:
    loaded_svm_rbf = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Label encoder for time of day
time_of_day_encoder = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    subject = request.form['subject']
    body = request.form['body']

    # Feature Extraction
    subject_length = len(subject)
    body_length = len(body)
    num_of_links = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body))
    
    spam_keywords = ['gift card', 'claim', 'urgent', 'prize', 'promotion']
    spam_keywords_count = sum([body.lower().count(keyword) for keyword in spam_keywords])
    
    contains_suspicious_words = any(word in body.lower() for word in ['click here', 'unsubscribed', 'survey', 'limited time'])
    sender_reputation_score = 50  # Placeholder value
    
    time_of_day = 'Afternoon' if 12 <= datetime.now().hour < 18 else 'Morning'
    time_of_day_encoded = time_of_day_encoder.fit_transform([time_of_day])[0]  # Ensure same encoding as training
    
    email_domain = 'gmail.com'  # Placeholder
    recipient_address_count = 1  # Placeholder
    unsubscribe_link_present = 1 if 'Unsubscribe Link' in body else 0

    # Transform the body text using the loaded TF-IDF vectorizer
    textual_features = vectorizer.transform([body]).toarray().flatten()

    # Ensure that the number of textual features is consistent
    max_text_features = 3  # Adjust based on your trained model's expected input
    textual_features = textual_features[:max_text_features]

    # Combine all features into a single array
    features = np.array([subject_length, body_length, num_of_links, spam_keywords_count, 
                         contains_suspicious_words, sender_reputation_score, time_of_day_encoded,
                         recipient_address_count, unsubscribe_link_present] + list(textual_features))

    # Ensure the features have the correct shape
    if features.shape[0] != 12:
        return "Error: Incorrect number of features."

    # Debugging: Print features before scaling
    print("Features before scaling:", features)

    # Reshape the feature array to fit the model (1 sample, n features)
    features = features.reshape(1, -1)

    # Scale the features using the saved scaler
    scaled_features = loaded_scaler.transform(features)

    # Debugging: Print scaled features
    print("Scaled features:", scaled_features)

    # Predict using the trained model
    prediction = loaded_svm_rbf.predict(scaled_features)

    # Debugging: Print prediction
    print("Prediction:", prediction)

    # Output the result
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
