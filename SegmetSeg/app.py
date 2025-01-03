from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)

# Load the pre-trained model and vectorizer (Replace these with your actual files)
model = pickle.load(open('model.pkl', 'rb'))  # Naive Bayes model
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))  # TfidfVectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the sentence from the form input
        sentence = request.form['sentence']
        
        # Vectorize the input sentence
        sentence_vectorized = vectorizer.transform([sentence])
        
        # Predict sentiment
        prediction = model.predict(sentence_vectorized)
        
        # Convert prediction to string (positive/negative)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return render_template('index.html', prediction_text=f"The sentiment of the input is: {sentiment}")

if __name__ == "__main__":
    app.run(debug=True)
