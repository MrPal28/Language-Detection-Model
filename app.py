from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and CountVectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    cv = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        data = cv.transform([text]).toarray()
        
        # Get the predicted language and its probability
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        accuracy = np.max(probabilities) * 100  # Accuracy as percentage

        return render_template('result.html', language=prediction, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
