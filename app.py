from flask import Flask, render_template, request
import pickle
import numpy as np
from googletrans import Translator

app = Flask(__name__)

# Load the model and CountVectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# Initialize the Translator
translator = Translator()

# Function to translate text
def translate_text(text="type", src="English", dest="Hindi"):
    text_1 = text
    src_1 = src.lower().split('_', 1)[0]  # Get the language code for source
    dest_1 = dest.lower().split('_', 1)[0]  # Get the language code for destination
    
    # Translate the text
    translated = translator.translate(text_1, src=src_1, dest=dest_1)
    
    return translated.text

@app.route('/')
def home():
    languages = ['Estonian', 'Swedish', 'Thai', 'Tamil', 'Dutch', 'Japanese', 'Turkish', 'Latin', 'Urdu', 
                 'Indonesian', 'Portuguese', 'French', 'Chinese', 'Korean', 'Hindi', 'Spanish', 'Pushto', 
                 'Persian', 'Romanian', 'Russian', 'English', 'Arabic']

    return render_template('index.html', languages=languages)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        dest = request.form['language_option']  # Get the destination language selected by the user
        data = cv.transform([text]).toarray()
        
        # Get the predicted language and its probability
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        accuracy = np.max(probabilities) * 100  # Accuracy as percentage
        
        # Check if the confidence score is above 0.7
        confidence_score = np.max(probabilities)
        if confidence_score > 0.7:
            # Translate the text to the selected destination language
            translated_text = translate_text(text, src=prediction, dest=dest)
        else:
            # Detect the language if confidence is low
            detected = translator.detect(text)
            detected_lang = detected.lang
            translated_text = translate_text(text, src=detected_lang, dest=dest)

        # Render the result template with the prediction, accuracy, and translated text
        return render_template('result.html', language=prediction, accuracy=accuracy, translated_text=translated_text, dest=dest)


if __name__ == "__main__":
    app.run(debug=True)
