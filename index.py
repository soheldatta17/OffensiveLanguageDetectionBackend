from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)


MODEL_PATH = "lstm_text_classification_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_LENGTH = 25  

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer loaded successfully!")


@app.route('/', methods=['GET'])
def home():
    return "Offensive Language Detector Backend is Running."


@app.route('/check', methods=['GET','POST'])
def check_sentence():
    try:
        
        data = request.get_json()
        sentences = data.get("sentences", [])

        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400

        
        tokenized_sentences = tokenizer.texts_to_sequences(sentences)
        padded_sentences = pad_sequences(tokenized_sentences, maxlen=MAX_LENGTH)

        
        predictions = model.predict(padded_sentences)

        
        results = []
        for i, sentence in enumerate(sentences):
            predicted_class = int(np.argmax(predictions[i]))
            probabilities = predictions[i].tolist()
            print(probabilities)
            if max(probabilities) == probabilities[0]:
                results.append(f'The sentence "{sentence}" is Non-Abusive')
            elif max(probabilities) == probabilities[1]:
                results.append(f'The sentence "{sentence}" is Abusive')
            else:
                results.append(f'The sentence "{sentence}" is Unable to Determine')

        
        return results

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask Server for Offensive Language Detector")
    app.run(host='0.0.0.0', port=10000)
