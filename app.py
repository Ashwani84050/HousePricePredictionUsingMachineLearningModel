from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Ensure the model folder exists
MODEL_PATH = "model/model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Train the model first!")

# Load trained model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])

        return render_template('result.html', prediction_text=f'Estimated House Price: ${prediction[0] * 100000:.2f}')
    except Exception as e:
        return f"Error in prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
