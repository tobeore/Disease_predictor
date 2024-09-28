from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Define steps for each diagnosis
treatment_steps = {
    'Malaria': [
        "Take antimalarial drugs such as artemisinin-based combination therapies (ACTs).",
        "Ensure you stay hydrated and get plenty of rest.",
        "Monitor your symptoms and seek medical attention if they worsen.",
        "Prevent mosquito bites using insect repellents and sleeping under mosquito nets."
    ],
    'Dengue': [
        "Stay hydrated by drinking plenty of fluids.",
        "Use pain relievers such as acetaminophen (avoid aspirin).",
        "Rest and avoid strenuous activity.",
        "Seek medical attention if you experience severe symptoms such as bleeding or breathing difficulties."
    ],
    'Chikungunya': [
        "Take nonsteroidal anti-inflammatory drugs (NSAIDs) to relieve pain.",
        "Rest and drink plenty of fluids.",
        "Apply cold compresses to painful joints.",
        "Prevent mosquito bites to stop the spread of the virus."
    ],
    # Add more prognoses and treatment steps here
}

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = [float(x) for x in request.form.values()]
    data = np.array(form_data).reshape(1, -1)

    prediction = model.predict(data)[0]

    # Get the treatment steps for the predicted disease
    steps = treatment_steps.get(prediction, ["Consult a healthcare professional for more information."])

    # Return the result page with the prediction and treatment steps
    return render_template('result.html', prediction=prediction, steps=steps)

if __name__ == '__main__':
    app.run(debug=True)
