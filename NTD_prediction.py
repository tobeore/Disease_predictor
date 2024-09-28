from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('Support_Vector')

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
    # List of symptoms (features)
    symptoms = ['urination_loss', 'yellow_eyes', 'slow_heart_rate', 'loss_of_appetite',
       'yellow_skin', 'light_sensitivity', 'abdominal_pain', 'weakness',
       'back_pain', 'coma', 'red_eyes', 'paralysis', 'inflammation',
       'neck_pain', 'jaundice', 'irritability', 'digestion_trouble',
       'toenail_loss', 'weight_loss', 'stomach_pain', 'gum_bleed', 'diziness',
       'microcephaly', 'tremor', 'facial_distortion', 'skin_lesions',
       'lymph_swells', 'stiff_neck', 'myalgia', 'orbital_pain', 'ulcers',
       'confusion', 'itchiness', 'swelling', 'hyperpyrexia', 'fatigue',
       'gastro_bleeding', 'pleural_effusion', 'ascites', 'hypotension',
       'prostraction', 'bitter_tongue', 'breathing_restriction',
       'finger_inflammation', 'convulsion', 'anemia', 'cocacola_urine']

    return render_template('Diagnosis_form.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mapping of numerical predictions to the diseases name
        disease_mapping = {
            0: 'West Nile fever',
            1: 'Rift Valley fever',
            2: 'Tungiasis',
            3: 'Chikungunya',
            4: 'Dengue',
            5: 'Yellow Fever',
            6: 'Plague',
            7: 'Malaria',
            
        }
        # Convert form inputs to float )
        form_data = [float(x) for x in request.form.values()]
        
        # Reshape the data for the model
        data = np.array(form_data).reshape(1, -1)
        data1 = np.array()

        # Predict the disease using the model
        prediction = model.predict(data)[0]
        print(prediction)
        
      # Map the index to the disease name
        Diagnosis_name = disease_mapping.get(prediction, "Unknown Disease")

        # Get treatment steps for the predicted disease
        steps = treatment_steps.get(Diagnosis_name, ["Consult a healthcare professional for more information."])
        
        # Render the result page with the prediction and treatment steps
        return render_template('result.html', prediction= Diagnosis_name, steps=steps)
    
    except ValueError:
        # Handle any conversion errors
        return "Invalid input: There was an error processing your form data.", 400


if __name__ == '__main__':
    app.run(debug=True)
