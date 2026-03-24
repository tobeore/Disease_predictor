# Neglected Tropical Disease (NTD) & Arbovirus Predictor
📌 Project Overview
The NTD Predictor is a machine learning-based web application designed to predict Neglected Tropical Diseases and other vector-borne illnesses (such as Malaria, Dengue, Chikungunya, Yellow Fever, Plague, and West Nile fever) based on a patient's symptoms.

The project encompasses a full data science pipeline: from analyzing real-world epidemiological data and handling class imbalances, to training various classification models. The best-performing models (Support Vector Classifier and Multi-Layer Perceptron) were saved, and the SVC is currently deployed using a Flask web framework. Upon prediction, the application provides immediate treatment steps and healthcare recommendations.

✨ Key Features
Symptom-Based Diagnosis: Accepts input for over 60 distinct symptoms (e.g., sudden fever, yellow eyes, joint pain, rash) to predict the likely illness.

Web Interface: A user-friendly HTML/CSS form served via Flask for easy patient data entry.

Treatment Guidance: Automatically maps the predicted disease to a set of recommended healthcare actions.

Robust Machine Learning Backend: Utilizes a trained Support Vector Classifier (SVC) for inference, with an alternative Neural Network (MLP) available.

Epidemiological Analysis: Includes real-world arbovirus notification data from Brazil for geographical and demographic analysis.

🛠️ Tech Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, Imbalanced-learn (SMOTE), XGBoost, CatBoost

Data Manipulation & Analysis: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Model Serialization: Joblib, Pickle

📂 Repository Structure
NTD_prediction.py: The main Flask application script. It handles routing, loads the trained model, parses form inputs, and renders the predictions and treatments.

Support_Vector: The serialized Support Vector Machine (SVC) model used for inference by the web app.

best_mlp_model.pkl: A serialized Multi-Layer Perceptron (Neural Network) classifier, saved as a high-performing alternative model.

CSC476.ipynb & Untitled.ipynb: Jupyter Notebooks used for the Data Science pipeline. They cover exploratory data analysis, SMOTE for handling class imbalances, and the training/evaluation of multiple models (Random Forest, Logistic Regression, Naive Bayes, etc.).

test.csv: The testing dataset containing patient symptom flags (1.0 for presence, 0.0 for absence) used to validate model accuracy.

all_arb_cid.csv: Epidemiological dataset containing real-world arbovirus notification records (e.g., Dengue, Chikungunya in Minas Gerais, Brazil).

templates/: (Required by Flask) Contains the HTML files (Diagnosis_form.html and result.html) for the web interface.

🚀 Installation & Setup
1. Clone the repository:

Bash
git clone <your-repository-url>
cd <your-repository-folder>
2. Install dependencies:
Ensure you have Python 3.x installed. Install the required Python libraries using pip:

Bash
pip install Flask scikit-learn pandas numpy joblib imbalanced-learn matplotlib seaborn
3. Run the Web App:
Start the Flask development server by running the main Python script:

Bash
python NTD_prediction.py
4. Access the Application:
Open your web browser and navigate to:

Plaintext
http://127.0.0.1:5000/
🧠 Model Training Details
The machine learning models were trained using scikit-learn. Because medical datasets often have unequal class distributions, data preprocessing included the Synthetic Minority Over-sampling Technique (SMOTE) to ensure the models learned to predict rare diseases just as well as common ones.

The final prediction pipeline takes a NumPy array of numerical symptom flags (derived from inputs matching the structure of test.csv) and outputs an index mapped to the predicted disease class.
