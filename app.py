import numpy as np
import pandas as pd
import pickle   
from flask import Flask, request, render_template

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model1.pkl", "rb"))

# Specify the feature names used during training
feature_names = ['industrial_risk', 'management_risk', 'financial_flexibility', 'credibility', 'competitiveness', 'operating_risk']

@app.route("/")
def Home():
    return render_template("index41.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract form values
    form_values = request.form.values()

    # Check for empty strings and convert to float
    float_features = [float(x) if x.strip() != '' else 0.0 for x in form_values]

    features = [np.array(float_features)]

    # Set feature names for prediction


    prediction = model.predict(features)

    # Determine the prediction text based on the prediction
    if prediction == 0:
        prediction_text = "The company is Going towards Bankruptcy."
    else:
        prediction_text = "The company is Going towards Non-Bankrupt."

    return render_template("index41.html",prediction_text = " {}".format(prediction_text))

if __name__ == "__main__":
    app.run(debug=True)