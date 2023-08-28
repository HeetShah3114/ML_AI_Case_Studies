from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the trained model and other necessary data
model = joblib.load("linear_regression_model.joblib")
# You need to save the model using joblib.dump(model, "linear_regression_model.joblib") before loading it here

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        years_experience = float(request.form['years_experience'])
        prediction = model.predict(np.array([[years_experience]]))[0]
        return render_template('index.html', prediction=prediction)
    except:
        return render_template('index.html', error="Invalid input. Please enter a valid number.")

if __name__ == '__main__':
    app.run(debug=True)
