import pickle
from flask import Flask, request, Response, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_preparation import preparing_data
import json


app = Flask(__name__, template_folder="templates")

try:
    stock_prediction_model = load_model('stock_price_prediction.h5')
except FileNotFoundError:
    print("Model file not found. Please ensure 'stock_price_prediction.keras' is in the correct directory.")
    stock_prediction_model = None

# stock_prediction_model = pickle.load(open('stock_price_prediction_model.pkl', "rb"))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['GET', 'POST'])
def predict_api():
    
    stock_id = "AAPL"
    if 'stock_id' in request.form.keys():
        stock_id = request.form['stock_id']

    X, y = preparing_data(stock_id)
    # Add prediction logic here using stock_prediction_model

    print(X.shape)
    return jsonify({"X": json.dumps(X)})




if __name__ == "__main__":
    app.run(debug=True)



