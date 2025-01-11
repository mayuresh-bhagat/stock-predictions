import pickle
from flask import Flask, request, Response, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from data_preparation import preparing_data
import yfinance as yf # type: ignore


app = Flask(__name__, template_folder="templates")


try:
    stock_prediction_model = pickle.load(open('stock_price_prediction_model.pkl', "rb"))
except FileNotFoundError:   
    print("Scaler file not found. Please ensure 'stock_price_prediction_model.pkl' is in the correct directory.")
    stock_prediction_model = None

try:
    stock_prediction_scaler = pickle.load(open('stock_price_prediction_scaler.pkl', "rb"))
except FileNotFoundError:
    print("Scaler file not found. Please ensure 'stock_price_prediction_scaler.pkl' is in the correct directory.")
    stock_prediction_scaler = None


def preparing_data(stock_id, start_data="2022-01-01"):
    """
    Prepares the data for stock prediction by downloading stock prices, splitting the data into training and testing sets,
    and scaling the data for model input.
    Args:
        stock_id (str): The stock ticker symbol.
        start_data (str, optional): The start date for downloading stock data. Defaults to "2022-01-01".
        end_data (datetime, optional): The end date for downloading stock data. Defaults to today's date.
    Returns:
        tuple: A tuple containing the input data (X) and the output data (y) as numpy arrays.
    """
    if stock_prediction_scaler is None:
        raise ValueError("Scaler is not loaded. Cannot proceed with data preparation.")
        
    df = yf.download(stock_id, start_data)

    
    sequence_length = 200

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.40)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.40): int(len(df))])

    past_200_days = data_training.tail(sequence_length)
    final_df = pd.concat([past_200_days,data_testing], ignore_index=True)
    input_data = stock_prediction_scaler.fit_transform(final_df)

    output_df = final_df
    output_df["MA_100_Days"] = final_df.rolling(100).mean()
    output_df["MA_200_Days"] = final_df.rolling(200).mean()

    x_test, y_test = [], []
    for i in range(sequence_length, input_data.shape[0]):
        x_test.append(input_data[i - sequence_length:i])
        y_test.append(input_data[i, 0])
    X_test, y_test = np.array(x_test), np.array(y_test)
    

    print(stock_prediction_scaler.inverse_transform(y_test.reshape(-1, 1))[-1])
    return X_test, y_test, output_df


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    stock_id = "AAPL"
    if 'stock_id' in request.args.keys():
        stock_id = request.args['stock_id']

    
    ## preparing the data
    X, y, df = preparing_data(stock_id)
    
    ## Prediction
    prediction = stock_prediction_model.predict(X)

    ## Unscalling
    prediction_unscaled = stock_prediction_scaler.inverse_transform(prediction.reshape(-1, 1))
    y_unscaled = stock_prediction_scaler.inverse_transform(y.reshape(-1, 1))

    ## Mean Different between Acutal data and Predicted data
    mean_diff = np.mean(y_unscaled - prediction_unscaled)

    ## Output
    output = {
        "prediction": prediction_unscaled[-1].tolist(),
        "actual": y_unscaled[-1].tolist(),
        "mean_diff": mean_diff.tolist(),
        "new_prediction": float(prediction_unscaled[-1]) + float(mean_diff)
    }

    return jsonify(output)




if __name__ == "__main__":
    app.run(debug=True)



