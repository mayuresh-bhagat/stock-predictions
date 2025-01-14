import pickle
from flask import Flask, request, Response, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
# from data_preparation import preparing_data
import yfinance as yf
import matplotlib.pyplot as plt
import os
import requests
from pathlib import Path


app = Flask(__name__, template_folder="templates")




IMAGES_PATH = Path() / "static"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    return path


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


def preparing_data(stock_id, start_data="2022-01-01", chart=False):
    """
    Prepares data for stock prediction and optionally generates a chart of the stock's closing price with EMA indicators.
    Parameters:
    stock_id (str): The stock identifier (ticker symbol) to download data for.
    start_data (str): The start date for downloading stock data in the format 'YYYY-MM-DD'. Default is "2022-01-01".
    chart (bool): If True, generates a chart of the stock's closing price with 100 and 200 days EMA. Default is False.
    Returns:
    tuple: A tuple containing:
        - X_test (numpy.ndarray): The input data for testing the model.
        - y_test (numpy.ndarray): The actual closing prices for testing the model.
        - ema_chart_path_100_200 (str or None): The file path to the generated EMA chart if `chart` is True, otherwise None.
        - date (pandas.Index): The dates corresponding to the last 10 entries in the downloaded stock data.
    Raises:
    ValueError: If the scaler is not loaded.
    """
    
    if stock_prediction_scaler is None:
        raise ValueError("Scaler is not loaded. Cannot proceed with data preparation.")
        
    
    # url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_id}&outputsize=full&apikey=YG8JN0I1UN1ARP9T'
    # r = requests.get(url, verify=False)
    # data = r.json()

    # df = pd.DataFrame(data['Time Series (Daily)']).T
    # df.columns = ['Open', 'High', 'Low', 'Close', 'Volumn']
    # df.sort_index(ascending=True, inplace=True)
    # df = yf.download(stock_id, start_data)

    # import yfinance as yf

    df = yf.Ticker('RELIANCE.NS')
    df = df.history(period="5y")
    
    date = df.tail(1).index


    ## EMA Chart of 100 & 200
    if chart :

        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(df.Close, 'y', label='Closing Price')
        plt.plot(ema100, 'g', label='EMA 100')
        plt.plot(ema200, 'r', label='EMA 200')
        plt.title("Closing Price vs Time (100 & 200 Days EMA)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        ema_chart_path_100_200 = save_fig("ema_100_200")
        
    else :
        ema_chart_path_100_200 = None

    sequence_length = 200

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.40)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.40): int(len(df))])

    past_200_days = data_training.tail(sequence_length)
    final_df = pd.concat([past_200_days,data_testing], ignore_index=True)
    input_data = stock_prediction_scaler.fit_transform(final_df)


    x_test, y_test = [], []
    for i in range(sequence_length, input_data.shape[0]):
        x_test.append(input_data[i - sequence_length:i])
        y_test.append(input_data[i, 0])
    X_test, y_test = np.array(x_test), np.array(y_test)
    

    # print(stock_prediction_scaler.inverse_transform(y_test.reshape(-1, 1))[-1])
    return X_test, y_test, ema_chart_path_100_200, date


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    if 'stock_id' in request.args.keys():
        stock_id = request.args['stock_id']


        ## preparing the data
        X, y, ema_chart_path_100_200, date = preparing_data(stock_id)
        
        ## Prediction
        prediction = stock_prediction_model.predict(X)

        ## Unscalling
        prediction_unscaled = stock_prediction_scaler.inverse_transform(prediction.reshape(-1, 1))
        y_unscaled = stock_prediction_scaler.inverse_transform(y.reshape(-1, 1))

        ## Mean Different between Acutal data and Predicted data
        mean_diff = np.mean(y_unscaled - prediction_unscaled)

        ## Output
        output = {
            "prediction": prediction_unscaled[-1][0].tolist(),
            "actual": y_unscaled[-1][0].tolist(),
            "mean_diff": mean_diff.tolist(),
            "new_prediction": float(prediction_unscaled[-1]) + float(mean_diff)
        }

        return jsonify(output)
    else :
        return jsonify({'message' : 'Please pass the stock id'})
    
    


@app.route('/predict', methods=['POST'])
def predict():
    
    stock_id = "AAPL"
    if 'stock_id' in request.form.keys():
        stock_id = request.form['stock_id']
        
    else :
        return jsonify({'message' : 'Please pass the stock id'})
    
    ## preparing the data
    X, y, ema_chart_path_100_200, date = preparing_data(stock_id, chart=True)
    
    
    ## Prediction
    prediction = stock_prediction_model.predict(X)

    ## Unscalling
    prediction_unscaled = stock_prediction_scaler.inverse_transform(prediction.reshape(-1, 1))
    y_unscaled = stock_prediction_scaler.inverse_transform(y.reshape(-1, 1))

    ## Actual vs Predicted Chart
    plt.figure(figsize=(12, 6))
    plt.plot(y_unscaled, 'r', label='Actual Price')
    plt.plot(prediction_unscaled, 'g', label='Predicted Price')
    plt.title("Acutal Vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    actual_predicted_path = save_fig("acutal_predicted")
    
    

    ## Mean Different between Acutal data and Predicted data
    mean_diff = np.mean(y_unscaled - prediction_unscaled)

    output = {
        "Stock" : stock_id,
        "prediction": prediction_unscaled[-1].tolist()[0],
        "actual": y_unscaled[-1].tolist()[0],
        "mean_diff": mean_diff.tolist(),
        "new_prediction (predicted + meand_diff)": float(prediction_unscaled[-1]) + float(mean_diff),
        "date" : date.tolist()[0]
    }

    print(output)
    df = pd.DataFrame([output]).T

    print(df)

    return render_template('index.html', 
                           df=df.to_html(classes="table table-bordered"),
                           stock_id = stock_id,
                           ema_chart_100_200=ema_chart_path_100_200, 
                           actual_predicted=actual_predicted_path
                           )


if __name__ == "__main__":
    app.run(debug=True)



