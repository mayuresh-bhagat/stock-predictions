import yfinance as  yf
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


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

    X_data = df.Close

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.40)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.40): int(len(df))])

    past_100_days = data_training.tail(sequence_length)
    final_df = pd.concat([past_100_days,data_testing], ignore_index=True)
    input_data = stock_prediction_scaler.fit_transform(final_df)

    # output_df = final_df
    # output_df["MA_100_Days"] = final_df.rolling(100).mean()
    # output_df["MA_200_Days"] = final_df.rolling(200).mean()

    x_test, y_test = [], []
    for i in range(sequence_length, input_data.shape[0]):
        x_test.append(input_data[i - sequence_length:i])
        y_test.append(input_data[i, 0])
    X_test, y_test = np.array(x_test), np.array(y_test)
    

    print(stock_prediction_scaler.inverse_transform(y_test.reshape(-1, 1))[-1])
    return X_test, y_test

X, y = preparing_data("AAPL")

# print(df.head())